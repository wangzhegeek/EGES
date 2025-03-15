import numpy as np
import networkx as nx
import torch
import time
import random
from collections import defaultdict, deque
import threading
import queue
from tqdm import tqdm


class StreamingWalker:
    """
    流式随机游走实现，支持增量图构建和在线更新
    """
    def __init__(self, p=0.25, q=2, window_size=5, max_graph_size=1000000, 
                 buffer_size=10000, update_interval=1000, device='cpu'):
        """
        初始化流式随机游走器
        
        参数:
        p: 返回参数，控制游走回到之前节点的概率
        q: 进出参数，控制游走探索新节点的概率
        window_size: 上下文窗口大小
        max_graph_size: 图的最大节点数，超过此数量将使用LRU策略移除旧节点
        buffer_size: 样本缓冲区大小
        update_interval: 图更新间隔（处理多少条数据后更新图）
        device: 计算设备
        """
        self.p = p
        self.q = q
        self.window_size = window_size
        self.max_graph_size = max_graph_size
        self.buffer_size = buffer_size
        self.update_interval = update_interval
        self.device = device
        
        # 初始化图
        self.G = nx.Graph()
        
        # 节点映射
        self.node_map = {}  # 原始ID到索引的映射
        self.reverse_node_map = {}  # 索引到原始ID的映射
        self.next_node_idx = 0
        
        # LRU缓存，用于跟踪节点访问时间
        self.node_last_access = {}
        
        # 样本缓冲区
        self.sample_buffer = deque(maxlen=buffer_size)
        
        # 会话缓冲区，用于临时存储会话
        self.session_buffer = []
        self.processed_count = 0
        
        # 线程安全的队列，用于异步处理
        self.session_queue = queue.Queue(maxsize=1000)
        self.sample_queue = queue.Queue(maxsize=buffer_size)
        
        # 控制标志
        self.is_running = False
        self.walker_thread = None
        self.processor_thread = None
    
    def _get_node_idx(self, node_id):
        """获取节点索引，如果不存在则创建"""
        if node_id not in self.node_map:
            self.node_map[node_id] = self.next_node_idx
            self.reverse_node_map[self.next_node_idx] = node_id
            self.next_node_idx += 1
        
        # 更新节点访问时间
        self.node_last_access[node_id] = time.time()
        
        return self.node_map[node_id]
    
    def _prune_graph(self):
        """
        使用LRU策略裁剪图，移除最近最少使用的节点
        """
        if len(self.G) <= self.max_graph_size:
            return
        
        # 计算需要移除的节点数量
        num_to_remove = len(self.G) - self.max_graph_size
        
        # 按最后访问时间排序
        nodes_by_access_time = sorted(
            self.node_last_access.items(), 
            key=lambda x: x[1]
        )
        
        # 移除最早访问的节点
        nodes_to_remove = [node for node, _ in nodes_by_access_time[:num_to_remove]]
        
        for node_id in nodes_to_remove:
            if node_id in self.node_map:
                node_idx = self.node_map[node_id]
                
                # 从图中移除节点
                if node_idx in self.G:
                    self.G.remove_node(node_idx)
                
                # 清理映射
                del self.reverse_node_map[node_idx]
                del self.node_map[node_id]
                del self.node_last_access[node_id]
    
    def process_session(self, session):
        """
        处理单个会话，更新图
        
        参数:
        session: 会话序列，包含多个商品ID
        """
        if len(session) <= 1:
            return
        
        # 将原始ID映射为索引
        mapped_session = [self._get_node_idx(item) for item in session]
        
        # 更新图
        for i in range(len(mapped_session) - 1):
            source = mapped_session[i]
            target = mapped_session[i + 1]
            
            # 添加边或更新边权重
            if self.G.has_edge(source, target):
                self.G[source][target]['weight'] += 1
            else:
                self.G.add_edge(source, target, weight=1)
        
        # 将会话添加到缓冲区
        self.session_buffer.append(mapped_session)
        self.processed_count += 1
        
        # 如果达到更新间隔，生成随机游走
        if self.processed_count >= self.update_interval:
            self._generate_walks_from_buffer()
            self.processed_count = 0
            
            # 裁剪图
            self._prune_graph()
    
    def _generate_walks_from_buffer(self):
        """从会话缓冲区生成随机游走"""
        if not self.session_buffer:
            return
        
        # 获取缓冲区中的所有节点
        nodes = set()
        for session in self.session_buffer:
            nodes.update(session)
        
        # 为每个节点生成随机游走
        walks = []
        for node in nodes:
            if node in self.G:
                # 生成一次随机游走
                walk = self._node2vec_walk(node, length=10)
                if len(walk) > 1:
                    walks.append(walk)
        
        # 生成上下文对
        pairs = self._generate_context_pairs(walks)
        
        # 将样本添加到缓冲区
        self.sample_buffer.extend(pairs)
        
        # 清空会话缓冲区
        self.session_buffer = []
    
    def _node2vec_walk(self, start_node, length):
        """
        从指定节点开始生成一次随机游走
        
        参数:
        start_node: 起始节点
        length: 游走长度
        
        返回:
        walk: 游走序列
        """
        G = self.G
        walk = [start_node]
        
        # 如果节点没有邻居，直接返回
        if len(list(G.neighbors(start_node))) == 0:
            return walk
        
        for _ in range(length - 1):
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            
            if len(cur_nbrs) == 0:
                break
            
            if len(walk) == 1:
                # 第一步随机选择邻居
                next_node = random.choice(cur_nbrs)
            else:
                # 计算转移概率
                prev = walk[-2]
                probs = []
                
                for nbr in cur_nbrs:
                    if nbr == prev:  # 返回到上一个节点
                        prob = G[cur][nbr].get('weight', 1) * (1.0 / self.p)
                    elif G.has_edge(nbr, prev):  # 是上一个节点的邻居
                        prob = G[cur][nbr].get('weight', 1)
                    else:  # 探索新节点
                        prob = G[cur][nbr].get('weight', 1) * (1.0 / self.q)
                    
                    probs.append(prob)
                
                # 归一化概率
                sum_probs = sum(probs)
                if sum_probs > 0:
                    probs = [p / sum_probs for p in probs]
                    next_node = random.choices(cur_nbrs, weights=probs, k=1)[0]
                else:
                    next_node = random.choice(cur_nbrs)
            
            walk.append(next_node)
        
        return walk
    
    def _generate_context_pairs(self, walks):
        """
        根据游走序列生成上下文对
        
        参数:
        walks: 随机游走序列列表
        
        返回:
        pairs: 上下文对列表
        """
        pairs = []
        for walk in walks:
            if len(walk) <= 1:
                continue
                
            for i in range(len(walk)):
                for j in range(max(0, i - self.window_size), min(len(walk), i + self.window_size + 1)):
                    if i != j:
                        # 将索引转换回原始ID
                        source = self.reverse_node_map[walk[i]]
                        target = self.reverse_node_map[walk[j]]
                        pairs.append((source, target))
        
        return pairs
    
    def get_samples(self, batch_size):
        """
        获取一批样本
        
        参数:
        batch_size: 批次大小
        
        返回:
        samples: 样本列表
        """
        if len(self.sample_buffer) < batch_size:
            # 如果缓冲区样本不足，生成更多样本
            self._generate_walks_from_buffer()
        
        # 如果仍然不足，返回所有可用样本
        if len(self.sample_buffer) < batch_size:
            samples = list(self.sample_buffer)
            self.sample_buffer.clear()
            return samples
        
        # 从缓冲区中随机抽取样本
        indices = random.sample(range(len(self.sample_buffer)), batch_size)
        samples = [self.sample_buffer[i] for i in indices]
        
        # 移除已使用的样本
        new_buffer = deque(maxlen=self.buffer_size)
        for i, sample in enumerate(self.sample_buffer):
            if i not in indices:
                new_buffer.append(sample)
        
        self.sample_buffer = new_buffer
        return samples
    
    def start_async_processing(self):
        """启动异步处理线程"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # 启动会话处理线程
        self.processor_thread = threading.Thread(target=self._async_session_processor)
        self.processor_thread.daemon = True
        self.processor_thread.start()
        
        # 启动随机游走生成线程
        self.walker_thread = threading.Thread(target=self._async_walk_generator)
        self.walker_thread.daemon = True
        self.walker_thread.start()
    
    def stop_async_processing(self):
        """停止异步处理线程"""
        self.is_running = False
        
        if self.processor_thread:
            self.processor_thread.join(timeout=1)
        
        if self.walker_thread:
            self.walker_thread.join(timeout=1)
    
    def _async_session_processor(self):
        """异步处理会话的线程函数"""
        while self.is_running:
            try:
                # 从队列中获取会话
                session = self.session_queue.get(timeout=0.1)
                
                # 处理会话
                self.process_session(session)
                
                # 标记任务完成
                self.session_queue.task_done()
            except queue.Empty:
                # 队列为空，等待一段时间
                time.sleep(0.01)
            except Exception as e:
                print(f"会话处理错误: {e}")
    
    def _async_walk_generator(self):
        """异步生成随机游走的线程函数"""
        while self.is_running:
            try:
                # 如果样本队列未满且有足够的会话数据，生成更多样本
                if (self.sample_queue.qsize() < self.buffer_size / 2 and 
                    len(self.session_buffer) >= self.update_interval / 2):
                    
                    # 生成随机游走和样本
                    self._generate_walks_from_buffer()
                    
                    # 将样本添加到队列
                    for sample in self.sample_buffer:
                        if not self.is_running:
                            break
                        self.sample_queue.put(sample, timeout=0.1)
                    
                    # 清空缓冲区
                    self.sample_buffer.clear()
                
                # 等待一段时间
                time.sleep(0.1)
            except Exception as e:
                print(f"随机游走生成错误: {e}")
    
    def add_session(self, session):
        """
        添加会话到处理队列
        
        参数:
        session: 会话序列，包含多个商品ID
        """
        if len(session) <= 1:
            return
        
        if self.is_running:
            # 异步模式：添加到队列
            try:
                self.session_queue.put(session, timeout=1)
            except queue.Full:
                print("警告: 会话队列已满，跳过此会话")
        else:
            # 同步模式：直接处理
            self.process_session(session)
    
    def get_graph_info(self):
        """获取图的统计信息"""
        return {
            'num_nodes': len(self.G),
            'num_edges': self.G.number_of_edges(),
            'node_map_size': len(self.node_map),
            'buffer_size': len(self.sample_buffer),
            'session_buffer_size': len(self.session_buffer)
        }
    
    def get_node_maps(self):
        """获取节点映射"""
        return self.node_map, self.reverse_node_map


class StreamingDataProcessor:
    """
    流式数据处理器，用于处理实时数据流
    """
    def __init__(self, walker, time_window=30, session_timeout=30):
        """
        初始化流式数据处理器
        
        参数:
        walker: StreamingWalker实例
        time_window: 会话时间窗口（分钟）
        session_timeout: 会话超时时间（分钟）
        """
        self.walker = walker
        self.time_window = time_window
        self.session_timeout = session_timeout
        
        # 用户会话缓存
        self.user_sessions = defaultdict(list)
        self.last_action_time = {}
    
    def process_action(self, user_id, item_id, timestamp, action_type=None):
        """
        处理单个用户行为
        
        参数:
        user_id: 用户ID
        item_id: 商品ID
        timestamp: 行为时间戳
        action_type: 行为类型
        """
        # 检查是否需要结束当前会话
        if user_id in self.last_action_time:
            last_time = self.last_action_time[user_id]
            time_diff = (timestamp - last_time).total_seconds() / 60
            
            if time_diff > self.session_timeout:
                # 会话超时，结束当前会话
                if len(self.user_sessions[user_id]) > 1:
                    self.walker.add_session(self.user_sessions[user_id])
                
                # 清空会话
                self.user_sessions[user_id] = []
        
        # 更新最后行为时间
        self.last_action_time[user_id] = timestamp
        
        # 添加商品到会话
        self.user_sessions[user_id].append(item_id)
        
        # 如果是购买行为或特定行为类型，结束当前会话
        if action_type in [2, 'purchase']:
            if len(self.user_sessions[user_id]) > 1:
                self.walker.add_session(self.user_sessions[user_id])
            
            # 清空会话
            self.user_sessions[user_id] = []
    
    def process_batch(self, actions):
        """
        处理一批用户行为
        
        参数:
        actions: 用户行为列表，每个元素是(user_id, item_id, timestamp, action_type)元组
        """
        for action in actions:
            user_id, item_id, timestamp, action_type = action
            self.process_action(user_id, item_id, timestamp, action_type)
    
    def flush_sessions(self):
        """
        强制结束所有会话并处理
        """
        for user_id, session in self.user_sessions.items():
            if len(session) > 1:
                self.walker.add_session(session)
        
        # 清空会话
        self.user_sessions.clear()
        self.last_action_time.clear()
    
    def process_dataframe(self, df, user_col='user_id', item_col='sku_id', 
                         time_col='action_time', type_col='type', batch_size=1000):
        """
        处理DataFrame格式的用户行为数据
        
        参数:
        df: 包含用户行为的DataFrame
        user_col: 用户ID列名
        item_col: 商品ID列名
        time_col: 时间列名
        type_col: 行为类型列名
        batch_size: 批处理大小
        """
        # 确保时间列是datetime类型
        if not pd.api.types.is_datetime64_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])
        
        # 按时间排序
        df = df.sort_values(by=time_col)
        
        # 分批处理
        total_rows = len(df)
        num_batches = (total_rows + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="处理数据批次"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_rows)
            
            batch = df.iloc[start_idx:end_idx]
            
            actions = []
            for _, row in batch.iterrows():
                actions.append((
                    row[user_col],
                    row[item_col],
                    row[time_col],
                    row[type_col] if type_col in row else None
                ))
            
            self.process_batch(actions)
        
        # 处理完所有数据后，刷新剩余会话
        self.flush_sessions() 