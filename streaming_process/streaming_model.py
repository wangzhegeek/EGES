import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import time
from collections import defaultdict
import threading
import queue


class StreamingEGES(nn.Module):
    """
    流式增强图嵌入模型（Streaming Enhanced Graph Embedding with Side Information）
    支持在线学习和增量更新
    """
    def __init__(self, num_nodes, embedding_dim=128, side_info_dims=None, 
                 side_info_nums=None, use_attention=True, device='cpu'):
        """
        初始化流式EGES模型
        
        参数:
        num_nodes: 初始节点数量
        embedding_dim: 嵌入维度
        side_info_dims: 侧面信息的嵌入维度列表
        side_info_nums: 每种侧面信息的类别数量列表
        use_attention: 是否使用注意力机制
        device: 计算设备
        """
        super(StreamingEGES, self).__init__()
        
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.device = device
        self.use_attention = use_attention
        
        # 节点嵌入
        self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)
        nn.init.normal_(self.node_embeddings.weight, mean=0, std=0.01)
        
        # 侧面信息嵌入
        self.side_info_dims = side_info_dims if side_info_dims else []
        self.side_info_nums = side_info_nums if side_info_nums else []
        self.side_embeddings = nn.ModuleList()
        
        for i, (dim, num) in enumerate(zip(self.side_info_dims, self.side_info_nums)):
            embedding = nn.Embedding(num, dim)
            nn.init.normal_(embedding.weight, mean=0, std=0.01)
            self.side_embeddings.append(embedding)
        
        # 注意力网络
        if use_attention:
            # 注意力权重，包括节点嵌入和所有侧面信息
            self.attention = nn.Linear(1, 1 + len(self.side_info_dims))
            nn.init.xavier_uniform_(self.attention.weight)
        
        # 节点映射，用于处理新节点
        self.node_map = {}
        self.reverse_node_map = {}
        self.next_node_idx = 0
        
        # 侧面信息映射
        self.side_info_maps = [defaultdict(int) for _ in range(len(self.side_info_dims))]
        self.side_info_reverse_maps = [{} for _ in range(len(self.side_info_dims))]
        self.side_info_next_idx = [1 for _ in range(len(self.side_info_dims))]  # 从1开始，0留给未知
        
        # 节点侧面信息缓存
        self.node_side_info = {}
        
        # 优化器
        self.optimizer = None
        
        # 线程安全队列，用于异步更新
        self.update_queue = queue.Queue(maxsize=10000)
        self.is_updating = False
        self.update_thread = None
        
        # 移动模型到指定设备
        self.to(device)
    
    def _get_node_idx(self, node_id, create_if_not_exists=True):
        """获取节点索引，如果不存在则创建"""
        if node_id in self.node_map:
            return self.node_map[node_id]
        
        if not create_if_not_exists:
            return None
        
        # 创建新节点
        if self.next_node_idx >= self.num_nodes:
            self._expand_embeddings()
        
        idx = self.next_node_idx
        self.node_map[node_id] = idx
        self.reverse_node_map[idx] = node_id
        self.next_node_idx += 1
        
        return idx
    
    def _get_side_info_idx(self, side_idx, value, create_if_not_exists=True):
        if side_idx >= len(self.side_info_maps):
            return 0  # 未知类别
        
        if value in self.side_info_maps[side_idx]:
            return self.side_info_maps[side_idx][value]
        
        if not create_if_not_exists:
            return 0  # 未知类别
        
        # 创建新类别
        if self.side_info_next_idx[side_idx] >= self.side_info_nums[side_idx]:
            self._expand_side_embeddings(side_idx)
        
        idx = self.side_info_next_idx[side_idx]
        self.side_info_maps[side_idx][value] = idx
        self.side_info_reverse_maps[side_idx][idx] = value
        self.side_info_next_idx[side_idx] += 1
        
        return idx
    
    def _expand_embeddings(self):
        """扩展节点嵌入容量"""
        old_num_nodes = self.num_nodes
        self.num_nodes = max(1, int(old_num_nodes * 1.5))  # 增加50%容量
        
        # 创建新的嵌入层
        new_embeddings = nn.Embedding(self.num_nodes, self.embedding_dim)
        nn.init.normal_(new_embeddings.weight, mean=0, std=0.01)
        
        # 复制旧的嵌入权重
        with torch.no_grad():
            new_embeddings.weight[:old_num_nodes] = self.node_embeddings.weight
        
        # 替换嵌入层
        self.node_embeddings = new_embeddings.to(self.device)
        
        print(f"扩展节点嵌入容量: {old_num_nodes} -> {self.num_nodes}")
    
    def _expand_side_embeddings(self, side_idx):
        """扩展侧面信息嵌入容量"""
        old_num = self.side_info_nums[side_idx]
        self.side_info_nums[side_idx] = max(1, int(old_num * 1.5))  # 增加50%容量
        
        # 创建新的嵌入层
        dim = self.side_info_dims[side_idx]
        new_embeddings = nn.Embedding(self.side_info_nums[side_idx], dim)
        nn.init.normal_(new_embeddings.weight, mean=0, std=0.01)
        
        # 复制旧的嵌入权重
        with torch.no_grad():
            new_embeddings.weight[:old_num] = self.side_embeddings[side_idx].weight
        
        # 替换嵌入层
        self.side_embeddings[side_idx] = new_embeddings.to(self.device)
        
        print(f"扩展侧面信息{side_idx}嵌入容量: {old_num} -> {self.side_info_nums[side_idx]}")
    
    def update_side_info(self, node_id, side_info_list):
        """
        更新节点的侧面信息
        
        参数:
        node_id: 节点ID
        side_info_list: 侧面信息列表，每个元素对应一种侧面信息
        """
        # 确保侧面信息列表长度正确
        if len(side_info_list) != len(self.side_info_dims):
            side_info_list = side_info_list + [None] * (len(self.side_info_dims) - len(side_info_list))
        
        # 获取节点索引
        node_idx = self._get_node_idx(node_id)
        
        # 获取侧面信息索引
        side_indices = []
        for i, value in enumerate(side_info_list):
            if value is not None:
                idx = self._get_side_info_idx(i, value)
            else:
                idx = 0  # 未知类别
            side_indices.append(idx)
        
        # 更新节点侧面信息缓存
        self.node_side_info[node_id] = side_indices
    
    def get_node_embedding(self, node_id):
        """
        获取节点的嵌入向量
        
        参数:
        node_id: 节点ID
        
        返回:
        embedding: 节点嵌入向量
        """
        node_idx = self._get_node_idx(node_id, create_if_not_exists=False)
        if node_idx is None:
            return torch.zeros(self.embedding_dim, device=self.device)
        
        # 获取节点嵌入
        node_emb = self.node_embeddings(torch.tensor([node_idx], device=self.device))
        
        # 如果没有侧面信息或不使用注意力，直接返回节点嵌入
        if not self.use_attention or node_id not in self.node_side_info:
            return node_emb.squeeze(0)
        
        # 获取侧面信息嵌入
        side_indices = self.node_side_info[node_id]
        side_embs = []
        
        for i, idx in enumerate(side_indices):
            if i < len(self.side_embeddings):
                side_emb = self.side_embeddings[i](torch.tensor([idx], device=self.device))
                # 如果侧面信息嵌入维度与节点嵌入不同，进行线性投影
                if self.side_info_dims[i] != self.embedding_dim:
                    # 简单的线性投影
                    side_emb = F.linear(
                        side_emb, 
                        torch.randn(self.embedding_dim, self.side_info_dims[i], device=self.device)
                    )
                side_embs.append(side_emb)
        
        # 计算注意力权重
        attention_input = torch.ones(1, 1, device=self.device)  # 简单起见，使用常数输入
        attention_weights = F.softmax(self.attention(attention_input).squeeze(0), dim=0)
        
        # 应用注意力权重
        node_weight = attention_weights[0]
        weighted_emb = node_weight * node_emb
        
        for i, side_emb in enumerate(side_embs):
            if i < len(attention_weights) - 1:
                side_weight = attention_weights[i + 1]
                weighted_emb = weighted_emb + side_weight * side_emb
        
        return weighted_emb.squeeze(0)
    
    def forward(self, nodes, pos_neighbors, neg_neighbors):
        """
        前向传播
        
        参数:
        nodes: 中心节点索引
        pos_neighbors: 正样本邻居索引
        neg_neighbors: 负样本邻居索引
        
        返回:
        pos_loss: 正样本损失
        neg_loss: 负样本损失
        """
        # 获取节点嵌入
        node_embs = self.node_embeddings(nodes)
        pos_embs = self.node_embeddings(pos_neighbors)
        neg_embs = self.node_embeddings(neg_neighbors)
        
        # 如果使用注意力机制，融合侧面信息
        if self.use_attention and len(self.side_embeddings) > 0:
            # 获取侧面信息嵌入
            side_embs_list = []
            for i, side_embedding in enumerate(self.side_embeddings):
                # 为每个节点获取对应的侧面信息索引
                side_indices = torch.zeros_like(nodes)
                for j, node_idx in enumerate(nodes.cpu().numpy()):
                    node_id = self.reverse_node_map.get(node_idx.item())
                    if node_id in self.node_side_info:
                        side_idx = self.node_side_info[node_id][i]
                        side_indices[j] = side_idx
                
                side_embs = side_embedding(side_indices)
                
                # 如果侧面信息嵌入维度与节点嵌入不同，进行线性投影
                if self.side_info_dims[i] != self.embedding_dim:
                    # 简单的线性投影
                    side_embs = F.linear(
                        side_embs, 
                        torch.randn(self.embedding_dim, self.side_info_dims[i], device=self.device)
                    )
                
                side_embs_list.append(side_embs)
            
            # 计算注意力权重
            batch_size = nodes.size(0)
            attention_input = torch.ones(batch_size, 1, device=self.device)
            attention_weights = F.softmax(self.attention(attention_input), dim=1)
            
            # 应用注意力权重
            node_weights = attention_weights[:, 0].unsqueeze(1)
            weighted_embs = node_weights * node_embs
            
            for i, side_embs in enumerate(side_embs_list):
                if i < attention_weights.size(1) - 1:
                    side_weights = attention_weights[:, i + 1].unsqueeze(1)
                    weighted_embs = weighted_embs + side_weights * side_embs
            
            node_embs = weighted_embs
        
        # 计算正样本得分
        pos_scores = torch.sum(node_embs * pos_embs, dim=1)
        pos_loss = -F.logsigmoid(pos_scores).mean()
        
        # 计算负样本得分
        neg_scores = torch.sum(node_embs * neg_embs, dim=1)
        neg_loss = -F.logsigmoid(-neg_scores).mean()
        
        return pos_loss, neg_loss
    
    def train_batch(self, samples, batch_size=64, num_neg_samples=5, lr=0.01):
        """
        训练一个批次
        
        参数:
        samples: 样本列表，每个样本是(source, target)对
        batch_size: 批次大小
        num_neg_samples: 负采样数量
        lr: 学习率
        
        返回:
        loss: 训练损失
        """
        if not samples:
            return 0.0
        
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # 准备批次数据
        np.random.shuffle(samples)
        
        total_loss = 0.0
        num_batches = (len(samples) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(samples))
            batch_samples = samples[start_idx:end_idx]
            
            # 提取源节点和目标节点
            sources = [sample[0] for sample in batch_samples]
            targets = [sample[1] for sample in batch_samples]
            
            # 获取节点索引
            source_indices = torch.tensor([self._get_node_idx(src) for src in sources], device=self.device)
            target_indices = torch.tensor([self._get_node_idx(tgt) for tgt in targets], device=self.device)
            
            # 负采样
            neg_indices = self._negative_sampling(source_indices, num_neg_samples)
            
            # 前向传播
            self.optimizer.zero_grad()
            pos_loss, neg_loss = self.forward(source_indices, target_indices, neg_indices)
            loss = pos_loss + neg_loss
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / num_batches
    
    def _negative_sampling(self, source_indices, num_samples):
        """
        负采样
        
        参数:
        source_indices: 源节点索引
        num_samples: 负采样数量
        
        返回:
        neg_indices: 负样本索引
        """
        batch_size = source_indices.size(0)
        
        # 从所有节点中随机采样
        neg_indices = torch.randint(0, self.next_node_idx, (batch_size, num_samples), device=self.device)
        
        # 确保负样本不包含源节点
        for i in range(batch_size):
            for j in range(num_samples):
                while neg_indices[i, j] == source_indices[i]:
                    neg_indices[i, j] = torch.randint(0, self.next_node_idx, (1,), device=self.device)
        
        return neg_indices.view(-1)
    
    def start_async_updating(self):
        """启动异步更新线程"""
        if self.is_updating:
            return
        
        self.is_updating = True
        self.update_thread = threading.Thread(target=self._async_update_processor)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def stop_async_updating(self):
        """停止异步更新线程"""
        self.is_updating = False
        
        if self.update_thread:
            self.update_thread.join(timeout=1)
    
    def _async_update_processor(self):
        """异步处理更新的线程函数"""
        while self.is_updating:
            try:
                # 从队列中获取更新数据
                update_data = self.update_queue.get(timeout=0.1)
                
                # 处理更新
                if update_data['type'] == 'train':
                    samples = update_data['samples']
                    batch_size = update_data.get('batch_size', 64)
                    num_neg_samples = update_data.get('num_neg_samples', 5)
                    lr = update_data.get('lr', 0.01)
                    
                    self.train_batch(samples, batch_size, num_neg_samples, lr)
                
                elif update_data['type'] == 'side_info':
                    node_id = update_data['node_id']
                    side_info = update_data['side_info']
                    
                    self.update_side_info(node_id, side_info)
                
                # 标记任务完成
                self.update_queue.task_done()
            except queue.Empty:
                # 队列为空，等待一段时间
                time.sleep(0.01)
            except Exception as e:
                print(f"更新处理错误: {e}")
    
    def add_train_task(self, samples, batch_size=64, num_neg_samples=5, lr=0.01):
        """
        添加训练任务到更新队列
        
        参数:
        samples: 样本列表，每个样本是(source, target)对
        batch_size: 批次大小
        num_neg_samples: 负采样数量
        lr: 学习率
        """
        if not samples:
            return
        
        if self.is_updating:
            # 异步模式：添加到队列
            try:
                self.update_queue.put({
                    'type': 'train',
                    'samples': samples,
                    'batch_size': batch_size,
                    'num_neg_samples': num_neg_samples,
                    'lr': lr
                }, timeout=1)
            except queue.Full:
                print("警告: 更新队列已满，跳过此训练任务")
        else:
            # 同步模式：直接处理
            self.train_batch(samples, batch_size, num_neg_samples, lr)
    
    def add_side_info_task(self, node_id, side_info):
        """
        添加侧面信息更新任务到更新队列
        
        参数:
        node_id: 节点ID
        side_info: 侧面信息列表
        """
        if self.is_updating:
            # 异步模式：添加到队列
            try:
                self.update_queue.put({
                    'type': 'side_info',
                    'node_id': node_id,
                    'side_info': side_info
                }, timeout=1)
            except queue.Full:
                print("警告: 更新队列已满，跳过此侧面信息更新任务")
        else:
            # 同步模式：直接处理
            self.update_side_info(node_id, side_info)
    
    def save_model(self, path):
        """
        保存模型
        
        参数:
        path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型参数
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_nodes': self.num_nodes,
            'embedding_dim': self.embedding_dim,
            'side_info_dims': self.side_info_dims,
            'side_info_nums': self.side_info_nums,
            'use_attention': self.use_attention,
            'node_map': self.node_map,
            'reverse_node_map': self.reverse_node_map,
            'next_node_idx': self.next_node_idx,
            'side_info_maps': self.side_info_maps,
            'side_info_reverse_maps': self.side_info_reverse_maps,
            'side_info_next_idx': self.side_info_next_idx,
            'node_side_info': self.node_side_info
        }, path)
    
    def load_model(self, path):
        """
        加载模型
        
        参数:
        path: 模型路径
        """
        if not os.path.exists(path):
            print(f"模型文件不存在: {path}")
            return False
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # 更新模型参数
        self.num_nodes = checkpoint['num_nodes']
        self.embedding_dim = checkpoint['embedding_dim']
        self.side_info_dims = checkpoint['side_info_dims']
        self.side_info_nums = checkpoint['side_info_nums']
        self.use_attention = checkpoint['use_attention']
        self.node_map = checkpoint['node_map']
        self.reverse_node_map = checkpoint['reverse_node_map']
        self.next_node_idx = checkpoint['next_node_idx']
        self.side_info_maps = checkpoint['side_info_maps']
        self.side_info_reverse_maps = checkpoint['side_info_reverse_maps']
        self.side_info_next_idx = checkpoint['side_info_next_idx']
        self.node_side_info = checkpoint['node_side_info']
        
        # 重新创建嵌入层
        self.node_embeddings = nn.Embedding(self.num_nodes, self.embedding_dim)
        self.side_embeddings = nn.ModuleList()
        
        for i, (dim, num) in enumerate(zip(self.side_info_dims, self.side_info_nums)):
            embedding = nn.Embedding(num, dim)
            self.side_embeddings.append(embedding)
        
        # 如果使用注意力，重新创建注意力网络
        if self.use_attention:
            self.attention = nn.Linear(1, 1 + len(self.side_info_dims))
        
        # 加载模型状态
        self.load_state_dict(checkpoint['model_state_dict'])
        
        # 移动模型到指定设备
        self.to(self.device)
        
        return True
    
    def save_embeddings(self, path):
        """
        保存嵌入向量
        
        参数:
        path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 获取所有节点的嵌入向量
        embeddings = {}
        for node_id in self.node_map:
            embeddings[node_id] = self.get_node_embedding(node_id).cpu().detach().numpy()
        
        # 保存嵌入向量
        np.save(path, embeddings)
    
    def get_all_embeddings(self):
        """
        获取所有节点的嵌入向量
        
        返回:
        embeddings: 节点ID到嵌入向量的字典
        """
        embeddings = {}
        for node_id in self.node_map:
            embeddings[node_id] = self.get_node_embedding(node_id).cpu().detach().numpy()
        
        return embeddings 