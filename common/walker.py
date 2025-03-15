import numpy as np
import networkx as nx
import torch
import torch_geometric as pyg
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import from_networkx
import time
from tqdm import tqdm


class FastGraphWalker:
    def __init__(self, p=1, q=1, device=None):
        """
        初始化随机游走器
        
        参数:
        p: 返回参数，控制立即重访节点的可能性
        q: 进出参数，允许搜索区分"向内"和"向外"节点
        device: 计算设备
        """
        self.p = p
        self.q = q
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
    
    def build_graph(self, session_list):
        """
        构建图并直接在GPU上进行处理
        
        参数:
        session_list: 会话列表
        
        返回:
        pyg_data: PyG图数据
        node_maps: 节点映射元组 (node_map, reverse_node_map)
        """
        print("构建图...")
        start_time = time.time()
        
        # 提取所有边
        edges = []
        for session in session_list:
            if len(session) > 1:
                for i in range(len(session) - 1):
                    # 确保节点是整数类型
                    u = int(session[i])
                    v = int(session[i + 1])
                    edges.append((u, v))
        
        print(f"提取的边数量: {len(edges)}")
        if len(edges) == 0:
            print("警告: 没有提取到边，无法构建图")
            return None, None
        
        # 创建NetworkX图
        G = nx.Graph()
        G.add_edges_from(edges)
        
        # 获取所有唯一节点
        nodes = list(G.nodes())
        node_map = {node: i for i, node in enumerate(nodes)}
        reverse_node_map = {i: node for i, node in enumerate(nodes)}
        
        # 重新映射节点ID
        G_relabeled = nx.relabel_nodes(G, node_map)
        
        # 转换为PyG图
        pyg_data = from_networkx(G_relabeled).to(self.device)
        
        end_time = time.time()
        print(f"图构建完成，耗时: {end_time - start_time:.2f}秒")
        print(f"图包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边")
        
        return pyg_data, (node_map, reverse_node_map)
    
    def generate_walks(self, pyg_data, num_walks, walk_length, window_size=5):
        """
        使用PyG的Node2Vec生成随机游走，并生成与SimpleWalker类似的样本对
        
        参数:
        pyg_data: PyG图数据
        num_walks: 每个节点的游走次数
        walk_length: 每次游走的长度
        window_size: 上下文窗口大小，默认为5
        
        返回:
        all_pairs: 所有样本对
        """
        print(f"生成随机游走 (p={self.p}, q={self.q})...")
        start_time = time.time()
        
        # 使用PyG的Node2Vec
        model = Node2Vec(
            pyg_data.edge_index,
            embedding_dim=64,  # 这个值不重要，因为我们只使用游走部分
            walk_length=walk_length,
            context_size=window_size,  # 使用传入的窗口大小
            walks_per_node=num_walks,
            p=self.p,
            q=self.q,
            num_negative_samples=1  # 这个值不重要，因为我们自己处理负采样
        ).to(self.device)
        
        # 获取完整的随机游走序列，而不仅仅是样本对
        walks = []
        
        # 生成随机游走
        # 注意：PyG的Node2Vec不直接暴露随机游走序列，我们需要自己生成
        # 创建一个临时的NetworkX图用于生成随机游走
        edge_index = pyg_data.edge_index.cpu().numpy()
        G = nx.Graph()
        for i in range(edge_index.shape[1]):
            G.add_edge(edge_index[0, i], edge_index[1, i])
        
        # 使用NetworkX生成随机游走
        nodes = list(G.nodes())
        for _ in range(num_walks):
            np.random.shuffle(nodes)
            for node in tqdm(nodes, desc="生成随机游走"):
                walk = [node]
                for _ in range(walk_length - 1):
                    curr = walk[-1]
                    neighbors = list(G.neighbors(curr))
                    if len(neighbors) == 0:
                        break
                    
                    # 实现Node2Vec的偏向随机游走
                    if len(walk) > 1:
                        prev = walk[-2]
                        probs = []
                        for nbr in neighbors:
                            if nbr == prev:  # 返回到上一个节点
                                prob = 1.0 / self.p
                            elif G.has_edge(nbr, prev):  # 保持在同一社区
                                prob = 1.0
                            else:  # 探索新社区
                                prob = 1.0 / self.q
                            probs.append(prob)
                        
                        # 归一化概率
                        sum_probs = sum(probs)
                        probs = [p / sum_probs for p in probs]
                        
                        next_node = np.random.choice(neighbors, p=probs)
                    else:
                        next_node = np.random.choice(neighbors)
                    
                    walk.append(next_node)
                
                walks.append(walk)
        
        print(f"生成的随机游走序列数量: {len(walks)}")
        
        # 使用与SimpleWalker相同的方法生成上下文对
        from common.data_process import get_graph_context_all_pairs
        all_pairs = get_graph_context_all_pairs(walks, window_size)  # 使用传入的窗口大小
        
        end_time = time.time()
        print(f"随机游走完成，耗时: {end_time - start_time:.2f}秒")
        print(f"生成的样本对数量: {len(all_pairs)}")
        
        return all_pairs


class SimpleWalker:
    def __init__(self, p=1, q=1):
        """
        初始化简单随机游走器
        
        参数:
        p: 返回参数
        q: 进出参数
        """
        self.p = p
        self.q = q
    
    def build_graph(self, session_list):
        """
        构建图
        
        参数:
        session_list: 会话列表
        
        返回:
        G: NetworkX图
        node_maps: 节点映射元组 (node_map, reverse_node_map)
        """
        print("构建图...")
        start_time = time.time()
        
        # 提取所有边
        edges = []
        for session in session_list:
            if len(session) > 1:
                for i in range(len(session) - 1):
                    u = int(session[i])
                    v = int(session[i + 1])
                    edges.append((u, v))
        
        print(f"提取的边数量: {len(edges)}")
        if len(edges) == 0:
            print("警告: 没有提取到边，无法构建图")
            return None, None
        
        # 创建NetworkX图
        G = nx.Graph()
        G.add_edges_from(edges)
        
        # 获取所有唯一节点
        nodes = list(G.nodes())
        node_map = {node: i for i, node in enumerate(nodes)}
        reverse_node_map = {i: node for i, node in enumerate(nodes)}
        
        end_time = time.time()
        print(f"图构建完成，耗时: {end_time - start_time:.2f}秒")
        print(f"图包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边")
        
        return G, (node_map, reverse_node_map)
    
    def generate_walks(self, G, num_walks, walk_length):
        """
        生成随机游走
        
        参数:
        G: NetworkX图
        num_walks: 每个节点的游走次数
        walk_length: 每次游走的长度
        
        返回:
        walks: 随机游走序列
        """
        print(f"生成随机游走 (p={self.p}, q={self.q})...")
        start_time = time.time()
        
        walks = []
        nodes = list(G.nodes())
        
        for _ in range(num_walks):
            np.random.shuffle(nodes)
            for node in tqdm(nodes):
                walk = [node]
                for _ in range(walk_length - 1):
                    curr = walk[-1]
                    neighbors = list(G.neighbors(curr))
                    if len(neighbors) == 0:
                        break
                    walk.append(np.random.choice(neighbors))
                walks.append(walk)
        
        end_time = time.time()
        print(f"随机游走完成，耗时: {end_time - start_time:.2f}秒")
        print(f"生成的游走序列数量: {len(walks)}")
        
        return walks 