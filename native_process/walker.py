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
    
    
    def generate_walks(self, pyg_data, num_walks, walk_length, window_size):
        print(f"生成随机游走 (p={self.p}, q={self.q})...")
        start_time = time.time()

        # 使用PyG的Node2Vec
        model = Node2Vec(
            pyg_data.edge_index,
            embedding_dim=64,
            walk_length=walk_length,
            context_size=5,
            walks_per_node=num_walks,  # 直接设置总游走次数
            p=self.p,
            q=self.q,
            sparse=True
        ).to(self.device)

        # 直接在GPU上生成游走
        loader = model.loader(batch_size=128, shuffle=True)
        all_walks = []
        
        for walk_batch in tqdm(loader, desc="生成游走"):
            pos_rw, _ = walk_batch  # 关键修正：解包元组
            all_walks.append(pos_rw.cpu().numpy())

        # 合并结果
        all_walks = np.concatenate(all_walks, axis=0)
        
        # 生成上下文对（保持原有逻辑）
        print("生成上下文对...")
        all_pairs = []
        for walk in all_walks:
            for i in range(len(walk)):
                for j in range(max(0, i-window_size), min(len(walk), i+window_size+1)):
                    if i != j:
                        all_pairs.append((walk[i], walk[j]))
        
        print(f"生成样本对数量: {len(all_pairs)}")
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