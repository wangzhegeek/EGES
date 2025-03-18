import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import networkx as nx
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import from_networkx
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os
import sys

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import plot_loss_curve, visualize_embeddings


class EGES(nn.Module):
    """
    集成版EGES模型，将游走采样和模型训练整合在一起
    """
    def __init__(self, edge_index, num_nodes, feature_dim, feature_lens, 
                 embedding_dim=128, walk_length=10, context_size=5, 
                 walks_per_node=10, p=1.0, q=1.0, lr=0.001, device=None):
        """
        初始化EGES模型
        
        参数:
        edge_index: 图的边索引
        num_nodes: 节点数量
        feature_dim: 特征维度
        feature_lens: 每个特征的长度列表
        embedding_dim: 嵌入维度
        walk_length: 随机游走长度
        context_size: 上下文窗口大小
        walks_per_node: 每个节点的游走次数
        p: 返回参数
        q: 进出参数
        lr: 学习率
        device: 计算设备
        """
        super(EGES, self).__init__()
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # 模型参数
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.feature_lens = feature_lens
        self.embedding_dim = embedding_dim
        self.lr = lr
        
        # 确保边索引在正确的设备上
        if edge_index.device != self.device:
            edge_index = edge_index.to(self.device)
        
        # 初始化Node2Vec模型
        self.node2vec = Node2Vec(
            edge_index,
            embedding_dim=embedding_dim,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            p=p,
            q=q,
            sparse=True,
            num_negative_samples=5  # 增加负样本数量
        ).to(self.device)
        
        # 初始化特征嵌入层
        self.embedding_layers = nn.ModuleList()
        for i in range(self.feature_dim):
            embedding_layer = nn.Embedding(self.feature_lens[i], self.embedding_dim)
            nn.init.xavier_uniform_(embedding_layer.weight)
            self.embedding_layers.append(embedding_layer)
        
        # 注意力网络
        self.attention_network = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.Softmax(dim=1)
        )
        
        # 将所有模块移动到指定设备
        self.to(self.device)
        
        # 初始化优化器
        self.optimizer = None
        self.scheduler = None
        
    def init_optimizer(self, lr=None):
        """
        初始化优化器
        
        参数:
        lr: 学习率，如果为None则使用默认值
        """
        if lr is not None:
            self.lr = lr
        
        # 使用SparseAdam优化器，它支持稀疏梯度
        self.optimizer = torch.optim.SparseAdam(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            verbose=True
        )
    
    def forward(self, features=None):
        """
        前向传播
        
        参数:
        features: 节点特征，如果为None则只返回Node2Vec嵌入
        
        返回:
        embeddings: 节点嵌入
        """
        # 获取Node2Vec嵌入
        node2vec_emb = self.node2vec()
        
        # 如果没有提供特征，直接返回Node2Vec嵌入
        if features is None:
            return node2vec_emb
        
        # 确保特征在正确的设备上
        features = [f.to(self.device, non_blocking=True) for f in features]
        
        # 处理特征
        batch_size = features[0].size(0)
        
        # 对每个特征进行嵌入查找
        embed_list = []
        for i in range(self.feature_dim):
            # 确保索引在有效范围内
            valid_indices = torch.clamp(features[i], 0, self.feature_lens[i] - 1)
            # 使用非阻塞传输
            valid_indices = valid_indices.to(self.device, non_blocking=True)
            embed_list.append(self.embedding_layers[i](valid_indices))
        
        # 堆叠嵌入 [batch_size, embedding_dim, num_feat]
        stacked_embeds = torch.stack(embed_list, dim=2)
        
        # 计算注意力权重
        feature_ids = torch.arange(self.feature_dim, device=self.device).expand(batch_size, self.feature_dim)
        attention_weights = self.attention_network(feature_ids.float())
        
        # 应用注意力权重 [batch_size, embedding_dim, 1]
        attention_weights = attention_weights.unsqueeze(1)
        weighted_embeds = torch.matmul(stacked_embeds, attention_weights.transpose(1, 2))
        
        # 最终嵌入 [batch_size, embedding_dim]
        feature_emb = weighted_embeds.squeeze(2)
        
        # 获取对应节点的Node2Vec嵌入
        node_indices = torch.clamp(features[0], 0, self.num_nodes - 1)  # 确保节点索引在有效范围内
        node_emb = node2vec_emb[node_indices]
        
        # 融合嵌入
        final_emb = node_emb + feature_emb
        
        return final_emb
    
    def get_loader(self, batch_size=128, shuffle=True):
        """
        获取数据加载器
        
        参数:
        batch_size: 批次大小
        shuffle: 是否打乱数据
        
        返回:
        loader: 数据加载器
        """
        return self.node2vec.loader(batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    
    def train_step(self, pos_rw, neg_rw=None, features=None, update_params=True):
        """
        训练一步
        
        参数:
        pos_rw: 正样本随机游走
        neg_rw: 负样本随机游走，如果为None则自动生成
        features: 节点特征，如果为None则不使用特征
        update_params: 是否更新参数，默认为True
        
        返回:
        loss: 损失值
        """
        self.train()
        self.optimizer.zero_grad()
        
        # 确保数据在正确的设备上，并使用non_blocking=True
        pos_rw = pos_rw.to(self.device, non_blocking=True)
        
        # 如果没有提供负样本，在GPU上直接生成
        if neg_rw is None:
            neg_rw = torch.randint(0, self.num_nodes, pos_rw.size(), 
                                 dtype=torch.long, device=self.device)
        else:
            neg_rw = neg_rw.to(self.device, non_blocking=True)
        
        # 计算Node2Vec损失
        loss = self.node2vec.loss(pos_rw, neg_rw)
        
        # 反向传播
        loss.backward()
        
        # 手动裁剪稀疏梯度
        for param in self.parameters():
            if param.grad is not None:
                if param.grad.is_sparse:
                    # 对于稀疏梯度，我们只能逐个处理非零元素
                    grad_values = param.grad._values()
                    grad_norm = torch.norm(grad_values)
                    if grad_norm > 5.0:  # max_norm
                        grad_values.mul_(5.0 / grad_norm)
                else:
                    # 对于稠密梯度，使用常规裁剪
                    torch.nn.utils.clip_grad_norm_([param], max_norm=5.0)
        
        # 更新参数
        if update_params:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return loss.item()
    
    def update_lr(self, val_loss):
        """
        更新学习率
        
        参数:
        val_loss: 验证损失
        """
        if self.scheduler is not None:
            self.scheduler.step(val_loss)
    
    def get_embeddings(self, features=None, indices=None):
        """
        获取节点嵌入
        
        参数:
        features: 节点特征，如果为None则只返回Node2Vec嵌入
        indices: 要获取嵌入的节点索引，如果为None则获取所有节点
        
        返回:
        embeddings: 节点嵌入
        """
        with torch.no_grad():
            # 如果没有提供特征，直接返回Node2Vec嵌入
            if features is None:
                embeddings = self.node2vec()
                if indices is not None:
                    # 确保索引在有效范围内
                    valid_indices = torch.clamp(indices, 0, self.num_nodes - 1)
                    return embeddings[valid_indices]
                return embeddings
            
            # 确保特征在正确的设备上
            features = [f.to(self.device, non_blocking=True) for f in features]
            
            # 处理特征
            batch_size = features[0].size(0)
            
            # 对每个特征进行嵌入查找
            embed_list = []
            for i in range(self.feature_dim):
                # 确保索引在有效范围内
                valid_indices = torch.clamp(features[i], 0, self.feature_lens[i] - 1)
                # 使用非阻塞传输
                valid_indices = valid_indices.to(self.device, non_blocking=True)
                embed_list.append(self.embedding_layers[i](valid_indices))
            
            # 堆叠嵌入 [batch_size, embedding_dim, num_feat]
            stacked_embeds = torch.stack(embed_list, dim=2)
            
            # 计算注意力权重
            feature_ids = torch.arange(self.feature_dim, device=self.device).expand(batch_size, self.feature_dim)
            attention_weights = self.attention_network(feature_ids.float())
            
            # 应用注意力权重 [batch_size, embedding_dim, 1]
            attention_weights = attention_weights.unsqueeze(1)
            weighted_embeds = torch.matmul(stacked_embeds, attention_weights.transpose(1, 2))
            
            # 最终嵌入 [batch_size, embedding_dim]
            feature_emb = weighted_embeds.squeeze(2)
            
            # 获取对应节点的Node2Vec嵌入
            if indices is not None:
                # 确保索引在有效范围内
                valid_indices = torch.clamp(indices, 0, self.num_nodes - 1)
                node_emb = self.node2vec()[valid_indices]
            else:
                node_emb = self.node2vec()[torch.clamp(features[0], 0, self.num_nodes - 1)]
            
            # 融合嵌入
            final_emb = node_emb + feature_emb
            
            return final_emb


class EGESTrainer:
    """
    EGES模型训练器
    """
    def __init__(self, session_list, side_info=None, embedding_dim=128, 
                 walk_length=10, context_size=5, walks_per_node=10, 
                 p=1.0, q=1.0, lr=0.001, device=None, prefetch_factor=2,
                 G=None, node_map=None, reverse_node_map=None, pyg_data=None):
        """
        初始化训练器
        
        参数:
        session_list: 会话列表
        side_info: 侧信息
        embedding_dim: 嵌入维度
        walk_length: 随机游走长度
        context_size: 上下文窗口大小
        walks_per_node: 每个节点的游走次数
        p: 返回参数
        q: 进出参数
        lr: 学习率
        device: 计算设备
        prefetch_factor: 预取因子，控制数据预加载的批次数
        G: 预构建的图
        node_map: 预构建的节点映射
        reverse_node_map: 预构建的反向节点映射
        pyg_data: 预构建的PyG数据
        """
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.prefetch_factor = prefetch_factor
        
        # 使用预构建的图和节点映射，或者构建新的图
        if G is not None and node_map is not None and reverse_node_map is not None:
            self.G = G
            self.node_map = node_map
            self.reverse_node_map = reverse_node_map
            self.num_nodes = self.G.number_of_nodes()
            
            # 使用预构建的PyG数据或者转换为PyG图
            if pyg_data is not None:
                self.pyg_data = pyg_data.to(self.device)
            else:
                self.pyg_data = from_networkx(self.G).to(self.device)
        else:
            # 构建图
            self.G, self.node_maps = self._build_graph(session_list)
            self.node_map, self.reverse_node_map = self.node_maps
            
            # 设置节点数量
            self.num_nodes = self.G.number_of_nodes()
            
            # 转换为PyG图，并直接放在GPU上
            self.pyg_data = from_networkx(self.G).to(self.device)
        
        # 处理侧信息
        self.side_info = side_info
        self.side_info_dict = {}
        
        if side_info is not None:
            # 创建特征长度列表
            self.feature_lens = []
            for i in range(side_info.shape[1]):
                tmp_len = len(set(side_info[:, i])) + 1
                self.feature_lens.append(tmp_len)
            
            # 创建侧信息字典并预先将数据转移到GPU
            side_info_tensor = torch.tensor(side_info, dtype=torch.long, device=self.device)
            for i in range(len(side_info)):
                sku_id = side_info[i][0]
                if sku_id in self.node_map:
                    self.side_info_dict[self.node_map[sku_id]] = side_info_tensor[i]
        else:
            self.feature_lens = [self.num_nodes + 1]
        
        # 创建模型
        self.model = EGES(
            edge_index=self.pyg_data.edge_index,
            num_nodes=self.num_nodes,
            feature_dim=len(self.feature_lens),
            feature_lens=self.feature_lens,
            embedding_dim=embedding_dim,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            p=p,
            q=q,
            lr=lr,
            device=self.device
        )
        
        # 初始化优化器
        self.model.init_optimizer(lr)
        
        # 创建数据缓冲区
        self.data_buffer = []
        
    def _build_graph(self, session_list):
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
        
        # 重新映射节点ID
        G_relabeled = nx.relabel_nodes(G, node_map)
        
        end_time = time.time()
        print(f"图构建完成，耗时: {end_time - start_time:.2f}秒")
        print(f"图包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边")
        
        return G_relabeled, (node_map, reverse_node_map)
    
    def _prepare_features(self, nodes):
        """
        准备特征，优化版本直接在GPU上处理
        
        参数:
        nodes: 原始节点ID列表
        
        返回:
        features: 特征列表
        """
        if self.side_info is None:
            # 将原始节点ID转换为内部索引
            node_indices = [self.node_map.get(node, 0) for node in nodes]
            nodes_tensor = torch.tensor(node_indices, dtype=torch.long, device=self.device)
            return [torch.clamp(nodes_tensor, 0, self.feature_lens[0] - 1)]
        
        # 将原始节点ID转换为内部索引
        node_indices = [self.node_map.get(node, 0) for node in nodes]
        nodes_tensor = torch.tensor(node_indices, dtype=torch.long, device=self.device)
        features = [torch.clamp(nodes_tensor, 0, self.feature_lens[0] - 1)]
        
        # 使用预计算的映射矩阵
        if not hasattr(self, '_feature_mapping'):
            # 第一次调用时创建映射矩阵
            self._feature_mapping = []
            for i in range(1, len(self.feature_lens)):
                mapping = torch.zeros(self.num_nodes, dtype=torch.long, device=self.device)
                for node_id, info in self.side_info_dict.items():
                    if node_id < self.num_nodes:  # 确保索引在有效范围内
                        mapping[node_id] = info[i]
                self._feature_mapping.append(mapping)
        
        # 使用预计算的映射矩阵快速获取特征
        for i, mapping in enumerate(self._feature_mapping):
            # 使用高效的索引操作，同时确保索引在有效范围内
            valid_nodes = torch.clamp(nodes_tensor, 0, self.num_nodes - 1)
            feature_col = mapping[valid_nodes]
            # 确保特征索引在有效范围内
            feature_col = torch.clamp(feature_col, 0, self.feature_lens[i + 1] - 1)
            features.append(feature_col)
        
        return features
        
    def _prefetch_data(self, loader, num_batches):
        """
        预取数据到缓冲区
        
        参数:
        loader: 数据加载器
        num_batches: 预取的批次数
        """
        try:
            for _ in range(num_batches):
                batch = next(loader)
                self.data_buffer.append(batch)
        except StopIteration:
            pass
            
    def train(self, epochs=10, batch_size=128, output_dir='./output/integrated', plot_loss=True):
        """
        训练模型
        
        参数:
        epochs: 训练轮数
        batch_size: 批次大小
        output_dir: 输出目录，用于保存中间结果
        plot_loss: 是否绘制损失曲线
        
        返回:
        model: 训练好的模型
        """
        print("开始训练...")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取数据加载器
        loader = self.model.get_loader(batch_size=batch_size, shuffle=True)
        
        # 记录每个epoch的损失
        epoch_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            batch_count = 0
            
            # 创建迭代器
            loader_iter = iter(loader)
            
            # 预取数据
            self._prefetch_data(loader_iter, self.prefetch_factor)
            
            # 使用tqdm显示进度条
            pbar = tqdm(total=len(loader), desc=f"Epoch {epoch+1}/{epochs}")
            
            while True:
                # 如果缓冲区为空，重新预取数据
                if not self.data_buffer:
                    self._prefetch_data(loader_iter, self.prefetch_factor)
                    if not self.data_buffer:
                        break
                
                # 获取一批数据
                pos_rw, neg_rw = self.data_buffer.pop(0)
                
                # 训练一步 - 不使用特征，简化训练过程
                loss = self.model.train_step(pos_rw, neg_rw)
                
                total_loss += loss
                batch_count += 1
                
                # 更新进度条
                pbar.set_postfix({'loss': f'{loss:.4f}'})
                pbar.update(1)
                
                # 在处理当前批次时，异步预取下一批数据
                if len(self.data_buffer) < self.prefetch_factor:
                    self._prefetch_data(loader_iter, 1)
            
            pbar.close()
            
            # 计算平均损失
            avg_loss = total_loss / batch_count
            epoch_losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            
            # 更新学习率
            self.model.update_lr(avg_loss)
            
            # 清空数据缓冲区
            self.data_buffer.clear()
        
        print("训练完成！")
        
        # 绘制损失下降曲线（如果需要）
        if plot_loss:
            plot_loss_curve(epoch_losses, output_dir=output_dir)
        
        # 存储损失值供后续使用
        self.epoch_losses = epoch_losses
        
        return self.model
    
    def get_embeddings(self):
        """
        获取所有节点的嵌入
        
        返回:
        embeddings_dict: 节点嵌入字典，键为原始节点ID，值为嵌入向量
        """
        self.model.eval()
        
        # 获取所有节点的嵌入 - 分批处理以减少内存压力
        batch_size = 1024  # 使用更大的批次大小
        num_nodes = len(self.reverse_node_map)
        embeddings_dict = {}
        
        with torch.no_grad():
            # 分批处理节点
            for i in range(0, num_nodes, batch_size):
                batch_indices = list(range(i, min(i + batch_size, num_nodes)))
                
                # 准备特征（如果有）
                if self.side_info is not None:
                    batch_nodes = [self.reverse_node_map[idx] for idx in batch_indices]  # 获取原始节点ID
                    features = self._prepare_features(batch_nodes)
                else:
                    features = None
                
                # 获取这批节点的嵌入
                indices_tensor = torch.tensor(batch_indices, device=self.device)
                try:
                    batch_embeddings = self.model.get_embeddings(features, indices=indices_tensor)
                    
                    # 将嵌入移到CPU并转换为NumPy
                    batch_embeddings = batch_embeddings.cpu().numpy()
                    
                    # 将嵌入映射回原始节点ID
                    for j, idx in enumerate(batch_indices):
                        node_id = self.reverse_node_map[idx]
                        embeddings_dict[node_id] = batch_embeddings[j]
                except Exception as e:
                    print(f"处理批次 {i} 时出错: {str(e)}")
                    continue
        
        return embeddings_dict
    
    def save_model(self, path):
        """
        保存模型
        
        参数:
        path: 保存路径
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.model.optimizer.state_dict(),
            'node_map': self.node_map,
            'reverse_node_map': self.reverse_node_map
        }, path)
    
    def load_model(self, path):
        """
        加载模型
        
        参数:
        path: 加载路径
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.node_map = checkpoint['node_map']
        self.reverse_node_map = checkpoint['reverse_node_map'] 