import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist


class EGES_Model(nn.Module):
    def __init__(self, num_nodes, num_feat, feature_lens, embedding_dim=128, lr=0.001):
        """
        初始化EGES模型
        
        参数:
        num_nodes: 节点数量
        num_feat: 特征数量
        feature_lens: 每个特征的长度列表
        embedding_dim: 嵌入维度
        lr: 学习率
        """
        super(EGES_Model, self).__init__()
        self.num_feat = num_feat
        self.feature_lens = feature_lens
        self.embedding_dim = embedding_dim
        self.num_nodes = num_nodes
        self.lr = lr
        
        # 初始化嵌入层
        self.embedding_layers = nn.ModuleList()
        for i in range(self.num_feat):
            embedding_layer = nn.Embedding(self.feature_lens[i], self.embedding_dim)
            # 使用Xavier初始化以提高收敛速度
            nn.init.xavier_uniform_(embedding_layer.weight)
            self.embedding_layers.append(embedding_layer)
        
        # 注意力网络 - 使用更高效的实现
        self.attention_network = nn.Sequential(
            nn.Linear(self.num_feat, self.num_feat),
            nn.ReLU(),
            nn.Linear(self.num_feat, self.num_feat),
            nn.Softmax(dim=1)
        )
        
        # 输出层 - 使用嵌入矩阵共享权重
        self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)
        nn.init.xavier_uniform_(self.node_embeddings.weight)
        
        # 学习率调度器
        self.scheduler = None
        
    def init_optimizer(self, lr=None, distributed=False):
        """
        初始化优化器，支持分布式训练
        
        参数:
        lr: 学习率，如果为None则使用默认值
        distributed: 是否使用分布式训练
        """
        if lr is not None:
            self.lr = lr
            
        # 优化器 - 使用带权重衰减的Adam
        self.optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=1e-5,  # 添加L2正则化
            betas=(0.9, 0.999)  # 使用默认动量参数
        )
        
        # 学习率调度器 - 使用ReduceLROnPlateau
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=2, 
            verbose=True
        )
        
    def forward(self, inputs, context_indices=None):
        """
        前向传播
        
        参数:
        inputs: 包含多个特征列的列表
        context_indices: 上下文节点索引，如果提供则同时计算上下文嵌入
        
        返回:
        node_embeddings: 节点嵌入
        context_embeddings: 上下文嵌入（如果提供了context_indices）
        """
        batch_size = inputs[0].size(0)
        
        # 对每个特征进行嵌入查找
        embed_list = []
        for i in range(self.num_feat):
            # 确保索引在有效范围内
            valid_indices = torch.clamp(inputs[i], 0, self.feature_lens[i] - 1)
            embed_list.append(self.embedding_layers[i](valid_indices))
        
        # 堆叠嵌入 [batch_size, embedding_dim, num_feat]
        stacked_embeds = torch.stack(embed_list, dim=2)
        
        # 计算注意力权重
        # 为每个节点创建特征ID向量
        feature_ids = torch.arange(self.num_feat, device=inputs[0].device).expand(batch_size, self.num_feat)
        attention_weights = self.attention_network(feature_ids.float())
        
        # 应用注意力权重 [batch_size, embedding_dim, 1]
        attention_weights = attention_weights.unsqueeze(1)
        weighted_embeds = torch.matmul(stacked_embeds, attention_weights.transpose(1, 2))
        
        # 最终嵌入 [batch_size, embedding_dim]
        node_embeddings = weighted_embeds.squeeze(2)
        
        # 如果提供了上下文索引，同时计算上下文嵌入
        if context_indices is not None:
            # 确保上下文索引在有效范围内
            valid_context_indices = torch.clamp(context_indices, 0, self.num_nodes - 1)
            context_embeddings = self.node_embeddings(valid_context_indices)
            return node_embeddings, context_embeddings
        
        return node_embeddings
    
    def compute_loss(self, node_embeddings, context_embeddings, labels=None):
        """
        计算损失函数
        使用负采样的Skip-gram模型
        
        参数:
        node_embeddings: 节点嵌入
        context_embeddings: 上下文嵌入
        labels: 标签（可选）
        
        返回:
        loss: 损失值
        """
        batch_size = node_embeddings.size(0)
        
        # 计算正样本得分
        pos_score = torch.sum(node_embeddings * context_embeddings, dim=1)
        pos_score = torch.sigmoid(pos_score)
        
        # 计算负样本得分
        # 使用批次内其他样本作为负样本
        neg_score = torch.matmul(node_embeddings, context_embeddings.t())
        neg_score = torch.sigmoid(neg_score)
        
        # 创建标签矩阵
        neg_mask = torch.ones((batch_size, batch_size), device=node_embeddings.device) - torch.eye(batch_size, device=node_embeddings.device)
        
        # 计算正样本损失
        pos_loss = -torch.log(pos_score + 1e-10).mean()
        
        # 计算负样本损失
        neg_loss = -torch.sum(torch.log(1 - neg_score + 1e-10) * neg_mask) / (batch_size * (batch_size - 1))
        
        # 总损失
        loss = pos_loss + neg_loss
        
        return loss
    
    def get_embeddings(self, inputs):
        """
        获取节点嵌入
        
        参数:
        inputs: 包含多个特征列的列表
        
        返回:
        embeddings: 节点嵌入
        """
        with torch.no_grad():
            return self.forward(inputs)
    
    def train_step(self, inputs, labels):
        """
        训练一步
        
        参数:
        inputs: 包含多个特征列的列表
        labels: 标签
        
        返回:
        loss: 损失值
        """
        self.train()
        self.optimizer.zero_grad()
        
        # 获取节点嵌入和上下文嵌入
        node_embeddings, context_embeddings = self.forward(inputs, labels)
        
        # 计算损失
        loss = self.compute_loss(node_embeddings, context_embeddings, labels)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        
        # 更新参数
        self.optimizer.step()
        
        return loss.item()
    
    def update_lr(self, val_loss):
        """
        更新学习率
        
        参数:
        val_loss: 验证损失
        """
        if self.scheduler is not None:
            self.scheduler.step(val_loss)
    
    def sync_parameters(self):
        """
        在分布式训练中同步模型参数
        """
        for param in self.parameters():
            dist.broadcast(param.data, 0)
    
    def reduce_gradients(self):
        """
        在分布式训练中归约梯度
        """
        for param in self.parameters():
            if param.requires_grad:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= dist.get_world_size() 