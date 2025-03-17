import pandas as pd
import numpy as np
from itertools import chain
import time
import networkx as nx
import torch
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import multiprocessing
from datetime import datetime


def cnt_session(data, time_cut=30, cut_type=2):
    """
    根据时间间隔和操作类型划分会话
    
    参数:
    data: 包含用户行为数据的DataFrame
    time_cut: 时间间隔阈值（分钟）
    cut_type: 切分会话的行为类型
    
    返回:
    session: 会话列表
    """
    sku_list = data['sku_id']
    time_list = data['action_time']
    type_list = data['type']
    
    # 确保时间列表中的元素是datetime对象
    time_list = [pd.to_datetime(t) if isinstance(t, str) else t for t in time_list]
    
    session = []
    tmp_session = []
    for i, item in enumerate(sku_list):
        # 检查是否是最后一个元素或者是否是切分类型
        if type_list[i] == cut_type or i == len(sku_list)-1:
            tmp_session.append(item)
            session.append(tmp_session)
            tmp_session = []
        # 检查与下一个元素的时间间隔
        elif i < len(sku_list)-1:
            # 计算时间差（分钟）
            time_diff = (time_list[i+1] - time_list[i]).total_seconds() / 60
            if time_diff > time_cut:
                tmp_session.append(item)
                session.append(tmp_session)
                tmp_session = []
            else:
                tmp_session.append(item)
    
    return session


def get_session(action_data, use_type=None, verbose=True):
    """
    获取会话列表
    
    参数:
    action_data: 用户行为数据
    use_type: 要使用的行为类型列表
    verbose: 是否打印详细信息
    
    返回:
    flattened_sessions: 展平后的会话列表
    """
    if verbose:
        print("原始数据形状:", action_data.shape)
        print("原始数据列:", action_data.columns)
    
    # 将时间字段转换为datetime类型
    action_data['action_time'] = pd.to_datetime(action_data['action_time'])
    
    if use_type is None:
        use_type = [1, 2, 3, 5]
    action_data = action_data[action_data['type'].isin(use_type)]
    if verbose:
        print("过滤后数据形状:", action_data.shape)
    
    action_data = action_data.sort_values(by=['user_id', 'action_time'], ascending=True)
    group_action_data = action_data.groupby('user_id').agg(list)
    if verbose:
        print("分组后数据形状:", group_action_data.shape)
    
    session_list = group_action_data.apply(cnt_session, axis=1)
    session_list = session_list.to_numpy()
    if verbose:
        print("会话列表长度:", len(session_list))
    
    # 展平会话列表
    flattened_sessions = []
    for sessions in session_list:
        flattened_sessions.extend(sessions)
    if verbose:
        print("处理后的会话数量:", len(flattened_sessions))
    
    return flattened_sessions


def get_graph_context_all_pairs(walks, window_size):
    """
    根据游走序列生成上下文对
    使用并行处理提高效率
    
    参数:
    walks: 随机游走序列
    window_size: 上下文窗口大小
    
    返回:
    all_pairs: 所有上下文对
    """
    # 确定CPU核心数
    num_cores = multiprocessing.cpu_count()
    # 将walks分成num_cores份
    chunk_size = max(1, len(walks) // num_cores)
    chunks = [walks[i:i+chunk_size] for i in range(0, len(walks), chunk_size)]
    
    # 创建进程池
    pool = multiprocessing.Pool(processes=num_cores)
    
    # 并行处理每个chunk
    results = pool.starmap(_process_walk_chunk, [(chunk, window_size) for chunk in chunks])
    
    # 关闭进程池
    pool.close()
    pool.join()
    
    # 合并结果
    all_pairs = []
    for result in results:
        all_pairs.extend(result)
    
    return all_pairs


def _process_walk_chunk(walks_chunk, window_size):
    """
    处理一个walks chunk，生成上下文对
    
    参数:
    walks_chunk: 随机游走序列的一部分
    window_size: 上下文窗口大小
    
    返回:
    pairs: 上下文对列表
    """
    pairs = []
    for walk in walks_chunk:
        if len(walk) <= 1:
            continue
            
        for i in range(len(walk)):
            for j in range(max(0, i - window_size), min(len(walk), i + window_size + 1)):
                if i != j:
                    pairs.append((walk[i], walk[j]))
    
    return pairs


class GraphDataset(Dataset):
    """
    图数据集类，用于加载图嵌入训练数据
    """
    def __init__(self, side_info, pairs, node_map):
        """
        初始化数据集
        
        参数:
        side_info: 节点sideinfo
        pairs: 节点对
        node_map: 节点映射
        """
        self.side_info = side_info
        self.pairs = pairs
        self.node_map = node_map
        
        # 创建SKU ID到索引的映射
        self.sku_to_idx = {}
        for i, row in enumerate(side_info):
            self.sku_to_idx[row[0]] = i
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        node, context = self.pairs[idx]
        
        # 确保节点在side_info中
        if node in self.sku_to_idx:
            node_idx = self.sku_to_idx[node]
            node_features = self.side_info[node_idx]
            
            # 确保上下文节点在node_map中
            if context in self.node_map:
                context_idx = self.node_map[context]
                # 将特征转换为张量并确保是长整型
                return torch.tensor(node_features, dtype=torch.long), torch.tensor(context_idx, dtype=torch.long)
        
        # 如果节点不在side_info中或上下文节点不在node_map中，返回一个默认值
        # 使用第一个节点作为默认值
        return torch.tensor(self.side_info[0], dtype=torch.long), torch.tensor(0, dtype=torch.long)


def create_dataloader(side_info, pairs, node_map, batch_size=512, num_workers=4, distributed=False, world_size=1, rank=0, drop_last=False):
    """
    创建数据加载器
    支持分布式训练
    
    参数:
    side_info: 节点sideinfo
    pairs: 节点对
    node_map: 节点映射
    batch_size: 批次大小
    num_workers: 工作进程数
    distributed: 是否使用分布式训练
    world_size: 总进程数
    rank: 当前进程的排名
    drop_last: 是否丢弃最后一个不完整的批次
    
    返回:
    dataloader: 数据加载器
    """
    dataset = GraphDataset(side_info, pairs, node_map)
    
    if distributed:
        # 创建分布式采样器
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            dataset, 
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last
        )
    
    return dataloader 