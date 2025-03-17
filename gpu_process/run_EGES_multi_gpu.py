import pandas as pd
import numpy as np
import torch
import time
import argparse
import os
import sys
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import pickle
import networkx as nx
from torch_geometric.utils import from_networkx
import io
import logging

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 导入集成版EGES
from gpu_process.EGES_module import EGES, EGESTrainer
from utils import set_seed, setup, cleanup, get_free_port, write_embedding, visualize_embeddings
from data_process import get_session


# 设置日志
def setup_logger(rank, log_dir=None):
    """
    设置日志
    
    参数:
    rank: 进程排名
    log_dir: 日志目录
    """
    logger = logging.getLogger(f"rank_{rank}")
    logger.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter(f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(console_handler)
    
    # 如果指定了日志目录，也添加文件处理器
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, f"rank_{rank}.log"))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def build_global_graph(session_list, logger):
    """
    构建全局图，确保所有进程使用相同的图结构
    
    参数:
    session_list: 会话列表
    logger: 日志记录器
    
    返回:
    G: NetworkX图
    node_maps: 节点映射元组 (node_map, reverse_node_map)
    """
    logger.info("构建全局图...")
    start_time = time.time()
    
    # 提取所有边 - 使用集合去重
    edge_set = set()
    for session in session_list:
        if len(session) > 1:
            for i in range(len(session) - 1):
                u = int(session[i])
                v = int(session[i + 1])
                edge_set.add((u, v))
    
    edges = list(edge_set)
    logger.info(f"提取的边数量: {len(edges)}")
    if len(edges) == 0:
        logger.warning("警告: 没有提取到边，无法构建图")
        return None, None
    
    # 创建NetworkX图
    logger.info("创建NetworkX图...")
    G = nx.Graph()
    G.add_edges_from(edges)
    
    # 获取所有唯一节点
    logger.info("创建节点映射...")
    nodes = list(G.nodes())
    node_map = {node: i for i, node in enumerate(nodes)}
    reverse_node_map = {i: node for i, node in enumerate(nodes)}
    
    # 重新映射节点ID
    logger.info("重新映射节点ID...")
    G_relabeled = nx.relabel_nodes(G, node_map)
    
    end_time = time.time()
    logger.info(f"全局图构建完成，耗时: {end_time - start_time:.2f}秒")
    logger.info(f"全局图包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边")
    
    return G_relabeled, (node_map, reverse_node_map)


def fast_serialize(obj):
    """
    快速序列化对象
    
    参数:
    obj: 要序列化的对象
    
    返回:
    bytes: 序列化后的字节
    """
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.getvalue()


def fast_deserialize(bytes_obj):
    """
    快速反序列化对象
    
    参数:
    bytes_obj: 序列化后的字节
    
    返回:
    obj: 反序列化后的对象
    """
    buffer = io.BytesIO(bytes_obj)
    return torch.load(buffer)


def train_model(rank, world_size, args):
    """
    在指定GPU上训练模型
    
    参数:
    rank: 当前进程的排名
    world_size: 总进程数
    args: 命令行参数
    """
    # 设置日志
    log_dir = os.path.join(args.output_dir, 'logs')
    logger = setup_logger(rank, log_dir)
    
    # 设置分布式训练环境
    logger.info(f"初始化分布式环境...")
    start_time = time.time()
    setup(rank, world_size, args.master_addr, args.master_port)
    logger.info(f"分布式环境初始化完成，耗时: {time.time() - start_time:.2f}秒")
    
    # 设置随机种子 - 每个进程使用不同的种子以增加多样性
    set_seed(args.seed + rank)
    
    # 设置设备
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # 数据加载和预处理 - 只在主进程进行
    if rank == 0:
        logger.info(f"开始数据加载和预处理")
        
        # 读取数据
        start_time = time.time()
        action_data = pd.read_csv(os.path.join(args.data_path, 'action_head.csv'))
        logger.info(f"读取数据完成，耗时: {time.time() - start_time:.2f}秒")
        
        # 构建会话
        start_time = time.time()
        session_list = get_session(action_data)
        logger.info(f"构建会话完成，耗时: {time.time() - start_time:.2f}秒")
        
        # 读取SKUsideinfo
        start_time = time.time()
        sku_info = pd.read_csv(os.path.join(args.data_path, 'jdata_product.csv'))
        side_info = sku_info[['sku_id', 'cate', 'brand', 'shop_id']].values
        logger.info(f"读取SKUsideinfo完成，耗时: {time.time() - start_time:.2f}秒")
        
        # 构建全局图 - 确保所有进程使用相同的图结构
        G, node_maps = build_global_graph(session_list, logger)
        node_map, reverse_node_map = node_maps
        
        # 转换为PyG图
        logger.info("转换为PyG图...")
        start_time = time.time()
        pyg_data = from_networkx(G)
        logger.info(f"PyG图转换完成，耗时: {time.time() - start_time:.2f}秒")
        
        # 分割数据 - 为每个进程准备一部分数据
        if world_size > 1:
            logger.info("分割会话数据...")
            start_time = time.time()
            # 计算每个进程的会话数量
            sessions_per_process = len(session_list) // world_size
            session_splits = []
            
            # 分割会话列表
            for i in range(world_size):
                start_idx = i * sessions_per_process
                end_idx = (i + 1) * sessions_per_process if i < world_size - 1 else len(session_list)
                session_splits.append(session_list[start_idx:end_idx])
            
            # 更新主进程的会话列表
            session_list = session_splits[0]
            logger.info(f"会话数据分割完成，耗时: {time.time() - start_time:.2f}秒")
            logger.info(f"每个进程的会话数量: {[len(split) for split in session_splits]}")
        
        # 创建训练器
        logger.info("创建训练器...")
        start_time = time.time()
        trainer = EGESTrainer(
            session_list=session_list,
            side_info=side_info,
            embedding_dim=args.embedding_dim,
            walk_length=args.walk_length,
            context_size=args.context_size,
            walks_per_node=args.walks_per_node,
            p=args.p,
            q=args.q,
            lr=args.lr,
            device=device,
            prefetch_factor=args.prefetch_factor,
            # 使用预构建的图和节点映射
            G=G,
            node_map=node_map,
            reverse_node_map=reverse_node_map,
            pyg_data=pyg_data
        )
        logger.info(f"训练器创建完成，耗时: {time.time() - start_time:.2f}秒")
        
        # 准备要广播的数据
        if world_size > 1:
            logger.info("准备广播数据...")
            start_time = time.time()
            
            # 准备图数据
            graph_data = {
                'edge_index': trainer.pyg_data.edge_index.cpu(),
                'num_nodes': trainer.num_nodes,
                'node_map': trainer.node_map,
                'reverse_node_map': trainer.reverse_node_map,
                'feature_lens': trainer.feature_lens
            }
            
            # 如果有侧信息，也保存
            if trainer.side_info is not None:
                graph_data['side_info_dict'] = {k: v.cpu() for k, v in trainer.side_info_dict.items()}
            
            # 保存模型状态
            model_state = trainer.model.state_dict()
            
            # 序列化数据 - 使用快速序列化
            graph_data_bytes = fast_serialize(graph_data)
            model_state_bytes = fast_serialize(model_state)
            side_info_bytes = fast_serialize(side_info) if side_info is not None else None
            session_splits_bytes = fast_serialize(session_splits)
            
            # 准备广播列表
            broadcast_data = [
                graph_data_bytes,
                model_state_bytes,
                side_info_bytes,
                session_splits_bytes
            ]
            
            logger.info(f"广播数据准备完成，耗时: {time.time() - start_time:.2f}秒")
            logger.info(f"广播数据大小: {sum(len(x) if x else 0 for x in broadcast_data) / (1024 * 1024):.2f} MB")
        else:
            broadcast_data = None
            session_splits = None
    else:
        # 非主进程初始化为None
        trainer = None
        session_list = None
        broadcast_data = [None, None, None, None]
        side_info = None
    
    # 广播数据到所有进程
    if world_size > 1:
        logger.info("开始广播数据...")
        start_time = time.time()
        
        # 广播数据大小
        if rank == 0:
            sizes = [len(x) if x else 0 for x in broadcast_data]
        else:
            sizes = [0, 0, 0, 0]
        
        # 使用all_gather广播大小
        size_tensor = torch.tensor(sizes, dtype=torch.long, device=device)
        dist.broadcast(size_tensor, 0)
        sizes = size_tensor.tolist()
        
        # 根据大小创建接收缓冲区
        if rank != 0:
            broadcast_data = [
                torch.zeros(sizes[0], dtype=torch.uint8, device=device) if sizes[0] > 0 else None,
                torch.zeros(sizes[1], dtype=torch.uint8, device=device) if sizes[1] > 0 else None,
                torch.zeros(sizes[2], dtype=torch.uint8, device=device) if sizes[2] > 0 else None,
                torch.zeros(sizes[3], dtype=torch.uint8, device=device) if sizes[3] > 0 else None
            ]
        else:
            # 转换为张量
            broadcast_data = [
                torch.frombuffer(x, dtype=torch.uint8).to(device) if x else None
                for x in broadcast_data
            ]
        
        # 广播数据
        for i in range(4):
            if sizes[i] > 0:
                dist.broadcast(broadcast_data[i], 0)
        
        # 反序列化数据
        if rank != 0:
            # 转换为字节
            broadcast_data = [
                x.cpu().numpy().tobytes() if x is not None else None
                for x in broadcast_data
            ]
            
            # 反序列化
            graph_data = fast_deserialize(broadcast_data[0])
            model_state = fast_deserialize(broadcast_data[1])
            side_info = fast_deserialize(broadcast_data[2]) if broadcast_data[2] else None
            session_splits = fast_deserialize(broadcast_data[3])
            
            # 使用分配给当前进程的会话列表
            session_list = session_splits[rank]
            
            # 创建训练器
            logger.info("创建训练器...")
            start_time_trainer = time.time()
            
            # 从图数据中提取图和节点映射
            G = nx.Graph()
            edge_index = graph_data['edge_index']
            edges = [(edge_index[0, i].item(), edge_index[1, i].item()) for i in range(edge_index.size(1))]
            G.add_edges_from(edges)
            
            # 创建训练器 - 使用预构建的图和节点映射
            trainer = EGESTrainer(
                session_list=session_list,
                side_info=side_info,
                embedding_dim=args.embedding_dim,
                walk_length=args.walk_length,
                context_size=args.context_size,
                walks_per_node=args.walks_per_node,
                p=args.p,
                q=args.q,
                lr=args.lr,
                device=device,
                prefetch_factor=args.prefetch_factor,
                # 使用预构建的图和节点映射，确保所有进程使用相同的图结构
                G=G,
                node_map=graph_data['node_map'],
                reverse_node_map=graph_data['reverse_node_map'],
                pyg_data=None  # 稍后会更新
            )
            
            # 更新图结构和节点映射
            trainer.num_nodes = graph_data['num_nodes']
            trainer.node_map = graph_data['node_map']
            trainer.reverse_node_map = graph_data['reverse_node_map']
            trainer.feature_lens = graph_data['feature_lens']
            
            # 更新PyG数据
            trainer.pyg_data = from_networkx(G).to(device)
            trainer.pyg_data.edge_index = edge_index.to(device)
            
            # 重新创建模型以确保结构一致
            trainer.model = EGES(
                edge_index=trainer.pyg_data.edge_index,
                num_nodes=trainer.num_nodes,
                feature_dim=len(trainer.feature_lens),
                feature_lens=trainer.feature_lens,
                embedding_dim=args.embedding_dim,
                walk_length=args.walk_length,
                context_size=args.context_size,
                walks_per_node=args.walks_per_node,
                p=args.p,
                q=args.q,
                lr=args.lr,
                device=device
            )
            
            # 初始化优化器
            trainer.model.init_optimizer(args.lr)
            
            # 如果有侧信息，也更新
            if 'side_info_dict' in graph_data:
                trainer.side_info_dict = {k: v.to(device) for k, v in graph_data['side_info_dict'].items()}
            
            # 加载模型状态
            trainer.model.load_state_dict(model_state)
            
            logger.info(f"训练器创建完成，耗时: {time.time() - start_time_trainer:.2f}秒")
        
        logger.info(f"数据广播完成，耗时: {time.time() - start_time:.2f}秒")
    
    # 同步所有进程
    if world_size > 1:
        dist.barrier()
        logger.info("所有进程同步完成，准备开始训练")
    
    # 训练模型
    if rank == 0:
        logger.info("开始训练...")
    
    # 获取数据加载器
    loader = trainer.model.get_loader(batch_size=args.batch_size, shuffle=True)
    
    # 记录每个epoch的损失
    epoch_losses = []
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        trainer.model.train()
        total_loss = 0
        batch_count = 0
        
        # 使用tqdm显示进度条（仅在主进程）
        if rank == 0:
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        else:
            pbar = loader
        
        # 每个进程独立训练，不同步梯度
        for pos_rw, neg_rw in pbar:
            # 训练一步
            loss = trainer.model.train_step(pos_rw, neg_rw)
            
            # 累积损失和批次计数
            total_loss += loss
            batch_count += 1
            
            # 更新进度条（仅在主进程）
            if rank == 0 and isinstance(pbar, tqdm):
                pbar.set_postfix({'loss': f'{loss:.4f}'})
        
        # 计算平均损失
        avg_loss = total_loss / batch_count
        
        # 同步损失
        if world_size > 1:
            avg_loss_tensor = torch.tensor([avg_loss], device=device)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss_tensor.item() / world_size
        
        # 记录损失
        epoch_losses.append(avg_loss)
        
        # 更新学习率
        trainer.model.update_lr(avg_loss)
        
        # 打印每轮的平均损失（仅在主进程）
        if rank == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}")
        
        # 同步模型参数 - 使用模型平均
        if world_size > 1:
            logger.info(f"开始同步模型参数...")
            sync_start_time = time.time()
            
            # 收集所有进程的模型参数
            model_state = trainer.model.state_dict()
            
            # 对于每个参数，进行all_reduce操作
            for key in model_state:
                if 'embedding' in key or 'attention' in key:  # 只平均嵌入和注意力参数
                    # 确保参数在GPU上
                    param = model_state[key].to(device).float()
                    
                    # 使用all_reduce操作平均参数
                    dist.all_reduce(param, op=dist.ReduceOp.SUM)
                    param.div_(world_size)
                    
                    # 更新模型状态
                    model_state[key] = param
            
            # 将平均后的参数加载回模型
            trainer.model.load_state_dict(model_state)
            
            logger.info(f"模型参数同步完成，耗时: {time.time() - sync_start_time:.2f}秒")
        
        # 清理缓存并等待所有GPU操作完成
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        
        logger.info(f"Epoch {epoch+1} 完成，总耗时: {time.time() - epoch_start_time:.2f}秒")
    
    # 保存模型和嵌入（仅在主进程）
    if rank == 0:
        # 保存模型
        checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        trainer.save_model(os.path.join(checkpoint_dir, 'model_final.pt'))
        
        # 绘制损失曲线
        from utils import plot_loss_curve
        plot_loss_curve(epoch_losses, output_dir=args.output_dir)
        
        # 获取嵌入
        logger.info("保存嵌入...")
        embedding_dir = os.path.join(args.output_dir, 'embedding')
        os.makedirs(embedding_dir, exist_ok=True)
        
        # 获取节点嵌入 - 使用较大的批次大小
        node_embeddings = trainer.get_embeddings()
        
        # 使用numpy的高效操作处理嵌入
        sorted_node_ids = sorted(node_embeddings.keys())
        embedding_array = np.stack([node_embeddings[node_id] for node_id in sorted_node_ids])
        
        # 并行保存不同格式的嵌入
        np.save(os.path.join(embedding_dir, "node_embeddings.npy"), embedding_array)
        write_embedding(embedding_array, os.path.join(embedding_dir, "node_embeddings.txt"))
        
        # 如果需要可视化，则调用可视化函数
        if args.visualize:
            visualize_embeddings(node_embeddings, args.data_path, args.output_dir)
        
        logger.info("训练完成！")
    
    # 清理分布式环境
    cleanup()


def main():
    parser = argparse.ArgumentParser(description='多GPU版集成EGES实现')
    parser.add_argument('--data_path', type=str, default='./data/', help='数据文件路径')
    parser.add_argument('--output_dir', type=str, default='./output/integrated_multi_gpu/', help='输出目录')
    parser.add_argument('--p', type=float, default=0.25, help='返回参数')
    parser.add_argument('--q', type=float, default=2, help='进出参数')
    parser.add_argument('--walk_length', type=int, default=10, help='随机游走长度')
    parser.add_argument('--context_size', type=int, default=5, help='上下文窗口大小')
    parser.add_argument('--walks_per_node', type=int, default=10, help='每个节点的游走次数')
    parser.add_argument('--embedding_dim', type=int, default=128, help='嵌入维度')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--gpus', type=int, default=-1, help='使用的GPU数量，-1表示使用所有可用GPU')
    parser.add_argument('--master_addr', type=str, default='localhost', help='主节点地址')
    parser.add_argument('--master_port', type=str, default=None, help='主节点端口')
    parser.add_argument('--visualize', action='store_true', help='是否可视化嵌入向量')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='数据预取因子')
    args = parser.parse_args()
    
    # 设置CUDA后端为NCCL
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(torch.cuda.device_count())))
    torch.backends.cudnn.benchmark = True  # 启用cuDNN自动调优
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 确定可用的GPU数量
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("没有可用的GPU，将使用CPU训练")
        args.gpus = 0
    elif args.gpus == -1 or args.gpus > num_gpus:
        args.gpus = num_gpus
        print(f"将使用所有 {num_gpus} 个可用的GPU")
    else:
        print(f"将使用 {args.gpus} 个GPU")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 如果没有指定端口，获取一个可用端口
    if args.master_port is None:
        args.master_port = get_free_port()
    
    # 单GPU或CPU训练
    if args.gpus <= 1:
        train_model(0, 1, args)
    else:
        # 多GPU训练
        mp.spawn(
            train_model,
            args=(args.gpus, args),
            nprocs=args.gpus,
            join=True
        )


if __name__ == "__main__":
    main() 