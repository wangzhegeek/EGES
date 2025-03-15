import pandas as pd
import numpy as np
import torch
import time
import argparse
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import sys
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入公共模块
from common.EGES_model import EGES_Model
from common.utils import set_seed, setup, cleanup, get_free_port, save_dict_to_file, write_embedding, visualize_embeddings
from common.data_process import get_session, create_dataloader, get_graph_context_all_pairs
from common.walker import FastGraphWalker, SimpleWalker


def train_model(rank, world_size, args):
    """
    在指定GPU上训练模型
    
    参数:
    rank: 当前进程的排名
    world_size: 总进程数
    args: 命令行参数
    """
    # 设置分布式训练环境
    setup(rank, world_size, args.master_addr, args.master_port)
    
    # 设置随机种子 - 所有进程使用相同的种子以确保一致性
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device(f"cuda:{rank}")
    
    # 数据加载和预处理
    if rank == 0:
        print(f"进程 {rank}: 开始数据加载和预处理")
        
        # 读取数据
        start_time = time.time()
        action_data = pd.read_csv(os.path.join(args.data_path, 'action_head.csv'))
        print(f"读取数据完成，耗时: {time.time() - start_time:.2f}秒")
        
        # 构建会话
        start_time = time.time()
        session_list = get_session(action_data)
        print(f"构建会话完成，耗时: {time.time() - start_time:.2f}秒")
        
        # 构建图并生成随机游走
        start_time = time.time()
        if args.use_simple_walker:
            # 使用与单GPU版本相同的SimpleWalker
            walker = SimpleWalker(p=args.p, q=args.q)
            G, node_maps = walker.build_graph(session_list)
            
            if G is None:
                print("图构建失败，退出程序")
                cleanup()
                return
            
            node_map, reverse_node_map = node_maps
            
            # 生成随机游走
            walks = walker.generate_walks(G, args.num_walks, args.walk_length)
            
            # 生成上下文对
            all_pairs = get_graph_context_all_pairs(walks, args.window_size)
        else:
            # 使用FastGraphWalker
            walker = FastGraphWalker(p=args.p, q=args.q, device=device)
            pyg_data, node_maps = walker.build_graph(session_list)
            
            if pyg_data is None:
                print("图构建失败，退出程序")
                cleanup()
                return
            
            node_map, reverse_node_map = node_maps
            
            # 生成随机游走和样本对
            all_pairs = walker.generate_walks(pyg_data, args.num_walks, args.walk_length)
        
        print(f"图构建和随机游走完成，耗时: {time.time() - start_time:.2f}秒")
        print(f"生成的样本对数量: {len(all_pairs)}")
        
        # 读取SKUsideinfo
        start_time = time.time()
        sku_info = pd.read_csv(os.path.join(args.data_path, 'jdata_product.csv'))
        
        # 提取特征
        side_info = sku_info[['sku_id', 'cate', 'brand', 'shop_id']].values
        
        # 创建特征长度列表
        feature_lens = []
        for i in range(side_info.shape[1]):
            tmp_len = len(set(side_info[:, i])) + 1  # 加1是为了处理未知值
            feature_lens.append(tmp_len)
        
        print(f"读取SKUsideinfo完成，耗时: {time.time() - start_time:.2f}秒")
    else:
        # 非主进程等待主进程完成数据加载
        session_list = None
        all_pairs = None
        side_info = None
        node_map = None
        reverse_node_map = None
        feature_lens = None
    
    # 广播数据到所有进程
    if world_size > 1:
        if rank == 0:
            print("广播数据到所有进程...")
        
        # 将数据转换为张量以便广播
        if rank == 0:
            # 将数据转换为张量并移动到GPU
            all_pairs_tensor = torch.tensor(all_pairs, dtype=torch.long).to(device).contiguous()
            side_info_tensor = torch.tensor(side_info, dtype=torch.long).to(device).contiguous()
            feature_lens_tensor = torch.tensor(feature_lens, dtype=torch.long).to(device).contiguous()
            
            # 将字典转换为列表以便广播
            node_map_keys = torch.tensor(list(node_map.keys()), dtype=torch.long).to(device).contiguous()
            node_map_values = torch.tensor(list(node_map.values()), dtype=torch.long).to(device).contiguous()
            
            reverse_node_map_keys = torch.tensor(list(reverse_node_map.keys()), dtype=torch.long).to(device).contiguous()
            reverse_node_map_values = torch.tensor(list(reverse_node_map.values()), dtype=torch.long).to(device).contiguous()
            
            # 广播张量大小
            all_pairs_size = torch.tensor(all_pairs_tensor.size(), dtype=torch.long).to(device).contiguous()
            side_info_size = torch.tensor(side_info_tensor.size(), dtype=torch.long).to(device).contiguous()
            node_map_size = torch.tensor(len(node_map), dtype=torch.long).to(device).contiguous()
        else:
            # 非主进程创建空张量
            all_pairs_size = torch.zeros(2, dtype=torch.long).to(device).contiguous()
            side_info_size = torch.zeros(2, dtype=torch.long).to(device).contiguous()
            node_map_size = torch.zeros(1, dtype=torch.long).to(device).contiguous()
        
        # 广播张量大小
        dist.broadcast(all_pairs_size, 0)
        dist.broadcast(side_info_size, 0)
        dist.broadcast(node_map_size, 0)
        
        # 非主进程根据大小创建张量
        if rank != 0:
            all_pairs_tensor = torch.zeros(all_pairs_size[0], all_pairs_size[1], dtype=torch.long).to(device).contiguous()
            side_info_tensor = torch.zeros(side_info_size[0], side_info_size[1], dtype=torch.long).to(device).contiguous()
            feature_lens_tensor = torch.zeros(side_info_size[1], dtype=torch.long).to(device).contiguous()
            
            node_map_keys = torch.zeros(node_map_size.item(), dtype=torch.long).to(device).contiguous()
            node_map_values = torch.zeros(node_map_size.item(), dtype=torch.long).to(device).contiguous()
            
            reverse_node_map_keys = torch.zeros(node_map_size.item(), dtype=torch.long).to(device).contiguous()
            reverse_node_map_values = torch.zeros(node_map_size.item(), dtype=torch.long).to(device).contiguous()
        
        # 广播数据
        dist.broadcast(all_pairs_tensor, 0)
        dist.broadcast(side_info_tensor, 0)
        dist.broadcast(feature_lens_tensor, 0)
        dist.broadcast(node_map_keys, 0)
        dist.broadcast(node_map_values, 0)
        dist.broadcast(reverse_node_map_keys, 0)
        dist.broadcast(reverse_node_map_values, 0)
        
        # 将张量转换回原始格式
        all_pairs = all_pairs_tensor.cpu().numpy().tolist()
        side_info = side_info_tensor.cpu().numpy()
        feature_lens = feature_lens_tensor.cpu().numpy().tolist()
        
        # 重建字典
        node_map = {k.item(): v.item() for k, v in zip(node_map_keys.cpu(), node_map_values.cpu())}
        reverse_node_map = {k.item(): v.item() for k, v in zip(reverse_node_map_keys.cpu(), reverse_node_map_values.cpu())}
    
    # 创建数据加载器
    dataloader = create_dataloader(
        side_info=side_info,
        pairs=all_pairs,
        node_map=node_map,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed=(world_size > 1),
        world_size=world_size,
        rank=rank,
        drop_last=False
    )
    
    # 根据GPU数量调整学习率
    effective_lr = args.lr
    if world_size > 1:
        effective_lr = args.lr * world_size
        if rank == 0:
            print(f"根据GPU数量调整学习率: {args.lr} -> {effective_lr}")
    
    # 创建模型
    model = EGES_Model(
        num_nodes=len(node_map),
        num_feat=side_info.shape[1],
        feature_lens=feature_lens,
        embedding_dim=args.embedding_dim,
        lr=effective_lr
    ).to(device)
    
    # 如果是分布式训练，使用DDP包装模型
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # 初始化优化器
    model.module.init_optimizer() if world_size > 1 else model.init_optimizer()
    
    # 训练模型
    if rank == 0:
        print("开始训练...")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        # 设置数据加载器的epoch
        if world_size > 1 and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
        
        # 使用tqdm显示进度条（仅在主进程）
        if rank == 0:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        else:
            pbar = dataloader
        
        for batch_idx, (features, contexts) in enumerate(pbar):
            # 将数据移到设备上
            features = features.to(device)
            contexts = contexts.to(device)
            
            # 将特征拆分为多个列
            feature_columns = [features[:, i] for i in range(features.size(1))]
            
            # 训练一步
            if world_size > 1:
                # 在DDP模式下，forward和backward是自动同步的
                loss = model.module.train_step(feature_columns, contexts)
                
                # 确保梯度同步 (DDP已经自动处理，但为了确保一致性，我们可以显式同步)
                if args.explicit_sync and batch_idx % args.sync_every == 0:
                    for param in model.parameters():
                        if param.requires_grad and param.grad is not None:
                            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                            param.grad.data /= world_size
            else:
                loss = model.train_step(feature_columns, contexts)
            
            total_loss += loss
            
            # 更新进度条（仅在主进程）
            if rank == 0 and isinstance(pbar, tqdm):
                pbar.set_postfix({'loss': f'{loss:.4f}'})
        
        # 打印每轮的平均损失（仅在主进程）
        avg_loss = total_loss / len(dataloader)
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}")
        
        # 更新学习率
        if world_size > 1:
            model.module.update_lr(avg_loss)
        else:
            model.update_lr(avg_loss)
        
        # 保存模型（仅在主进程）
        if rank == 0 and ((epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1):
            checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            model_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
            
            # 保存模型状态
            model_state = model.module.state_dict() if world_size > 1 else model.state_dict()
            optimizer_state = model.module.optimizer.state_dict() if world_size > 1 else model.optimizer.state_dict()
            
            torch.save({
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer_state,
                'epoch': epoch,
                'loss': avg_loss,
                'node_map': node_map,
                'reverse_node_map': reverse_node_map
            }, model_path)
            
            # 如果是最后一轮，保存为final模型
            if epoch == args.epochs - 1:
                final_model_path = os.path.join(checkpoint_dir, "model_final.pt")
                torch.save({
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer_state,
                    'epoch': epoch,
                    'loss': avg_loss,
                    'node_map': node_map,
                    'reverse_node_map': reverse_node_map
                }, final_model_path)
    
    # 保存嵌入（仅在主进程）
    if rank == 0:
        print("保存嵌入...")
        embedding_dir = os.path.join(args.output_dir, 'embedding')
        os.makedirs(embedding_dir, exist_ok=True)
        
        # 获取嵌入
        embeddings = model.module.node_embeddings.weight.detach().cpu().numpy() if world_size > 1 else model.node_embeddings.weight.detach().cpu().numpy()
        
        # 将嵌入映射回原始节点ID
        node_embeddings = {}
        for idx, node_id in reverse_node_map.items():
            node_embeddings[node_id] = embeddings[idx]
        
        # 保存嵌入到文件
        np.save(os.path.join(embedding_dir, "node_embeddings.npy"), node_embeddings)
        
        # 保存节点映射
        save_dict_to_file(node_map, os.path.join(embedding_dir, "node_map.txt"))
        save_dict_to_file(reverse_node_map, os.path.join(embedding_dir, "reverse_node_map.txt"))
        
        # 将嵌入写入文本文件
        embedding_file = os.path.join(embedding_dir, "node_embeddings.txt")
        write_embedding([node_embeddings[node_id] for node_id in sorted(node_embeddings.keys())], embedding_file)
        
        # 如果需要可视化，则调用可视化函数
        if args.visualize:
            visualize_embeddings(node_embeddings, args.data_path, args.output_dir)
        
        print("训练完成！")
    
    # 清理分布式环境
    cleanup()


def main():
    parser = argparse.ArgumentParser(description='多GPU版EGES实现')
    parser.add_argument('--data_path', type=str, default='./data/', help='数据文件路径')
    parser.add_argument('--output_dir', type=str, default='./output/multi_gpu/', help='输出目录')
    parser.add_argument('--p', type=float, default=0.25, help='返回参数')
    parser.add_argument('--q', type=float, default=2, help='进出参数')
    parser.add_argument('--num_walks', type=int, default=10, help='每个节点的游走次数')
    parser.add_argument('--walk_length', type=int, default=10, help='每次游走的长度')
    parser.add_argument('--window_size', type=int, default=5, help='上下文窗口大小')
    parser.add_argument('--embedding_dim', type=int, default=128, help='嵌入维度')
    parser.add_argument('--batch_size', type=int, default=512, help='批次大小')
    parser.add_argument('--epochs', type=int, default=2, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--gpus', type=int, default=-1, help='使用的GPU数量，-1表示使用所有可用GPU')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载的工作进程数')
    parser.add_argument('--save_every', type=int, default=1, help='每多少轮保存一次模型')
    parser.add_argument('--master_addr', type=str, default='localhost', help='主节点地址')
    parser.add_argument('--master_port', type=str, default=None, help='主节点端口')
    parser.add_argument('--visualize', action='store_true', help='是否可视化嵌入向量')
    parser.add_argument('--use_simple_walker', action='store_true', help='使用SimpleWalker而不是FastGraphWalker')
    parser.add_argument('--explicit_sync', action='store_true', help='是否显式同步梯度')
    parser.add_argument('--sync_every', type=int, default=10, help='每多少批次同步一次梯度')
    parser.add_argument('--extra_epochs', action='store_true', help='是否增加训练轮数')
    args = parser.parse_args()
    
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
    
    # 如果使用多GPU，增加训练轮数
    if args.gpus > 1 and args.extra_epochs:
        original_epochs = args.epochs
        args.epochs = int(args.epochs * 1.5)  # 增加50%的训练轮数
        print(f"多GPU训练: 增加训练轮数 {original_epochs} -> {args.epochs}")
    
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