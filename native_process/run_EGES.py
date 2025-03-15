import pandas as pd
import numpy as np
import torch
import time
import argparse
import os
import sys

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入公共模块
from common.EGES_model import EGES_Model
from common.utils import set_seed, plot_embeddings, write_embedding, save_dict_to_file, visualize_embeddings
from common.data_process import get_session, create_dataloader, get_graph_context_all_pairs
from common.walker import SimpleWalker
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='EGES原生实现')
    parser.add_argument('--data_path', type=str, default='../data/')
    parser.add_argument('--output_dir', type=str, default='./')
    parser.add_argument('--p', type=float, default=0.25, help='返回参数')
    parser.add_argument('--q', type=float, default=2, help='进出参数')
    parser.add_argument('--num_walks', type=int, default=10, help='每个节点的游走次数')
    parser.add_argument('--walk_length', type=int, default=10, help='每次游走的长度')
    parser.add_argument('--window_size', type=int, default=5, help='上下文窗口大小')
    parser.add_argument('--embedding_dim', type=int, default=128, help='嵌入维度')
    parser.add_argument('--batch_size', type=int, default=8192, help='批次大小')
    parser.add_argument('--epochs', type=int, default=1, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--visualize', action='store_true', help='是否可视化嵌入向量')
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'embedding'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'data_cache'), exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据加载和预处理
    print("开始数据加载和预处理")
    
    # 读取数据
    print("读取数据...")
    start_time = time.time()
    action_data = pd.read_csv(args.data_path + 'action_head.csv')
    end_time = time.time()
    print(f"读取数据完成，耗时: {end_time - start_time:.2f}秒")
    
    # 构建会话
    print("构建会话...")
    start_time = time.time()
    session_list = get_session(action_data)
    end_time = time.time()
    print(f"构建会话完成，耗时: {end_time - start_time:.2f}秒")
    
    # 构建图并生成随机游走
    walker = SimpleWalker(p=args.p, q=args.q)
    G, node_maps = walker.build_graph(session_list)
    
    if G is None:
        print("图构建失败，退出程序")
        return
    
    node_map, reverse_node_map = node_maps
    
    # 生成随机游走
    walks = walker.generate_walks(G, args.num_walks, args.walk_length)
    
    # 生成上下文对
    print("生成上下文对...")
    start_time = time.time()
    all_pairs = get_graph_context_all_pairs(walks, args.window_size)
    end_time = time.time()
    print(f"生成上下文对完成，耗时: {end_time - start_time:.2f}秒")
    print(f"生成的样本对数量: {len(all_pairs)}")
    
    # 读取SKUsideinfo
    print("读取SKUsideinfo...")
    start_time = time.time()
    sku_info = pd.read_csv(args.data_path + 'jdata_product.csv')
    print(f"SKU信息形状: {sku_info.shape}")
    
    # 提取特征
    side_info = sku_info[['sku_id', 'cate', 'brand', 'shop_id']].values
    print(f"sideinfo形状: {side_info.shape}")
    
    # 创建特征长度列表
    feature_lens = []
    for i in range(side_info.shape[1]):
        tmp_len = len(set(side_info[:, i])) + 1  # 加1是为了处理未知值
        feature_lens.append(tmp_len)
    
    end_time = time.time()
    print(f"读取SKUsideinfo完成，耗时: {end_time - start_time:.2f}秒")
    
    # 创建数据加载器
    dataloader = create_dataloader(
        side_info=side_info,
        pairs=all_pairs,
        node_map=node_map,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    # 创建模型
    model = EGES_Model(
        num_nodes=len(node_map),
        num_feat=side_info.shape[1],
        feature_lens=feature_lens,
        embedding_dim=args.embedding_dim,
        lr=args.lr
    ).to(device)
    
    # 初始化优化器
    model.init_optimizer()
    
    # 训练模型
    print("开始训练...")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        # 使用tqdm显示进度条
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, (features, contexts) in enumerate(pbar):
            # 将数据移到设备上
            features = features.to(device)
            contexts = contexts.to(device)
            
            # 将特征拆分为多个列
            feature_columns = [features[:, i] for i in range(features.size(1))]
            
            # 训练一步
            loss = model.train_step(feature_columns, contexts)
            
            total_loss += loss
            
            # 打印进度
            if (batch_idx + 1) % 100 == 0:
                print(f"Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss:.4f}")
        
        # 打印每轮的平均损失
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}")
        
        # 更新学习率
        model.update_lr(avg_loss)
        
        # 保存模型
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            model_path = os.path.join(args.output_dir, 'checkpoints', f"model_epoch_{epoch+1}.pt")
            
            # 保存模型状态
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'node_map': node_map,
                'reverse_node_map': reverse_node_map
            }, model_path)
            
            # 如果是最后一轮，保存为final模型
            if epoch == args.epochs - 1:
                final_model_path = os.path.join(args.output_dir, 'checkpoints', "model_final.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': model.optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': avg_loss,
                    'node_map': node_map,
                    'reverse_node_map': reverse_node_map
                }, final_model_path)
    
    # 保存嵌入
    print("保存嵌入...")
    
    # 获取嵌入
    embeddings = model.node_embeddings.weight.detach().cpu().numpy()
    
    # 将嵌入映射回原始节点ID
    node_embeddings = {}
    for idx, node_id in reverse_node_map.items():
        node_embeddings[node_id] = embeddings[idx]
    
    # 保存嵌入到文件
    np.save(os.path.join(args.output_dir, 'embedding', "node_embeddings.npy"), node_embeddings)
    
    # 保存节点映射
    save_dict_to_file(node_map, os.path.join(args.output_dir, 'embedding', "node_map.txt"))
    save_dict_to_file(reverse_node_map, os.path.join(args.output_dir, 'embedding', "reverse_node_map.txt"))
    
    # 将嵌入写入文本文件
    print("将嵌入写入文本文件...")
    embedding_file = os.path.join(args.output_dir, 'embedding', "node_embeddings.txt")
    write_embedding([node_embeddings[node_id] for node_id in sorted(node_embeddings.keys())], embedding_file)
    
    # 如果需要可视化，则调用可视化函数
    if args.visualize:
        visualize_embeddings(node_embeddings, args.data_path, args.output_dir)
    
    print("训练完成！")


if __name__ == "__main__":
    main() 