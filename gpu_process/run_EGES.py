import pandas as pd
import numpy as np
import torch
import time
import argparse
import os
import sys
from tqdm import tqdm

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 导入集成版EGES
from gpu_process.EGES_module import EGESTrainer
from utils import set_seed, plot_loss_curve, write_embedding, visualize_embeddings
from data_process import get_session


def main():
    parser = argparse.ArgumentParser(description='集成版EGES实现')
    parser.add_argument('--data_path', type=str, default='./data/', help='数据文件路径')
    parser.add_argument('--output_dir', type=str, default='./output/integrated/', help='输出目录')
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
    parser.add_argument('--gpu', type=int, default=0, help='使用的GPU ID，-1表示使用CPU')
    parser.add_argument('--visualize', action='store_true', help='是否进行向量聚类可视化')
    parser.add_argument('--n_clusters', type=int, default=8, help='聚类的簇数量')
    parser.add_argument('--full_data', action='store_true', help='是否使用完整数据集进行训练')
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"使用GPU: {args.gpu}")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 读取数据
    print("开始数据加载和预处理")
    start_time = time.time()
    
    # 根据参数决定使用完整数据集还是样本数据
    if args.full_data:
        action_file = 'action.csv'
        print("使用完整数据集进行训练")
    else:
        action_file = 'action_head.csv'
        print("使用样本数据集进行训练")
        
    action_data = pd.read_csv(os.path.join(args.data_path, action_file))
    print(f"读取数据完成，耗时: {time.time() - start_time:.2f}秒")
    print(f"数据形状: {action_data.shape}")
    
    # 构建会话
    start_time = time.time()
    session_list = get_session(action_data)
    print(f"构建会话完成，耗时: {time.time() - start_time:.2f}秒")
    print(f"会话数量: {len(session_list)}")
    
    # 读取SKUsideinfo
    start_time = time.time()
    sku_info = pd.read_csv(os.path.join(args.data_path, 'jdata_product.csv'))
    side_info = sku_info[['sku_id', 'cate', 'brand', 'shop_id']].values
    print(f"读取SKUsideinfo完成，耗时: {time.time() - start_time:.2f}秒")
    
    # 创建训练器
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
        device=device
    )
    
    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )

    # 保存模型
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    trainer.save_model(os.path.join(checkpoint_dir, 'model_final.pt'))
    
    # 获取嵌入
    print("保存嵌入...")
    embedding_dir = os.path.join(args.output_dir, 'embedding')
    os.makedirs(embedding_dir, exist_ok=True)
    
    # 获取节点嵌入
    node_embeddings = trainer.get_embeddings()
    
    # 保存嵌入到文件
    np.save(os.path.join(embedding_dir, "node_embeddings.npy"), node_embeddings)
    
    # 将嵌入写入文本文件
    embedding_file = os.path.join(embedding_dir, "node_embeddings.txt")
    write_embedding([node_embeddings[node_id] for node_id in sorted(node_embeddings.keys())], embedding_file)
    
    # 使用utils中的可视化函数
    if args.visualize:
        print(f"使用数据路径 {args.data_path} 进行可视化...")
        visualize_embeddings(node_embeddings, args.data_path, args.output_dir)
    
    print("训练完成！")


if __name__ == "__main__":
    main() 