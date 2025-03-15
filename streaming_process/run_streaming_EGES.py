#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import time
import datetime
import pandas as pd
import numpy as np
import torch
import random
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入流式处理模块
from streaming_process.streaming_walker import StreamingWalker, StreamingDataProcessor
from streaming_process.streaming_model import StreamingEGES

# 导入公共模块
from common.metapath2vec_dataset import Metapath2vecDataset


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="流式EGES模型训练")
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='./data/',
                        help='数据路径')
    parser.add_argument('--output_dir', type=str, default='./output/streaming/',
                        help='输出目录')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='日志打印间隔')
    
    # 模型参数
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='嵌入维度')
    parser.add_argument('--side_info_dims', type=str, default='64,32,32',
                        help='sideinfo嵌入维度，逗号分隔')
    parser.add_argument('--initial_nodes', type=int, default=100000,
                        help='初始节点数量')
    parser.add_argument('--use_attention', action='store_true',
                        help='是否使用注意力机制')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='计算设备')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--num_neg_samples', type=int, default=5,
                        help='负采样数量')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='学习率')
    parser.add_argument('--save_interval', type=int, default=10000,
                        help='模型保存间隔（处理多少条数据后保存）')
    
    # 流式处理参数
    parser.add_argument('--window_size', type=int, default=5,
                        help='上下文窗口大小')
    parser.add_argument('--max_graph_size', type=int, default=1000000,
                        help='图的最大节点数')
    parser.add_argument('--buffer_size', type=int, default=10000,
                        help='样本缓冲区大小')
    parser.add_argument('--update_interval', type=int, default=1000,
                        help='图更新间隔')
    parser.add_argument('--session_timeout', type=int, default=30,
                        help='会话超时时间（分钟）')
    parser.add_argument('--async_mode', action='store_true',
                        help='是否使用异步处理模式')
    
    # 模拟流式数据参数
    parser.add_argument('--simulate_stream', action='store_true',
                        help='是否模拟流式数据')
    parser.add_argument('--stream_batch_size', type=int, default=100,
                        help='流式数据批次大小')
    parser.add_argument('--stream_interval', type=float, default=0.1,
                        help='流式数据间隔（秒）')
    
    # 可视化参数
    parser.add_argument('--visualize', action='store_true',
                        help='是否可视化嵌入')
    parser.add_argument('--vis_interval', type=int, default=50000,
                        help='可视化间隔（处理多少条数据后可视化）')
    parser.add_argument('--vis_num_nodes', type=int, default=1000,
                        help='可视化节点数量')
    
    # 加载预训练模型
    parser.add_argument('--load_model', type=str, default='',
                        help='加载预训练模型路径')
    
    return parser.parse_args()


def setup_logging(output_dir):
    """设置日志"""
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, f'streaming_eges_{time.strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger()


def load_product_data(data_path):
    """加载商品数据"""
    product_file = os.path.join(data_path, 'product.txt')
    
    if not os.path.exists(product_file):
        logging.warning(f"商品数据文件不存在: {product_file}")
        return None
    
    logging.info(f"加载商品数据: {product_file}")
    
    # 读取商品数据
    product_data = pd.read_csv(product_file, sep='\t')
    
    # 确保列名正确
    if 'sku_id' not in product_data.columns:
        if 'asin' in product_data.columns:
            product_data.rename(columns={'asin': 'sku_id'}, inplace=True)
        else:
            logging.warning("商品数据缺少sku_id列")
            return None
    
    # 检查sideinfo列
    side_info_cols = ['cate', 'brand', 'shop']
    for col in side_info_cols:
        if col not in product_data.columns:
            logging.warning(f"商品数据缺少{col}列")
            product_data[col] = 'unknown'
    
    return product_data


def load_behavior_data(data_path, simulate_stream=False):
    """加载用户行为数据"""
    behavior_file = os.path.join(data_path, 'user_behavior.txt')
    
    if not os.path.exists(behavior_file):
        logging.warning(f"用户行为数据文件不存在: {behavior_file}")
        return None
    
    logging.info(f"加载用户行为数据: {behavior_file}")
    
    # 读取用户行为数据
    behavior_data = pd.read_csv(behavior_file, sep='\t')
    
    # 确保列名正确
    required_cols = ['user_id', 'sku_id', 'action_time', 'type']
    for col in required_cols:
        if col not in behavior_data.columns:
            if col == 'sku_id' and 'asin' in behavior_data.columns:
                behavior_data.rename(columns={'asin': 'sku_id'}, inplace=True)
            else:
                logging.warning(f"用户行为数据缺少{col}列")
                return None
    
    # 确保时间列是datetime类型
    if not pd.api.types.is_datetime64_dtype(behavior_data['action_time']):
        behavior_data['action_time'] = pd.to_datetime(behavior_data['action_time'])
    
    # 按时间排序
    behavior_data = behavior_data.sort_values(by='action_time')
    
    # 如果模拟流式数据，只返回前10000条
    if simulate_stream:
        return behavior_data.head(10000)
    
    return behavior_data


def visualize_embeddings(model, output_dir, step, num_nodes=1000):
    """可视化嵌入向量"""
    logging.info(f"可视化嵌入向量 (步骤: {step})")
    
    # 获取所有嵌入
    embeddings = model.get_all_embeddings()
    
    # 如果节点数量过多，随机采样
    if len(embeddings) > num_nodes:
        sampled_keys = random.sample(list(embeddings.keys()), num_nodes)
        embeddings = {k: embeddings[k] for k in sampled_keys}
    
    # 提取节点ID和嵌入向量
    node_ids = list(embeddings.keys())
    embedding_matrix = np.array(list(embeddings.values()))
    
    # 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embedding_matrix)
    
    # 绘制散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)
    
    # 添加一些节点标签
    num_labels = min(20, len(node_ids))
    for i in random.sample(range(len(node_ids)), num_labels):
        plt.annotate(str(node_ids[i]), (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
    
    plt.title(f'节点嵌入可视化 (步骤: {step})')
    plt.xlabel('t-SNE维度1')
    plt.ylabel('t-SNE维度2')
    
    # 保存图像
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    plt.savefig(os.path.join(vis_dir, f'embedding_vis_step_{step}.png'))
    plt.close()


def simulate_streaming_data(behavior_data, batch_size=100, interval=0.1):
    """模拟流式数据"""
    total_rows = len(behavior_data)
    num_batches = (total_rows + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_rows)
        
        batch = behavior_data.iloc[start_idx:end_idx]
        
        yield batch
        
        # 模拟数据流间隔
        time.sleep(interval)


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 设置日志
    logger = setup_logging(args.output_dir)
    logger.info("启动流式EGES模型训练")
    logger.info(f"参数: {args}")
    
    # 加载数据
    product_data = load_product_data(args.data_path)
    behavior_data = load_behavior_data(args.data_path, args.simulate_stream)
    
    if product_data is None or behavior_data is None:
        logger.error("数据加载失败，退出程序")
        return
    
    logger.info(f"商品数据: {len(product_data)}条")
    logger.info(f"用户行为数据: {len(behavior_data)}条")
    
    # 解析sideinfo维度
    side_info_dims = [int(dim) for dim in args.side_info_dims.split(',')]
    
    # 计算sideinfo类别数量
    side_info_cols = ['cate', 'brand', 'shop']
    side_info_nums = [product_data[col].nunique() + 1 for col in side_info_cols]  # +1 for unknown
    
    logger.info(f"sideinfo维度: {side_info_dims}")
    logger.info(f"sideinfo类别数量: {side_info_nums}")
    
    # 创建流式随机游走器
    walker = StreamingWalker(
        p=0.25,
        q=2,
        window_size=args.window_size,
        max_graph_size=args.max_graph_size,
        buffer_size=args.buffer_size,
        update_interval=args.update_interval,
        device=args.device
    )
    
    # 创建流式数据处理器
    data_processor = StreamingDataProcessor(
        walker=walker,
        time_window=60,  # 1小时
        session_timeout=args.session_timeout
    )
    
    # 创建流式EGES模型
    model = StreamingEGES(
        num_nodes=args.initial_nodes,
        embedding_dim=args.embedding_dim,
        side_info_dims=side_info_dims,
        side_info_nums=side_info_nums,
        use_attention=args.use_attention,
        device=args.device
    )
    
    # 加载预训练模型
    if args.load_model and os.path.exists(args.load_model):
        logger.info(f"加载预训练模型: {args.load_model}")
        model.load_model(args.load_model)
    
    # 预处理商品sideinfo
    logger.info("预处理商品sideinfo")
    for _, row in tqdm(product_data.iterrows(), total=len(product_data), desc="处理商品sideinfo"):
        sku_id = row['sku_id']
        side_info = [row[col] for col in side_info_cols]
        model.update_side_info(sku_id, side_info)
    
    # 启动异步处理
    if args.async_mode:
        logger.info("启动异步处理模式")
        walker.start_async_processing()
        model.start_async_updating()
    
    # 处理流式数据
    logger.info("开始处理流式数据")
    
    processed_count = 0
    start_time = time.time()
    
    # 模拟流式数据或一次性处理
    if args.simulate_stream:
        data_iterator = simulate_streaming_data(
            behavior_data,
            batch_size=args.stream_batch_size,
            interval=args.stream_interval
        )
    else:
        # 一次性处理所有数据
        data_iterator = [behavior_data]
    
    for batch in data_iterator:
        # 处理一批数据
        actions = []
        for _, row in batch.iterrows():
            actions.append((
                row['user_id'],
                row['sku_id'],
                row['action_time'],
                row['type']
            ))
        
        # 更新图和生成样本
        data_processor.process_batch(actions)
        
        # 获取样本并训练模型
        samples = walker.get_samples(args.batch_size * 10)
        if samples:
            model.add_train_task(
                samples=samples,
                batch_size=args.batch_size,
                num_neg_samples=args.num_neg_samples,
                lr=args.learning_rate
            )
        
        # 更新处理计数
        processed_count += len(batch)
        
        # 打印日志
        if processed_count % args.log_interval == 0:
            elapsed = time.time() - start_time
            logger.info(f"已处理 {processed_count} 条数据，耗时 {elapsed:.2f} 秒，"
                       f"速度 {processed_count / elapsed:.2f} 条/秒")
            
            # 打印图信息
            graph_info = walker.get_graph_info()
            logger.info(f"图信息: 节点数={graph_info['num_nodes']}, "
                       f"边数={graph_info['num_edges']}, "
                       f"样本缓冲区大小={graph_info['buffer_size']}")
        
        # 保存模型
        if processed_count % args.save_interval == 0:
            model_path = os.path.join(args.output_dir, f'model_step_{processed_count}.pt')
            logger.info(f"保存模型: {model_path}")
            model.save_model(model_path)
            
            # 保存嵌入向量
            emb_path = os.path.join(args.output_dir, f'embeddings_step_{processed_count}.npy')
            logger.info(f"保存嵌入向量: {emb_path}")
            model.save_embeddings(emb_path)
        
        # 可视化嵌入
        if args.visualize and processed_count % args.vis_interval == 0:
            visualize_embeddings(model, args.output_dir, processed_count, args.vis_num_nodes)
    
    # 处理完所有数据后，刷新剩余会话
    logger.info("刷新剩余会话")
    data_processor.flush_sessions()
    
    # 获取最后的样本并训练模型
    samples = walker.get_samples(args.batch_size * 10)
    if samples:
        model.add_train_task(
            samples=samples,
            batch_size=args.batch_size,
            num_neg_samples=args.num_neg_samples,
            lr=args.learning_rate
        )
    
    # 停止异步处理
    if args.async_mode:
        logger.info("停止异步处理")
        walker.stop_async_processing()
        model.stop_async_updating()
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, 'model_final.pt')
    logger.info(f"保存最终模型: {final_model_path}")
    model.save_model(final_model_path)
    
    # 保存最终嵌入向量
    final_emb_path = os.path.join(args.output_dir, 'embeddings_final.npy')
    logger.info(f"保存最终嵌入向量: {final_emb_path}")
    model.save_embeddings(final_emb_path)
    
    # 最终可视化
    if args.visualize:
        visualize_embeddings(model, args.output_dir, 'final', args.vis_num_nodes)
    
    # 打印总结
    total_time = time.time() - start_time
    logger.info(f"处理完成，共处理 {processed_count} 条数据，总耗时 {total_time:.2f} 秒，"
               f"平均速度 {processed_count / total_time:.2f} 条/秒")
    
    # 打印最终图信息
    graph_info = walker.get_graph_info()
    logger.info(f"最终图信息: 节点数={graph_info['num_nodes']}, "
               f"边数={graph_info['num_edges']}")


if __name__ == "__main__":
    main() 