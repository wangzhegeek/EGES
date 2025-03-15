import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import os
import random
import socket
import torch.distributed as dist
import pandas as pd


def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def setup(rank, world_size, master_addr='localhost', master_port='12355'):
    """
    设置分布式训练环境
    """
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"进程 {rank} 初始化完成")


def cleanup():
    """
    清理分布式训练环境
    """
    dist.destroy_process_group()


def get_free_port():
    """
    获取一个可用的端口号
    """
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return str(port)


def plot_embeddings(embebed_mat, side_info_mat, output_dir='./data_cache'):
    """
    使用t-SNE可视化嵌入向量
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model = TSNE(n_components=2)
    node_pos = model.fit_transform(embebed_mat)
    brand_color_idx, shop_color_idx, cate_color_idx = {}, {}, {}
    for i in range(len(node_pos)):
        brand_color_idx.setdefault(side_info_mat[i, 1], [])
        brand_color_idx[side_info_mat[i, 1]].append(i)
        shop_color_idx.setdefault(side_info_mat[i, 2], [])
        shop_color_idx[side_info_mat[i, 2]].append(i)
        cate_color_idx.setdefault(side_info_mat[i, 3], [])
        cate_color_idx[side_info_mat[i, 3]].append(i)

    plt.figure(figsize=(10, 8))
    for c, idx in brand_color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.title('Brand Distribution')
    plt.savefig(os.path.join(output_dir, 'brand_dist.png'))

    plt.figure(figsize=(10, 8))
    for c, idx in shop_color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.title('Shop Distribution')
    plt.savefig(os.path.join(output_dir, 'shop_dist.png'))

    plt.figure(figsize=(10, 8))
    for c, idx in cate_color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.title('Category Distribution')
    plt.savefig(os.path.join(output_dir, 'cate_dist.png'))


def visualize_embeddings(node_embeddings, data_path, output_dir):
    """
    可视化嵌入向量
    
    参数:
    node_embeddings: 节点嵌入字典，键为节点ID，值为嵌入向量
    data_path: 数据路径，用于读取SKU侧面信息
    output_dir: 输出目录，用于保存可视化结果
    """
    print("可视化嵌入...")
    try:
        # 准备嵌入矩阵和侧面信息矩阵
        embebed_mat = []
        side_info_mat = []
        
        # 读取SKU侧面信息
        sku_info = pd.read_csv(data_path + 'jdata_product.csv')
        sku_info_dict = {row['sku_id']: row for _, row in sku_info.iterrows()}
        
        for node_id in sorted(node_embeddings.keys()):
            if node_id in sku_info_dict:
                embebed_mat.append(node_embeddings[node_id])
                row = sku_info_dict[node_id]
                side_info_mat.append([node_id, row['brand'], row['shop_id'], row['cate']])
        
        if len(embebed_mat) > 0:
            embebed_mat = np.array(embebed_mat)
            side_info_mat = np.array(side_info_mat)
            
            # 可视化嵌入
            plot_dir = os.path.join(output_dir, 'embedding', 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            plot_embeddings(embebed_mat, side_info_mat, output_dir=plot_dir)
            print(f"嵌入可视化完成，结果保存在 {plot_dir}")
        else:
            print("没有找到匹配的节点进行可视化")
    except Exception as e:
        print(f"可视化嵌入时出错: {e}")


def write_embedding(embedding_result, output_file):
    """
    将嵌入向量写入文件
    """
    with open(output_file, 'w') as f:
        for i in range(len(embedding_result)):
            s = " ".join(str(val) for val in embedding_result[i].tolist())
            f.write(s + "\n")


def save_dict_to_file(dict_obj, output_file):
    """
    将字典保存到文件
    """
    with open(output_file, 'w') as f:
        for key, value in dict_obj.items():
            f.write(f"{key}\t{value}\n")


def load_dict_from_file(input_file):
    """
    从文件加载字典
    """
    dict_obj = {}
    with open(input_file, 'r') as f:
        for line in f:
            key, value = line.strip().split('\t')
            dict_obj[int(key)] = int(value)
    return dict_obj 