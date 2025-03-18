import pandas as pd
import numpy as np
import os
import sys
import time
import torch
import signal
import argparse
from datetime import datetime, timedelta
from tqdm import tqdm
from torch_geometric.utils import from_networkx
import networkx as nx
import threading

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_process import get_session
from gpu_process.EGES_module import EGES, EGESTrainer
from utils import save_embeddings

# 全局变量，用于信号处理
stop_streaming = False

# 信号处理函数
def signal_handler(sig, frame):
    global stop_streaming
    print("\n接收到停止信号，正在安全退出...")
    stop_streaming = True

# 注册信号处理
signal.signal(signal.SIGINT, signal_handler)

class StreamingEGES:
    """
    流式EGES实现，按时间戳划分数据
    """
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
        self.time_interval = args.time_interval  # 时间间隔（小时）
        
        # 模型相关参数
        self.embedding_dim = args.embedding_dim
        self.walk_length = args.walk_length
        self.context_size = args.context_size
        self.walks_per_node = args.walks_per_node
        self.p = args.p
        self.q = args.q
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        
        # 初始化节点映射和图
        self.G = None
        self.node_map = {}
        self.reverse_node_map = {}
        self.side_info = None
        self.side_info_dict = {}
        self.feature_lens = None
        self.model = None
        self.trainer = None
        
        # 添加窗口计数器
        self.window_count = 0
        self.save_interval = args.save_interval  # 每10个窗口保存一次
        
        # 初始化时间窗口
        self.current_time = None
        self.next_window_time = None
        
        # 创建输出目录
        self.output_dir = os.path.join(args.output_dir, 'streaming')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载商品侧信息
        self._load_side_info()
    
    def _load_side_info(self):
        """加载商品侧信息"""
        print("加载商品侧信息...")
        product_file = os.path.join(self.args.data_path, 'jdata_product.csv')
        if not os.path.exists(product_file):
            print(f"错误: 商品侧信息文件 {product_file} 不存在!")
            return

        side_info_df = pd.read_csv(product_file)
        side_info_df = side_info_df.fillna(0)
        
        # 将时间类型转换为时间戳
        if 'market_time' in side_info_df.columns:
            side_info_df['market_time'] = pd.to_datetime(side_info_df['market_time'])
            side_info_df['market_time'] = side_info_df['market_time'].astype(int) // 10**9
        
        # 保留数值类型特征列
        numeric_cols = ['sku_id', 'brand', 'shop_id', 'cate']
        if all(col in side_info_df.columns for col in numeric_cols):
            self.side_info = side_info_df[numeric_cols].values.astype(np.int64)
            print(f"加载的侧信息形状: {self.side_info.shape}")
        else:
            print("警告: 侧信息列不完整，使用默认列")
            cols = [col for col in numeric_cols if col in side_info_df.columns]
            self.side_info = side_info_df[cols].values.astype(np.int64)

    def _initialize_model(self):
        """初始化或更新模型"""
        # 如果模型已存在，则保存旧模型的状态
        old_state_dict = None
        if self.model is not None:
            old_state_dict = self.model.state_dict()
        
        # 创建特征长度列表
        if self.feature_lens is None and self.side_info is not None:
            self.feature_lens = []
            for i in range(self.side_info.shape[1]):
                tmp_len = len(set(self.side_info[:, i])) + 1
                self.feature_lens.append(tmp_len)
        
        # 创建PyG图
        if self.G is not None:
            pyg_data = from_networkx(self.G).to(self.device)
            
            # 创建EGESTrainer
            self.trainer = EGESTrainer(
                session_list=None,  # 不需要再次构建图
                side_info=self.side_info,
                embedding_dim=self.embedding_dim,
                walk_length=self.walk_length,
                context_size=self.context_size,
                walks_per_node=self.walks_per_node,
                p=self.p,
                q=self.q,
                lr=self.lr,
                device=self.device,
                prefetch_factor=2,
                G=self.G,
                node_map=self.node_map,
                reverse_node_map=self.reverse_node_map,
                pyg_data=pyg_data
            )
            
            self.model = self.trainer.model
            
            # 如果有旧模型的状态，加载到新模型
            if old_state_dict is not None:
                # 尝试加载兼容的层
                try:
                    self.model.load_state_dict(old_state_dict, strict=False)
                    print("已加载先前的模型权重")
                except Exception as e:
                    print(f"加载先前的模型权重时出错: {e}")
                    print("使用新初始化的模型继续")
    
    def _update_graph(self, session_list):
        """更新图结构"""
        print("更新图结构...")
        start_time = time.time()
        
        # 提取所有边
        edges = []
        for session in session_list:
            if len(session) > 1:
                for i in range(len(session) - 1):
                    u = int(session[i])
                    v = int(session[i + 1])
                    edges.append((u, v))
        
        print(f"当前时间窗口的边数量: {len(edges)}")
        if len(edges) == 0:
            print("警告: 当前窗口没有边，无法更新图")
            return False
        
        # 如果是第一次构建图
        if self.G is None:
            self.G = nx.Graph()
            self.G.add_edges_from(edges)
            
            # 获取所有唯一节点
            nodes = list(self.G.nodes())
            self.node_map = {node: i for i, node in enumerate(nodes)}
            self.reverse_node_map = {i: node for i, node in enumerate(nodes)}
            
            # 重新映射节点ID
            self.G = nx.relabel_nodes(self.G, self.node_map)
        else:
            # 添加新的边到现有图
            new_nodes = set()
            relabeled_edges = []
            
            for u, v in edges:
                # 处理新节点
                if u not in self.node_map:
                    new_nodes.add(u)
                if v not in self.node_map:
                    new_nodes.add(v)
            
            # 为新节点分配ID
            current_max_id = max(self.node_map.values()) if self.node_map else -1
            for i, node in enumerate(new_nodes):
                node_id = current_max_id + 1 + i
                self.node_map[node] = node_id
                self.reverse_node_map[node_id] = node
            
            # 将边转换为内部ID
            for u, v in edges:
                relabeled_edges.append((self.node_map[u], self.node_map[v]))
            
            # 添加到现有图
            self.G.add_edges_from(relabeled_edges)
        
        end_time = time.time()
        print(f"图更新完成，耗时: {end_time - start_time:.2f}秒")
        print(f"当前图节点数: {self.G.number_of_nodes()}, 边数: {self.G.number_of_edges()}")
        return True
    
    def _process_time_window(self, action_data, window_start, window_end):
        """处理一个时间窗口的数据"""
        print(f"\n处理时间窗口: {window_start} 到 {window_end}")
        
        # 过滤当前时间窗口的数据
        window_data = action_data[(action_data['action_time'] >= window_start) & 
                                  (action_data['action_time'] < window_end)]
        
        if window_data.empty:
            print(f"时间窗口 {window_start} 到 {window_end} 没有数据")
            return False
        
        print(f"当前窗口数据量: {len(window_data)}")
        
        # 获取会话列表
        session_list = get_session(window_data, verbose=False)
        
        if not session_list or len(session_list) == 0:
            print("警告: 当前窗口没有有效会话")
            return False
        
        # 更新图结构
        if not self._update_graph(session_list):
            return False
        
        # 初始化模型
        self._initialize_model()
        
        # 训练模型
        print(f"使用当前窗口数据训练模型 ({window_start} 到 {window_end})...")
        
        # 不再为每个时间窗口创建输出目录，只记录时间标记
        self.current_window_time = window_start
        
        self.model = self.trainer.train(
            epochs=self.epochs,
            batch_size=self.batch_size,
            output_dir=self.output_dir,  # 使用流处理的主输出目录
            plot_loss=False  # 禁用损失曲线绘制
        )
        
        # 递增窗口计数
        self.window_count += 1
        
        # 每save_interval个窗口保存一次模型和嵌入
        if self.window_count % self.save_interval == 0:
            time_str = self.current_window_time.strftime('%Y%m%d%H')
            self._save_checkpoint(f"checkpoint_{self.window_count}_{time_str}")
            # 获取并保存嵌入
            embed_dir = os.path.join(self.output_dir, f"embeddings_{self.window_count}_{time_str}")
            self._save_embeddings(embed_dir)
        
        return True
    
    def _save_checkpoint(self, checkpoint_name):
        """保存模型检查点"""
        if self.model is None:
            return
        
        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.pt")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'window_count': self.window_count,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, checkpoint_path)
        
        print(f"模型检查点已保存到 {checkpoint_path}")
    
    def _save_embeddings(self, output_dir):
        """保存节点嵌入"""
        print("保存节点嵌入...")
        
        # 创建嵌入目录
        embed_dir = os.path.join(output_dir, 'embedding')
        os.makedirs(embed_dir, exist_ok=True)
        
        # 确保模型处于评估模式
        self.model.eval()
        
        # 获取所有节点的嵌入
        with torch.no_grad():
            embeddings = self.model.forward()
        
        # 转换为CPU上的numpy数组
        embeddings_np = embeddings.detach().cpu().numpy()
        
        # 保存嵌入
        save_embeddings(embeddings_np, self.reverse_node_map, embed_dir)
        
        # 不再进行可视化
        
        print(f"嵌入已保存到 {embed_dir}")
    
    def stream_process(self):
        """流式处理数据"""
        global stop_streaming
        
        # 加载用户行为数据
        action_file = os.path.join(self.args.data_path, 'action_head.csv')
        if not os.path.exists(action_file):
            print(f"错误: 用户行为数据文件 {action_file} 不存在!")
            return
        
        # 读取用户行为数据
        print("加载用户行为数据...")
        action_data = pd.read_csv(action_file)
        
        # 确保action_time列是datetime类型
        action_data['action_time'] = pd.to_datetime(action_data['action_time'])
        
        # 获取数据的时间范围
        min_time = action_data['action_time'].min()
        max_time = action_data['action_time'].max()
        
        print(f"数据时间范围: {min_time} 到 {max_time}")
        
        # 初始化当前时间窗口
        self.current_time = min_time
        self.next_window_time = min_time + timedelta(hours=self.time_interval)
        
        # 循环处理每个时间窗口的数据，直到数据处理完或接收到停止信号
        while self.current_time < max_time and not stop_streaming:
            if self._process_time_window(action_data, self.current_time, self.next_window_time):
                print(f"时间窗口 {self.current_time} 到 {self.next_window_time} 处理完成")
            
            # 更新时间窗口
            self.current_time = self.next_window_time
            self.next_window_time = self.current_time + timedelta(hours=self.time_interval)
        
        if stop_streaming:
            print("流处理被用户中断")
        else:
            print("所有数据已处理完毕")
        
        # 保存最终模型
        if self.model is not None:
            # 使用当前时间作为最终模型的时间标记
            time_str = datetime.now().strftime('%Y%m%d%H%M')
            final_output_dir = os.path.join(self.output_dir, f'final_{time_str}')
            os.makedirs(final_output_dir, exist_ok=True)
            
            # 保存模型权重
            checkpoint_dir = os.path.join(final_output_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'window_count': self.window_count,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, os.path.join(checkpoint_dir, 'model_final.pt'))
            
            # 保存最终嵌入
            embed_dir = os.path.join(final_output_dir, 'embeddings')
            self._save_embeddings(embed_dir)
            
            print(f"最终模型已保存到 {final_output_dir}")


def main():
    parser = argparse.ArgumentParser(description='流式EGES实现')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='./data/',
                      help='数据文件路径')
    parser.add_argument('--output_dir', type=str, default='./output/',
                      help='输出目录')
    parser.add_argument('--time_interval', type=int, default=1,
                      help='时间窗口间隔(小时)')
    parser.add_argument('--save_interval', type=int, default=100,
                      help='每10个窗口保存一次')
    # 模型参数
    parser.add_argument('--embedding_dim', type=int, default=128,
                      help='嵌入维度')
    parser.add_argument('--walk_length', type=int, default=10,
                      help='随机游走长度')
    parser.add_argument('--context_size', type=int, default=5,
                      help='上下文窗口大小')
    parser.add_argument('--walks_per_node', type=int, default=10,
                      help='每个节点的游走次数')
    parser.add_argument('--p', type=float, default=1.0,
                      help='返回参数')
    parser.add_argument('--q', type=float, default=1.0,
                      help='进出参数')
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=0.001,
                      help='学习率')
    parser.add_argument('--batch_size', type=int, default=128,
                      help='批次大小')
    parser.add_argument('--epochs', type=int, default=1,
                      help='每个时间窗口的训练轮数')
    parser.add_argument('--cpu', action='store_true',
                      help='是否使用CPU训练')
    
    # 不再需要可视化参数
    # parser.add_argument('--visualize', action='store_true',
    #                   help='是否可视化嵌入向量')
    
    args = parser.parse_args()
    
    # 打印配置信息
    print("\n=== 流式EGES训练配置 ===")
    print(f"数据路径: {args.data_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"时间窗口间隔: {args.time_interval}小时")
    print(f"嵌入维度: {args.embedding_dim}")
    print(f"使用设备: {'CPU' if args.cpu else 'GPU'}")
    print(f"每个窗口训练轮数: {args.epochs}")
    print("====================\n")
    
    # 创建并启动流式处理
    streaming_eges = StreamingEGES(args)
    streaming_eges.stream_process()


if __name__ == "__main__":
    main() 