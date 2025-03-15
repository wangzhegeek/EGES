# EGES: Enhanced Graph Embedding with Side Information

本项目实现了EGES（Enhanced Graph Embedding with Side Information）模型，该模型结合了图嵌入和侧面信息，用于电子商务推荐系统。项目支持单机单GPU和多GPU分布式训练，提供了高效的图构建、随机游走和嵌入训练功能。

## 项目特点

- **多种训练模式**：支持单机单GPU和多GPU分布式训练
- **高效图处理**：提供SimpleWalker和FastGraphWalker两种随机游走实现
- **侧面信息融合**：结合商品的类别、品牌、店铺等侧面信息进行嵌入
- **注意力机制**：使用注意力机制学习不同特征的重要性
- **可视化支持**：训练完成后自动将嵌入向量可视化，支持按品牌、店铺和类别进行可视化
- **模块化设计**：代码结构清晰，易于扩展和修改

## 项目结构

```
EGES/
├── common/                # 公共模块
│   ├── __init__.py       # 包初始化文件
│   ├── EGES_model.py     # 模型定义
│   ├── data_process.py   # 数据处理
│   ├── walker.py         # 随机游走
│   └── utils.py          # 工具函数
├── native_process/       # 单机单GPU实现
│   ├── run_EGES.py       # 训练脚本
│   └── README.md         # 使用说明
├── multi_gpu_process/    # 多GPU分布式实现
│   ├── run_EGES.py       # 分布式训练脚本
│   └── README.md         # 使用说明
├── data/                 # 数据目录
├── README.md             # 项目说明
└── requirements.txt      # 依赖库
```

## 环境要求

- Python 3.7+
- PyTorch 1.8+
- PyTorch Geometric
- NetworkX
- Pandas
- NumPy
- scikit-learn
- tqdm

## 安装

1. 克隆仓库
```bash
git clone https://github.com/yourusername/EGES.git
cd EGES
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 准备数据
将数据文件放在`data/`目录下，包括：
- `action_head.csv`：用户行为数据
- `jdata_product.csv`：商品信息数据

## 使用方法

### 单机单GPU训练

```bash
python native_process/run_EGES.py \
    --data_path ./data/ \
    --output_dir ./output/native/ \
    --embedding_dim 128 \
    --batch_size 512 \
    --epochs 10 \
    --visualize
```

### 多GPU分布式训练

```bash
python multi_gpu_process/run_EGES.py \
    --data_path ./data/ \
    --output_dir ./output/multi_gpu/ \
    --use_simple_walker \
    --explicit_sync \
    --extra_epochs \
    --visualize
```

## 实现细节

### 模型架构

EGES模型结合了图嵌入和侧面信息，主要包括以下组件：

1. **多个嵌入层**：用于嵌入不同的特征（商品ID、类别、品牌、店铺等）
2. **注意力网络**：用于学习不同特征的重要性
3. **输出层**：用于生成最终的节点嵌入

### 训练流程

1. **数据预处理**：读取用户行为数据和商品信息数据
2. **会话构建**：根据用户行为数据构建会话序列
3. **图构建**：基于会话序列构建图
4. **随机游走**：使用Node2Vec算法生成随机游走序列
5. **上下文生成**：根据随机游走序列生成上下文对
6. **模型训练**：使用Skip-gram模型训练节点嵌入
7. **嵌入保存**：将训练好的嵌入向量保存为多种格式
8. **嵌入可视化**：使用t-SNE算法将嵌入向量可视化

### 分布式训练

多GPU分布式训练使用PyTorch的DistributedDataParallel (DDP)框架，主要步骤如下：

1. 使用`torch.multiprocessing.spawn`启动多个进程，每个进程对应一个GPU
2. 在每个进程中，使用`torch.distributed.init_process_group`初始化进程组
3. 使用`DistributedSampler`对数据进行分片，确保每个进程处理不同的数据
4. 使用`DistributedDataParallel`包装模型，实现梯度同步
5. 在训练过程中，只在主进程（rank=0）上保存模型和打印日志

## 性能优化

1. **多GPU并行训练**：支持使用多个GPU进行分布式训练，提高训练速度
2. **高效随机游走**：提供FastGraphWalker实现，使用PyTorch Geometric加速随机游走
3. **多进程数据处理**：使用多进程并行处理数据，提高数据加载和预处理速度
4. **梯度同步优化**：提供显式梯度同步选项，提高分布式训练的稳定性
5. **学习率调度**：使用ReduceLROnPlateau调度器，根据验证损失自动调整学习率

## 参考文献

- [EGES: Enhanced Graph Embedding with Side Information](https://dl.acm.org/doi/10.1145/3219819.3219869)
- [Node2Vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [PyTorch Distributed](https://pytorch.org/tutorials/intermediate/dist_tuto.html)

## 许可证

MIT License 