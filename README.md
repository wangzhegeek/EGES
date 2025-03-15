# EGES: Enhanced Graph Embedding with Side Information

本项目实现了EGES（Enhanced Graph Embedding with Side Information）模型，该模型结合了图嵌入和侧面信息，用于电子商务推荐系统。项目支持单机单GPU、多GPU分布式训练以及流式在线学习，提供了高效的图构建、随机游走和嵌入训练功能。

## 项目特点

- **多种训练模式**：支持单机单GPU、多GPU分布式训练和流式在线学习
- **高效图处理**：提供SimpleWalker、FastGraphWalker和StreamingWalker三种随机游走实现
- **侧面信息融合**：结合商品的类别、品牌、店铺等侧面信息进行嵌入
- **注意力机制**：使用注意力机制学习不同特征的重要性
- **可视化支持**：训练完成后自动将嵌入向量可视化，支持按品牌、店铺和类别进行可视化
- **模块化设计**：代码结构清晰，易于扩展和修改
- **实时更新**：流式版本支持实时处理数据流并更新模型

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
├── streaming_process/    # 流式在线学习实现
│   ├── run_streaming_EGES.py  # 流式训练脚本
│   ├── streaming_walker.py    # 流式随机游走
│   ├── streaming_model.py     # 流式模型
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

### 流式在线学习

```bash
python streaming_process/run_streaming_EGES.py \
    --data_path ./data/ \
    --output_dir ./output/streaming/ \
    --embedding_dim 128 \
    --side_info_dims 64,32,32 \
    --use_attention \
    --async_mode \
    --visualize
```

## 实现细节

### 模型架构

EGES模型结合了图嵌入和侧面信息，主要包括以下组件：

1. **多个嵌入层**：用于嵌入不同的特征（商品ID、类别、品牌、店铺等）
2. **注意力网络**：用于学习不同特征的重要性
3. **输出层**：用于生成最终的节点嵌入

### 训练流程

#### 批处理训练（单GPU和多GPU）

1. **数据预处理**：读取用户行为数据和商品信息数据
2. **会话构建**：根据用户行为数据构建会话序列
3. **图构建**：基于会话序列构建图
4. **随机游走**：使用Node2Vec算法生成随机游走序列
5. **上下文生成**：根据随机游走序列生成上下文对
6. **模型训练**：使用Skip-gram模型训练节点嵌入
7. **嵌入保存**：将训练好的嵌入向量保存为多种格式
8. **嵌入可视化**：使用t-SNE算法将嵌入向量可视化

#### 流式在线学习

1. **初始化**：创建流式随机游走器和流式EGES模型
2. **数据流处理**：实时处理用户行为数据流
3. **会话感知**：基于时间窗口和用户行为自动划分会话
4. **增量图构建**：根据会话序列增量更新图结构
5. **在线随机游走**：实时生成随机游走样本
6. **在线模型更新**：使用生成的样本增量更新模型
7. **动态内存管理**：使用LRU策略自动管理图大小
8. **实时可视化**：定期可视化当前嵌入向量

## 性能优化

1. **多GPU并行训练**：支持使用多个GPU进行分布式训练，提高训练速度
2. **高效随机游走**：提供FastGraphWalker实现，使用PyTorch Geometric加速随机游走
3. **多进程数据处理**：使用多进程并行处理数据，提高数据加载和预处理速度
4. **梯度同步优化**：提供显式梯度同步选项，提高分布式训练的稳定性
5. **学习率调度**：使用ReduceLROnPlateau调度器，根据验证损失自动调整学习率
6. **异步处理架构**：流式版本使用多线程异步处理数据流、图更新和模型训练
7. **动态内存管理**：流式版本使用LRU策略自动管理图大小，适合处理大规模数据

## 版本对比

| 特性 | 单GPU版本 | 多GPU版本 | 流式版本 |
|------|----------|----------|----------|
| 训练方式 | 批处理 | 批处理 | 在线学习 |
| 数据处理 | 一次性加载 | 一次性加载 | 增量处理 |
| 图构建 | 一次性构建 | 一次性构建 | 增量构建 |
| 内存占用 | 高 | 分布式 | 可控 |
| 训练速度 | 中等 | 快 | 实时 |
| 适用场景 | 离线训练 | 大规模离线训练 | 实时推荐 |
| 更新频率 | 低 | 低 | 高 |

## 参考文献

- [EGES: Enhanced Graph Embedding with Side Information](https://dl.acm.org/doi/10.1145/3219819.3219869)
- [Node2Vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [PyTorch Distributed](https://pytorch.org/tutorials/intermediate/dist_tuto.html)
- [Online Learning for Graph Embeddings](https://arxiv.org/abs/1810.10046)
- [Streaming Graph Neural Networks](https://arxiv.org/abs/1810.10627)