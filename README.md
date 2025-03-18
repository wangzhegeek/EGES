# update 2025.3 重构项目，更新到Pytorch新版本

# EGES (Enhanced Graph Embedding with Side Information)

这是一个基于PyTorch实现的EGES（Enhanced Graph Embedding with Side Information）模型。该模型通过结合图结构和节点的侧信息来学习更好的节点表示。

## 功能特点

- 支持单GPU和多GPU分布式训练
- 集成了Node2Vec随机游走采样
- 支持节点侧信息的融合
- 实现了高效的数据加载和批处理
- 提供了嵌入可视化功能
- 支持模型checkpointing和结果保存
- 新增流式数据处理功能，适用于推荐系统场景

## 环境要求

- Python 3.7+
- PyTorch 1.8+
- CUDA (推荐用于GPU训练)
- PyTorch Geometric (用于图数据处理)

## 安装

1. 克隆仓库：
```bash
git clone [repository_url]
cd EGES
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 数据格式

项目需要两个主要的输入文件：

1. `action_head.csv`：用户行为数据，包含以下列：
   - user_id: 用户ID
   - sku_id: 商品ID
   - action_time: 行为时间
   - module_id: 模块ID
   - type: 行为类型

2. `jdata_product.csv`：商品侧信息数据，包含以下列：
   - sku_id: 商品ID
   - cate: 类别ID
   - brand: 品牌ID
   - shop_id: 店铺ID

## 使用方法

### 单GPU训练

```bash
python gpu_process/run_EGES.py \
    --data_path ./data/ \
    --output_dir ./output/single_gpu/ \
    --epochs 2 \
    --batch_size 128 \
    --embedding_dim 128 \
    --visualize
```

### 多GPU分布式训练

```bash
python gpu_process/run_EGES_multi_gpu.py \
    --data_path ./data/ \
    --output_dir ./output/multi_gpu/ \
    --epochs 10 \
    --batch_size 128 \
    --gpus -1 \
    --embedding_dim 128 \
    --visualize
```

### 流式数据处理和训练

新增流式处理功能，按时间戳划分数据进行增量训练，适用于推荐系统场景：

```bash
python gpu_process/stream_EGES.py \
    --data_path ./data/ \
    --output_dir ./output/streaming/ \
    --time_interval 1 \
    --save_interval 10 \
    --epochs 1 \
    --batch_size 128 \
    --embedding_dim 128
```

### 主要参数说明

#### 通用参数
- `--data_path`: 数据文件路径
- `--output_dir`: 输出目录
- `--epochs`: 训练轮数
- `--batch_size`: 批次大小
- `--embedding_dim`: 嵌入维度
- `--walk_length`: 随机游走长度
- `--context_size`: 上下文窗口大小
- `--walks_per_node`: 每个节点的游走次数
- `--p`: 返回参数
- `--q`: 进出参数
- `--lr`: 学习率

#### 多GPU训练参数
- `--gpus`: 使用的GPU数量，-1表示使用所有可用GPU
- `--sync_gradients`: 是否同步梯度（多GPU训练时）
- `--sync_params`: 是否在每个epoch后同步模型参数（多GPU训练时）
- `--visualize`: 是否可视化嵌入向量

#### 流式处理参数
- `--time_interval`: 时间窗口间隔(小时)
- `--save_interval`: 每多少个窗口保存一次模型和嵌入
- `--cpu`: 是否使用CPU进行训练(不指定则使用GPU)

## 输出说明

### 批量训练输出
训练完成后，模型会在指定的输出目录下生成以下文件：

1. `/checkpoints/`
   - `model_final.pt`: 训练完成的模型权重

2. `/embedding/`
   - `node_embeddings.npy`: NumPy格式的节点嵌入
   - `/plots/`: 嵌入可视化结果（如果启用可视化）
     - `cate_dist.png`: 按类别分布的可视化
     - `brand_dist.png`: 按品牌分布的可视化
     - `shop_dist.png`: 按店铺分布的可视化

### 流式处理输出
流式处理会在指定的输出目录下生成以下文件：

1. `/checkpoints/`
   - `checkpoint_{window_count}_{time}.pt`: 定期保存的模型检查点
   - `model_final.pt`: 最终的模型权重

2. `/embeddings_{window_count}_{time}/embedding/`
   - `node_embeddings.npy`: 定期保存的节点嵌入向量

3. `/final_{timestamp}/`
   - 流式处理结束时保存的最终模型和嵌入向量

## 项目结构

```
EGES/
├── data/                     # 数据目录
├── gpu_process/              # GPU训练相关代码
│   ├── EGES_module.py        # EGES模型核心实现
│   ├── run_EGES.py           # 单GPU训练脚本
│   ├── run_EGES_multi_gpu.py # 多GPU训练脚本
│   └── stream_EGES.py        # 流式数据处理实现
├── native_process/           # CPU训练相关代码
│   ├── EGES_model.py         # 原生EGES模型实现
│   ├── run_EGES.py           # CPU训练脚本
│   ├── walker.py             # 随机游走实现
│   └── README.md             # CPU版本说明
├── utils.py                  # 工具函数
├── data_process.py           # 数据处理函数
├── requirements.txt          # 项目依赖
└── README.md                 # 项目说明
```

## 版本说明

1. **GPU单机版**: 使用PyTorch Geometric实现，支持高效的图数据处理和模型训练。
2. **GPU多机多卡版**: 支持分布式训练，可以利用多台机器的多个GPU加速训练过程。
3. **Native CPU版**: 使用原生PyTorch实现，适用于没有GPU资源的环境。
4. **流式处理版**: 按时间戳划分数据，支持增量训练，适用于推荐系统场景。

## 流式处理特点

- 按时间戳将数据划分为固定时间间隔（默认1小时）的窗口
- 逐个处理时间窗口，增量更新图结构和模型
- 定期保存模型检查点和嵌入向量
- 支持通过Ctrl+C安全中断处理流程
- 处理完所有数据或中断时，保存最终模型

## 注意事项

1. 多GPU训练时，学习率会根据GPU数量自动调整
2. 建议在使用多GPU训练时启用梯度同步（--sync_gradients）
3. 可视化功能会消耗较多内存，对于大规模数据集可能需要调整batch_size
4. 确保数据文件格式正确，且列名与要求一致
5. 流式处理中的时间窗口大小需要根据数据特性和业务需求进行调整

## 引用

如果您使用了本项目的代码，请引用原始EGES论文：

```bibtex
@inproceedings{wang2018billion,
  title={Billion-scale commodity embedding for e-commerce recommendation in alibaba},
  author={Wang, Jizhe and Huang, Pipei and Zhao, Huan and Zhang, Zhibo and Zhao, Binqiang and Lee, Dik Lun},
  booktitle={Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={839--848},
  year={2018}
}
```