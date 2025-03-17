# EGES单机实现

这是EGES（Enhanced Graph Embedding with Side Information）模型的单机单GPU实现版本。该实现基于原始论文，使用PyTorch框架开发，支持商品推荐场景下的图嵌入学习。

## 功能特点

- 支持商品会话序列的图构建
- 实现基于Node2Vec的随机游走策略
- 集成商品sideinfo（类别、品牌、店铺等）
- 支持模型训练和检查点保存
- 提供嵌入向量的保存和可视化功能

## 使用方法

### 基本用法

```bash
python native_process/run_EGES.py
```

### 使用自定义参数

```bash
python native_process/run_EGES.py \
    --data_path ./data/ \
    --output_dir ./output/native/ \
    --embedding_dim 128 \
    --batch_size 8192 \
    --epochs 10 \
    --visualize
```

## 参数说明

- `--data_path`：数据文件路径，默认为 './data/'
- `--output_dir`：输出目录，默认为 './output/native/'
- `--p`：Node2Vec返回参数，默认为0.25
- `--q`：Node2Vec进出参数，默认为2
- `--num_walks`：每个节点的游走次数，默认为10
- `--walk_length`：每次游走的长度，默认为10
- `--window_size`：上下文窗口大小，默认为5
- `--embedding_dim`：嵌入向量维度，默认为128
- `--batch_size`：训练批次大小，默认为512
- `--epochs`：训练轮数，默认为10
- `--lr`：学习率，默认为0.001
- `--seed`：随机种子，默认为42
- `--visualize`：是否可视化嵌入向量

## 输出说明

训练过程会生成以下文件：

- `checkpoints/`：模型检查点目录
  - `model_epoch_N.pt`：每5轮保存的模型
  - `model_final.pt`：最终模型
- `embedding/`：嵌入向量目录
  - `node_embeddings.npy`：节点嵌入向量
  - `node_embeddings.txt`：文本格式的嵌入向量
  - `node_map.txt`：节点ID映射
  - `reverse_node_map.txt`：反向节点ID映射
  - `plots/`：可视化结果（如果启用）

## 性能优化

- 使用多进程数据加载
- 支持GPU加速
- 实现了高效的随机游走算法
- 优化了内存使用

## 常见问题

1. **内存不足**
   - 减小batch_size
   - 减少num_walks和walk_length
   - 关闭可视化功能

2. **训练速度慢**
   - 增加batch_size
   - 使用更快的GPU
   - 减少epochs数量

3. **模型效果不理想**
   - 调整p和q参数
   - 增加embedding_dim
   - 增加训练轮数 