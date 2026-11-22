# SpeckleLeJEPA 当前代码行为

当前项目具备两条可独立调用的路径。`CellBatchStream` 将 NPY 索引转换为预先组好的细胞视图；`SpeckleLeJEPA` 接收视图张量并返回自监督损失。这里描述现有源码行为，复现范围和测试状态见 [交接说明](handoff.md)。

## 索引、解码与缓存

1. `load_image_index()` 读取 `ImageIndex.save_to_npy()` 的三个字典，按基因名查板位，再取 `row[3]` 的 RGB PNG 与 `row[4]` 的实例 PNG。基因 ID 字典随格式读入，组批使用基因名。路径组件 `Savedir` 转为 `SavedirPolished`，相对路径以 NPY 所在目录为基准。格式使用 pickle，入口接收可信索引。 [索引代码](../speckle_lejepa/data.py#L48)
2. RGB 解码接受 8 位或 16 位三通道 PNG，mask 要求同尺寸的单通道 `uint16`。任何非零 ID 都按单个实例处理，即使同一 ID 的像素空间上不连通。空板位会跳过，同基因同路径的重复索引会合并。图像缺失、格式错误或同一 RGB 的基因或 mask 归属冲突会报错。 [解码与实例定位](../speckle_lejepa/data.py#L97)
3. `CellBatchStream` 从基因队列延迟加载原图，以 `seed` 和 `set_epoch()` 的轮次决定打散及裁剪随机性。每个进程最多缓存 `ceil(batch_size / cells_per_image)` 对原图与 mask 的原始整型数组。补图优先级随活跃图的基因计数变化，同优先级随机选择。剩余原图数不够时可能连续产出多个小批。一个 epoch 内同图的一个实例至多产出一次。 [数据流与缓存](../speckle_lejepa/data.py#L141)

在 1000×1000、RGB 三通道 `uint16`、mask 单通道 `uint16` 的假设下，`B=256, n=1` 的单份图像缓存约 **1.91 GiB**。若 RGB 为三通道 `uint8`、mask 仍为 `uint16`，约 **1.19 GiB**。这些值只计解码后的原图与标签数组，实际进程内存还包含索引、临时裁剪和已组好的浮点视图等开销。

## 裁剪与张量接口

`CellViews` 根据实例外接框居中裁出正方形，边缘越界时补零，过大的实例会随正方形缩放到底图尺寸。RGB 以原 dtype 的最大值归一化到 `[0, 1]`。每种视图随机选满足面积覆盖和前景比例的窗口。RGB 采用抗锯齿双线性缩放，二值掩膜采用最近邻取样，缩放后再次检查前景比例并把目标外像素置零。任何必需视图找不到合格窗口时，该细胞计入 `stream.stats.skipped_cells`。 [视图实现](../speckle_lejepa/views.py)

模型接收同设备、同 dtype 的浮点张量 `global_views[B,Vg,3,96,96]` 和 `local_views[B,Vl,3,48,48]`。同一 `B` 索引需对应同一细胞。`batch.cells` 保存 `(gene, image_path, label)` 供追踪。数据流已完成组批，直接迭代或用 `DataLoader(stream, batch_size=None, num_workers=0)`。`stream.stats` 包含解码图数、空图数、产出和跳过细胞数、缓存图对数峰值。 [调用方式](../README.md#细胞视图与组批)

## 网络、损失与精度

- torchvision ResNet-50 V1.5 随机初始化，移除分类层，输出池化后的 2048 维特征。分辨率分别前向、共享骨干参数。两组特征合并后调用三层 Projector，投影头 BatchNorm 覆盖本进程所有视图。骨干的 BatchNorm 分别在两种分辨率的前向中计算。`model.encode()` 返回骨干特征。 [模型代码](../speckle_lejepa/model.py)
- `LeJEPALoss` 收到 `[V,B,D]` 形状的投影后，用前 `Vg` 个全局视图计算每个细胞的中心，再对所有视图计算均方损失。中心参与反传。同一步共享重采样的单位方向，SIGReg 对每个视图沿细胞轴计算经验特征函数、进行高斯加权积分，并按实际样本数缩放。返回的 `total`、`prediction`、`sigreg` 均为 FP32 标量。 [损失代码](../speckle_lejepa/objective.py)
- 训练调用方通过 `MixedPrecision.for_device(device)` 获取 policy，在 `autocast()` 中调用模型，并用 `grad_scaler()` 完成优化。CPU 使用 FP32，CUDA 原生 BF16 使用 BF16 autocast，FP16 启用动态缩放。预测项和 SIGReg 计算都在关闭 autocast 的 FP32 段执行。 [精度代码](../speckle_lejepa/precision.py)
- 模型的 `distributed=True` 路径使用默认进程组，从 rank 0 广播方向，并以可反传的 SUM 汇总预测误差、特征函数统计量与细胞数。统计按全局实际细胞数计算，支持某一步各 rank 本地细胞数不等。通信要求各 rank 按相同顺序参加前向和反传。数据流当前没有 rank 分片或步数对齐。梯度累积仍由各个前向独立形成 SIGReg 统计 batch。 [分布式调用约定](../README.md#混合精度与调用方式)
