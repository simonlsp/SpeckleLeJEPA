# SpeckleLeJEPA

SpeckleLeJEPA 是显微细胞图像的 LeJEPA 自监督预训练项目。数据流从 NPY 索引读取三通道原图和实例标签，按基因分散组批，再把细胞的全局和局部视图交给共享 ResNet-50 和三层投影头。模型和数据流可以分别调用。

跨会话接手请先读 [交接说明](docs/handoff.md)，并参阅 [已确认决策](docs/decisions.md)和 [当前代码行为](docs/implementation.md)。

## 安装

使用 Python 3.10 或以上版本。远端训练环境应先安装适配其驱动的 PyTorch 和 torchvision。项目要求 PyTorch 2.5 及以上、torchvision 0.20 及以上，使用 PyTorch 安装指引为两者选择配套版本。 [PyTorch 安装指引](https://pytorch.org/get-started/locally/)

在 `SpeckleLeJEPA` 目录执行：

```bash
python -m pip install -e .
```

## 输入与输出

| 参数 | 形状 | 含义 |
| --- | --- | --- |
| `global_views` | `[B, Vg, 3, 96, 96]` | 每个细胞的全局视图 |
| `local_views` | `[B, Vl, 3, 48, 48]` | 同一批细胞的局部视图 |

`B` 是当前进程的细胞数，`Vg` 和 `Vl` 分别由张量的第二维读取，两者均须至少为 1。`96` 和 `48` 是输入分辨率。同一索引 `b` 在两组张量中须对应同一个细胞。输入为同设备、同 dtype 的浮点像素张量，推荐以 FP32 传入，由混合精度上下文决定计算精度。数据流会应用下面的像素缩放和掩膜处理。

`model(global_views, local_views)` 返回三个可反传的 FP32 标量 `total`、`prediction` 和 `sigreg`。用 `total` 执行反传，另外两项可用于记录训练状态。

## NPY 索引与图像读取

`CellBatchStream` 接收 `ImageIndex.save_to_npy()` 保存的 NPY。这个文件依次包含基因名到板位、基因 ID 到板位、板位到图像记录的三个字典。数据流从基因名映射遍历所有有图像记录的板位，对每条记录读取 `row[3]` 的 RGB PNG 和 `row[4]` 的实例 PNG。mask 路径中的目录名 `Savedir` 改为 `SavedirPolished`，与 `ImageAnalyser.py` 的 polished mask 选择一致。相对路径从 NPY 所在目录解析。请使用可信的 NPY，因为这一旧索引格式需要启用 NumPy pickle 反序列化。

`load_image_index()` 可以单独解析索引，保留跨平台绝对路径的格式。例如，在 Windows 上解析 Linux 路径时保留 `/home/...`。训练时请在可以读取索引中图像的设备上运行。

RGB 接受三通道 8 位或 16 位 PNG。实例标签要求同尺寸、单通道 16 位 PNG。像素为 `0` 时是背景，每个非零整数 ID 对应一个实例。解码保留 RGB 位深和实例 ID，缓存保留原始整数 dtype。已在 `Speckle_LeJEPA` 的 torchvision 0.20.1 构建验证 16 位读入。若其他构建将 16 位 PNG 降为 8 位，数据流会报错。索引中空板位会跳过，同基因的重复路径会合并，同一原图的基因或 mask 有冲突时会报错。图像缺失、位深错误或尺寸不匹配时，迭代报错并给出路径。

## 细胞视图与组批

每个实例以其外接框中心裁成 128×128 的细胞底图，越界补零，其他实例和背景置零。外接框超出 128 时，先用能装下该实例的正方形裁剪，再缩放到 128。RGB 按其整数 dtype 的最大值缩放至 `[0, 1]`。视图裁剪的边长比例以下表中的细胞外接框最长边为基准，且受底图尺寸限制。

| 视图 | 默认个数 | 裁剪边长比例 | 至少保留的实例面积 | 裁剪窗口内最小前景比例 |
| --- | ---: | ---: | ---: | ---: |
| 全局 96×96 | 2 | 0.90–1.20 | 75% | 15% |
| 局部 48×48 | 2 | 0.35–0.65 | 5% | 45% |

裁剪时随机选择满足约束的窗口，缩放后再次检查前景比例，并按对应 mask 将实例外置零。若某一类视图没有可用窗口，该细胞计入 `stats.skipped_cells`。这些阈值和视图数可以通过 `CellViews` 与 `CropRule` 调整。

默认 `batch_size=256`、`cells_per_image=1`，每个进程最多缓存 `ceil(batch_size / cells_per_image)` 对解码后的原图和实例标签。补充原图时优先选当前池里未出现的基因，必要时选出现最少的基因。每轮打散图像和图内细胞顺序。每张图在同批最多贡献 `cells_per_image` 个不同实例，每个合格实例每轮产出一次。剩余原图不足时会产生较小的 batch，可能连续出现多个。`batch.cells` 的基因、RGB 路径及实例 ID 用于追踪。`stats` 记录解码图数、空图数、产出与跳过的细胞数，以及缓存的原图对数峰值。

```python
from speckle_lejepa import CellBatchStream

stream = CellBatchStream("/path/to/ImageIndex.npy", batch_size=256, cells_per_image=1, seed=0)
stream.set_epoch(0)
for batch in stream:
    global_views = batch.global_views  # [B, 2, 3, 96, 96]
    local_views = batch.local_views    # [B, 2, 3, 48, 48]
    # 将两个张量转到训练设备并传给模型。
    break
```

调用 `set_epoch()` 切换轮次随机种子。数据流已完成组批，可直接迭代，也可以通过 `DataLoader(stream, batch_size=None, num_workers=0)` 读取。当前数据流由单个采样进程持有；多卡数据切分和各 rank 的步数对齐需要由训练入口处理。

## 网络与损失

ResNet-50 使用 torchvision 的标准 V1.5 实现，从随机参数初始化。骨干的全局平均池化输出 2048 维特征。96 和 48 的视图各自展平为单一分辨率批次，通过同一个骨干。随后合并两组特征并调用同一个投影头，使投影头中的批归一化在该进程的全部视图上计算。骨干中的批归一化在两种分辨率各自的前向内计算。 [torchvision ResNet-50](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)

投影头默认是 `2048 → 2048 → 2048 → 128`。前两层线性映射后接 BatchNorm 和 ReLU，第三层直接输出投影。损失内部使用 `[Vg + Vl, B, D]` 的顺序，前 `Vg` 个视图为全局视图。

- **预测损失**先计算每个细胞的全局视图均值，再减去中心并对所有视图、细胞和特征的平方偏差取均值。中心参与反传。
- **SIGReg**每次前向重采样单位方向，并供该步所有视图共用。每个视图沿细胞轴估计经验特征函数。采用高斯窗和正半轴梯形积分，权重计入对称的负半轴。对视图和切片平均后的统计量乘以细胞数。
- **总损失**为 `(1 - reg_weight) * prediction + reg_weight * sigreg`。初始 `reg_weight=0.05` 来自论文的经验值。SIGReg 默认 256 个切片及 `[0, 3]` 上包含端点的 17 个积分点，沿用官方最小示例。隐藏维 `hidden_dim=2048` 和输出维 `projector_dim=128` 同样取自该示例的模型定义。

预测项采用论文 §5.1 的中心实现。当前项目的全局和局部视图来自两个分辨率输入。SIGReg 的数值积分和两项损失的归约保持 FP32。 [LeJEPA §5.1](https://arxiv.org/html/2511.08544v3#S5.SS1) · [作者发布的最小示例](https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/MINIMAL.md)

## 混合精度与调用方式

`MixedPrecision.for_device(device)` 检查所选 CUDA 设备的原生 BF16 支持。设备支持时选择 BF16 autocast，其余 CUDA 设备使用 FP16 autocast 并启用 `GradScaler`。CPU 使用 FP32，可用于检查张量接口。模型参数应保留 FP32，由 autocast 为合适的算子选择计算精度。 [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html)

以下示例使用随机张量展示一次前向和优化器更新。`B=2` 是示例输入大小，实际数据由训练入口传入。

```python
import torch

from speckle_lejepa import MixedPrecision, SpeckleLeJEPA

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SpeckleLeJEPA().to(device).train()
precision = MixedPrecision.for_device(device)
scaler = precision.grad_scaler()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

global_views = torch.randn(2, 2, 3, 96, 96, device=device)
local_views = torch.randn(2, 2, 3, 48, 48, device=device)

optimizer.zero_grad(set_to_none=True)
with precision.autocast():
    losses = model(global_views, local_views)
scaler.scale(losses.total).backward()
scaler.step(optimizer)
scaler.update()
```

在 DDP 中，先初始化默认进程组并为每个进程选择设备，再用 `DistributedDataParallel` 包装 `SpeckleLeJEPA(distributed=True)`。所有 rank 需按相同顺序执行前向和反传，并使用相同的视图数、投影维度和 SIGReg 设置。每个 rank 至少传入一个细胞。每步由 rank 0 发布同一组方向，再通过可反传的求和归约合并特征函数、预测误差和细胞数。这样即使各 rank 本地细胞数不等，损失仍按全局实际细胞数计算。骨干和投影头沿用局部 BatchNorm，归约作用于它们已经生成的投影。使用梯度累积时，每次前向各自形成一个 SIGReg 统计批次。

## 命令行训练

在 `SpeckleLeJEPA` 目录中运行。脚本自动选择可用的 CUDA 设备或 CPU，可以用 `--device cuda:0` 或 `--device cpu` 指定设备。

```bash
conda activate Speckle_LeJEPA
python train.py /data/ImageIndex.npy --batch-size 256 --epochs 10 --output runs/pretrain
```

预实验可以用两个独立的上限缩小样本量。

```bash
python train.py /data/ImageIndex.npy --batch-size 4 --max-image-per-gene 2 --max-cell-per-image 8 --output runs/preview
```

`--max-image-per-gene` 按 `--seed` 为每个基因随机选择图像，所选图像在各轮固定。`--max-cell-per-image` 每轮从每张图取至多指定数量的实例参加裁剪，裁剪失败时产出的细胞数可能更少。省略任一参数时，对应范围内的样本全部参与。`--batch-size` 控制单设备每批最多处理的细胞数。

训练当前批次时，后台会准备一个下一批次。`--cell-workers` 默认使用 4 个线程，从图片池中并行处理不同图片的细胞并生成遮罩视图。`--image-workers` 默认使用 2 个独立线程预读和解码候选补充图像，最多额外保存 2 对 RGB 和 mask。组批顺序和每图上限由同一个采样器控制。启用细胞线程后，裁剪使用按任务分配的随机种子，固定配置可复现，正数线程数之间结果一致。若续训需要沿用旧版串行随机序列，传入 `--cell-workers 0 --image-workers 0`。

脚本默认使用 AdamW、学习率 `5e-4` 和随机种子 `0`。`--output` 是输出目录，默认为 `checkpoints/`。每轮训练结束会保存 `epoch_0001.pt`、`epoch_0002.pt` 等 checkpoint，其中 `model` 字段是模型的 `state_dict()`。训练日志同时输出到终端并追加到 `train.log`，记录训练配置、损失、耗时、细胞吞吐、显存峰值以及 checkpoint 操作。`--log_interval N` 可调整每隔多少批记录一次进度，默认为 100，每轮第一批也会记录。

用 `--resume` 从输出目录中 epoch 编号最大的 checkpoint 继续，恢复模型、优化器、梯度缩放器和随机数状态。续训时传入与前次一致的 batch、采样上限和种子，可延续原采样配置。`--epochs` 表示最终要达到的 epoch 编号，省略时再训练一轮，新训练从第 1 轮开始。可用 `--max_saved_checkpoints N` 在每次成功保存后只保留最新的 N 个 `epoch_*.pt`，省略时保留全部。新训练请使用新的输出目录，继续使用已有 checkpoint 的目录时添加 `--resume`。

```bash
python train.py /data/ImageIndex.npy --batch-size 256 --output runs/pretrain --resume --epochs 20 --max_saved_checkpoints 3
```

## 导出骨干特征

下游使用 `model.encode(images)`，输入是 `[N, 3, H, W]` 的浮点张量，输出为 `[N, 2048]` 的池化骨干特征。导出时将模型切换为 `eval()`，并使用 `torch.inference_mode()`。

```python
model.eval()
with torch.inference_mode():
    features = model.encode(global_views[:, 0])
```

保存模型时，可用 `model.state_dict()` 保存骨干和投影头权重，并在训练配置中一同记录投影维度、SIGReg 积分配置和精度设置。单独部署骨干时可保存 `model.backbone.state_dict()`。

## 数据流测试

在项目根目录运行 `python -m unittest discover -s tests -v`。测试在临时目录生成 NPY 和 16 位 PNG，覆盖路径解析、超过 255 的实例 ID、视图背景与前景比例、按基因分散组批、缓存上限和尾批遍历。
