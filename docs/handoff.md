# SpeckleLeJEPA 会话交接

工作区为 `D:\develop\Chenghong_LEJEPA`。项目在 `SpeckleLeJEPA/`，已实现网络、损失、自动混合精度和 NPY 数据流。数据流已与默认模型跑通一次优化器更新。下一次会话可据用户指示继续训练入口或用真实索引验证读取与裁剪。先查这份交接、[已确认决策](decisions.md)和 [当前代码行为](implementation.md)，公开接口与使用示例见 [README](../README.md)。

## 文件与参考位置

| 位置 | 用途 |
| --- | --- |
| [LeJEPA-v3-notes.md](../../LeJEPA-v3-notes.md) | 论文阅读、公式差异、官方代码核对 |
| [官方仓库副本](../../lejepa/README.md) | `c293d291ca87cd4fddee9d3fffe4e914c7272052`，最小示例见 `MINIMAL.md` |
| `D:\develop\ChengHongCluster\SpeckleSegmentationV7\ImageAnalyser.py` | RGB 和 polished mask 路径处理参考 |
| `D:\develop\ChengHongCluster\SpeckleSegmentationV7\GetImageWithIndex.py` | `ImageIndex` 三字典 NPY 格式来源 |
| [项目源码](../speckle_lejepa/__init__.py) | `data.py`、`views.py`、`model.py`、`objective.py`、`precision.py` 的公开入口 |
| [tests/test_data.py](../tests/test_data.py) | 使用临时 NPY 和 PNG 的数据流回归测试 |

官方副本包含 SIGReg 库和 Imagenette 最小示例。其中消融启动命令引用的 `scripts/je.py` 与配置目录不在所用提交中，因此目前以论文定义和最小示例实现为依据。 [官方实现核对](../../LeJEPA-v3-notes.md#官方代码核对)

## 环境和验证记录

用户提供的本机环境名称为 `Speckle_LeJEPA`，目录是 `H:\Simonlsp\mambaforge\envs\Speckle_LeJEPA`。已确认 Python `3.12.16`、PyTorch `2.5.1`、torchvision `0.20.1`、CUDA 构建 `12.1`。用户环境配置还包含 torchaudio `2.5.1`，当前项目不调用它。PowerShell 未将 `conda` 放入 PATH 时，从工作区根目录运行：

```powershell
Set-Location -LiteralPath 'D:\develop\Chenghong_LEJEPA\SpeckleLeJEPA'
& 'H:\Simonlsp\mambaforge\Scripts\conda.exe' run --no-capture-output -n Speckle_LeJEPA python -B -m unittest discover -s tests -v
```

- 3 项数据测试已通过。覆盖 16 位 RGB 与超过 255 的实例 ID（包括 65535）、相对路径和 `SavedirPolished`、目标外置零、前景约束、按基因分散、`n` 限制、缓存上限、尾批及不可裁剪实例计数。
- 合成 NPY 经 `DataLoader(batch_size=None, num_workers=0)` 送入默认模型，CPU FP32 与本机 Tesla P4 的 CUDA FP16 均已完成前向、反传、参数更新和 2048 维特征导出。P4 上动态缩放降低初始 scale 后完成更新。BF16 硬件检测路径用模拟返回值验证，原生 BF16 未实测。
- 多卡求和归约已做公式核对，未运行多 rank 前反向。没有在实际 NPY、真实图片集或远端训练环境上实测。以上记录仅证明接口和功能，不用于估算远端性能。

## 当前接缝与待确认事项

- 项目当前有可迭代的数据流与单步模型接口，完整训练入口、日志、优化器调度、checkpoint 和断点续训尚待实现。README 的 AdamW 与学习率是用法示例。训练入口应通过 `MixedPrecision` policy 调用模型，并对 `Losses.total` 反传。
- 正式使用前的输入信息是实际 `ImageIndex.npy` 路径和远端可访问的图像路径。当前支持索引所在目录下的相对路径和可直接访问的绝对路径。真实索引尚未用于验证路径映射、位深和重复归属处理。
- 当前数据流每轮按真实图像内容产出，未预建实例数量表，也没有固定 `__len__`。实际数据可用 `stream.stats` 校验空图、跳过率和小批。用户若指定每轮步数或多卡，需确定全局与每卡 batch 口径、rank 分片和同步尾批。模型在 `distributed=True` 时要求所有 rank 参加每步通信。
- 保持已确定的按基因分散组批与细胞等权。裁剪阈值、视图数量、训练轮数和远端资源参数目前应从 [已确认决策](decisions.md#当前实现默认值) 区分为实现默认值或待确认信息。
