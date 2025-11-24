# Repository Guidelines

## 项目结构与模块组织
- 代码主包位于 `src/polaris`：`backbone/`（ViT/ResNet/Transformer 编码器与量化模块）、`models/`（MERL/MELP/ECGFM 与微调头）、`datasets/`（Lightning DataModule、增强与数据集读取）、`utils/`（学习率与损失工具）、`prompt/`（提示与标签 JSON）、`paths.py`（路径常量）。
- 训练与评估脚本在 `scripts/`：`pretrain/main_pretrain.py`、`finetune/main_finetune.py`、`zeroshot/test_zeroshot.py`、`preprocess/preprocess_mimic_iv_ecg.py`。
- 日志默认写入 `logs/polaris*`；数据与划分按 `paths.py` 约定，预期位于仓库上级的 `datasets/`。若运行前发现 `RAW_DATA_PATH`、`PROCESSED_DATA_PATH`、`SPLIT_DIR`、`VQNSP_CKPT_PATH` 未在 `paths.py` 定义，请先补充。

## 构建、运行与调试命令
- 安装（开发模式）：`python -m pip install -e .`（建议先创建并激活虚拟环境）。
- 预训练示例：`CUDA_VISIBLE_DEVICES=0,1 python scripts/pretrain/main_pretrain.py --model_name melp --num_devices 2 --batch_size 64 --lr 2e-4 --train_data_pct 1 --val_dataset_list ptbxl-super ptbxl-sub`。
- 微调示例：`CUDA_VISIBLE_DEVICES=0 python scripts/finetune/main_finetune.py --model_name melp --dataset_name icbeb --ckpt_path <预训练权重> --num_devices 1`。
- 零样本评估：`python scripts/zeroshot/test_zeroshot.py --model_name merl --ckpt_path <权重> --test_sets icbeb chapman --batch_size 128 --save_results`。

## 代码风格与命名
- 遵循 PEP8，4 空格缩进；函数/变量用 snake_case，类名用 PascalCase；Lightning 模型与 DataModule 保持现有接口命名。
- 保持脚本 argparse 参数与文档示例一致；新增路径或常量集中添加到 `src/polaris/paths.py`。
- 在核心训练/评估函数添加简短 docstring，并使用类型注解。

## 依赖安装（ECGFM 专用）
- 若需要 `ECGFMModel`，需额外安装 `fairseq-signals` 提供的 `fairseq_signals_backbone`：
  ```
  git clone https://github.com/Jwoo5/fairseq-signals.git ../fairseq-signals
  cd ../fairseq-signals
  pip install --editable .
  ```
- 注意：上游 README 建议 Python ≤3.9；本项目使用 Python ≥3.10 时，如出现编译/依赖冲突，可尝试创建单独 3.9 虚拟环境专跑 ECGFM，或在 3.10 下用 `pip install --no-build-isolation --editable .` 观察是否可用。
- 安装后确保 `fairseq_signals_backbone` 在 `PYTHONPATH` 中（上述 editable 安装会自动添加）。未安装时仍可训练 MELP/MERL，不受影响。

## 测试与验证
- 目前未配置单元测试；运行 `python scripts/zeroshot/test_zeroshot.py ...` 或 `python scripts/finetune/main_finetune.py ... --ckpt_path ...` 作为回归检查。
- 如新增数据流程，优先为 `datasets/` 下的数据加载与划分逻辑补充 `pytest` 用例，并固定随机种子以与现有脚本行为一致。

## 提交与 PR
- Git 历史采用英文祈使句短说明（例如 “Remove unused imports”），保持不超过 72 字符。
- PR 请包含：改动摘要、动机/影响、运行过的命令与关键指标（如 AUC）、使用的数据/权重位置（相对路径），必要时附截图或日志路径 `logs/...`。
- 若涉及新路径或外部依赖，请在 PR 描述中明确如何设置 `paths.py` 及所需目录结构。
