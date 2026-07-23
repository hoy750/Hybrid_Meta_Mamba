# Hybrid Meta-Mamba



- 9 个基学习器（论文表 1 中五大类共 9 种代表模型）：
  - 线性模型：LR
  - 核方法：SVM
  - 距离模型：KNN
  - Bagging：Random Forest
  - Boosting：XGBoost
  - 概率模型：Naive Bayes
  - 神经网络：MLP
  - 表格深度学习：TabNet
  - 状态空间模型：Mamba
- K-Fold OOF 元特征构建，避免 Stacking 信息泄露
- 8 个 Meta Learner 与非 Mamba 的 8 个 baseline 一一对应，外加重点实现的 Meta-Mamba
- 7 组实验入口：Baseline、Meta Learner、Top-K、Stage-wise、OOF、Remove Mamba、Remove Traditional

## 目录说明

```text
Hybrid-Meta-Mamba/
├── configs/                  # 默认配置与四阶段配置
├── datasets/                 # 运行预处理后生成 stage1-stage4 数据
├── preprocessing/            # 阶段特征切片、标准化、划分
├── models/                   # 传统模型、深度模型、元学习器
├── stacking/                 # OOF 概率与元特征矩阵
├── experiments/              # 7 组论文实验入口
├── evaluation/               # 指标、图表、统计辅助函数
├── results/                  # 实验输出
├── figures/                  # 论文图
├── paper/                    # 论文表格/结果材料
└── tests/                    # 最小安全测试
```

## 数据约定

`data/4阶段_2分类_88特征/student_course_v3_4s2c.csv` 是保持不变的 88 维原始特征总表。预处理从总表生成 Stage 1-4 累计阶段数据，并保留统一的样本划分。

标签列固定为 `risk_2cls`，其中 `0=Normal`，`1=At Risk`。当前工作区中的总表是完整的 32,593 行数据，重建后的训练/测试分割为 26,074/6,519。

## 推荐流程

完整实验按以下顺序执行。前两条命令会重建阶段数据及其训练/测试划分：

```bash
python preprocessing/build_stage_dataset.py --config configs/default.yaml
python preprocessing/split_train_test.py --config configs/default.yaml
python experiments/exp01_baseline.py --config configs/default.yaml
python stacking/generate_oof.py --config configs/default.yaml --stage 4
python experiments/exp02_meta.py --config configs/default.yaml
python experiments/exp03_topk.py --config configs/default.yaml
python experiments/exp04_stage.py --config configs/default.yaml
python experiments/exp05_oof.py --config configs/default.yaml
python experiments/exp06_remove_mamba.py --config configs/default.yaml
python experiments/exp07_remove_traditional.py --config configs/default.yaml
```

如果当前环境还没有装好 Mamba，只想先验证其他 baseline：

```bash
python experiments/exp01_baseline.py --config configs/default.yaml --exclude mamba
```

如果只想做流程检查、不写 `results/` 文件：

```bash
python experiments/exp01_baseline.py --config configs/default.yaml --exclude mamba --no-save
```

## Mamba 说明

`models/deep/mamba.py` 会优先调用 `mamba-ssm` 中的 `Mamba2`。如果环境未安装或 CUDA 版本不匹配，默认会退回到一个轻量 GRU 序列块，保证接口可用。正式论文复现实验建议安装 `mamba-ssm` 并在配置中设置 `require_mamba: true`，这样缺依赖时会直接报错，避免误把 fallback 结果当作 Mamba-2 结果。

## 统一模型接口

所有模型都遵循：

```python
model.fit(X_train, y_train)
prob = model.predict_proba(X_test)[:, 1]
pred = model.predict(X_test)
```

实验脚本只负责编排流程，模型实现放在 `models/`，OOF 逻辑放在 `stacking/`，指标计算放在 `evaluation/`。

## 输出约定

每个实验保存两类文件：

- `metrics.csv`：Accuracy、Macro-F1、AUC、Normal Recall、Risk Recall、threshold
- `prediction.csv`：样本序号、真实标签、风险概率、预测标签

这些输出会落在 `results/` 下与论文实验章节对应的目录中。
