# 4 阶段 2 分类 88 维特征数据集

> **数据集**：OULAD 学业危机预测 · V3_4s2c 优化版
> **样本数**：32,593（学生-课程对）；当前工作区已恢复完整 CSV
> **特征数**：88 维（不含标签）
> **标签**：`risk_2cls` ∈ {0=Normal, 1=At Risk}
> **修改记录**：在 V3_4s2c（83 维）基础上，删除 4 个弱相关/有争议特征 + 新增 3 类派生特征（每阶段 3 个）= 88 维

---

## 生成阶段数据说明

本目录中的 `student_course_v3_4s2c.csv` 始终保持原始 88 维特征口径。运行：

```bash
python preprocessing/build_stage_dataset.py --config configs/default.yaml
python preprocessing/split_train_test.py --config configs/default.yaml
```

预处理生成 `datasets/stage1` 至 `datasets/stage4` 共享阶段数据。训练入口根据模型选择输入视图：普通 baseline 使用背景信息和已观察阶段行为，Base Mamba 使用相同阶段序列及额外的共享动态上下文。具体列选择以 `preprocessing/schema.py` 为准，所有模型沿用同一组样本和数据划分。

---

## 0. 修改总览（相对于原 83 维版本）

| 操作 | 特征 | 原 corr (S4) | 新 corr (S4) | 理由 |
|---|---|---|---|---|
| ❌ **删除** | `s_acceleration` | -0.020 | — | 与 risk 几乎无关 |
| ❌ **删除** | `s_trend_score` | -0.028 | — | 与 risk 几乎无关 |
| ❌ **删除** | `s_k_oucontent_ratio` (×4) | -0.219 | — | 冗余特征，与 8 click 强共线 |
| ❌ **删除** | `s1_is_cramming` | +0.046 (反方向) | — | 命名暗示风险但实际是保护性信号；28% 信息可被其他特征覆盖 |
| ➕ **新增** | `s_k_sqrt_active_days` (×4) | — | **-0.678** | sqrt 压缩长尾，捕捉边际递减 |
| ➕ **新增** | `s_k_log_total_clicks` (×4) | — | **-0.599** | 聚合 8 click，替代共线问题 |
| ➕ **新增** | `s_k_click_entropy` (×4) | — | **-0.360** | 香农熵，反映资源多样性 |
| 🔄 **变换** | 8 click 计数 `s_k_xx_clicks` (×32) | — | — | `log1p` 压缩长尾 |

**净变化**：83 → 88 维（+5）

### 关于 `s1_is_cramming` 的删除决定

该特征虽然在 OULAD V2 派生数据中保留，但在本研究中被删除，理由：
1. **命名反直觉**——名为"突击式访问"暗示风险行为，但实证显示 cramming=1 学生通过率反而更高（57.1% vs 46.3%）
2. **业务含义不清**——与"临时抱佛脚"的教学直觉相反，更像是"短期高效学习"模式
3. **可替代性高**——其他 11 个 Stage 1 特征（active_days、n_sites、8 类 click、3 类 score）能解释其 72% 的方差；剩余 28% 的独立信息（来自点击时间分布）在论文中暂作放弃
4. **被原作者质疑**——Mamba 训练脚本（`mamba_4s2c.py:17`）已明确标注"业务意义不清，已从输入移除"

---

## 1. 阶段维度总览（超集结构）

本数据集采用**超集结构**：每一阶段的特征集是前一阶段的**严格超集**。
早期模型的特征空间被晚期模型完全包含，从而支持阶段间的可解释比较与滚动融合。

| 阶段 | 时间窗 | 该阶段独有特征 | 累计特征数 |
|---|---|---|---|
| **Stage 1** | 0-25% | 13 VLE + 3 score + 3 new | 9+19 = **28** |
| **Stage 2** | 0-50% | 13 VLE + 3 score + 3 new | +19 = **47** |
| **Stage 3** | 0-75% | 13 VLE + 3 score + 3 new | +19 = **66** |
| **Stage 4** | 0-100% | 13 VLE + 3 score + 3 new | +19 = **85** |
| 跨阶段 | 4 阶段全程 | 3 动态特征 | +3 = **88** |
| 标签 | — | `risk_2cls` | +1 = **89 列** |

> 注：跨阶段动态特征（s_trend_clicks / s_drop_flag / s_total_clicks）只放在最后，
> 因为它们的定义需要全部 4 个阶段的数据才能计算完成。

---

## 2. Stage 1 详细特征（共 28 维）

### 2.1 身份/课程背景（9 维，所有阶段共有）

| # | 字段名 | 中文 | 取值 | 说明 |
|---|---|---|---|---|
| 1 | `disability_Y` | 残障标识 | 0/1 | 是否有公开声明的残障情况 |
| 2 | `age_band_ord` | 年龄段 | 0-5 | 0=≤35, 1=35-39, ..., 5=55+（有序编码）|
| 3 | `highest_education_ord` | 最高学历 | 0-4 | 0=无, 1=初中, 2=高中, 3=本科, 4=研究生 |
| 4 | `imd_band_ord` | IMD 贫困分位 | 0-9 | 0=最贫困 10%, ..., 9=最富裕 10% |
| 5 | `studied_credits` | 已修学分 | int | 学生历史累计学分 |
| 6 | `num_of_prev_attempts` | 既往修读次数 | int | 该课程之前修过几次 |
| 7 | `days_to_register` | 注册延迟天数 | int | 课程开始后多少天才注册 |
| 8 | `module_presentation_length` | 课程呈现周期 | int | 课程总天数（OULAD = 268 天）|
| 9 | `cutoff_s4` | Stage 4 截止线 | int | 课程结束后的评估截止日（天）|

### 2.2 VLE 行为特征（13 维，Stage 1 独有）

| # | 字段名 | 中文 | 取值范围 | 计算方式 | 物理意义 |
|---|---|---|---|---|---|
| 10 | `s1_active_days` | 活跃天数 | 0-268 | 该阶段有 VLE 交互的不同天数 | **学习持续性**（corr = -0.465, S1 阶段最强负信号）|
| 11 | `s1_n_sites` | 访问资源类型数 | 0-8 | 该阶段点过的不同 VLE 资源类型数量 | **学习广度** |
| 12-19 | `s1_xx_clicks` (8 列) | 8 类资源点击量 | log1p(0+)=0, log1p(7000)≈8.9 | 各资源类型总点击量（log1p 压缩）| 8 类资源分别点了几次 |
| 20 | `s1_clicks_per_active` | 日均点击量 | float | 总点击 / 活跃天数 | **学习强度**（剔除天数差异）|
| 21 | `s1_longest_streak` | 最长连续活跃天数 | 0-268 | 日期序列最长连续区间 | **学习习惯连续性** |
| 22 | `s1_is_early_drop` | 是否早期骤降 | 0/1 | 该阶段后半段 < 前半段 50% 则 1 | **弃课信号**（corr = +0.154，正向风险）|

#### 8 类 VLE 资源 click 详解（字段 12-19）

| # | 字段名 | 资源类型 | 中文 | OULAD 原 activity_type |
|---|---|---|---|---|
| 12 | `s1_oucontent_clicks` | 课程内容页 | OU 课程核心内容 | `oucontent` |
| 13 | `s1_resource_clicks` | 资源页 | 课程附件资源 | `resource` |
| 14 | `s1_forumng_clicks` | 论坛 | 师生/同学讨论区 | `forumng` |
| 15 | `s1_homepage_clicks` | 课程主页 | 课程入口页 | `homepage` |
| 16 | `s1_quiz_clicks` | 测验 | 测验题页面 | `quiz` |
| 17 | `s1_subpage_clicks` | 子页面 | 内容子页 | `subpage` |
| 18 | `s1_ouwiki_clicks` | 维基 | OU Wiki 词条 | `ouwiki` |
| 19 | `s1_url_clicks` | 外链 | 外部链接 | `url` |

**注意**：原始 OULAD 有 14 种 activity_type，本数据集**只单独记录 8 种**。
被合并/忽略的 6 种：glossary、dataplus、dualpane、externalquiz、folder、htmlactivity、repeatactivity、sharedsubpage（部分并入 subpage_clicks 或忽略）。

### 2.3 阶段成绩（3 维）

| # | 字段名 | 中文 | 取值 | 说明 |
|---|---|---|---|---|
| 23 | `s1_avg_score` | 已提交作业平均分 | 0-100 | 该阶段已提交作业的算术平均 |
| 24 | `s1_n_submitted` | 已提交作业数 | 0-TMA数 | 该阶段提交了几次作业 |
| 25 | `s1_max_score` | 阶段最大分 | 0-100 | 该阶段所有作业中的最高分 |

### 2.4 新增派生特征（3 维）

| # | 字段名 | 中文 | 公式 | 物理意义 |
|---|---|---|---|---|
| 26 | `s1_sqrt_active_days` | sqrt(活跃天数) | `√max(0, active_days)` | 长尾压缩，捕捉"边际递减" |
| 27 | `s1_log_total_clicks` | log(总点击量) | `log(1 + Σ 8类clicks)` | 聚合 8 click，解决共线 |
| 28 | `s1_click_entropy` | 点击香农熵 | `-Σ (share_i × log(share_i))` | 资源多样性 |

**为什么需要这三个**：
- `sqrt_active_days` (corr=-0.499) 比原始 `active_days` (corr=-0.465) **强 3.4 个点**
- `log_total_clicks` (corr=-0.466) 比 8 click 之和还强
- `click_entropy` 提供"广度"独立信号

---

## 3. Stage 2 特征（新增 19 维，累计 47）

在 Stage 1 的 28 维基础上，新增 Stage 2 独有的 19 维：

| 字段段 | 数量 | 命名 | 内容 |
|---|---|---|---|
| `s2_xx_clicks` + `s2_xx_score` + `s2_xx_derived` | 19 | s2_ 前缀 | 同 Stage 1 结构（VLE 13 + score 3 + new 3）|

**Stage 2 全部 19 维**：

| # | 字段 | 类别 |
|---|---|---|
| 29-41 | `s2_active_days` ... `s2_is_early_drop` (13 列) | VLE 行为 |
| 42-44 | `s2_avg_score` / `s2_n_submitted` / `s2_max_score` | 阶段成绩 |
| 45-47 | `s2_sqrt_active_days` / `s2_log_total_clicks` / `s2_click_entropy` | 新增派生 |

---

## 4. Stage 3 特征（新增 19 维，累计 66）

与 Stage 2 完全相同的 19 维结构，仅前缀 `s3_`：

| # | 字段段 | 数量 |
|---|---|---|
| 48-60 | `s3_active_days` ... `s3_is_early_drop` | 13 VLE |
| 61-63 | `s3_avg_score` / `s3_n_submitted` / `s3_max_score` | 3 score |
| 64-66 | `s3_sqrt_active_days` / `s3_log_total_clicks` / `s3_click_entropy` | 3 new |

---

## 5. Stage 4 特征（新增 19 维，累计 85）

与 Stage 2/3 完全相同的 19 维结构，仅前缀 `s4_`：

| # | 字段段 | 数量 |
|---|---|---|
| 67-79 | `s4_active_days` ... `s4_is_early_drop` | 13 VLE |
| 80-82 | `s4_avg_score` / `s4_n_submitted` / `s4_max_score` | 3 score |
| 83-85 | `s4_sqrt_active_days` / `s4_log_total_clicks` / `s4_click_entropy` | 3 new |

---

## 6. 跨阶段动态（3 维，累计 88）

| # | 字段名 | 中文 | 公式 | 物理意义 | risk 相关性 |
|---|---|---|---|---|---|
| 86 | `s_trend_clicks` | 点击量趋势 | 跨阶段动态汇总 | 点击量是涨是跌 | **-0.518** |
| 87 | `s_drop_flag` | 全程无活动标志 | 跨阶段动态汇总 | 是否"挂机"了 | **+0.318** |
| 88 | `s_total_clicks` | 累计总点击 | 跨阶段动态汇总 | 总学习投入 | **-0.436** |

**已删除的 2 个弱相关动态特征**（原 V3_4s2c 有 5 个）：

| 字段 | 原 corr | 删除理由 |
|---|---|---|
| ~~`s_acceleration`~~ | -0.020 | 成绩曲线凹凸与 risk 几乎无关 |
| ~~`s_trend_score`~~ | -0.028 | 成绩趋势与 risk 几乎无关 |
| ~~`s_k_oucontent_ratio`~~ (×4) | -0.219 | 与 8 click 强共线，是冗余特征 |
| ~~`s1_is_cramming`~~ | +0.046 | 命名反直觉，72% 信息可被替代 |

---

## 7. 标签（1 维）

| # | 字段名 | 中文 | 取值 | 分布 |
|---|---|---|---|---|
| 89 | `risk_2cls` | 二分类学业风险 | 0=Normal, 1=At Risk | 0: 15,385 (47.2%) · 1: 17,208 (52.8%) |

**标签映射**（基于 OULAD final_result）：
- `Distinction` / `Pass` → 0 (Normal)
- `Fail` / `Withdrawn` → 1 (At Risk)

---

## 8. 关键相关性速查（S4 阶段，32,593 样本）

### VLE 行为（已优化）— Top 8 强负相关
| 特征 | corr | 类别 |
|---|---|---|
| `s4_sqrt_active_days` | **-0.678** | 新增（sqrt 变换）|
| `s4_active_days` | -0.626 | 原 14 维 |
| `s4_log_total_clicks` | **-0.599** | 新增（log 聚合）|
| `s4_n_sites` | -0.533 | 原 14 维 |
| `s4_homepage_clicks` | -0.435 | 原 14 维（log1p 变换）|
| `s4_oucontent_clicks` | -0.380 | 原 14 维（log1p 变换）|
| `s4_subpage_clicks` | -0.373 | 原 14 维（log1p 变换）|
| `s4_click_entropy` | **-0.360** | 新增（香农熵）|

### 跨阶段动态 — 强预测力
| 特征 | corr | 类别 |
|---|---|---|
| `s_trend_clicks` | **-0.518** | 保留 |
| `s_total_clicks` | **-0.436** | 保留 |
| `s_drop_flag` | **+0.318** | 保留（正相关 = 风险）|

---

## 9. 使用建议

### 推荐消融实验设计

| 方案 | VLE 维数 | 跨阶段 | 用途 |
|---|---|---|---|
| ① 基线（原 83 维）| 56 | 5 | 重现原结果 |
| ② 砍 oucontent_ratio | 52 | 5 | 验证 ratio 冗余 |
| ③ + 砍 s_acc/s_trend_score | 52 | 3 | 验证动态冗余 |
| ④ + 砍 is_cramming | 51 | 3 | 验证命名争议特征 |
| ⑤ + 加 sqrt/log/entropy | 64 | 3 | **本数据集**（推荐主表）|
| ⑥ ⑤ + 8 click 全 log1p | 64 | 3 | 进一步压缩长尾 |

### 给不同模型的最佳特征子集

| 模型 | 推荐特征子集 |
|---|---|
| **Mamba**（受益最大）| 全部 88 维 + log1p 变换（已内置）|
| **MLP / LSTM** | 全部 88 维 |
| **XGBoost / LightGBM / CatBoost** | 全部 88 维（log1p 几乎不影响）|
| **LR** | 全部 88 维（标准化前）|
| **RF / TabNet** | 全部 88 维 |

---

## 10. 数据集统计

```
样本数 (N):           32,593
特征数 (含标签):      89
特征数 (不含标签):    88
正样本 (At Risk):     17,208 (52.8%)
负样本 (Normal):      15,385 (47.2%)
缺失值:               0
Stage 4 完整覆盖:     32,593 (100%)
```

### 各阶段样本数（相同，全部 32,593，因为是同一组学生）

每个学生都有 s1, s2, s3, s4 全部 4 个阶段的行为记录（缺失日补 0 处理）。

---

## 11. 文件清单

| 文件 | 说明 |
|---|---|
| `student_course_v3_4s2c.csv` | 88 维特征主数据文件，~29 MB |
| `_build_optimized.py` | 数据集构建脚本（可重跑）|
| `README.md` | 本文档 |

---

## 12. 数据来源与修改记录

| 版本 | 日期 | 变化 |
|---|---|---|
| V3_4s2c.0 (83 维) | 2026-06-23 | 初版，83 特征 + 1 标签 |
| V3_4s2c.1 (88 维，本版) | 2026-06-26 | 删除 4 个（s_acceleration, s_trend_score, oucontent_ratio×4, s1_is_cramming）+ 新增 3 类派生（每阶段 3 个）+ 8 click log1p 变换 |

**主要参考文献**：
> Kuzilek J, Hlosta M, Zdrahal Z. Open University Learning Analytics dataset. *Scientific Data*. 2017;4:170171.
> Zhan H, Meng X, Asif M. Risk Early Warning of a Dynamic Ideological and Political Education System Based on LSTM-MLP. *Mobile Networks and Applications*. 2024;29(1).
