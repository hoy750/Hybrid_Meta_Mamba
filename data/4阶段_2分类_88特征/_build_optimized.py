"""
Build optimized V3_4s2c dataset by:
  1. Removing weak/redundant features:
       - s_acceleration (cross-stage dynamic, |corr| = 0.02)
       - s_trend_score  (cross-stage dynamic, |corr| = 0.03)
       - oucontent_ratio (VLE behavior,        |corr| = 0.22, redundant)
  2. Adding 3 per-stage engineered features:
       - sqrt(active_days)              : compresses long-tail active_days
       - log(1+total_clicks)            : aggregates 8 click counts (solves collinearity)
       - click_entropy                  : Shannon entropy of 8 click types
  3. Transforming 8 click counts with log1p:
       - oucontent/resource/forumng/homepage/quiz/subpage/ouwiki/url clicks
       - long-tail → compressed scale (Mamba/MLP-friendly)

Input  : C:/Users/17769/Desktop/论文撰写所需材料/数据集_4阶段_2分类_83特征/student_course_v3_4s2c.csv
Output : C:/Users/17769/Desktop/4阶段_2分类_92特征/student_course_v3_4s2c.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path(r'C:\Users\17769\Desktop\论文撰写所需材料\数据集_4阶段_2分类_83特征\student_course_v3_4s2c.csv')
DST_DIR = Path(r'C:\Users\17769\Desktop\4阶段_2分类_92特征')
DST_DIR.mkdir(parents=True, exist_ok=True)
DST = DST_DIR / 'student_course_v3_4s2c.csv'

CLICK_TYPES = ['oucontent', 'resource', 'forumng', 'homepage',
               'quiz', 'subpage', 'ouwiki', 'url']

# Load
df = pd.read_csv(SRC)
print(f'Loaded: {df.shape}')

# ----- Step 1: per-stage feature engineering -----
for s in [1, 2, 3, 4]:
    # 1.1 sqrt(active_days)
    ad = df[f's{s}_active_days']
    df[f's{s}_sqrt_active_days'] = np.sqrt(ad.clip(lower=0))

    # 1.2 log(1 + total_clicks)  (sum of 8 click types)
    total_clicks = sum(df[f's{s}_{c}_clicks'] for c in CLICK_TYPES)
    df[f's{s}_log_total_clicks'] = np.log1p(total_clicks)

    # 1.3 click_entropy (Shannon entropy of share distribution)
    shares = np.array([df[f's{s}_{c}_clicks'].values for c in CLICK_TYPES], dtype=float).T
    shares_sum = shares.sum(axis=1, keepdims=True) + 1e-10
    p = shares / shares_sum
    entropy = -(p * np.log(p + 1e-10)).sum(axis=1)
    df[f's{s}_click_entropy'] = entropy

    # 1.4 log1p on each of 8 click columns (replace, not add)
    for c in CLICK_TYPES:
        col = f's{s}_{c}_clicks'
        df[col] = np.log1p(df[col].clip(lower=0))

# ----- Step 2: drop weak features -----
TO_DROP = ['s_acceleration', 's_trend_score', 'oucontent_ratio']
# oucontent_ratio is per-stage in original CSV (only s1_oucontent_ratio? check)
ratio_cols = [c for c in df.columns if c.endswith('_oucontent_ratio') and c != 'oucontent_ratio']
# Actually original has only s1_oucontent_ratio in some places, let me just drop any *_oucontent_ratio
oucontent_ratio_cols = [c for c in df.columns if c.endswith('oucontent_ratio')]
print(f'Found oucontent_ratio columns: {oucontent_ratio_cols}')

TO_DROP_ALL = ['s_acceleration', 's_trend_score'] + oucontent_ratio_cols
df = df.drop(columns=[c for c in TO_DROP_ALL if c in df.columns])
print(f'Dropped: {TO_DROP_ALL}')
print(f'After drop: {df.shape}')

# ----- Step 3: reorder columns by stage/category for readability -----
identity_cols = [
    'disability_Y', 'age_band_ord', 'highest_education_ord', 'imd_band_ord',
    'studied_credits', 'num_of_prev_attempts', 'days_to_register',
    'module_presentation_length', 'cutoff_s4',
]
vle_cols = ['active_days', 'n_sites', 'oucontent_clicks', 'resource_clicks',
            'forumng_clicks', 'homepage_clicks', 'quiz_clicks', 'subpage_clicks',
            'ouwiki_clicks', 'url_clicks', 'clicks_per_active',
            'longest_streak', 'is_early_drop']
score_cols = ['avg_score', 'n_submitted', 'max_score']
new_per_stage = ['sqrt_active_days', 'log_total_clicks', 'click_entropy']
dynamic_cols_kept = ['s_trend_clicks', 's_drop_flag', 's_total_clicks']  # s_acceleration, s_trend_score 已删除

ordered = list(identity_cols)
for s in [1, 2, 3, 4]:
    for v in vle_cols:
        c = f's{s}_{v}'
        if c in df.columns: ordered.append(c)
    for sc in score_cols:
        c = f's{s}_{sc}'
        if c in df.columns: ordered.append(c)
    # s1_is_cramming 已删除 (业务含义不清, 命名与实际方向相反)
    # if s == 1 and 's1_is_cramming' in df.columns:
    #     ordered.append('s1_is_cramming')
    for nf in new_per_stage:
        c = f's{s}_{nf}'
        if c in df.columns: ordered.append(c)
# 跨阶段动态（仅 3 个保留的）
for dc in dynamic_cols_kept:
    if dc in df.columns:
        ordered.append(dc)
# label last
if 'risk_2cls' in df.columns:
    ordered.append('risk_2cls')

# Keep only columns that exist
ordered = [c for c in ordered if c in df.columns]
df = df[ordered]
print(f'After reorder: {df.shape}')

# Sanity
print(f'Final columns ({len(df.columns)}):')
for i, c in enumerate(df.columns, 1):
    print(f'  {i:>3}. {c}')

# Save
df.to_csv(DST, index=False, encoding='utf-8-sig')
print(f'\nSaved: {DST}')

# Sanity check
df2 = pd.read_csv(DST, nrows=3)
print('\nFirst 3 rows preview:')
print(df2.head().to_string())
print(f'\nFile size: {DST.stat().st_size:,} bytes')
