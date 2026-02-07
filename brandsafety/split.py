import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit

def split_by_bucket(
    df,
    *,
    label_col,
    bucket_col = "sample_bucket",
    anchor_value = "anchor_unbiased",
    train_values = ("mined_keywords", "diversity_long_tail"),
    anchor_cal_frac = 0.5,
    seed = 42,
    anchor_split = "stratified",      # "stratified" or "group_target_root"
    group_col = "target_root"
):
    if bucket_col not in df.columns:
        raise ValueError(f"bucket_col='{bucket_col}' not found in df")
    if label_col not in df.columns:
        raise ValueError(f"label_col='{label_col}' not found in df")

    df_anchor = df[df[bucket_col] == anchor_value].copy()
    df_trainp = df[df[bucket_col].isin(train_values)].copy()

    if df_anchor.shape[0] < 200:
        raise ValueError(f"Anchor set too small for cal/test: n={df_anchor.shape[0]}")
    if df_trainp.shape[0] < 200:
        raise ValueError(f"Train pool too small: n={df_trainp.shape[0]}")

    y_anchor = df_anchor[label_col].astype(str).values

    if anchor_split == "stratified":
        df_cal, df_test = train_test_split(
            df_anchor,
            test_size = 1.0 - float(anchor_cal_frac),
            random_state = seed,
            stratify = y_anchor
        )
    elif anchor_split == "group_target_root":
        if group_col not in df_anchor.columns:
            raise ValueError(f"group_col='{group_col}' not in df (required for group split)")
        groups = df_anchor[group_col].astype(str).values
        gss = GroupShuffleSplit(n_splits = 1, test_size = 1.0 - float(anchor_cal_frac), random_state = seed)
        cal_idx, test_idx = next(gss.split(df_anchor, y_anchor, groups))
        df_cal = df_anchor.iloc[cal_idx].copy()
        df_test = df_anchor.iloc[test_idx].copy()
    else:
        raise ValueError("anchor_split must be 'stratified' or 'group_target_root'")

    return df_trainp.reset_index(drop = True), df_cal.reset_index(drop = True), df_test.reset_index(drop = True)
