# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

DATA_DIR = Path("HW-1")
OUT_DIR = Path("splits")
OUT_DIR.mkdir(exist_ok=True)

SEED = 42  


df = pd.read_csv(DATA_DIR / "nyt.csv")
print("原始数据:", df.shape)


print("\n类别数量:", df["label"].nunique())
print("\n各类别样本数:")
print(df["label"].value_counts())


df["label_id"] = df["label"].astype("category").cat.codes
label_map = dict(enumerate(df["label"].astype("category").cat.categories))
print("\n标签 -> 数字 映射:", label_map)


try:
    train_df, rest_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["label_id"]
    )
    val_df, test_df = train_test_split(
        rest_df, test_size=0.5, random_state=SEED, stratify=rest_df["label_id"]
    )
except ValueError:
   
    print("警告：分层划分失败（可能某类样本太少），改用普通随机划分")
    train_df, rest_df = train_test_split(df, test_size=0.2, random_state=SEED)
    val_df, test_df = train_test_split(rest_df, test_size=0.5, random_state=SEED)

print("\n划分结果:")
for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"{name}: {len(d)} 条")


train_df.to_csv(OUT_DIR / "train.csv", index=False)
val_df.to_csv(OUT_DIR / "val.csv", index=False)
test_df.to_csv(OUT_DIR / "test.csv", index=False)
print("\n已保存: splits/train.csv, val.csv, test.csv")
