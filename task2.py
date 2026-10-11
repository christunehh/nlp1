# -*- coding: utf-8 -*-

import argparse
import csv
import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


try:
    from nltk import word_tokenize
    word_tokenize("test sentence")
except Exception:
    try:
        import nltk
        nltk.download("punkt", quiet=True)
        from nltk import word_tokenize
        word_tokenize("test sentence")
    except Exception:
        from gensim.utils import simple_preprocess
        USE_NLTK = False
        print("nltk 不可用，改用 gensim 自带分词")


def tokenize(text):
    text = str(text)
    if USE_NLTK:
        return [t.lower() for t in word_tokenize(text)]
    return simple_preprocess(text)  


def save_result(method, acc, f1):
    file_exists = os.path.exists("results.csv")
    with open("results.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["method", "accuracy", "macro_f1"])
        writer.writerow([method, f"{acc:.4f}", f"{f1:.4f}"])


_nyt_tokens = {}


def get_nyt_tokens(split):
    if split not in _nyt_tokens:
        df = pd.read_csv(f"splits/{split}.csv")
        print(f"  对 NYT {split} 集分词中...")
        t0 = time.time()
        _nyt_tokens[split] = [tokenize(t) for t in df["text"]]
        print(f"  完成，用时 {time.time() - t0:.1f}s")
    return _nyt_tokens[split]


def mean_pool(tokens, embed, dim=100):
   
    vecs = [embed[t] for t in tokens if t in embed]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0)


def doc_matrix(token_lists, embed, dim=100):
    return np.vstack([mean_pool(t, embed, dim) for t in token_lists])


def coverage(token_lists, embed):
   
    total = sum(len(t) for t in token_lists)
    hit = sum(1 for t in token_lists for w in t if w in embed)
    return hit / total


def run_experiment(name, Xtr, Xva, Xte, ytr, yva, yte):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)

    va_pred = clf.predict(Xva)
    va_acc = accuracy_score(yva, va_pred)
    va_f1 = f1_score(yva, va_pred, average="macro")

    te_pred = clf.predict(Xte)
    te_acc = accuracy_score(yte, te_pred)
    te_f1 = f1_score(yte, te_pred, average="macro")

    print(f"\n===== {name} =====")
    print(f"验证集:  Acc={va_acc:.4f}   Macro-F1={va_f1:.4f}")
    print(f"测试集:  Acc={te_acc:.4f}   Macro-F1={te_f1:.4f}")
    save_result(name, te_acc, te_f1)


def load_glove():
    """优先用本地文件；没有就通过 gensim 自动下载 glove.6B 的 100 维版本"""
    from gensim.models import KeyedVectors
    local = "HW-1/glove.6B.100d.txt"
    if os.path.exists(local):
        print("找到本地 GloVe 文件，直接加载")
        return KeyedVectors.load_word2vec_format(local, no_header=True)
    import gensim.downloader as api
    print("正在下载 GloVe (glove-wiki-gigaword-100，约 340MB，请耐心等待)...")
    return api.load("glove-wiki-gigaword-100")


def load_ag_texts():
    ag = pd.read_csv("HW-1/ag.csv")
    print(f"ag.csv 形状: {ag.shape}，列名: {list(ag.columns)}")
    if "text" in ag.columns:
        return ag["text"].tolist()
    if "description" in ag.columns:
        print("未找到 text 列，改用 title + description 拼接")
        return (ag["title"].fillna("").astype(str) + " "
                + ag["description"].fillna("").astype(str)).tolist()
    raise ValueError("ag.csv 里既没有 text 也没有 description 列，把列名发给我看看")


def train_w2v(token_lists, dim=100):
    from gensim.models import Word2Vec
    print(f"  训练 Word2Vec：{len(token_lists)} 篇文档，{dim} 维，5 个 epoch...")
    t0 = time.time()
    model = Word2Vec(token_lists, vector_size=dim, window=5,
                     min_count=5, workers=4, epochs=5, seed=42)
    print(f"  完成，用时 {time.time() - t0:.1f}s，学到 {len(model.wv)} 个词")
    return model.wv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="all", choices=["all", "glove", "ag", "nyt"])
    args = parser.parse_args()

    ytr = pd.read_csv("splits/train.csv")["label_id"].values
    yva = pd.read_csv("splits/val.csv")["label_id"].values
    yte = pd.read_csv("splits/test.csv")["label_id"].values

    toks_tr = get_nyt_tokens("train")
    toks_va = get_nyt_tokens("val")
    toks_te = get_nyt_tokens("test")

    if args.exp in ("all", "glove"):
        print("\n========== 实验 2-1: 预训练 GloVe ==========")
        kv = load_glove()
        print(f"  GloVe 词表大小: {len(kv.key_to_index)}")
        print(f"  NYT 训练集词汇覆盖率: {coverage(toks_tr, kv):.1%}")
        Xtr = doc_matrix(toks_tr, kv)
        Xva = doc_matrix(toks_va, kv)
        Xte = doc_matrix(toks_te, kv)
        run_experiment("Task2-GloVe", Xtr, Xva, Xte, ytr, yva, yte)

    if args.exp in ("all", "ag"):
        print("\n========== 实验 2-2: AG News 自训练 Word2Vec ==========")
        ag_texts = load_ag_texts()
        print("  对 AG News 分词中...")
        t0 = time.time()
        ag_tokens = [tokenize(t) for t in ag_texts]
        print(f"  完成，用时 {time.time() - t0:.1f}s")
        wv = train_w2v(ag_tokens)
        print(f"  NYT 训练集词汇覆盖率: {coverage(toks_tr, wv):.1%}")
        Xtr = doc_matrix(toks_tr, wv)
        Xva = doc_matrix(toks_va, wv)
        Xte = doc_matrix(toks_te, wv)
        run_experiment("Task2-W2V-AGNews", Xtr, Xva, Xte, ytr, yva, yte)

    if args.exp in ("all", "nyt"):
        print("\n========== 实验 2-3: NYT 自训练 Word2Vec ==========")
        
        wv = train_w2v(toks_tr)
        print(f"  NYT 训练集词汇覆盖率: {coverage(toks_tr, wv):.1%}")
        Xtr = doc_matrix(toks_tr, wv)
        Xva = doc_matrix(toks_va, wv)
        Xte = doc_matrix(toks_te, wv)
        run_experiment("Task2-W2V-NYT", Xtr, Xva, Xte, ytr, yva, yte)

    print("\n全部完成，结果已追加写入 results.csv")
