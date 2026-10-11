# -*- coding: utf-8 -*-

import csv
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


train = pd.read_csv("splits/train.csv")
val = pd.read_csv("splits/val.csv")
test = pd.read_csv("splits/test.csv")

X_train, y_train = train["text"], train["label_id"]
X_val, y_val = val["text"], val["label_id"]
X_test, y_test = test["text"], test["label_id"]


def save_result(method, acc, f1):
    
    file_exists = os.path.exists("results.csv")
    with open("results.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["method", "accuracy", "macro_f1"])
        writer.writerow([method, f"{acc:.4f}", f"{f1:.4f}"])


def run_experiment(name, vectorizer):
   
    X_tr = vectorizer.fit_transform(X_train)
    X_va = vectorizer.transform(X_val)
    X_te = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y_train)

    va_pred = clf.predict(X_va)
    va_acc = accuracy_score(y_val, va_pred)
    va_f1 = f1_score(y_val, va_pred, average="macro")

    te_pred = clf.predict(X_te)
    te_acc = accuracy_score(y_test, te_pred)
    te_f1 = f1_score(y_test, te_pred, average="macro")

    print(f"\n===== {name} =====")
    print(f"词表大小: {len(vectorizer.vocabulary_)}")
    print(f"验证集:  Acc={va_acc:.4f}   Macro-F1={va_f1:.4f}")
    print(f"测试集:  Acc={te_acc:.4f}   Macro-F1={te_f1:.4f}")

    save_result(name, te_acc, te_f1)
    return te_acc, te_f1



run_experiment("Task1-BinaryBoW", CountVectorizer(binary=True))


run_experiment("Task1-WordFreq", CountVectorizer())


run_experiment("Task1-TFIDF", TfidfVectorizer())

print("\n结果已写入 results.csv")
