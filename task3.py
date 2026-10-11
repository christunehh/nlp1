# -*- coding: utf-8 -*-

import csv
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score


MODEL_NAME = "google-bert/bert-base-uncased"
MAX_LEN = 64    
EPOCHS = 3       
BATCH = 32        
LR = 2e-5        
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)
if device.type == "cuda":
    print("GPU 型号:", torch.cuda.get_device_name(0))


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class NewsDataset(Dataset):
    
    def __init__(self, df):
        self.enc = tokenizer(
            df["text"].astype(str).tolist(),
            truncation=True,          
            padding="max_length",     
            max_length=MAX_LEN,
        )
        self.y = df["label_id"].values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (torch.tensor(self.enc["input_ids"][i]),
                torch.tensor(self.enc["attention_mask"][i]),
                torch.tensor(self.y[i], dtype=torch.long))


train_ds = NewsDataset(pd.read_csv("splits/train.csv"))
val_ds = NewsDataset(pd.read_csv("splits/val.csv"))
test_ds = NewsDataset(pd.read_csv("splits/test.csv"))

train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH)
test_dl = DataLoader(test_ds, batch_size=BATCH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3
)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


def evaluate(dataloader):
   
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for input_ids, mask, y in dataloader:
            input_ids, mask = input_ids.to(device), mask.to(device)
            logits = model(input_ids, attention_mask=mask).logits
            preds.extend(logits.argmax(dim=-1).cpu().numpy())
            golds.extend(y.numpy())
    return accuracy_score(golds, preds), f1_score(golds, preds, average="macro")



print(f"\n开始训练: {EPOCHS} epochs, 每 epoch {len(train_dl)} 步")
for epoch in range(1, EPOCHS + 1):
    model.train()
    t0 = time.time()
    total_loss = 0.0
    for step, (input_ids, mask, y) in enumerate(train_dl):
        input_ids, mask, y = input_ids.to(device), mask.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(input_ids, attention_mask=mask, labels=y)
        out.loss.backward()
        optimizer.step()
        total_loss += out.loss.item()
        if (step + 1) % 100 == 0:
            print(f"  epoch {epoch}, step {step + 1}/{len(train_dl)}, "
                  f"当前 loss {out.loss.item():.4f}")
    va_acc, va_f1 = evaluate(val_dl)
    print(f"epoch {epoch} 完成, 平均 loss {total_loss / len(train_dl):.4f}, "
          f"验证集 Acc={va_acc:.4f} Macro-F1={va_f1:.4f}, 用时 {time.time() - t0:.1f}s")


te_acc, te_f1 = evaluate(test_dl)
print(f"\n===== Task3-BERT =====")
print(f"测试集:  Acc={te_acc:.4f}   Macro-F1={te_f1:.4f}")

file_exists = os.path.exists("results.csv")
with open("results.csv", "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["method", "accuracy", "macro_f1"])
    writer.writerow(["Task3-BERT", f"{te_acc:.4f}", f"{te_f1:.4f}"])


model.save_pretrained("bert_nyt")
tokenizer.save_pretrained("bert_nyt")
print("模型已保存到 bert_nyt/，结果已写入 results.csv")
