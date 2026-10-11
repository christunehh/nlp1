# 作业一：文本分类实现 —— 代码运行说明

## 1. 环境要求

- Python 3.9+
- 依赖包：pandas、numpy、scikit-learn、nltk、gensim、torch、transformers
- Task 3（BERT）建议使用 NVIDIA GPU；纯 CPU 亦可运行但耗时显著增加

安装命令：

```bash
pip install pandas numpy scikit-learn nltk gensim torch transformers
```

## 2. 文件结构

```
taskone/
├── HW-1/
│   ├── nyt.csv            # NYT 新闻数据集
│   ├── ag.csv             # AG News 数据集
│   └── glove.6B.100d.txt  # GloVe 100 维词向量
├── datasplit.py     # 第 1 步：数据划分（80/10/10）
├── task1.py        # Task 1：词袋模型 ×3 + 逻辑回归
├── task2.py   # Task 2：词向量 ×3 + 逻辑回归
├── task3.py       # Task 3：BERT 微调
├── splits/                # 划分结果
├── results.csv            # 各组实验的 Accuracy 与 Macro-F1
└── bert_nyt/              # 微调后的 BERT 模型
```

## 3. 运行步骤

按顺序执行以下命令：

```bash
# 第 1 步：数据划分
python datasplit.py

# Task 1：三种词袋模型 + 逻辑回归
python task1.py

# Task 2：三组词向量实验 + 逻辑回归
python task2.py --exp all
# 也可分组运行：--exp glove / --exp ag / --exp nyt

# Task 3：BERT 微调
python task3.py
```


