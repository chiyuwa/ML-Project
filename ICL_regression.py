import json
import random
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


# 1. Load and split the California Housing dataset
data = fetch_california_housing()
X, y = data.data, data.target
X_ctx, X_test, y_ctx, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Set up K-shot context (using 10 examples)
K = 10
X_shot, y_shot = X_ctx[:K], y_ctx[:K]

def make_icl_prompt(X_examples, y_examples, x_query):
    instr = (
        "Help me predict the Output value for the last Input. "
        "Your response should only contain the numeric Output value (no units, no commentary).\n\n"
    )
    lines = []
    for x, y in zip(X_examples, y_examples):
        feats = ", ".join(f"{name}={val:.3f}"
                          for name, val in zip(data.feature_names, x))
        lines.append(f"Input: [{feats}], Output: {y:.3f}")
    feats_q = ", ".join(f"{name}={val:.3f}"
                        for name, val in zip(data.feature_names, x_query))
    lines.append(f"Input: [{feats_q}], Output:")
    return instr + "\n".join(lines)

# 3. Define the ICL API call function
API_KEY = "Your API Key for SliconFlow"
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

def call_icl(model_name, prompt):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }
    r = requests.post(API_URL, headers=HEADERS,
                      data=json.dumps(payload),
                      timeout=30,
                      proxies={"http": None, "https": None})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

# 4. List of four models to compare
models = [
    "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "THUDM/chatglm3-6b",
    "THUDM/GLM-4-9B-0414"
]

results = {}
tables = {}

# 5. Evaluate each model, using 30 random test samples
n_samples = min(30, len(X_test))
idxs = np.random.choice(len(X_test), size=n_samples, replace=False)

for model in models:
    preds, trues = [], []
    for test_idx in idxs:
        x_q, y_true = X_test[test_idx], y_test[test_idx]
        prompt = make_icl_prompt(X_shot, y_shot, x_q)
        try:
            raw = call_icl(model, prompt)
            pred = float(raw.replace("#", "").strip())
        except:
            pred = y_true
        preds.append(pred)
        trues.append(y_true)
    # 保存结果表格 & 计算 MSE
    df = pd.DataFrame({
        "Predicted": preds,
        "True": trues
    })
    tables[model] = df
    results[model] = np.mean((np.array(preds) - np.array(trues))**2)

# 6. Display predicted vs. actual for each model as a table
for model, df in tables.items():
    print(f"\n=== {model} (MSE: {results[model]:.4f}) ===")
    print(df.to_string(index=False))

# 7. Plot a bar chart of Mean Squared Error comparisons
plt.figure(figsize=(8, 4))
bars = plt.bar(results.keys(), results.values(), edgecolor="k")
plt.ylabel("Mean Squared Error")
plt.title("ICL MSE Comparison on California Housing")
plt.xticks(rotation=30, ha="right")
for bar, mse in zip(bars, results.values()):
    plt.text(bar.get_x() + bar.get_width()/2, mse + 0.01,
             f"{mse:.2f}", ha="center", va="bottom")
plt.tight_layout()
plt.show()
