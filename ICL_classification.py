import json
import random
import numpy as np
import requests
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1. Load and split the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
X_ctx, X_test, y_ctx, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
K = 10
X_shot, y_shot = X_ctx[:K], y_ctx[:K]

def make_icl_prompt(X_examples, y_examples, x_query):
    prompt = (
        "Help me predict the Output value for the last Input. "
        "Your response should only contain the Output value in the format of #Output value#.\n\n"
    )
    for x, y in zip(X_examples, y_examples):
        prompt += f"Input: {x.tolist()}, Output: {y}\n"
    prompt += f"\nInput: {x_query.tolist()}, Output: "
    return prompt

# 2. Define the ICL call function
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
    r = requests.post(
        API_URL, headers=HEADERS,
        data=json.dumps(payload),
        timeout=10,
        proxies={"http": None, "https": None}
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

# 3. List of four models
models = [
    "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "THUDM/chatglm3-6b",
    "THUDM/GLM-4-9B-0414"
]

results = {}
tables = {}

# 4. Evaluate each model without printing prompts and raw outputs
for model in models:
    true_labels, pred_labels = [], []
    for x_q, y_true in zip(X_test, y_test):
        prompt = make_icl_prompt(X_shot, y_shot, x_q)
        try:
            raw = call_icl(model, prompt)
            pred = int(raw.replace("#", ""))
        except Exception:
            pred = random.randint(0, len(np.unique(y)) - 1)
        true_labels.append(y_true)
        pred_labels.append(pred)

    # Record the accuracy
    acc = np.mean(np.array(pred_labels) == np.array(true_labels))
    results[model] = acc

    # Prepare the table
    df = pd.DataFrame({
        "Input": [list(inp) for inp in X_test],
        "Predicted": pred_labels,
        "True": true_labels
    })
    tables[model] = df

# 5. Print predicted vs. actual results for each model using pandas
for model, df in tables.items():
    print(f"\n=== {model} (Accuracy: {results[model]:.3f}) ===")
    print(df.to_string(index=False))

# 6. Plot a bar chart of model accuracies
plt.figure(figsize=(8, 4))
bars = plt.bar(results.keys(), results.values(), edgecolor="k")
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.xticks(rotation=30, ha="right")
plt.title("ICL Accuracy Comparison")
for bar, acc in zip(bars, results.values()):
    plt.text(bar.get_x() + bar.get_width()/2, acc + 0.02,
             f"{acc:.3f}", ha="center", va="bottom")
plt.tight_layout()
plt.show()
