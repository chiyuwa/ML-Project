from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import random
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    classification_report
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve
from logging_system import logging_common as Logger

LOGNAME = "VisualIris"
log = Logger.get_logger(LOGNAME)
# 创建保存图片的目录
os.makedirs('figs', exist_ok=True)

def specificity(y_true, y_pred):
    """计算多分类特异度（按类分别计算）"""
    cm = confusion_matrix(y_true, y_pred)
    spec = []
    for i in range(n_classes):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(cm[:, i]) - cm[i, i]
        spec.append(tn / (tn + fp))
    return np.mean(spec)

os.makedirs('figs', exist_ok=True)

def plot_and_save_figures(model, X_test, y_test, model_name):
    """绘制并保存评估图表"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'figs/confusion_matrix_{model_name.lower().replace(" ", "_")}.png', 
                bbox_inches='tight')
    plt.close()
    
    # 绘制多分类ROC曲线
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc_score(y_test_bin[:, i], y_proba[:, i]):.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'figs/auroc_{model_name.lower().replace(" ", "_")}.png', 
                bbox_inches='tight')
    plt.close()
    
    # 绘制PR曲线
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
        plt.plot(recall, precision, 
                 label=f'{class_names[i]} (AP = {average_precision_score(y_test_bin[:, i], y_proba[:, i]):.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curves - {model_name}')
    plt.legend(loc="upper right")
    plt.savefig(f'figs/aupr_{model_name.lower().replace(" ", "_")}.png', 
                bbox_inches='tight')
    plt.close()


def evaluate_model(model, X_test, y_test, model_name, class_names):
    """评估模型性能"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # 基础指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    spec = specificity(y_test, y_pred)
    
    # AUROC和AUPR
    auroc = roc_auc_score(y_test_bin, y_proba, multi_class='ovr')
    aupr = average_precision_score(y_test_bin, y_proba)
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plot_and_save_figures(model, X_test, y_test, model_name)
    # 输出结果
    log.info(f"\n{model_name} Performance:")
    log.info(f"Accuracy: {accuracy:.4f}")
    log.info(f"Precision (macro): {precision:.4f}")
    log.info(f"Recall/Sensitivity (macro): {recall:.4f}")
    log.info(f"Specificity (macro): {spec:.4f}")
    log.info(f"AUROC (OvR): {auroc:.4f}")
    log.info(f"AUPR: {aupr:.4f}")
    log.info("\nClassification Report:")
    log.info(classification_report(y_test, y_pred, target_names=class_names))
    log.info("Confusion Matrix:\n")
    log.info(pd.DataFrame(cm, index=class_names, columns=class_names))
if __name__ == "__main__":
    iris = load_iris()

    # Features and target
    X = iris.data
    y = iris.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    class_names = iris.target_names  # 明确定义类别名称

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 二值化标签用于AUROC/AUPR计算
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = 3
    # 初始化模型
    models = {
        "SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    }

    # 训练和评估模型
    for name, model in models.items():
        model.fit(X_train, y_train)
        evaluate_model(model, X_test, y_test, name, class_names)