import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=["POLITICS", "SPORTS"])

    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.colorbar()

    plt.xticks([0, 1], ["POLITICS", "SPORTS"])
    plt.yticks([0, 1], ["POLITICS", "SPORTS"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_metric_comparison(results, metric, save_path):
    labels = [f"{r['feature']} + {r['model']}" for r in results]
    values = [r[metric] for r in results]

    plt.figure(figsize=(12, 5))
    bars = plt.bar(labels, values)

    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} Comparison")

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.4f}", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

