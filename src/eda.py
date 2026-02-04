import matplotlib.pyplot as plt
from collections import Counter
import os

def plot_class_distribution(sports, politics, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.bar(["SPORTS", "POLITICS"], [len(sports), len(politics)])
    plt.title("Class Distribution")
    plt.ylabel("Samples")
    plt.savefig(os.path.join(save_dir, "class_distribution.png"))
    plt.close()

def plot_text_length_distribution(sports, politics, save_dir):
    s_len = [len(t.split()) for t in sports]
    p_len = [len(t.split()) for t in politics]

    plt.figure()
    plt.hist(s_len, bins=30, alpha=0.6, label="SPORTS")
    plt.hist(p_len, bins=30, alpha=0.6, label="POLITICS")
    plt.legend()
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(save_dir, "text_length_distribution.png"))
    plt.close()

def plot_top_words(texts, title, save_path, top_n=20):
    words = []
    for t in texts:
        words.extend(t.split())

    freq = Counter(words).most_common(top_n)
    labels, values = zip(*freq)

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.xticks(rotation=45)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
