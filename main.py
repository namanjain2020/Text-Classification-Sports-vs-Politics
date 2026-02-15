import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from src.dataloader import load_data, balance_dataset
from src.preprocessing import preprocess_corpus
from src.features import bow_features, tf_idf_features, n_gram_features
from src.train_models import train_nb, train_lr, train_svm
from src.evaluate import evaluate
from src.visualize import plot_confusion_matrix, plot_metric_comparison
from src.eda import (plot_class_distribution, plot_text_length_distribution, plot_top_words)
import pickle
import os

# Loading the dataset
sports, politics = load_data()

# Balance dataset to handle the class imbalances.
sports, politics = balance_dataset(sports, politics)

# Combining the data so as to make a valid dataset for the problem given
X = sports + politics
y = ["SPORTS"] * len(sports) + ["POLITICS"] * len(politics)

# Dataset Shuffling 
combined = list(zip(X, y))
random.shuffle(combined)
X, y = zip(*combined)

# Text Pre-processing
X = preprocess_corpus(X)

# Exploratory data analysis
sports_clean = preprocess_corpus(sports)
politics_clean = preprocess_corpus(politics)

plot_class_distribution(sports_clean, politics, "report/figures")
plot_text_length_distribution(sports_clean, politics, "report/figures")
plot_top_words(sports_clean, "Top SPORTS Words", "report/figures/top_words_sports.png")
plot_top_words(politics_clean, "Top POLITICS Words", "report/figures/top_words_politics.png")

# K-Fold Cross Validation setup
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

feature_sets = {
    "BOW": bow_features,
    "TFIDF": tf_idf_features,
    "NGRAM": n_gram_features
}

models = {
    "Naive Bayes": train_nb,
    "Logistic Regression": train_lr,
    "SVM": train_svm
}

results = []
best_f1 = -1
best_feature = None
best_model_name = None

# Cross-validation loop and model training
for feat_name, feat_func in feature_sets.items():
    for model_name, train_func in models.items():

        acc_scores = []
        f1_scores = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):

            X_train = [X[i] for i in train_idx]
            X_test = [X[i] for i in test_idx]
            y_train = [y[i] for i in train_idx]
            y_test = [y[i] for i in test_idx]

            # Performing text vectorization inside each fold
            Xtr, Xte, vectorizer = feat_func(X_train, X_test)
            model = train_func(Xtr, y_train)
            accuracy, report_table, y_pred = evaluate(model, Xte, y_test)
            acc_scores.append(accuracy)

            # extracting SPORTS F1-score from table string
            for line in report_table.split("\n"):
                if line.strip().startswith("SPORTS"):
                    f1_scores.append(float(line.split()[3]))
                    break

        # Storing MEAN metrics across folds
        mean_f1 = np.mean(f1_scores)
        results.append({
        "feature": feat_name,
        "model": model_name,
        "accuracy": np.mean(acc_scores),
        "f1": mean_f1
        })

        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_feature = feat_name
            best_model_name = model_name

print("Best configuration selected:")
print("Feature:", best_feature)
print("Model:", best_model_name)
print("Best mean F1:", best_f1)

best_feature_func = feature_sets[best_feature]
best_train_func = models[best_model_name]

# Train on FULL dataset
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
Xtr_final, Xte_final, best_vectorizer = best_feature_func(X_train_final,X_test_final)

# FINAL evaluation of BEST model
best_model = best_train_func(Xtr_final, y_train_final)
final_accuracy, final_report, final_y_pred = evaluate(best_model, Xte_final, y_test_final)

# Save confusion matrix of BEST model
plot_confusion_matrix(y_test_final, final_y_pred, f"Confusion Matrix ({best_feature} + {best_model_name})", "report/figures/cm_best_model.png")

# Save full classification report of BEST model
with open("report/figures/best_model_classification_report.txt", "w") as f:
    f.write(final_report)

os.makedirs("saved_models", exist_ok=True)

with open("saved_models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("saved_models/best_vectorizer.pkl", "wb") as f:
    pickle.dump(best_vectorizer, f)

# Comparison plots
plot_metric_comparison(results,"accuracy","report/figures/accuracy_comparison.png")
plot_metric_comparison(results,"f1","report/figures/f1_comparison.png")
