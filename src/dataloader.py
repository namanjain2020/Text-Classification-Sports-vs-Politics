import kagglehub
import os
import json 
import random

random.seed(42)

# Loading the necessary data from the Kaggle datasets and then only including Sports and Political categories from it as in dataset.

def load_data():
    path = kagglehub.dataset_download("rmisra/news-category-dataset")
    data_file = os.path.join(path, "News_Category_Dataset_v3.json")
    
    sports_texts = []
    politics_texts = []
    
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            category = obj["category"]
            text = obj["headline"] + " " + obj["short_description"]

            if category == "SPORTS":
                sports_texts.append(text)
            elif category == "POLITICS":
                politics_texts.append(text)
    
    return sports_texts, politics_texts  

def balance_dataset(sports_texts, politics_texts):
    min_length = min(len(sports_texts), len(politics_texts))
    sports_balanced = random.sample(sports_texts, min_length)
    politics_balanced = random.sample(politics_texts, min_length)
    return sports_balanced, politics_balanced
    