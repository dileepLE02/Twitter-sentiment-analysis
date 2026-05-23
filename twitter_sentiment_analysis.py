"""
Twitter Sentiment Analysis using NLTK
======================================
Author  : Ballem Dileep
Dataset : Sentiment140 (1.6 million tweets)
GitHub  : https://github.com/dileepLE02/Twitter-sentiment-analysis.git
"""

# ─────────────────────────────────────────────
# PART 1 ─ EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────

# Step 1: Import Required Libraries
import re
import string
import warnings

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from wordcloud import WordCloud

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ─────────────────────────────────────────────
# Step 2: Load the Dataset
# ─────────────────────────────────────────────
# Update the path below to point to your local copy of the dataset.
DATASET_PATH = (
    r"C:\Users\balle\OneDrive\Desktop\project O\ds project"
    r"\Dataset_Sentiment Analysis of Twitter using NL toolkit.csv"
)

df = pd.read_csv(DATASET_PATH, encoding="latin-1", header=None)

# Step 3: Add Column Names
df.columns = ["target", "id", "date", "flag", "user", "text"]

# Step 4: Check Dataset Shape
print("Rows and Columns:", df.shape)

# Step 5: Inspect Data Types
print(df.dtypes)

# ─────────────────────────────────────────────
# Step 6-8: Missing Values & Duplicates
# ─────────────────────────────────────────────
print("\nNull counts:\n", df.isnull().sum())
missing_pct = (df.isnull().sum() / len(df)) * 100
print("\nMissing % :\n", missing_pct)
print("\nDuplicate rows:", df.duplicated().sum())

# Step 9-10: Remove Duplicates
df = df.drop_duplicates()
print("\nDataset after removing duplicates:", df.shape)

# Step 11: Map polarity to readable sentiment
df["sentiment"] = df["target"].replace({0: "Negative", 4: "Positive"})

# ─────────────────────────────────────────────
# Step 12: Visualise Sentiment Distribution
# ─────────────────────────────────────────────
colors = ["#6C8EBF", "#D9A66B"]
plt.figure(figsize=(10, 5))
sns.countplot(x="sentiment", data=df, palette=colors)
plt.title("Distribution of Sentiment Classes", fontsize=16, fontweight="bold")
plt.xlabel("Sentiment", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.tight_layout()
plt.savefig("plot_01_sentiment_distribution.png", dpi=150)
plt.show()
print("Insight: Balanced positive/negative classes – ideal for ML classification.")

# ─────────────────────────────────────────────
# Step 13: Tweet Length Distribution
# ─────────────────────────────────────────────
df["tweet_length"] = df["text"].apply(len)
plt.figure(figsize=(8, 5))
sns.histplot(df["tweet_length"], bins=30)
plt.title("Tweet Length Distribution")
plt.xlabel("Tweet Length (characters)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("plot_02_tweet_length.png", dpi=150)
plt.show()
print("Insight: Most tweets are short (< 150 chars), reflecting Twitter's format.")

# ─────────────────────────────────────────────
# Step 14: Word Count Distribution
# ─────────────────────────────────────────────
df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(10, 5))
sns.histplot(df["word_count"], bins=30, color="#6C8EBF")
plt.title("Word Count Distribution", fontsize=16, fontweight="bold")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("plot_03_word_count.png", dpi=150)
plt.show()
print("Insight: Majority of tweets contain 5-25 words.")


# ─────────────────────────────────────────────
# PART 2 ─ DATA TRANSFORMATION & MODEL BUILDING
# ─────────────────────────────────────────────

# Step 15: Download NLTK Resources
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")

# Step 16: Convert Polarity Labels  (4 → 1 for binary classification)
df["target"] = df["target"].replace(4, 1)

# Step 17: Create NLP Tools
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


# ─────────────────────────────────────────────
# Step 18: Text Cleaning Function
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Full NLP preprocessing pipeline for a single tweet."""
    text = text.lower()                                         # Lowercase
    text = re.sub(r"http\S+|www\S+", "", text)                 # Remove URLs
    text = re.sub(r"@\w+|#", "", text)                         # Remove mentions/hashtags
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = re.sub(r"\d+", "", text)                             # Remove numbers
    tokens = word_tokenize(text)                                # Tokenise
    tokens = [w for w in tokens if w not in stop_words]        # Remove stopwords
    tokens = [stemmer.stem(w) for w in tokens]                 # Stemming
    tokens = [lemmatizer.lemmatize(w) for w in tokens]         # Lemmatisation
    return " ".join(tokens)


# ─────────────────────────────────────────────
# Step 19: Apply Cleaning on a 50k Sample
# ─────────────────────────────────────────────
df_sample = df.sample(50_000, random_state=42)
print("Applying text cleaning – this may take a minute …")
df_sample["clean_text"] = df_sample["text"].apply(clean_text)
print(df_sample[["text", "clean_text"]].head())

# ─────────────────────────────────────────────
# Step 20: Train-Test Split (80 / 20)
# ─────────────────────────────────────────────
X = df_sample["clean_text"]
y = df_sample["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

# ─────────────────────────────────────────────
# Step 21: TF-IDF Vectorisation
# ─────────────────────────────────────────────
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ─────────────────────────────────────────────
# Step 22: Model 1 – Bernoulli Naive Bayes
# ─────────────────────────────────────────────
bnb_model = BernoulliNB()
bnb_model.fit(X_train_tfidf, y_train)
bnb_pred = bnb_model.predict(X_test_tfidf)
bnb_accuracy = accuracy_score(y_test, bnb_pred)
print(f"\nBernoulli Naive Bayes Accuracy : {bnb_accuracy:.4f}")

# ─────────────────────────────────────────────
# Step 23: Model 2 – Logistic Regression
# ─────────────────────────────────────────────
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
lr_pred = lr_model.predict(X_test_tfidf)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f"Logistic Regression Accuracy   : {lr_accuracy:.4f}")

# ─────────────────────────────────────────────
# Step 24: Model 3 – Support Vector Machine
# ─────────────────────────────────────────────
svm_model = LinearSVC()
svm_model.fit(X_train_tfidf, y_train)
svm_pred = svm_model.predict(X_test_tfidf)
svm_accuracy = accuracy_score(y_test, svm_pred)
print(f"SVM Accuracy                   : {svm_accuracy:.4f}")

# ─────────────────────────────────────────────
# Step 25: Word Clouds
# ─────────────────────────────────────────────
positive_tweets = df_sample[df_sample["target"] == 1]["clean_text"]
negative_tweets = df_sample[df_sample["target"] == 0]["clean_text"]
positive_text = " ".join(positive_tweets.astype(str))
negative_text = " ".join(negative_tweets.astype(str))

# Positive WordCloud
pos_wc = WordCloud(width=800, height=400, background_color="white", colormap="Blues").generate(
    positive_text
)
plt.figure(figsize=(10, 5))
plt.imshow(pos_wc, interpolation="bilinear")
plt.axis("off")
plt.title("Positive Tweet WordCloud", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("plot_04_positive_wordcloud.png", dpi=150)
plt.show()

# Negative WordCloud
neg_wc = WordCloud(width=800, height=400, background_color="white", colormap="Reds").generate(
    negative_text
)
plt.figure(figsize=(10, 5))
plt.imshow(neg_wc, interpolation="bilinear")
plt.axis("off")
plt.title("Negative Tweet WordCloud", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("plot_05_negative_wordcloud.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────
# Step 26: Model Accuracy Comparison
# ─────────────────────────────────────────────
models_list = ["Bernoulli NB", "Logistic Regression", "SVM"]
accuracies = [bnb_accuracy, lr_accuracy, svm_accuracy]

plt.figure(figsize=(8, 5))
sns.barplot(x=models_list, y=accuracies, palette=["#6C8EBF", "#D9A66B", "#8E7DBE"])
plt.title("Model Accuracy Comparison", fontsize=16, fontweight="bold")
plt.ylabel("Accuracy Score")
plt.xlabel("Models")
plt.ylim(0.7, 1.0)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=11)
plt.tight_layout()
plt.savefig("plot_06_accuracy_comparison.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────
# Step 27: Confusion Matrices
# ─────────────────────────────────────────────
for name, pred, cmap in [
    ("Bernoulli NB", bnb_pred, "Blues"),
    ("Logistic Regression", lr_pred, "Oranges"),
    ("SVM", svm_pred, "Purples"),
]:
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(f"plot_07_cm_{safe_name}.png", dpi=150)
    plt.show()

# ─────────────────────────────────────────────
# Step 28: ROC-AUC Curves
# ─────────────────────────────────────────────
# BNB
bnb_probs = bnb_model.predict_proba(X_test_tfidf)[:, 1]
fpr_bnb, tpr_bnb, _ = roc_curve(y_test, bnb_probs)
roc_auc_bnb = auc(fpr_bnb, tpr_bnb)

# LR
lr_probs = lr_model.predict_proba(X_test_tfidf)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
roc_auc_lr = auc(fpr_lr, tpr_lr)

# SVM (calibrated for probabilities)
svm_calibrated = CalibratedClassifierCV(svm_model)
svm_calibrated.fit(X_train_tfidf, y_train)
svm_probs = svm_calibrated.predict_proba(X_test_tfidf)[:, 1]
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_probs)
roc_auc_svm = auc(fpr_svm, tpr_svm)

plt.figure(figsize=(8, 6))
plt.plot(fpr_bnb, tpr_bnb, label=f"Bernoulli NB (AUC = {roc_auc_bnb:.2f})")
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {roc_auc_lr:.2f})")
plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC = {roc_auc_svm:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="red", label="Random Baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Curve Comparison", fontsize=16, fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig("plot_08_roc_auc.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────
# Step 29: Final Results Summary
# ─────────────────────────────────────────────
results = pd.DataFrame(
    {
        "Model": models_list,
        "Accuracy": [bnb_accuracy, lr_accuracy, svm_accuracy],
        "AUC": [roc_auc_bnb, roc_auc_lr, roc_auc_svm],
    }
)
print("\n===== Final Model Results =====")
print(results.to_string(index=False))
print("\nSelected Model: Logistic Regression")
print("Reason       : Highest accuracy (75.33%) and AUC (0.83), good scalability.")
