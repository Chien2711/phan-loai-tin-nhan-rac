import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Đọc dữ liệu
train_df = pd.read_csv("/kaggle/input/spam-message-classification/train.csv")
test_df = pd.read_csv("/kaggle/input/spam-message-classification/test.csv")
sample_submission = pd.read_csv("/kaggle/input/spam-message-classification/sample_submission.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print(train_df.head())


print(train_df['label'].value_counts())


X_train, X_val, y_train, y_val = train_test_split(
    train_df['sms'], train_df['label'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(test_df['sms'])

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_val_pred = model.predict(X_val_vec)
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

cm = confusion_matrix(y_val, y_val_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix - Validation")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

y_test_pred = model.predict(X_test_vec)

submission = sample_submission.copy()
submission['label'] = y_test_pred
submission.to_csv("submission.csv", index=False)
print("Submission file saved!")


