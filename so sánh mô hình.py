
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
reports = {}

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_val_vec)
    acc = accuracy_score(y_val, y_pred)
    results[name] = acc
    reports[name] = classification_report(y_val, y_pred, output_dict=True)
    
    print(f"\n==== {name} ====")
    print(classification_report(y_val, y_pred))


print("\n📊 So sánh độ chính xác:")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")

# Chuyển sang DataFrame để dễ xem và vẽ
results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
print("\nBảng kết quả:\n", results_df)

plt.figure(figsize=(8,5))
sns.barplot(data=results_df, x="Model", y="Accuracy", palette="viridis")
plt.title("So sánh Accuracy giữa các mô hình")
plt.ylim(0, 1)
for i, v in enumerate(results_df["Accuracy"]):
    plt.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
plt.show()


