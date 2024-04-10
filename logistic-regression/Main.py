from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Загрузка датасета breast cancer
data = load_breast_cancer()
X = data.data
y = data.target

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Стандартизация данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение метода k ближайших соседей
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)

# Считаем метрики качества для kNN
accuracy_knn = round(accuracy_score(y_test, y_pred_knn), 4)
precision_knn = round(precision_score(y_test, y_pred_knn), 4)
recall_knn = round(recall_score(y_test, y_pred_knn), 4)
f1_knn = round(f1_score(y_test, y_pred_knn), 4)

print("Метод k ближайших соседей:")
print(f"Accuracy: {accuracy_knn}")
print(f"Precision: {precision_knn}")
print(f"Recall: {recall_knn}")
print(f"F1 Score: {f1_knn}")

# Обучение логистической регрессии
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

# Считаем метрики качества для логистической регрессии
accuracy_lr = round(accuracy_score(y_test, y_pred_lr), 4)
precision_lr = round(precision_score(y_test, y_pred_lr), 4)
recall_lr = round(recall_score(y_test, y_pred_lr), 4)
f1_lr = round(f1_score(y_test, y_pred_lr), 4)

print("\nЛогистическая регрессия:")
print(f"Accuracy: {accuracy_lr}")
print(f"Precision: {precision_lr}")
print(f"Recall: {recall_lr}")
print(f"F1 Score: {f1_lr}")
