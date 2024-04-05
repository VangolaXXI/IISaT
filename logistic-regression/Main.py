from data_generation import generate_synthetic_data, train_test_split
from class_balancing import balance_classes
from models import LogisticRegression, KNearestNeighbors
from evaluation_metrics import accuracy, precision, recall, f1_score

# Генерация данных
X, y = generate_synthetic_data()

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Балансировка классов
X_train_balanced, y_train_balanced = balance_classes(X_train, y_train)

# Обучение и оценка модели логистической регрессии
model_lr = LogisticRegression()
model_lr.fit(X_train_balanced, y_train_balanced)
y_pred_lr = model_lr.predict(X_test)

# Обучение и оценка модели метода k ближайших соседей
model_knn = KNearestNeighbors(k=3)
y_pred_knn = model_knn.predict(X_test, X_train_balanced, y_train_balanced)

# Оценка качества моделей
print("Логистическая регрессия:")
print("Accuracy:", accuracy(y_test, y_pred_lr))
print("Precision:", precision(y_test, y_pred_lr))
print("Recall:", recall(y_test, y_pred_lr))
print("F1 Score:", f1_score(y_test, y_pred_lr))

print(f"\nМетод k ближайших соседей (k={model_knn.k}):")
print("Accuracy:", accuracy(y_test, y_pred_knn))
print("Precision:", precision(y_test, y_pred_knn))
print("Recall:", recall(y_test, y_pred_knn))
print("F1 Score:", f1_score(y_test, y_pred_knn))
