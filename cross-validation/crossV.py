from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
import numpy as np

# Загрузка датасета Iris
data = load_breast_cancer()
X = data.data
y = data.target

# Список значений k для kNN
k_values = list(range(1,11))

# Пустые списки для хранения результатов кросс-валидации
accuracy_scores = []

# Выполнение кросс-валидации для каждого значения k
for k in k_values:
    # Создание классификатора kNN
    knn = KNeighborsClassifier(n_neighbors=k)
    # Выполнение кросс-валидации с 5 фолдами и оценка качества
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    # Сохранение средней точности для данного значения k
    accuracy_scores.append(np.mean(scores))

# Вывод результатов
for k, accuracy in zip(k_values, accuracy_scores):
    print(f"Значение k: {k}, Средняя точность: {accuracy}")

# Выбор оптимального значения k
optimal_k = k_values[np.argmax(accuracy_scores)]
print(f"\nОптимальное значение k: {optimal_k}")