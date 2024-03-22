from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Загрузка данных
data = load_breast_cancer()
X = data.data
y = data.target

# Создание объекта классификатора kNN
knn = KNeighborsClassifier(n_neighbors=5)  # Выберите количество соседей

# Выполнение кросс-валидации с использованием 5 фолдов и оценка по точности
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')# Для полноты (Recall)

# Вывод результатов кросс-валидации
print("Точность для каждого фолда:")
for i, score in enumerate(scores):
    print(f"Фолд {i+1}: {score}")

print("Средняя точность: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
