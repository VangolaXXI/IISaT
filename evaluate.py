from sklearn.metrics import accuracy_score
from knn import KNN
from tabulate import tabulate


def evaluate_model(X_train, X_test, y_train, y_test, k_values, distance_metrics, random_states):
    results = []
    for random_state in random_states:
        for distance_metric in distance_metrics:
            for k in k_values:
                knn = KNN(k=k, distance_metric=distance_metric)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                results.append([k, distance_metric, accuracy, random_state])

    return results


if __name__ == "__main__":
    from compute import X_train, X_test, y_train, y_test

    # Оценка качества классификации
    k_values = [3, 5, 7]
    distance_metrics = ['euclidean', 'manhattan', 'chebyshev']
    random_states = [42, 1]

    print("Results:")
    all_results = []
    for random_state in random_states:
        all_results += evaluate_model(X_train, X_test, y_train, y_test, k_values, distance_metrics, [random_state])

    headers = ["k", "Distance Metric", "Accuracy", "Random State"]
    print(tabulate(all_results, headers=headers, tablefmt="pretty"))
