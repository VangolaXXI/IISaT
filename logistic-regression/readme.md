# Классификация Рака Груди

Этот репозиторий содержит код для обучения и оценки двух моделей машинного обучения, k-Nearest Neighbors (kNN) и логистической регрессии, на датасете Breast Cancer. Целью является классификация опухолей рака груди как злокачественных или доброкачественных на основе различных признаков.

## Датасет
Для данного проекта использовался датасет Breast Cancer, доступный в библиотеке sklearn. Он содержит признаки, такие как средний радиус, средняя текстура, средний периметр и другие.

## Модели

| Модель                           | Точность (Accuracy) | Точность (Precision) | Полнота (Recall) | F1-мера |
|----------------------------------|----------------------|----------------------|------------------|---------|
| k-Nearest Neighbors (kNN)        | 0.9474               | 0.9577               | 0.9577           | 0.9577  |
| Логистическая Регрессия          | 0.9737               | 0.9722               | 0.9859           | 0.979   |

Из результатов можно сделать следующие выводы:
1. Обе модели достигли высокой точности (Accuracy) в более чем 94%, что указывает на их хорошую обобщающую способность на данном датасете.
2. Логистическая регрессия показала лучшие показатели Recall и F1 Score, что говорит о ее лучшей способности обнаруживать злокачественные опухоли из всех реальных положительных случаев.
