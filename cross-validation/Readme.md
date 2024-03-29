# Кросс-валидация для классификации методом kNN

Этот проект представляет собой выполнение кросс-валидации для оценки качества классификации методом kNN (k-Nearest Neighbors) при различных значениях гиперпараметра k.

## Использованные библиотеки

- scikit-learn
- numpy

## Установка

Для запуска проекта необходимо установить библиотеку scikit-learn:
и numpy

## Использование

1. Запустите скрипт `knn_cross_validation.py`.
2. Результаты кросс-валидации и оптимальное значение k будут выведены в консоль.

## Результаты кросс-валидации
При `accuracy`

Значение k: 1, Средняя точность: 0.9051079024996118

Значение k: 2, Средняя точность: 0.9050768514205869

Значение k: 3, Средняя точность: 0.9191429902189101

Значение k: 4, Средняя точность: 0.9208818506443098

Значение k: 5, Средняя точность: 0.9279459711224964

Значение k: 6, Средняя точность: 0.9244216736531594

Значение k: 7, Средняя точность: 0.9261760596180716

Значение k: 8, Средняя точность: 0.9279459711224964

Значение k: 9, Средняя точность: 0.9314702685918336

Значение k: 10, Средняя точность: 0.9314702685918336

Оптимальное значение k: 9

## Описание файлов

- `knn_cross_validation.py`: основной скрипт, выполняющий кросс-валидацию для различных значений k и выводящий результаты.
- `README.md`: этот файл с описанием проекта и инструкциями по его использованию.

## Автор

Автор: [Ваше имя]

## Лицензия

Этот проект лицензирован по лицензии MIT. См. файл `LICENSE` для получения подробной информации.