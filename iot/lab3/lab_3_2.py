from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import scale
import numpy as np
import pandas as pd

# 1. Загрузка данных
url = '../../wine.csv'
wine_data = pd.read_csv(url)

# 2. Разделение данных на признаки и ответы
X = wine_data.iloc[:, 1:].values
y = wine_data.iloc[:, 0].values

# 3. Кросс-валидация
kf = KFold(n_splits=10, shuffle=True, random_state=100)

# 4. Вычисление качества классификации для метода KNN с изменением k от 1 до 100
best_k = 1
best_score = 0

for k in range(1, 101):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=kf, scoring='accuracy')
    if scores.mean() > best_score:
        best_k = k
        best_score = scores.mean()

print(f'Лучшее k: {best_k}, Качество: {best_score}')

# 6. Масштабирование признаков
X_scaled = scale(X)

# Снова нахождение лучшего k после масштабирования
best_k_scaled = 1
best_score_scaled = 0

for k in range(1, 101):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_scaled, y, cv=kf, scoring='accuracy')
    if scores.mean() > best_score_scaled:
        best_k_scaled = k
        best_score_scaled = scores.mean()

print(f'Лучшее k после масштабирования: {best_k_scaled}, Качество: {best_score_scaled}')


# Лучшее k: 1, Качество: 0.7513071895424837
# Лучшее k после масштабирования: 31, Качество: 0.9826797385620916
