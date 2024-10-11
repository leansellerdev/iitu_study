from sklearn.datasets import fetch_california_housing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score
import numpy as np

# 1. Загрузка данных
boston = fetch_california_housing()
X = boston.data
y = boston.target

# 2. Масштабирование признаков
X_scaled = scale(X)

# 3. Перебор значений параметра p
best_p = 1
best_score = float('inf')

for p in np.linspace(1, 20, 300):
    knn = KNeighborsRegressor(n_neighbors=6, weights='distance', p=p)
    scores = cross_val_score(knn, X_scaled, y, cv=10, scoring='neg_mean_squared_error')
    mean_score = -scores.mean()
    if mean_score < best_score:
        best_p = p
        best_score = mean_score

print(f'Лучшее значение p: {best_p}, Среднеквадратичная ошибка: {best_score}')
