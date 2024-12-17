import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('../ozon_dataset_cleaned_price.csv')

# Очистка столбца цены от нечисловых символов и преобразование в числовой формат
# data['price_clean'] = data['price'].str.replace(r'\D', '', regex=True).astype(float)

# Фильтрация корректных значений цены
price_data = data['price_clean'].dropna()

# Создание диапазонов цен:
# До 100,000 с шагом 10,000
low_price_bins = np.arange(10000, 100001, 10000)
# Выше 100,000 с шагом 100,000 (начинаем с 200,000, чтобы избежать дублирования границ)
high_price_bins = np.arange(200000, price_data.max() + 100000, 100000)
# Объединение диапазонов
combined_bins = np.concatenate((low_price_bins, high_price_bins))

# Создание меток для диапазонов
combined_labels = [f"{int(combined_bins[i])}-{int(combined_bins[i+1])}" for i in range(len(combined_bins) - 1)]

# Группировка цен по диапазонам
binned_prices_combined = pd.cut(price_data, bins=combined_bins, labels=combined_labels, right=False)

# Подсчёт количества товаров в каждом диапазоне
binned_counts_combined = binned_prices_combined.value_counts(sort=False)

# Вычисление PMF (относительная частота)
binned_pmf_combined = binned_counts_combined / binned_counts_combined.sum()

# Вычисление CDF (накопленная вероятность)
binned_cdf_combined = binned_pmf_combined.cumsum()

# Построение графика PMF
plt.figure(figsize=(12, 6))
plt.bar(combined_labels, binned_pmf_combined, alpha=0.6, label='PMF')
plt.title("Функция вероятности распределения (PMF) цен по диапазонам")
plt.xlabel("Диапазон цен (KZT)")
plt.ylabel("Вероятность")
plt.xticks(rotation=45)
plt.legend()
plt.grid()

# Построение графика CDF
plt.figure(figsize=(12, 6))
plt.step(combined_labels, binned_cdf_combined, where='mid', label='CDF')
plt.title("Кумулятивная функция распределения (CDF) цен по диапазонам")
plt.xlabel("Диапазон цен (KZT)")
plt.ylabel("Накопленная вероятность")
plt.xticks(rotation=45)
plt.legend()
plt.grid()

# Отображение графиков
plt.show()
