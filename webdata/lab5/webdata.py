import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Загрузка данных
data = pd.read_csv('../ozon_dataset_cleaned_price.csv')

# Очистка данных
clean_data = data[['price_clean', 'rating']].dropna()

### 1. Scatter Plot с прозрачностью ###
plt.figure(figsize=(12, 6))
sns.scatterplot(
    x=clean_data['price_clean'],
    y=clean_data['rating'],
    alpha=0.6,
    edgecolor=None
)
plt.title("Scatter Plot: Price vs. Rating")
plt.xlabel("Price (KZT)")
plt.ylabel("Rating")
plt.grid()
plt.show()

### 2. Корреляция ###
# Pearson correlation
pearson_corr, pearson_pval = stats.pearsonr(clean_data['price_clean'], clean_data['rating'])
# Spearman correlation
spearman_corr, spearman_pval = stats.spearmanr(clean_data['price_clean'], clean_data['rating'])

print("Корреляция:")
print(f"Коэффициент Пирсона: {pearson_corr:.4f}, p-значение: {pearson_pval:.4e}")
print(f"Коэффициент Спирмена: {spearman_corr:.4f}, p-значение: {spearman_pval:.4e}")

### 3. Ковариация ###
covariance = np.cov(clean_data['price_clean'], clean_data['rating'])[0, 1]
print(f"Ковариация между ценой и рейтингом: {covariance:.4f}")

### 4. Характеризация взаимосвязей ###
# Разделение данных на группы по ценам
bins = np.arange(0, clean_data['price_clean'].max() + 10000, 10000)
clean_data['price_bin'] = pd.cut(clean_data['price_clean'], bins=bins)

# Рассчёт среднего рейтинга в каждой группе
grouped = clean_data.groupby('price_bin')['rating'].agg(['mean', 'median', 'count']).reset_index()

# Построение CDF для групп
plt.figure(figsize=(12, 6))
for group in grouped.itertuples():
    group_data = clean_data[clean_data['price_bin'] == group.price_bin]['rating']
    cdf = np.sort(group_data)
    cdf_values = np.arange(1, len(cdf) + 1) / len(cdf)
    plt.step(cdf, cdf_values, where='post', label=f"Price {group.price_bin}")

plt.title("CDF: Rating within Price Groups")
plt.xlabel("Rating")
plt.ylabel("Cumulative Probability")
plt.legend()
plt.grid()
plt.show()

### 5. Характеристика групп ###
print("Характеристика групп по цене:")
print(grouped)
