import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, lognorm

# Загрузка данных
data = pd.read_csv('../ozon_dataset_cleaned_price.csv')

# Фильтрация корректных значений цены
price_data = data['price_clean'].dropna()

# Построение PDF и подгонка распределений
plt.figure(figsize=(12, 6))
sns.histplot(price_data, bins=50, kde=True, stat="density", label="Histogram (PDF)", alpha=0.6, color="blue")

# Нормальное распределение
mean, std = norm.fit(price_data)
x = np.linspace(price_data.min(), price_data.max(), 1000)
pdf_norm = norm.pdf(x, mean, std)

# Лог-нормальное распределение
shape, loc, scale = lognorm.fit(price_data, floc=0)
pdf_lognorm = lognorm.pdf(x, shape, loc, scale)

# Добавление линий распределений
plt.plot(x, pdf_norm, 'r-', label='Normal Distribution', linewidth=2)
plt.plot(x, pdf_lognorm, 'g-', label='Log-Normal Distribution', linewidth=2)

# Оформление графика
plt.title("PDF и подгонка распределений для цен")
plt.xlabel("Цена (KZT)")
plt.ylabel("Плотность вероятности")
plt.legend()
plt.grid()
plt.show()

# Параметры подогнанных распределений
print("Нормальное распределение:")
print(f"Среднее: {mean:.2f}, Стандартное отклонение: {std:.2f}")
print("Лог-нормальное распределение:")
print(f"Shape: {shape:.2f}, Loc: {loc:.2f}, Scale: {scale:.2f}")

# Моделирование данных
num_samples = 1000

# Генерация синтетических данных для нормального распределения
synthetic_normal = norm.rvs(loc=mean, scale=std, size=num_samples)

# Генерация синтетических данных для лог-нормального распределения
synthetic_lognorm = lognorm.rvs(shape, loc=loc, scale=scale, size=num_samples)

# Построение синтетических данных
plt.figure(figsize=(12, 6))
sns.histplot(synthetic_normal, bins=50, kde=True, stat="density", label="Synthetic Normal", alpha=0.6, color="red")
sns.histplot(synthetic_lognorm, bins=50, kde=True, stat="density", label="Synthetic Log-Normal", alpha=0.6, color="green")
plt.title("Синтетические данные на основе подогнанных распределений")
plt.xlabel("Цена (KZT)")
plt.ylabel("Плотность вероятности")
plt.legend()
plt.grid()
plt.show()

# Статистика синтетических данных
synthetic_normal_stats = {
    "mean": np.mean(synthetic_normal),
    "std_dev": np.std(synthetic_normal)
}

synthetic_lognorm_stats = {
    "mean": np.mean(synthetic_lognorm),
    "std_dev": np.std(synthetic_lognorm)
}

print("Статистика синтетических данных (Нормальное распределение):")
print(synthetic_normal_stats)
print("Статистика синтетических данных (Лог-нормальное распределение):")
print(synthetic_lognorm_stats)
