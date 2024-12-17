import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# 1. Загрузка данных
data = pd.read_csv('ozon_dataset.csv')  # замените на имя вашего файла

data = data[(data['review_count'] >= 0) & (data['review_count'] <= 10000)]

print(data.head())

# 2. Описание данных и базовая статистика
print("\nОписание данных:")
print(data.describe())

# 3. Визуализация распределений
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(data['rating'], bins=30, kde=True)
plt.title('Распределение рейтингов')

plt.subplot(1, 2, 2)
sns.histplot(data['review_count'], bins=30, kde=True)
plt.title('Распределение количества отзывов')

plt.show()

# Boxplot для поиска выбросов
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['rating'])
plt.title('Boxplot для рейтингов')
plt.show()

# 4. Изучение связи между rating и review_count
plt.figure(figsize=(8, 6))
sns.scatterplot(x='rating', y='review_count', data=data)
plt.title('Диаграмма рассеяния: rating vs review_count')
plt.show()

# Рассчитываем корреляцию
correlation = data['rating'].corr(data['review_count'])
print(f'Корреляция между rating и review_count: {correlation:.3f}')

# 5. Группировка данных по рейтингу
high_ratings = data[data['rating'] > 4.5]
low_ratings = data[data['rating'] <= 4.5]

print("\nСтатистика для высоких рейтингов (>4.5):")
print(high_ratings['review_count'].describe())

print("\nСтатистика для низких рейтингов (<=4.5):")
print(low_ratings['review_count'].describe())

# 6. Проверка гипотезы о равенстве средних значений
stat, p_value = ttest_ind(high_ratings['review_count'], low_ratings['review_count'], equal_var=False)
print("\nРезультаты t-теста:")
print(f"Статистика: {stat:.3f}, p-value: {p_value:.3f}")

if p_value < 0.05:
    print("Отвергаем нулевую гипотезу: средние значения различаются.")
else:
    print("Не удалось отвергнуть нулевую гипотезу: средние значения не различаются.")
