import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_csv('ozon_dataset.csv')

# Шаг 1: Предварительная обработка
df.dropna(inplace=True)  # Удаление пропущенных значений
df['rating'] = df['rating'].astype(float)
df['review_count'] = df['review_count'].astype(int)
df['article'] = df['article'].astype(str)

# df = df[(df['review_count'] >= 0) & (df['review_count'] <= 10000)]


# Шаг 2: Первичный анализ
print(df.head())
print(df.info())
print(df.describe())

# Шаг 3: Анализ данных
# 1. Средние значения
print("Средний рейтинг:", df['rating'].mean())
print("Медианное количество отзывов:", df['review_count'].median())

# 2. Минимальные и максимальные значения
print("Максимальный рейтинг:", df.loc[df['rating'].idxmax()])
print("Максимальное количество отзывов:", df.loc[df['review_count'].idxmax()])

# 3. Корреляция
correlation = df['rating'].corr(df['review_count'])
print("Корреляция между рейтингом и количеством отзывов:", correlation)

# Шаг 4: Визуализация
# Гистограммы
df['rating'].hist(bins=20, alpha=0.7, label='Rating')
plt.title("Распределение рейтингов")
plt.xlabel("Рейтинг")
plt.ylabel("Частота")
plt.legend()
plt.show()

df['review_count'].hist(bins=20, alpha=0.7, label='Review Count')
plt.title("Распределение количества отзывов")
plt.xlabel("Количество отзывов")
plt.ylabel("Частота")
plt.legend()
plt.show()

# Диаграмма рассеяния
plt.scatter(df['review_count'], df['rating'], alpha=0.5)
plt.title("Зависимость рейтинга от количества отзывов")
plt.xlabel("Количество отзывов")
plt.ylabel("Рейтинг")
plt.show()
