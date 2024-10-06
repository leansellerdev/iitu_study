import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('../../BTC-Hourly.csv')

# date to datetime
data['date'] = pd.to_datetime(data['date'])

# ГРАФИК РАСПРЕДЕЛЕНИЯ ЦЕН ЗАКРЫТИЯ
plt.figure(figsize=(10, 6))
sns.histplot(data=data['close'], bins=50, kde=True)
# bins = кол-во корзин(столбцов) на гистограмме
# kde = оценка плотности ядра (линия поверх поверх гистограммы)

plt.title('Распределение цены закрытия')
plt.show()

# ГРАФИК РАСПРЕДЕЛНИЯ ОБЪЕМА ТОРГОВ В BTC
plt.figure(figsize=(10, 6))
sns.histplot(data['Volume BTC'], bins=50, kde=True)
plt.title('Распределение объема торгов в BTC')
plt.show()

# Вычисляем корреляционную матрицу
corr_matrix = data[['open', 'high', 'low', 'close', 'Volume BTC', 'Volume USD']].corr()

# Визуализация корреляций
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Корреляционная матрица')
plt.show()

data['price_change_pct'] = (data['close'] - data['open']) / data['open'] * 100

# График распределения процентного изменения цены закрытия
plt.figure(figsize=(10, 6))
sns.histplot(data['price_change_pct'], bins=50, kde=True)
plt.title('Распределение процентного изменения цены закрытия')
plt.show()

# 50-дневное скользящее среднее
data['SMA_50'] = data['close'].rolling(window=50).mean()

# График скользящего среднего и цены закрытия
plt.figure(figsize=(10, 6))
plt.plot(data['date'], data['close'], label='Цена закрытия')
plt.plot(data['date'], data['SMA_50'], label='50-дневное скользящее среднее', color='red')
plt.title('Цена закрытия и 50-дневное скользящее среднее')
plt.legend()
plt.xticks(rotation=45)
plt.show()


