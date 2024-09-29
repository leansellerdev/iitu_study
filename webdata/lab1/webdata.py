import pandas as pd

# Чтение файла с данными
df = pd.read_csv('../../BTC-Hourly.csv')


# Просмотр первых строк
# print(df.head())
#
# # Информация о типах данных и размере датасета
# print(df.info())
#
# # Основная статистика по числовым столбцам
# print(df.describe())
#
# # Преобразование unix в дату
df['date'] = pd.to_datetime(df['unix'], unit='s')
#
# # Просмотр преобразованного столбца
df[['unix', 'date']].head()
#
# # Проверка наличия пропущенных значений
df.isnull().sum()
#
# # Удаление строк с пропущенными значениями
df.dropna(inplace=True)
#
# # Фильтрация данных для биткоина
df_btc = df[df['symbol'] == 'BTC/USD']
#
# # Просмотр первых строк
# print(df_btc.head())
#
# Минимальная и максимальная цена
min_price = df_btc['low'].min()
max_price = df_btc['high'].max()

print(f'Minimum Price: {min_price}')
print(f'Maximum Price: {max_price}')

# Средняя цена открытия и закрытия
mean_open = df_btc['open'].mean()
mean_close = df_btc['close'].mean()

print(f'Mean Open Price: {mean_open}')
print(f'Mean Close Price: {mean_close}')
#
# Поиск даты с максимальным объемом
max_volume_btc = df_btc['Volume BTC'].idxmax()
max_volume_usd = df_btc['Volume USD'].idxmax()

print(f'Date with max BTC volume: {df_btc.loc[max_volume_btc, "date"]}')
print(f'Date with max USD volume: {df_btc.loc[max_volume_usd, "date"]}')

# # Добавление столбца с месяцем
df_btc['month'] = df_btc['date'].dt.to_period('M')

# Группировка по месяцам
monthly_stats = df_btc.groupby('month').agg({
    'open': 'mean',
    'close': 'mean',
    'Volume BTC': 'sum',
    'Volume USD': 'sum'
})

print(monthly_stats)

# Рассчет волатильности
df_btc['volatility'] = df_btc['high'] - df_btc['low']

# Самые волатильные дни
most_volatile_days = df_btc[['date', 'volatility']].sort_values(by='volatility', ascending=False).head()

print(most_volatile_days)

