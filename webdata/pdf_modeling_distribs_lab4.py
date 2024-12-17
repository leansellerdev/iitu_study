import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, norm, lognorm, pareto, probplot

# Загрузка данных
data = pd.read_csv('ozon_dataset.csv')  # Заменить на твой датасет

data = data[(data['review_count'] >= 0) & (data['review_count'] <= 10000)]

# 1. Экспоненциальное распределение: CCDF на логарифмической шкале
def ccdf(data):
    sorted_data = np.sort(data)
    ccdf = 1.0 - np.arange(1, len(sorted_data)+1) / len(sorted_data)
    return sorted_data, ccdf

plt.figure(figsize=(10, 5))
rating_sorted, rating_ccdf = ccdf(data['rating'])
plt.plot(rating_sorted, rating_ccdf, label="Empirical CCDF")
plt.yscale('log')
plt.title("Экспоненциальное распределение: CCDF для Rating (Log Scale)")
plt.xlabel("Rating")
plt.ylabel("1 - CDF (log)")
plt.legend()
plt.show()

# 2. Нормальное распределение: Normal Probability Plot
plt.figure(figsize=(8, 6))
probplot(data['rating'], dist="norm", plot=plt)
plt.title("Normal Probability Plot для Rating")
plt.show()

# 3. Логнормальное распределение: CDF на логарифмической шкале
plt.figure(figsize=(8, 6))
log_data = np.log(data['rating'][data['rating'] > 0])  # Логарифмируем положительные значения
sorted_log, cdf_log = np.sort(log_data), np.arange(1, len(log_data)+1) / len(log_data)
plt.plot(sorted_log, cdf_log, label="Empirical Log-CDF")
plt.title("Логнормальное распределение: Log-CDF для Rating")
plt.xlabel("log(Rating)")
plt.ylabel("CDF")
plt.legend()
plt.show()

# 4. Парето распределение: CCDF на логарифмической шкале
xmin = data['rating'].min()
alpha = 3.0  # Пример параметра alpha

plt.figure(figsize=(10, 5))
rating_sorted, rating_ccdf = ccdf(data['rating'])
plt.plot(rating_sorted, rating_ccdf, label="Empirical CCDF")
plt.plot(rating_sorted, (xmin / rating_sorted)**alpha, label="Pareto Fit", linestyle='--')
plt.yscale('log')
plt.xscale('log')
plt.title("Парето распределение: CCDF на логарифмической шкале")
plt.xlabel("Rating (log scale)")
plt.ylabel("1 - CDF (log scale)")
plt.legend()
plt.show()


plt.figure(figsize=(10, 5))
review_sorted, review_ccdf = ccdf(data['review_count'])
plt.plot(review_sorted, review_ccdf, label="Empirical CCDF")
plt.yscale('log')
plt.title("Экспоненциальное распределение: CCDF для Review Count (Log Scale)")
plt.xlabel("Review Count")
plt.ylabel("1 - CDF (log)")
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
probplot(data['review_count'], dist="norm", plot=plt)
plt.title("Normal Probability Plot для Review Count")
plt.show()

plt.figure(figsize=(8, 6))
log_data = np.log(data['review_count'][data['review_count'] > 0])  # Логарифмируем положительные значения
sorted_log, cdf_log = np.sort(log_data), np.arange(1, len(log_data)+1) / len(log_data)
plt.plot(sorted_log, cdf_log, label="Empirical Log-CDF")
plt.title("Логнормальное распределение: Log-CDF для Review Count")
plt.xlabel("log(Review Count)")
plt.ylabel("CDF")
plt.legend()
plt.show()

# 4. Парето распределение: CCDF на логарифмической шкале
xmin = data['rating'].min()
alpha = 3.0  # Пример параметра alpha

plt.figure(figsize=(10, 5))
review_sorted, review_ccdf = ccdf(data['review_count'])
plt.plot(review_sorted, review_ccdf, label="Empirical CCDF")
plt.plot(review_sorted, (xmin / review_sorted)**alpha, label="Pareto Fit", linestyle='--')
plt.yscale('log')
plt.xscale('log')
plt.title("Парето распределение: CCDF на логарифмической шкале")
plt.xlabel("Review Count (log scale)")
plt.ylabel("1 - CDF (log scale)")
plt.legend()
plt.show()

