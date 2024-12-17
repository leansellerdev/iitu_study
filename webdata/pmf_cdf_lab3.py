import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('ozon_dataset.csv')  # Заменить на твой датасет

data = data[(data['review_count'] >= 0) & (data['review_count'] <= 10000)]


# PMF для rating
def compute_pmf(series):
    pmf = series.value_counts(normalize=True).sort_index()
    return pmf

rating_pmf = compute_pmf(data['rating'])
review_pmf = compute_pmf(data['review_count'])

# Построение PMF
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(rating_pmf.index, rating_pmf.values)
plt.title("PMF для Rating")
plt.xlabel("Rating")
plt.ylabel("Probability")

plt.subplot(1, 2, 2)
plt.bar(review_pmf.index, review_pmf.values)
plt.title("PMF для Review Count")
plt.xlabel("Review Count")
plt.ylabel("Probability")
plt.show()

# CDF для rating и review_count
def compute_cdf(series):
    sorted_data = np.sort(series)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf

rating_sorted, rating_cdf = compute_cdf(data['rating'])
review_sorted, review_cdf = compute_cdf(data['review_count'])

# Построение CDF
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(rating_sorted, rating_cdf)
plt.title("CDF для Rating")
plt.xlabel("Rating")
plt.ylabel("CDF")

plt.subplot(1, 2, 2)
plt.plot(review_sorted, review_cdf)
plt.title("CDF для Review Count")
plt.xlabel("Review Count")
plt.ylabel("CDF")
plt.show()

# Percentile-based statistics
rating_50th = np.percentile(data['rating'], 50)
review_iqr = np.percentile(data['review_count'], 75) - np.percentile(data['review_count'], 25)

print(f"Медиана для Rating: {rating_50th}")
print(f"IQR для Review Count: {review_iqr}")
