from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import time

# Загрузка выборки
digits = load_digits()

scaled_data = scale(digits.data)

print(digits.data.shape)
print(scaled_data.shape)

n_samples, n_features = scaled_data.shape

unique_targets = len(set(digits.target))


print(f"Размерность данных: {digits.data.shape}")
print(f"Количество объектов: {n_samples}")
print(f"Количество признаков: {n_features}")
print(f"Количество уникальных значений в целевой переменной: {unique_targets}")

kmeans = KMeans(init='k-means++', n_clusters=unique_targets, n_init=10)

start_time = time.time()

kmeans.fit(scaled_data)

end_time = time.time() - start_time

print('Центры кластеров: \n', kmeans.cluster_centers_)

kmeans.fit(scaled_data)

predicted_labels = kmeans.labels_
true_labels = digits.target

ari = adjusted_rand_score(true_labels, predicted_labels)
ami = adjusted_mutual_info_score(true_labels, predicted_labels)


print(f"ARI (Adjusted Rand Index): {ari:.4f}")
print(f"AMI (Adjusted Mutual Information): {ami:.4f}")

kmeans_random = KMeans(init='random', n_clusters=unique_targets, n_init=10)

start_time_random = time.time()

kmeans_random.fit(scaled_data)

end_time_random = time.time() - start_time_random

predicted_labels_random = kmeans_random.labels_

ari_random = adjusted_rand_score(true_labels, predicted_labels_random)
ami_random = adjusted_mutual_info_score(true_labels, predicted_labels_random)

print(f"ARI (Adjusted Rand Index) с init='random': {ari_random:.4f}")
print(f"AMI (Adjusted Mutual Information) с init='random': {ami_random:.4f}")

pca = PCA(n_components=unique_targets)
data_pca = pca.fit_transform(scaled_data)

print(f"Размерность данных после применения PCA: {data_pca.shape}")

kmeans_pca_init = KMeans(init=pca.components_, n_clusters=unique_targets, n_init=1)

start_time_pca_init = time.time()
kmeans_pca_init.fit(scaled_data)
end_time_pca_init = time.time() - start_time_pca_init

predicted_labels_pca_init = kmeans_pca_init.labels_

ari_pca_init = adjusted_rand_score(true_labels, predicted_labels_pca_init)
ami_pca_init = adjusted_mutual_info_score(true_labels, predicted_labels_pca_init)


# Вывод результатов
print(f"ARI (Adjusted Rand Index) с PCA init: {ari_pca_init:.4f}")
print(f"AMI (Adjusted Mutual Information) с PCA init: {ami_pca_init:.4f}")
print(f"Время работы алгоритма с PCA init: {end_time_pca_init:.4f} секунд")

# Вывод всех метрик и времени для каждого подхода

print("\n--- Сравнение KMeans моделей ---")
print(f"Метрики для init='k-means++':\nARI: {ari:.4f}, AMI: {ami:.4f}, Время: {end_time:.4f} секунд")
print(f"Метрики для init='random':\nARI: {ari_random:.4f}, AMI: {ami_random:.4f}, Время: {end_time_random:.4f} секунд")
print(f"Метрики для init=PCA.components_:\nARI: {ari_pca_init:.4f}, AMI: {ami_pca_init:.4f}, Время: {end_time_pca_init:.4f} секунд")


# Применим PCA для уменьшения данных до 2 компонент для визуализации
pca_2d = PCA(n_components=2)
data_2d = pca_2d.fit_transform(scaled_data)

# Получим центры кластеров
centers_2d = pca_2d.transform(kmeans_pca_init.cluster_centers_)

# Визуализация кластеров и центров
plt.figure(figsize=(10, 7))

# Отображение данных, предсказанных моделью KMeans с PCA init
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=predicted_labels_pca_init, s=50, cmap='viridis')

# Отображение центров кластеров
plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Центры кластеров')

# Настройки графика
plt.title('Визуализация кластеров и центров (KMeans + PCA)')
plt.xlabel('Первая главная компонента')
plt.ylabel('Вторая главная компонента')
plt.legend()
plt.show()
