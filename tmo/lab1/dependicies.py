import numpy as np
import matplotlib.pyplot as plt
from math import factorial

# Функция для вычисления основных характеристик системы
def P0(c, rho):
    sum_term = sum((c * rho) ** n / factorial(n) for n in range(c))
    return 1 / (sum_term + ((c * rho) ** c / (factorial(c) * (1 - rho))))

def queue_metrics(lmbda, mu, c):
    rho = lmbda / (c * mu)
    P0_value = P0(c, rho)
    Lq = ((c * rho) ** c * P0_value) / (factorial(c) * (1 - rho) ** 2)
    Wq = Lq / lmbda
    L = Lq + (lmbda / mu)
    W = Wq + (1 / mu)
    return L, Lq, W, Wq

# Диапазоны значений для параметров
lmbda_values = np.linspace(2, 20, 50)
mu_values = np.linspace(2, 10, 50)
c_values = np.arange(1, 11)

# Списки для хранения результатов
L_lmbda, Lq_lmbda, W_lmbda, Wq_lmbda = [], [], [], []
L_mu, Lq_mu, W_mu, Wq_mu = [], [], [], []
L_c, Lq_c, W_c, Wq_c = [], [], [], []

# Вычисление метрик для изменяющихся значений лямбда (λ)
for lmbda in lmbda_values:
    L, Lq, W, Wq = queue_metrics(lmbda, 3, 4)
    L_lmbda.append(L)
    Lq_lmbda.append(Lq)
    W_lmbda.append(W)
    Wq_lmbda.append(Wq)

# Вычисление метрик для изменяющихся значений мю (μ)
for mu in mu_values:
    L, Lq, W, Wq = queue_metrics(8, mu, 4)
    L_mu.append(L)
    Lq_mu.append(Lq)
    W_mu.append(W)
    Wq_mu.append(Wq)

# Вычисление метрик для изменяющегося числа серверов (c)
for c in c_values:
    L, Lq, W, Wq = queue_metrics(8, 3, c)
    L_c.append(L)
    Lq_c.append(Lq)
    W_c.append(W)
    Wq_c.append(Wq)

# Построение графиков зависимости от лямбда (λ)
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(lmbda_values, L_lmbda, label='L (system)')
plt.plot(lmbda_values, Lq_lmbda, label='Lq (queue)')
plt.xlabel('Lambda (λ)')
plt.ylabel('L, Lq')
plt.title('L и Lq от Lambda')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(lmbda_values, W_lmbda, label='W (system)')
plt.plot(lmbda_values, Wq_lmbda, label='Wq (queue)')
plt.xlabel('Lambda (λ)')
plt.ylabel('W, Wq')
plt.title('W и Wq от Lambda')
plt.legend()

# Построение графиков зависимости от мю (μ)
plt.subplot(2, 2, 3)
plt.plot(mu_values, L_mu, label='L (system)')
plt.plot(mu_values, Lq_mu, label='Lq (queue)')
plt.xlabel('Mu (μ)')
plt.ylabel('L, Lq')
plt.title('L и Lq от Mu')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(mu_values, W_mu, label='W (system)')
plt.plot(mu_values, Wq_mu, label='Wq (queue)')
plt.xlabel('Mu (μ)')
plt.ylabel('W, Wq')
plt.title('W и Wq от Mu')
plt.legend()

plt.tight_layout()
plt.show()

# Построение графиков зависимости от числа серверов (c)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(c_values, L_c, label='L (system)')
plt.plot(c_values, Lq_c, label='Lq (queue)')
plt.xlabel('Количество серверов (c)')
plt.ylabel('L, Lq')
plt.title('L и Lq от Количества серверов')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(c_values, W_c, label='W (system)')
plt.plot(c_values, Wq_c, label='Wq (queue)')
plt.xlabel('Количество серверов (c)')
plt.ylabel('W, Wq')
plt.title('W и Wq от Количества серверов')
plt.legend()

plt.tight_layout()
plt.show()
