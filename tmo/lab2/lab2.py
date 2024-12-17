import math


def poisson_distribution(lmbda: int, t: int, k: int) -> float:
    p = (math.exp(-lmbda * t) * (lmbda * t)**k) / math.factorial(k)

    return round(p, 4)

# Определите вероятность того, что в течение 2 минут в систему поступит ровно 8 клиентов.
print(poisson_distribution(5, 2, 8))

# Определите вероятность того, что в течение 5 минут в систему поступит не более 20 клиентов.
max_clients = 20
print(sum(poisson_distribution(5, 5, k) for k in range(max_clients + 1)))


def state_probability(lmbda: int, mu: int, states: int) -> list:
    p = [0] * (3 + 1)
    p[0] = 1

    for i in range(1, states + 1):
        p[i] = (lmbda / (i * mu)) * p[i - 1]

    p_sum = sum(p)
    P = [round((p / p_sum), 4)for p in p]

    return P

# Для системы с 3 операторами необходимо определить вероятности состояний P0, P1, P2 и P3
print(state_probability(5, 2, 3))


# Определите среднее время ожидания в очереди и вероятность того, что клиенту придется ждать
def wait_time(lmbda: int, mu: int) -> float:
    wq = lmbda / mu * (mu - lmbda)
    return wq

print(wait_time(0.6, 6))
