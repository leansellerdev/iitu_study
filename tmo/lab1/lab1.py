from math import factorial

# Given data
lmbda = 5   # интенсивность поступления заявок (lambda)
mu = 2      # интенсивность обслуживания одного сервера (mu)
c = 3       # количество серверов (c)

# 1. Коэффициент загрузки сервера (ρ)
def calc_rho(lmbda, mu, c):
    rho = lmbda / (c * mu)
    return rho

rho = calc_rho(lmbda, mu, c)

# 2. Вероятность того, что все серверы заняты(P0)
def calc_p0(c, rho):
    sum_part = sum([(c * rho) ** n / factorial(n) for n in range(c)])
    second_part = (c * rho) ** c / (factorial(c) * (1 - rho))
    return 1 / (sum_part + second_part)

p0 = calc_p0(c, rho)

# 3. Среднее количество заявок в очереди (Lq)
def calc_lq(rho, p0, c):
    lq = ((c * rho) ** c * p0) / (factorial(c) * (1 - rho) ** 2)
    return lq

lq = calc_lq(rho, p0, c)

# 4. Среднее время ожидания в очереди (Wq)
def calc_wq(lq, lmbda):
    wq = lq / lmbda
    return wq

wq = calc_wq(lq, lmbda)

# 5. Среднее количество заявок в системе (L)
def calc_l(lq, lmbda, mu):
    l = lq + (lmbda / mu)
    return l

l = calc_l(lq, lmbda, mu)

# 6. Среднее время пребывания заявки в системе (W)
def calc_w(wq, mu):
    w = wq + (1 / mu)
    return w

w = calc_w(wq, mu)

print(rho, p0, lq, wq, l, w, sep='\n')



"""
1 задание:
    1. Коэффициент загрузки сервера (p): 0.833
    2. Вероятность того, что все серверы заняты (p0): 0.045
    3. Среднее количество заявок в очереди (Lq): 4.213
    4. Среднее время ожидания в очереди (Wq): 0.843 минут
    5. Среднее количество заявок в системе (L): 6.713
    6. Среднее время пребывания заявки в системе (W): 1.343 минут
    
2 задание:
    1. Коэффициент загрузки сервера (p): 0.667
    2. Вероятность того, что все серверы заняты (p0): 0.060
    3. Среднее количество заявок в очереди (Lq): 1.135
    4. Среднее время ожидания в очереди (Wq): 0.142 минут
    5. Среднее количество заявок в системе (L): 3.802
    6. Среднее время пребывания заявки в системе (W): 0.475 минут
    
    Вопрос: Как изменятся показатели системы, если увеличить кол-во серверов до 5?
    Решение: 
        1. Коэффициент загрузки сервера (p): 0.533
        2. Вероятность того, что все серверы заняты (p0): 0.067
        3. Среднее количество заявок в очереди (Lq): 0.346
        4. Среднее время ожидания в очереди (Wq): 0.043 минут
        5. Среднее количество заявок в системе (L): 3.013
        6. Среднее время пребывания заявки в системе (W): 0.377 минут
    
    Ответ: Увеличение числа серверов снижает коэффициент загрузки и время ожидания, 
           но вероятность занятости всех серверов немного растет.
           
    Вопрос: Как изменятся показатели системы, если увеличить кол-во серверов до 5?
    Решение: 
        1. Коэффициент загрузки сервера (p): 0.5
        2. Вероятность того, что все серверы заняты (p0): 0.130
        3. Среднее количество заявок в очереди (Lq): 0.35
        4. Среднее время ожидания в очереди (Wq): 0.043 минут
        5. Среднее количество заявок в системе (L): 2.348
        6. Среднее время пребывания заявки в системе (W): 0.293 минут
    
    Ответ: Увеличение скорости обслуживания (интенсивности) значительно улучшает 
           время обработки заявок и снижает количество заявок в системе.
"""
