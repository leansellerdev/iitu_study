# In[1]:


import pandas
data = pandas.read_csv(filepath_or_buffer=r'/titanic.csv')

# In[2]:
# 1. Сколько мужчин было на корабле?
num_men = data[data['Sex'] == 'male'].shape[0]

# 2. Какая доля пассажиров (в %) выжила?
survived_percent = (data['Survived'].mean() * 100)

# 3. Какая доля пассажиров (в %) путешествовала во 2-ом классе?
second_class_percent = (data[data['Pclass'] == 2].shape[0] / data.shape[0] * 100)

# 4. Среднее и медиана возраста всех людей на корабле
mean_age = data['Age'].mean()
median_age = data['Age'].median()

# 5. Корреляция по Пирсону между признаками SibSp и Parch
correlation = data['SibSp'].corr(data['Parch'])

# 6. Самое популярное женское имя на корабле
women = data[data['Sex'] == 'female']

# Функция для извлечения имени
def extract_first_name(full_name):
    first_name_search = re.search(r"\((.*?)\)", full_name)
    if first_name_search:
        return first_name_search.group(1).split()[0]
    else:
        return full_name.split(",")[1].split()[1]

# Применение функции к колонке Name
women['FirstName'] = women['Name'].apply(extract_first_name)

# Подсчет наиболее популярного имени
popular_name = women['FirstName'].value_counts().idxmax()

# Вывод результатов
print(f"1. Количество мужчин на корабле: {num_men}")
print(f"2. Доля выживших пассажиров (в %): {survived_percent:.2f}%")
print(f"3. Доля пассажиров во 2-ом классе (в %): {second_class_percent:.2f}%")
print(f"4. Средний возраст пассажиров: {mean_age:.2f} лет, медиана возраста: {median_age:.2f} лет")
print(f"5. Корреляция по Пирсону между SibSp и Parch: {correlation:.2f}")
print(f"6. Самое популярное женское имя на корабле: {popular_name}")


correlation = data['SibSp'].corr(data['Parch'])
print("Корреляция по Пирсону между SibSp и Parch:", correlation)
