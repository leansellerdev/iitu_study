import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# 1. Загрузка данных
url = '../../titanic.csv'
data = pd.read_csv(url)

# 2. Отбор признаков
data = data[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']]

# 3. Преобразование строковых значений в числовые
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# 4. Удаление пропущенных значений
data.dropna(inplace=True)

# 5. Разделение на признаки и метки
X = data[['Pclass', 'Fare', 'Age', 'Sex']]
y = data['Survived']

# 6. Построение модели DecisionTreeClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
clf = DecisionTreeClassifier(random_state=100)
clf.fit(X_train, y_train)

# 7. Оценка важности признаков
y_pred = clf.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Наиболее важные признаки
importances = clf.feature_importances_
feature_names = X.columns
important_features = sorted(zip(importances, feature_names), reverse=True)
print(f'Важные признаки: {important_features[:2]}')

# Accuracy: 0.7534883720930232
# Важные признаки: [(np.float64(0.3074654119761512), 'Sex'), (np.float64(0.28559837913404784), 'Fare')]
