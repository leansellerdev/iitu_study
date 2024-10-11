import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
data = pd.read_csv("../../BTC-Hourly.csv")
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Создание признаков
data['price_change'] = data['close'].diff()
data['target_class'] = (data['price_change'] > 0).astype(int)

# Технические индикаторы
data['SMA_10'] = data['close'].rolling(window=10).mean()
data['SMA_50'] = data['close'].rolling(window=50).mean()

def compute_rsi(data, window=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    return RSI

data['RSI_14'] = compute_rsi(data)
data['EMA_12'] = data['close'].ewm(span=12, adjust=False).mean()
data['EMA_26'] = data['close'].ewm(span=26, adjust=False).mean()
data['MACD'] = data['EMA_12'] - data['EMA_26']
data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

# Удаление NaN значений
data.dropna(inplace=True)

# Выбор признаков и целевой переменной
features = ['open', 'high', 'low', 'close', 'Volume BTC', 'Volume USD',
            'SMA_10', 'SMA_50', 'RSI_14', 'MACD', 'Signal_Line']
X = data[features]
y = data['target_class']

# Разделение данных
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Обучение модели
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Настройка гиперпараметров
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(estimator=clf, param_grid=param_grid,
                           cv=5, n_jobs=-1, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

print("Лучшие параметры:", grid_search.best_params_)
print("Лучшая точность:", grid_search.best_score_)

best_clf = grid_search.best_estimator_

# Оценка модели
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = best_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.2f}")

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Матрица ошибок')
plt.xlabel('Предсказанные классы')
plt.ylabel('Истинные классы')
plt.show()

print(classification_report(y_test, y_pred))

# Визуализация дерева решений
from sklearn import tree

plt.figure(figsize=(20,10))
tree.plot_tree(best_clf, feature_names=features, class_names=['Падение', 'Рост'], filled=True, fontsize=12)
plt.title('Дерево решений')
plt.show()

# Визуализация важности признаков
importances = best_clf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices], y=np.array(features)[indices])
plt.title('Важность признаков')
plt.show()
