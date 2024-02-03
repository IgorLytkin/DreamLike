import numpy as np
import pyarrow as pa
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


vector_1 = np.array([1, 2, 3])
print(vector_1)

# Перемножение матриц
matrix_1 = np. array([[1, 2, 3], [3, 4, 5], [6, 7, 8]])
matrix_2 = np. array([[16, 22, 63], [31, 54, 51], [69, 70, 87]])
print(np. dot(matrix_1, matrix_2))

print(set(["1", "ml", "ml", "1", 2, 3, "ml"]))

# Создание массива
data = np.array([1, 2, 3, 4, 5])
print(data)
# Выполнение математических операций
mean = np.mean(data)
print(mean)
std_dev = np.std(data)
print(std_dev)

# Изменение формы массива
reshaped_data = data.reshape(5, 1)
print(reshaped_data)


# Создание DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'Salary': [50000, 60000, 75000]}
df = pd.DataFrame(data)
# Вывод первых нескольких строк
df.head()

filtered_data = df[df['Age'] > 30]
print(filtered_data)

# Выполнение операций с данными
average_salary = df['Salary'].mean()
print(average_salary)

# Создание графика
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt. plot(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Sine Wave')
plt.show()

# Загрузка данных
iris = load_iris()
iris

X, y = iris.data, iris. target
# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализация и обучение модели
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
# Предсказание на тестовом наборе и оценка точности
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

data = np.array([12, 34, 23, 55, 355])
result = np.mean(data)
print(result)