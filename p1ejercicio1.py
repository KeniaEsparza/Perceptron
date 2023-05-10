import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Lectura de los patrones de entrenamiento desde un archivo CSV
train_data = pd.read_csv("XOR_trn.csv", header=None)
train_X = train_data.iloc[:, :-1].values  # entradas
train_y = train_data.iloc[:, -1].values   # salidas

# Lectura de los patrones de prueba desde un archivo CSV
test_data = pd.read_csv("XOR_tst.csv", header=None)
test_X = test_data.iloc[:, :-1].values  # entradas
test_y = test_data.iloc[:, -1].values   # salidas
#print(test_X,test_y)

# Definición de la función de activación
def step_function(z):
    return np.where(z >= 0, 1, -1)

# Definición de la función de entrenamiento del perceptrón
def train_perceptron(X, y, learning_rate, epochs):
    # Inicialización de los pesos
    np.random.seed(0)
    weights = np.random.normal(loc=0.0, scale=0.01, size=X.shape[1]+1)
    
    # Adición de un sesgo a las entradas
    X = np.insert(X, 0, 1, axis=1)
    
    # Entrenamiento del perceptrón
    for epoch in range(epochs):
        errors = 0
        for xi, target in zip(X, y):
            output = step_function(np.dot(xi, weights))
            error = target - output
            weights += learning_rate * error * xi
            errors += int(error != 0.0)
        if errors == 0:
            break
    
    return weights

# Entrenamiento del perceptrón
learning_rate = 0.1
epochs = 50 #100 #50 es la buena, menos de 75 y mas de 50 dan buenos resultados
weights = train_perceptron(train_X, train_y, learning_rate, epochs)

# Prueba del perceptrón en los patrones de prueba
test_X = np.insert(test_X, 0, 1, axis=1)
predictions = step_function(np.dot(test_X, weights))
accuracy = np.mean(predictions == test_y)
print(f"Accuracy: {accuracy:.2f}")

# Visualización de los patrones de entrenamiento y la recta que los separa
# Graficar los patrones de entrenamiento y la recta que los separa
x1 = np.linspace(-2, 2, 10) ##checae esto
x2 = -(weights[0] + weights[1]*x1) / weights[2] #2
plt.plot(x1, x2, c='r', label='Separating line')
plt.scatter(train_X[:,0], train_X[:,1], c=train_y, cmap='coolwarm', label='Training data')
#plt.xlim(-2, 2)
#plt.ylim(-2, 2)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('XOR Problem')
plt.legend()
plt.show()

# Generar patrones de entrenamiento para el problema XOR
X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
y = np.array([1, -1, -1, 1])
noise = np.random.uniform(low=-0.05, high=0.05, size=X.shape)
X += noise
data = np.hstack((X, y[:, np.newaxis]))
#np.savetxt("XOR_tst.csv", data, delimiter=",")

