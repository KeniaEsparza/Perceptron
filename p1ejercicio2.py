import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Funcion para iniciar el entrenamiento del perceptron
def runPerceptron(training_data, generalization_data):
    #Entrenamiento de datos
    train_x = training_data.iloc[:, :-1].values #Inputs
    train_y = training_data.iloc[:, -1].values #Outputs

    #Test de datos
    test_x = generalization_data.iloc[:, :-1].values
    test_y = generalization_data.iloc[:, -1].values

    #Definicion de funcion de activacion
    def activation(z):
        return np.where(z >= 0, 1, -1)

    #TDefinicion de funcion de entrenamiento
    def trainingPerceptron(x, y, learning_rate, epochs):
        #Weight inicialization
        np.random.seed(0)
        weights = np.random.normal(0.0, 0.01, size=x.shape[1]+1)

        #Adicion de un sesgo a las entradas
        x = np.insert(x, 0, 1, axis=1)

        #Entrenamiento del perceptron
        for epoch in range(epochs):
            errors = 0
            for xi, target in zip(x,y):
                output = activation(np.dot(xi, weights))
                error = target - output
                weights += learning_rate * error * xi
                error += int(error != 0.0)
            if errors == 0:
                break

        return weights

    #Entrenamiento del perceptron
    learning_rate = 0.1 #Learning rate
    epochs = 100 #Epocas
    weights = trainingPerceptron(train_x,train_y,learning_rate, epochs)

    # Prueba del perceptrón en los patrones de prueba
    test_x = np.insert(test_x, 0, 1, axis=1)
    predictions = activation(np.dot(test_x, weights))
    accuracy = np.mean(predictions == test_y)
    print(f"Accuracy: {accuracy:.2f}")

    # Visualización de los patrones de entrenamiento y la recta que los separa
    # Graficar los patrones de entrenamiento y la recta que los separa
    x1 = np.linspace(-2, 2, 10)
    x2 = -(weights[0] + weights[1]*x1) / weights[2]
    plt.plot(x1, x2, c='r', label='Separating line')
    plt.scatter(train_x[:,0], train_x[:,1], c=train_y, cmap='coolwarm', label='Training data')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Ejercicio 2')
    plt.legend()
    plt.show()


#Elige con que patron trabajar
get_data = pd.read_csv("spheres1d10.csv", header=None)

print("Elige el archivo para entrenamiento: ")
print("1) Original.")
print("2) Datos pertubados en 10%.")
print("3) Datos pertubados en 50%.")
print("4) Datos pertubados en 70%.")
archivoElegido = input()

match archivoElegido:
    case "1":
        #Patron de entrenamiento original con perturbaciones 10%
        get_data = pd.read_csv("spheres1d10.csv", header=None) 
    case "2":
        #Patron de entrenamiento modificado con perturbaciones 10%
        get_data = pd.read_csv("spheres2d10.csv", header=None)
    case "3":
        #Patron de entrenamiento modificado con perturbaciones 50%
        get_data = pd.read_csv("spheres2d50.csv", header=None)
    case "4":
        #Patron de entrenamiento modificado con perturbaciones 70%
        get_data = pd.read_csv("spheres2d70.csv", header=None)
    case _:
        print("Esa opcion no existe")


#Escoge una particion
print("Escoge que tipo de particionado desea usar (1 a 10): ")
eleccionParticion = input()

match eleccionParticion:
    case "1":
        #Particion 1
        training_data = get_data.iloc[:799,:]
        generalization_data = get_data.iloc[800:,:]
    case "2":
        #Particion 2
        training_data = get_data.iloc[:599,:]
        generalization_data = get_data.iloc[600:799,:]
    case "3":
        #Particion 3
        training_data = get_data.iloc[:400,:]
        generalization_data = get_data.iloc[401:599,:]
    case "4":
        #Particion 4
        training_data = get_data.iloc[:-799,:]
        generalization_data = get_data.iloc[:200,:]
    case "5":
        #Particion 5
        training_data = get_data.iloc[:400,:]
        generalization_data = get_data.iloc[401:599,:]
    case "6":
        #Particion 6
        training_data = get_data.iloc[:800,:]
        generalization_data = get_data.iloc[801:,:]
    case "7":
        #Particion 7
        training_data = get_data.iloc[:500,:]  
        generalization_data = get_data.iloc[501:699,:]
    case "8":
        #Particion 8
        training_data = get_data.iloc[:800,:]
        generalization_data = get_data.iloc[200:,:]
    case "9":
        #Particion 9
        training_data = get_data.iloc[201:,:]
        generalization_data = get_data.iloc[:200,:]
    case "10":
        #Particion 10
        training_data = get_data.iloc[:700,:]
        generalization_data = get_data.iloc[800:,:]
    case _:
        print("Esa opcion no existe")

runPerceptron(training_data, generalization_data)

