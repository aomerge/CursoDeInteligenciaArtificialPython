import numpy as np
import matplotlib.pyplot as plt

class SumNeural:
    """
    A class representing a simple neural network for summing two inputs.

    Attributes:
        w (numpy.ndarray): Weights for each input.
        b (float): Bias term.

    Methods:
        forward(x): Performs the forward pass of the neural network.
        loss(y, y_pred): Calculates the loss between the predicted and target values.
        gradient(x, y, y_pred): Calculates the gradients of the weights and bias.
        train(X, Y, lr=0.01, n_epochs=100): Trains the neural network on the given data.

    """

    def __init__(self):
        self.w = np.random.rand(2)  # weights for each input
        self.b = np.random.rand()  # bias

    def forward(self, x):
        """
        Performs the forward pass of the neural network.

        Args:
            x (numpy.ndarray): Input values.

        Returns:
            float: The output of the neural network.

        """
        return np.dot(self.w, x) + self.b

    def loss(self, y, y_pred):
        """
        Calculates the loss between the predicted and target values.

        Args:
            y (float): Target value.
            y_pred (float): Predicted value.

        Returns:
            float: The loss value.

        """
        return ((y_pred - y) ** 2).mean()

    def gradient(self, x, y, y_pred):
        """
        Calculates the gradients of the weights and bias.

        Args:
            x (numpy.ndarray): Input values.
            y (float): Target value.
            y_pred (float): Predicted value.

        Returns:
            tuple: Gradients of the weights and bias.

        """
        return np.dot(2 * (y_pred - y), x), 2 * (y_pred - y)

    def train(self, X, Y, lr=0.01, n_epochs=100):
        """
        Trains the neural network on the given data.

        Args:
            X (numpy.ndarray): Input data.
            Y (numpy.ndarray): Target data.
            lr (float, optional): Learning rate. Defaults to 0.01.
            n_epochs (int, optional): Number of epochs. Defaults to 100.

        Returns:
            list: List of loss values during training.

        """
        losses = []
        for epoch in range(n_epochs):
            for x, y in zip(X, Y):
                y_pred = self.forward(x)
                l = self.loss(y, y_pred)
                w_grad, b_grad = self.gradient(x, y, y_pred)
                self.w -= lr * w_grad
                self.b -= lr * b_grad
            losses.append(self.loss(Y, [self.forward(x) for x in X]))
        return losses
    
    def save(self, path):
        np.savez(path, w=self.w, b=self.b)

    def load(self, path):
        data = np.load(path)
        self.w = data['w']
        self.b = data['b']

if __name__ == "__main__":
    # Inicializar la red neuronal
    model = SumNeural()

    # Datos de entrenamiento
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    Y = np.array([3, 7, 11, 15])

    # Entrenar el modelo
    losses = model.train(X, Y, lr=0.01, n_epochs=100)

    # Guardar el modelo
    model.save('model.npz')

    # Graficar la pérdida a lo largo de las iteraciones de entrenamiento
    plt.plot(losses)
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.show()    

