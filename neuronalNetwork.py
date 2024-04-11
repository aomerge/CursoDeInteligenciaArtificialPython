import numpy as np

# Función de activación ReLU
def relu(x):
    return np.maximum(0, x)

# Datos de entrada
X = np.array([1, 2, 3, 4])

# Dimensiones para la simplicidad
# Capa oculta: 4 entradas -> 64 salidas
# Capa de salida: 64 entradas -> 2 salidas

# Inicialización de pesos y biases
# Usaremos valores aleatorios pequeños para los pesos y ceros para los biases
W1 = np.random.rand(64, 4) * 0.01  # Pesos de la primera capa (64x4)
b1 = np.zeros((64,))               # Bias de la primera capa (64,)

W2 = np.random.rand(2, 64) * 0.01  # Pesos de la segunda capa (2x64)
b2 = np.zeros((2,))                # Bias de la segunda capa (2,)

# Forward pass
# Capa oculta
H = np.dot(W1, X) + b1  # H tiene forma (64,)
H = relu(H)  # Aplicación de ReLU

# Capa de salida
O = np.dot(W2, H) + b2  # O tiene forma (2,)

print("Salida de la red:", O)
