import numpy as np
import pytest
from sumNeural import SumNeural  # Asumiendo que tu clase SumNeural está en sum_neural.py

def test_sum_prediction():
    model = SumNeural()
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    Y = np.array([3, 7, 11, 15])
    model.train(X, Y, lr=0.01, n_epochs=100)

    test_input = np.array([9, 10])
    real_sum = 19  # Valor real
    predicted_sum = model.forward(test_input)

    # Afirmar que el error absoluto es menor que un umbral, por ejemplo, 1
    assert np.abs(real_sum - predicted_sum) < 1, "La predicción del modelo está fuera del margen de error aceptable."

# Puedes agregar más pruebas aquí...
