from collections import namedtuple
import random

Experiences = namedtuple('Experiences', ['obs', 'action', 'reward', 'new_state', 'done'])   

class ExpirenceMemory(object):
    """
    Un buffer que simula la memoria de un agente
    """
    def __init__(self, capacity = int(1e6)):
        """
        @param capacity: int: Capacidad de la memoria
        @praam memory: list: Lista de experiencias
        @param push_count: int: Número de experiencias almacenadas
        """
        self.capacity = capacity
        self.memory = []
        self.pushCount = 0

    def Store(self, experience):
        """
        @param experience: Experience: Experiencia a almacenar
        """
        self.memory.insert(self.pushCount % self.capacity, experience)
        self.pushCount += 1

    def sample(self, batch_size):
        """
        @param batch_size: int: Tamaño de la muestra
        @return list: Muestra de experiencias
        """
        assert batch_size <= self.getSize()
        return random.sample(self.memory, batch_size)

    def getSize(self):
        """
        @param batch_size: int: Tamaño de la muestra
        @return bool: True si el tamaño de la muestra es menor o igual al tamaño de la memoria
        """
        return len(self.memory)