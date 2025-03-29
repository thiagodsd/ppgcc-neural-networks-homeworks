from abc import ABC, abstractmethod
from typing import List, Optional, Union
import numpy as np


class Model(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class Perceptron(Model):
    """
    perceptron according to https://classroom.google.com/u/2/w/Njg4MzMyOTYxNzU0/t/all, p.20
    """
    def __init__(
            self,
            dim_input: int,
            dim_output: int,
        ):
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.bias = 1e0
        self.learning_rate = 1e-1
        self._is_fitted = False