from abc import ABC, abstractmethod
from typing import List, Optional, Union
import numpy as np
import pandas as pd

class Model(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class Adaline(Model):
    """
    todo
    """
    def __init__(self):
        self.weights = np.array([])
        self.epochs = 0
        self.errors = list()

    def sign(self, u: float) -> int:
        """
        todo
        """
        if u >= 0:
            return 1
        else:
            return -1

    def predict_raw(self, x) -> float:
        """
        todo
        """
        return np.dot(self.weights, np.insert(x, 0, 1.0))

    def mean_square_error(self, x, y) -> float:
        """
        todo
        """
        p = x.shape[0]
        E_qm = 0
        for i in range(p):
            u = self.predict_raw(x[i])
            E_qm += (y[i] - u) ** 2
        E_qm = E_qm / p
        return E_qm

    def fit(
            self,
            x: np.ndarray, 
            y: np.ndarray,
            learning_rate: float = 1e-1,
            epsilon: float = 1e-3,
            num_epochs: int = 100,
            seed: Optional[int] = 0
        ) -> Union[np.ndarray, int, List[float]]:
        """
        this is the train function required by https://classroom.google.com/u/2/w/Njg4MzMyOTYxNzU0/t/all
        
        inputs:
            X: input data
            y: target data
            learning_rate: learning rate
            num_epochs: number of epochs
            seed: random seed
        
        outputs:
            weights: np.ndarray
            epochs: int
            errors: List[float]

        """
        np.random.seed(seed)
        n_samples, n_features = x.shape
        self.weights = np.random.rand(n_features + 1) * 1e-3
        self.errors = list()
        self.epoch = 0
        E_qm = self.mean_square_error(x, y)
        self.errors.append(E_qm)
        while (self.epoch  < num_epochs):
            E_qm_old = E_qm
            for i in range(n_samples):
                u = self.predict_raw(x[i])
                self.weights += learning_rate * (y[i] - u) * np.insert(x[i], 0, 1.0)
            self.epochs += 1
            E_qm = self.mean_square_error(x, y)
            self.errors.append(E_qm)
            if abs(E_qm - E_qm_old) < epsilon:
                break
        return self.weights, self.epochs, self.errors
                

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        this is the train function required by https://classroom.google.com/u/2/w/Njg4MzMyOTYxNzU0/t/all

        inputs:
            x: input data

        outputs:
            predictions: np.ndarray
            accuracy: float
        """
        if len(self.weights) == 0:
            raise ValueError("Model has not been fitted yet.")
        n_samples = x.shape[0]
        pred = np.zeros(n_samples)
        for i in range(n_samples):
            u = self.predict_raw(x[i])
            pred[i] = self.sign(u)
        return pred
    

if __name__ == "__main__":
    train = pd.read_csv("../data/atividade_01/train_dataset1.csv")
    test = pd.read_csv("../data/atividade_01/test_dataset1.csv")
    print(train.head())
    print(test.head())