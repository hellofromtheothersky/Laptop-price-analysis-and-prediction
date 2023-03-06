from abc import ABC, abstractmethod
from sklearn.metrics import r2_score

class BaseModel(ABC):
    def predict(self, X):
        return self.model.predict([X_i for X_i in X])


    def save_model(self, model_name):
        pass
    
    @abstractmethod
    def grid_search(self, X, y):
        pass


    @abstractmethod
    def train_model(self, X, y):
        pass

    def evaluate(self, X, y_true):
        y_pred=self.model.predict(X)
        return r2_score(y_true, y_pred)