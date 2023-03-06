from BaseModel import BaseModel
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

class DecisionTree(BaseModel):
    def grid_search(self, X, y):
        param_grid={
        'max_depth': np.arange(1, 50, 1)
        }
        grid_mdecison=GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5)
        grid_mdecison.fit(X, y)
        return grid_mdecison.best_params_ , grid_mdecison.best_score_
    
    def train_model(self, X, y, config):
        max_depth=config['max_depth']
        input_pipe = [('Scale', StandardScaler()),
            ('model', DecisionTreeRegressor(max_depth=max_depth))]
        pipe = Pipeline(input_pipe)
        pipe.fit(X, y)
        self.model=pipe