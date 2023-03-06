from BaseModel import BaseModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class RandomForest(BaseModel):
    def grid_search(self, X, y):
        rfreg1=RandomForestRegressor(n_jobs=-1, bootstrap=True, random_state=0)
        param_grid={
            'max_samples':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  
            'max_features':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        }
        grid_rfreg=GridSearchCV(rfreg1, param_grid, cv=5)
        grid_rfreg.fit(X, y)
        return grid_rfreg.best_params_ , grid_rfreg.best_score_
    
    def train_model(self, X, y, config):
        max_features=config['max_features']
        max_samples=config['max_samples']
        input_pipe = [('Scale', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=500, max_features=max_features, max_samples=max_samples, n_jobs=-1, bootstrap=True, random_state=0))]
        pipe = Pipeline(input_pipe)
        pipe.fit(X, y)
        self.model=pipe