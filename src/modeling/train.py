INPUT_DIR='data/processed/'

from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
from DecisionTree import DecisionTree
from RandomForest import RandomForest

def gen_train_test():
    data = pd.read_csv(INPUT_DIR+'final_data.csv')

    #one hot encoding
    X = pd.get_dummies(
        data[[
         'weight',
         'Processor rank',
         'Graphics Coprocessor perf',
         'Brand','Laptop type',
         'Laptop purpose',
         'Hard Drive Type',
         'Memory Type',
         'Operating System']], 
        columns=['Brand', 'Laptop type', 'Laptop purpose', 'Hard Drive Type', 'Memory Type', 'Operating System']
        )
    y = data['price']

    #train test split
    return train_test_split(X, y, test_size=.2, random_state=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdname', type=str, default="rd", help="Model name: 'rf' or 'dc'")
    args = parser.parse_args()
    model_name=args.mdname
    X_train, X_test, y_train, y_test=gen_train_test()
    #best_params, best_score=model.grid_search(X_train, y_train)
    #print(best_params, best_score)
    found=True
    if model_name=='rd':
        model=RandomForest()
        config={'max_features': 0.6, 'max_samples': 0.7}
        print("Trainning Random Forest...")
    elif model_name=='dc':
        model=DecisionTree()
        config={'max_depth': 7}
        print("Trainning Decision Tree...")
    else:
        found=False
        print("Not found model")

    if found==True:
        model.train_model(X_train, y_train, config)
        print('R2 train score: ', model.evaluate(X_train, y_train)['r2'])
        print('R2 test score: ', model.evaluate(X_test, y_test)['r2'])
        print('--')
        print('RMSE train score: ', model.evaluate(X_train, y_train)['rmse'])
        print('RMSE test score: ', model.evaluate(X_test, y_test)['rmse'])


