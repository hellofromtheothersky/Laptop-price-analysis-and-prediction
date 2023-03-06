INPUT_DIR='data/processed/'

from sklearn.model_selection import train_test_split
import pandas as pd
from DecisionTree import DecisionTree

if __name__ == "__main__":
    data = pd.read_csv(INPUT_DIR+'final_data.csv')
    X = pd.get_dummies(data[['weight','Processor rank','Graphics Coprocessor perf','Brand','Laptop type','Laptop purpose','Hard Drive Type','Memory Type','Operating System', 'price']], 
                    columns=['Brand', 'Laptop type', 'Laptop purpose', 'Hard Drive Type', 'Memory Type', 'Operating System'])
    y = data['price']
    x_col = X.columns.to_list()
    x_col.remove('price')
    X_train, X_test, y_train, y_test = train_test_split(X[x_col], y, test_size=.2, random_state=1)

    model=DecisionTree()
    best_params, best_score=model.grid_search(X_train, y_train)
    # print(best_params, best_score)
    model.train_model(X_train, y_train, max_depth=7)
    print('R2 train score: ', model.evaluate(X_train, y_train))
    print('R2 test score: ', model.evaluate(X_test, y_test))


