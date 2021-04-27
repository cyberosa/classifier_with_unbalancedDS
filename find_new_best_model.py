import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from imblearn.over_sampling import SMOTE

def get_train_test_sets(df, imbalance_fix):
    X = df.drop(['label'], axis=1)
    y = df['label']  # Labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # 70% training and 30% test

    if imbalance_fix == 'down':
        print("Before undersampling: ", Counter(y_train))
        undersample = RandomUnderSampler(sampling_strategy='majority')
        X_train_under, y_train_under = undersample.fit_resample(X_train, y_train)
        print("After undersampling: ", Counter(y_train_under))
        return X_train_under, X_test, y_train_under, y_test

    if imbalance_fix == 'up':
        print("Before undersampling: ", Counter(y_train))
        smote = SMOTE()
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        print("After undersampling: ", Counter(y_train_smote))
        return X_train_smote, X_test, y_train_smote, y_test

    return X_train, X_test, y_train, y_test

def train_model(model_name, X_train, y_train, **best_p):
    '''

    :param model_name:
    :param X_train:
    :param y_train:
    :param best_p:
    :return:
    '''
    if model_name == 'rf_classifier':
        model = RandomForestClassifier(**best_p)

    fm = fit_model(model, X_train, y_train)
    paint_confusion_matrix_and_report(fm, X_test, y_test)

def train_eval_model(model_name, X_train, y_train, X_test, y_test, **best_p):
    '''

    :param model_name:
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param best_p:
    :return:
    '''
    start = time()
    if model_name == 'rf_classifier':
        model = RandomForestClassifier(**best_p)
    model.fit(X_train, y_train)
    end = time()
    result = end - start
    print('Training time = %.3f seconds' % result)
    # get predictions
    y_pred = model.predict(X_test)
    p = precision_score(y_pred, y_test, average=None)
    print("precision scores")
    print(p)
    r = recall_score(y_pred, y_test, average=None)
    print("recall scores")
    print(r)
    return p[0], r[0], p[1], r[1]

def fit_model(model, X0_train, y0_train):
    '''

    :param model:
    :param X0_train:
    :param y0_train:
    :return:
    '''
    start = time()
    model.fit(X0_train,y0_train)
    end = time()
    result = end - start
    print('Training time = %.3f seconds' % result)
    return model

def paint_confusion_matrix_and_report(model, X0_test, y0_test):
    '''

    :param model:
    :param X0_test:
    :param y0_test:
    :return:
    '''
    y_pred = model.predict(X0_test)
    cm2 = confusion_matrix(y0_test, y_pred.round())
    ax= plt.subplot()
    sns.heatmap(cm2, annot=True, ax = ax, fmt="d", cmap="YlGnBu")
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['no-conversion', 'conversion']); ax.yaxis.set_ticklabels(['no-conversion', 'conversion'])
    plt.show()
    # if we want to save it as a file plt.savefig( 'conf_mat.png' )
    prec_rec = classification_report(y_pred, y0_test, target_names=['no-conversion', 'conversion'])
    print(prec_rec)

def find_best_model_parameters(model_name, X_train, y_train):
    '''
    Method to find some initial best parameters given a model and some training data.
    The next step would be to perform a proper grid search
    :param model_name: base model to tune
    :param X_train:
    :param y_train:
    :return:
    '''
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=50, stop=400, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    if model_name == 'rf_classifier':
        model = RandomForestClassifier()
    # TODO add more models
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    s_random = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)  # Fit the random search model
    # let´s try first with the resample dataset
    start = time()
    s_random.fit(X_train, y_train)
    end = time()
    result = end - start
    print('Training time = %.3f seconds' % result)
    return s_random.best_params_

def perform_grid_search(model_name, param_grid, X_train, y_train):
    '''
    Function to perform a grid search given the param_grid, the model name and training data
    :param model_name:
    :param param_grid:
    :param X_train:
    :param y_train:
    :return:
    '''
    # Create a based model
    if model_name == 'rf_classifier':
        model = RandomForestClassifier()
    # TODO add more models
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_

if __name__ == '__main__':
    try:
        if len(sys.argv) < 3:
            print("error missing parameters")
            sys.exit()
        model_name = str(sys.argv[1])
        data_filename = str(sys.argv[2])
        imbalance_fix = str(sys.argv[3])

        # get the training data
        df = pd.read_pickle(data_filename)
        # split the sets
        X_train, X_test, y_train, y_test = get_train_test_sets(df, imbalance_fix)
        best_p = find_best_model_parameters(model_name, X_train, y_train)
        print(best_p)

        train_model(model_name, X_train, y_train, **best_p)
        # create a param_grid based on these results. For instance
        # param_grid = {
        #     'bootstrap': [False],
        #     'max_depth': [5, 10, 20],
        #     'max_features': ['auto'],
        #     'min_samples_leaf': [3, 4, 5],
        #     'min_samples_split': [4, 5, 10],
        #     'n_estimators': [200, 300, 400]
        # }
        #final_p = perform_grid_search(model_name, param_grid, X_train, y_train)
        # The way to train the model and paint parameters
        #fm = fit_model(final_model, X_train_under, y_train_under)
        #paint_confusion_matrix_and_report(fm, X_test, y_test)

    except Exception as e:
        print(e)