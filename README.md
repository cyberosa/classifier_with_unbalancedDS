## Description
The goal of this exercise is to build a prediction model using an unbalanced dataset. We will predict if a customer has a high probability of becoming a paid customer or not. The dataset provided has a strong unbalanced proportion of successful converted customers. That is why we will have to apply some sampling techniques to compensate this.

Besides a Flask API was built to launch prediction requests to the winning model.

## Organisation of the repo
-- Notebooks folder: it contains the notebooks used for exploration, building and training the model and the evaluation. Furthermore different approaches were analyzed to compare different models and to select the best one according to the chosen evaluation metric.
-- test folder: it contains some initial unittests for the python files.
-- data_processing.py : different cleaning/scaling methods used during the exploration.
-- find_new_best_model.py : functions to build and to train a classifier.
-- get_model_prediction.py : function to load a model and get the customer prediction.
-- pred_app.py : Flask api to process the GET requests from a local server.