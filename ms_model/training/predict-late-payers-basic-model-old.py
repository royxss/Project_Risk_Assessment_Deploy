# Import Libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Preprocessing and Pipeline libraries
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
import pickle

print("\nLoading training data...")
# load training data
test_data = pd.read_csv("training/data/peerLoanTraining.csv", engine='python', header=0)

# Separate out X and y
X_train = test_data.loc[:, test_data.columns != 'is_late']
y_train = test_data['is_late']

# load test data
test_data = pd.read_csv("training/data/peerLoanTest.csv", engine='python', header=0)

# Separate out X and y
X_test = test_data.loc[:, test_data.columns != 'is_late']
y_test = test_data['is_late']

# Preprocessing Steps
numeric_features = ['loan_amnt', 
                    'int_rate', 'annual_inc', 'revol_util', 
                    'dti', 'delinq_2yrs'
                   ]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ])

categorical_features = ['purpose','grade', 'emp_length', 'home_ownership']

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

preprocess = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
        ]
    )

# Combine preprocessing with classifier
latePaymentsModel = make_pipeline(
    preprocess,
    RandomForestClassifier(random_state=42))

# Fit the pipeline to the training data (fit is for both the preprocessing and the classifier)
print("\nTraining model ...")
latePaymentsModel.fit(X_train, y_train)

# Save the trained model as a pickle file
print("\nSaving model ...")
file = open('public/latePaymentsModel.pkl', 'wb')
pickle.dump(latePaymentsModel, file)
file.close()

# load the pickled model
print("\nLoading saved model to make example predictions...")
pickledModel = pickle.load(open('public/latePaymentsModel.pkl','rb'))

# Make a prediction for a likely on time payer
payOnTimePrediction = {
    'loan_amnt': [100],
    'int_rate': [0.02039],
    'purpose': ['credit_card'],
    'grade': ['A'],
    'annual_inc': [80000.00],
    'revol_util': [0.05],
    'emp_length': ['10+ years'],
    'dti': [1.46],
    'delinq_2yrs': [0],
    'home_ownership': ['RENT']
    }
payOnTimePredictionDf = pd.DataFrame.from_dict(payOnTimePrediction)

print("\nPredicting class probabilities for likely on-time payer:")
print(pickledModel.predict_proba(payOnTimePredictionDf))

# Prediction for a likely late payer
payLatePrediction = {
    'loan_amnt': [10000],
    'int_rate': [0.6],
    'purpose': ['credit_card'],
    'grade': ['D'],
    'annual_inc': [20000.00],
    'revol_util': [0.85],
    'emp_length': ['1 year'],
    'dti': [42.00],
    'delinq_2yrs': [4],
    'home_ownership': ['RENT']
    }
payLatePredictionDf = pd.DataFrame.from_dict(payLatePrediction)

print("\nPredicting class probabilities for a likely late payer:")
print(pickledModel.predict_proba(payLatePredictionDf))

# Predict class probabilities for a set of records using the test set
print("\nPredicting class probabilities for the test data set:")
print(pickledModel.predict_proba(X_test))

from sklearn.metrics import accuracy_score
print("Accuracy:\n%s" % accuracy_score(y_test, pickledModel.predict(X_test)))
