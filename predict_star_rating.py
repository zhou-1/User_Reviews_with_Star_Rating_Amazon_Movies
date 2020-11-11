# Import libraries 
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier

# Load data
data = pd.read_csv('train.csv')

# Detect missing values and make them blank for text and summary
data['Text'].loc[data['Text'].isna()] = ''
data['Summary'].loc[data['Summary'].isna()] = ''

Test_X = data[data.isnull().Score]

# Get train set for Text and Score
Train_X = data['Text']
Train_Y = data['Score']
# vectorization and TF_IDF data preprocessing
vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2)).fit(Train_X)
Train_X_vectorized = vectorizer.transform(Train_X)

# Get different models
mod1 = LogisticRegression(max_iter=500)
mod2 = LinearSVC(max_iter=1000)
mod3 = DecisionTreeClassifier(random_state = 0, max_depth=20)
mod4 = RandomForestClassifier(random_state = 0, max_depth=20)

# ensemble models
ensemble_models = VotingClassifier([('lr', mod1), ('svc', mod2), ('dtc', mod3), ('rfc', mod4)])
ensemble_models.fit(Train_X_vectorized, Train_Y)

# Predict the output based on samples
samples = pd.read_csv('test.csv')
predict_res = pd.DataFrame(samples)

predict_res['Score'] = ensemble_models.predict(vectorizer.transform(Test_X))
predict_res.to_csv('result.csv',index=False)
