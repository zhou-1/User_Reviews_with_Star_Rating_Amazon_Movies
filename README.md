# User_Reviews_with_Star_Rating_Amazon_Movies
The goal of this competition is to predict star rating associated with user reviews from Amazon Movie Reviews using the available features.   

CS 506 Midterm Report   
Zhou Shen    
Kaggle username: dragonpolice    
Kaggle display name: zhou    

The goal of this Kaggle competition is to predict star rating associated with user reviews from Amazon Movie Reviews using the available features.      
Useful reviews can be product ID, user ID, helpfulness numerator, helpfulness denominator, score (star rating), time of reviews, summary and text of reviews.       
First of all, import all needed libraries for linear algebra, data processing for CSV file I/O, different machine learning algorithms.   
```
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
```

Second, load data with panda’s read_csv and replace null value with text & summary in data.    
# Load data    
data = pd.read_csv('train.csv')     

# Detect missing values and make them blank for text and summary      
data['Text'].loc[data['Text'].isna()] = ''    
data['Summary'].loc[data['Summary'].isna()] = ''     


Third, create a Test_X for Score of data with null value.     
Test_X = data[data.isnull().Score]      


Forth, create train sets based on text and score of data. Then using TfidfVectorizer to do vectorization and TF_IDF data prep rocessing.   
```
# Get train set for Text and Score
Train_X = data['Text']
Train_Y = data['Score']
# vectorization and TF_IDF data preprocessing
vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2)).fit(Train_X)
Train_X_vectorized = vectorizer.transform(Train_X)
```


Fifth, get different models for Logistic Regression, Linear SVC, Decision Tree Classifier and Random Forest Classifier. Here, I tried different parameter inside of each function and kept the one I have for best result.      
```
# Get different models
mod1 = LogisticRegression(max_iter=500)
mod2 = LinearSVC(max_iter=1000)
mod3 = DecisionTreeClassifier(random_state = 0, max_depth=20)
mod4 = RandomForestClassifier(random_state = 0, max_depth=20)
```

Next, ensemble models.     
```
# ensemble models
ensemble_models = VotingClassifier([('lr', mod1), ('svc', mod2), ('dtc', mod3), ('rfc', mod4)])
ensemble_models.fit(Train_X_vectorized, Train_Y)
```

Last, predict the output based on samples in sample.csv file and convert to csv file.      
```
# Predict the output based on samples
samples = pd.read_csv('test.csv')
predict_res = pd.DataFrame(samples)

predict_res['Score'] = ensemble_models.predict(vectorizer.transform(Test_X))
predict_res.to_csv('result.csv',index=False)
```

Result.csv file is the one I used for final result.     
Model selections are below: Logistic Regression, Linear SVC, Decision Tree Classifier and Random Forest Classifier.      

If you wanna run the function, whole data set will cost you very long time. You can choose a subset of data or decrease the number of iterations of logistic regression.      
