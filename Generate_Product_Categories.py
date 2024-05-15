from google.colab import drive

drive.mount('/content/drive')

import numpy as np
import pandas as pd
import csv
import os
from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings, string
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
nltk.download('stopwords')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# Implementing the Hybrid Model using TF-IDF Vectorizer and KNN Algorithm

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Implement content-based filtering
tfidf_vectorizer = TfidfVectorizer()
product_features = tfidf_vectorizer.fit_transform(df['categories'] + str(df['Product_Type_ID']))

# Implement collaborative filtering
ratings = pd.DataFrame({'customer_id	': df['customer_id'], 'Product_Type_ID': df['Product_Type_ID'], 'Rating': 1})
pivot_table = ratings.pivot_table(index='customer_id	', columns='Product_Type_ID', values='Rating', fill_value=0)

knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(pivot_table)


customer_formats={}
def hybrid_recommendation(n_recommendations=5, content_weight=0.5):
    recommendations = {}
    unique_customer_ids = df['customer_id'].unique()
    for customer_id in unique_customer_ids:
        customer_data = df[df['customer_id'] == customer_id]

        # Content-Based Recommendations
        customer_profile = ' '.join(customer_data['categories'])
        customer_feature = tfidf_vectorizer.transform([customer_profile])

        content_indices = cosine_similarity(customer_feature, product_features).argsort()[0][-n_recommendations:]
        distances, indices = knn_model.kneighbors(pivot_table.loc[customer_id].values.reshape(1, -1), n_neighbors=n_recommendations)
        collaborative_indices = indices.squeeze()


        hybrid_indices = list(set(content_indices) | set(collaborative_indices))
        hybrid_scores = [(idx, content_weight * cosine_similarity(product_features[idx], customer_feature) + (1 - content_weight) * distance) for idx, distance in zip(hybrid_indices, distances.squeeze())]
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        hybrid_recommendations = [df.loc[idx, 'Product_Type_ID'] for idx, _ in hybrid_scores]

        recommendations[customer_id] = hybrid_recommendations[:n_recommendations]

        x=[]
        for i in hybrid_recommendations:
          x.append(product_id(i))

        x=remove_space(x)
        customer_formats[customer_id] = x

