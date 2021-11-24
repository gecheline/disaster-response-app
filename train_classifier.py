from classifier import SequentialClassifier
from tokenizer import tokenize

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import joblib
import pandas as pd
import numpy as np

def load_data(database_filepath):
    df = pd.read_csv(database_filepath).drop(columns=['Unnamed: 0', 'child_alone'])
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    
    rel2_inds = list(np.squeeze(np.argwhere(Y['related'].values==2)))
    Y = Y.drop(rel2_inds, axis=0)
    X = X.drop(rel2_inds, axis=0)
    return X, Y, Y.columns


def build_model():
    
    pipeline_bow = Pipeline([
    ('count_vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer())
    ])
    
    sequential_clf = SequentialClassifier(clf_related = LogisticRegression(max_iter=500, 
                                                                       class_weight='balanced',
                                                                       C = 1.0, 
                                                                       solver = 'lbfgs'), 
                                      clf_type = MultiOutputClassifier(LogisticRegression(max_iter=500, 
                                                                       class_weight='balanced',
                                                                       C = 1.0, 
                                                                       solver = 'lbfgs')),
                                      clf_aid = MultiOutputClassifier(LogisticRegression(max_iter=500, 
                                                                       class_weight='balanced',
                                                                       C = 2.0, 
                                                                       solver = 'lbfgs')), 
                                      clf_weather = MultiOutputClassifier(LogisticRegression(max_iter=500, 
                                                                       class_weight='balanced',
                                                                       C = 1.0, 
                                                                       solver = 'newton-cg')), 
                                      clf_infrastructure = MultiOutputClassifier(LogisticRegression(max_iter=500, 
                                                                       class_weight='balanced',
                                                                       C = 0.5, 
                                                                       solver = 'newton-cg')))
    pipeline = Pipeline([
        ('text_transform', pipeline_bow),
        ('clf', sequential_clf)
    ])
    
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    for i,category in enumerate(category_names):
        print(category, classification_report(Y_test.values[:,i], y_pred[:,i]))
        
if __name__ == "__main__":
    database_filepath, model_filepath = 'database_disaster.csv', 'classifier.pkl'
    X, Y, category_names = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    print('Building model...')
    model = build_model()
    
    print('Training model...')
    model.fit(X_train, Y_train)
    
    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)
    joblib.dump(model, model_filepath)
    