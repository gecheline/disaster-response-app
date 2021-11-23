import json
import plotly
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
import joblib
from sqlalchemy import create_engine

from sklearn.base import BaseEstimator, ClassifierMixin

app = Flask(__name__)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    df_filter = df.drop(columns=['id', 'message', 'original', 'genre'])
    
    # data for heatmap
    df_corr = df_filter.corr()

    #data for category occurences (bars with counts in each category)
    # drop the 188 rows with class related = 2
    rel2_inds = list(np.squeeze(np.argwhere(df_filter['related'].values==2)))
    df_filter = df_filter.drop(rel2_inds, axis=0)
    df_sum = df_filter.sum()
    category_names = list(df_sum.index)
    category_values = list(df_sum.values)
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_values
                )
            ],

            'layout': {
                'title': 'Category occurences',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        
        {
            'data': [
                Heatmap(
                    x = df_corr.index.tolist(),
                    y = df_corr.columns.tolist(),
                    z = df_corr.values.tolist()
                )
            ],

            'layout': {
                'title': 'Heatmap of the categories',
                'yaxis': {
                    'title': ""
                },
                'xaxis': {
                    'title': ""
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    
    class SequentialClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self, clf_related, 
                     clf_type, 
                     clf_aid,
                     clf_weather, 
                     clf_infrastructure,
                    ):

            # Initialize each classifier with their provided kwargs
            self.clf_related = clf_related
            self.clf_type = clf_type
            self.clf_aid = clf_aid
            self.clf_weather = clf_weather
            self.clf_infrastructure = clf_infrastructure


            self.all_columns = ['related', 'request', 'offer', 'aid_related', 'medical_help',
                               'medical_products', 'search_and_rescue', 'security', 'military',
                               'water', 'food', 'shelter', 'clothing', 'money',
                               'missing_people', 'refugees', 'death', 'other_aid',
                               'infrastructure_related', 'transport', 'buildings', 'electricity',
                               'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
                               'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
                               'other_weather', 'direct_report']
            self.type_columns = ['aid_related', 'weather_related', 'infrastructure_related', 
                                 'request', 'offer', 'direct_report']
            self.aid_columns = ['medical_help',
                               'medical_products', 'search_and_rescue', 'security', 'military',
                               'water', 'food', 'shelter', 'clothing', 'money',
                               'missing_people', 'refugees', 'death', 'other_aid']
            self.weather_columns = ['floods', 'storm', 'fire', 'earthquake', 'cold','other_weather']
            self.infrastructure_columns = ['transport', 'buildings', 'electricity',
                                           'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure']


        def fit(self, X, Y):

            Y1 = Y['related']

            self.clf_related.fit(X, Y1)

            related_mask = Y['related']==1
            self.clf_type.fit(X[related_mask], Y[related_mask][self.type_columns])

            aid_mask = Y['aid_related']==1
            self.clf_aid.fit(X[aid_mask], Y[aid_mask][self.aid_columns])

            weather_mask = Y['weather_related']==1
            self.clf_weather.fit(X[weather_mask], Y[weather_mask][self.weather_columns])

            infrastructure_mask = Y['infrastructure_related']==1
            self.clf_infrastructure.fit(X[infrastructure_mask], Y[infrastructure_mask][self.infrastructure_columns])

            return self

        def predict(self, X):

            #this is where the sequential part comes!
            y_predict = pd.DataFrame(np.zeros((X.shape[0], len(self.all_columns))), 
                                     columns= self.all_columns)

            y_predict['related'] = self.clf_related.predict(X)

            related_mask = y_predict['related'] == 1
            if X[related_mask].shape[0] > 0:
                y_types = self.clf_type.predict(X[related_mask])

                y_predict.loc[related_mask, 'aid_related'] = y_types[:,0]
                y_predict.loc[related_mask, 'weather_related'] = y_types[:,1]
                y_predict.loc[related_mask, 'infrastructure_related'] = y_types[:,2]
                y_predict.loc[related_mask, 'request'] = y_types[:,3]
                y_predict.loc[related_mask, 'offer'] = y_types[:,4]
                y_predict.loc[related_mask, 'direct_report'] = y_types[:,5]

                aid_mask = y_predict['aid_related'] == 1
                weather_mask = y_predict['weather_related'] == 1
                infrastructure_mask = y_predict['infrastructure_related'] == 1

                if X[aid_mask].shape[0] > 0:
                    y_aid_types = self.clf_aid.predict(X[aid_mask])
                    for i,col in enumerate(self.aid_columns):
                        y_predict.loc[aid_mask, col] = y_aid_types[:,i]

                if X[weather_mask].shape[0] > 0:
                    y_weather_types = self.clf_weather.predict(X[weather_mask])
                    for i,col in enumerate(self.weather_columns):
                        y_predict.loc[weather_mask, col] = y_weather_types[:,i]

                if X[infrastructure_mask].shape[0] > 0:
                    y_infrastructure_types = self.clf_infrastructure.predict(X[infrastructure_mask])
                    for i,col in enumerate(self.infrastructure_columns):
                        y_predict.loc[infrastructure_mask, col] = y_infrastructure_types[:,i]

            return y_predict.values

    def remove_stopwords(words):
        return [word for word in words if word not in stopwords.words('english')]

    def lemmatize(words):
        words = [WordNetLemmatizer().lemmatize(word, pos='n') for word in words]
        words = [WordNetLemmatizer().lemmatize(word, pos='v') for word in words]
        words = [WordNetLemmatizer().lemmatize(word, pos='a') for word in words]
        return words

    def tokenize_twitter(text):
        from nltk.tokenize import TweetTokenizer
        return TweetTokenizer().tokenize(text)

    def tokenize(text):
        return remove_stopwords(lemmatize(tokenize_twitter(text)))

    # load data
    df = pd.read_csv('database_disaster.csv')
    # load model
    model = joblib.load("classifier.pkl")
    
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()