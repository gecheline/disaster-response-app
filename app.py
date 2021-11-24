import json
import plotly
import pandas as pd
import numpy as np

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
import joblib
from classifier import SequentialClassifier
from tokenizer import tokenize

# load data
df = pd.read_csv('database_disaster.csv').drop(columns=['Unnamed: 0', 'child_alone'])
# load model
model = joblib.load("classifier.pkl")

app = Flask(__name__)
# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
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
    app.run()


if __name__ == '__main__':
    main()