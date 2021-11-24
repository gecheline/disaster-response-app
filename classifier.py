import pandas as pd
# import libraries
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin

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
        '''
        Predicts the categories for the input X.
        '''
        
        #this is where the sequential part comes!
        y_predict = pd.DataFrame(np.zeros((X.shape[0], len(self.all_columns))), 
                                 columns= self.all_columns)
        
        y_predict['related'] = self.clf_related.predict(X)
        
        if len(y_predict[y_predict['related']==1]) != 0:
            related_mask = y_predict['related'] == 1        
            y_types = self.clf_type.predict(X[related_mask])
            
            y_predict.loc[related_mask, 'aid_related'] = y_types[:,0]
            y_predict.loc[related_mask, 'weather_related'] = y_types[:,1]
            y_predict.loc[related_mask, 'infrastructure_related'] = y_types[:,2]
            y_predict.loc[related_mask, 'request'] = y_types[:,3]
            y_predict.loc[related_mask, 'offer'] = y_types[:,4]
            y_predict.loc[related_mask, 'direct_report'] = y_types[:,5]
            
            if len(y_predict[y_predict['aid_related']==1]) != 0:
                aid_mask = y_predict['aid_related'] == 1
                y_aid_types = self.clf_aid.predict(X[aid_mask])
                for i,col in enumerate(self.aid_columns):
                    y_predict.loc[aid_mask, col] = y_aid_types[:,i]
            
            if len(y_predict[y_predict['weather_related']==1]) != 0:
                weather_mask = y_predict['weather_related'] == 1
                y_weather_types = self.clf_weather.predict(X[weather_mask])
                for i,col in enumerate(self.weather_columns):
                    y_predict.loc[weather_mask, col] = y_weather_types[:,i]
                
            if len(y_predict[y_predict['infrastructure_related']==1]) != 0:
                infrastructure_mask = y_predict['infrastructure_related'] == 1
                y_infrastructure_types = self.clf_infrastructure.predict(X[infrastructure_mask])
                for i,col in enumerate(self.infrastructure_columns):
                    y_predict.loc[infrastructure_mask, col] = y_infrastructure_types[:,i]
            
        return y_predict.values
    
