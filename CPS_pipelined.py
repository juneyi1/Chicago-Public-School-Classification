# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, Imputer, FunctionTransformer, LabelBinarizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.base import TransformerMixin

CPS = pd.read_csv('school_data_training.csv')
CPS_test = pd.read_csv('school_data_test.csv')
y = CPS['probation']

multinomial = ['ZIP Code', 'ISAT Value Add Color Math', 
               'ISAT Value Add Color Read', 'General Services Route ',
               'Community Area Number', 'Community Area Name',
               'Ward', 'Police District'] 
binomial = ['Healthy Schools Certified?']
floats = ['Safety Score', 'Environment Score', 'Instruction Score', 
          'Rate of Misconducts (per 100 students) ', 
          'ISAT Exceeding Math %', 'ISAT Exceeding Reading % ',
          'ISAT Value Add Math', 'ISAT Value Add Read',
          'College Enrollment (number of students) '] #with NaN in it
objects = ['Family Involvement Score', 'Leaders Score ', 'Teachers Score', 
           'Parent Engagement Score', 'Parent Environment Score', 
           'Pk-2 Literacy %', 'Pk-2 Math %', 
           'Gr3-5 Grade Level Math %','Gr3-5 Grade Level Read % ',
           'Gr3-5 Keep Pace Read %', 'Gr3-5 Keep Pace Math %', 
           'Gr6-8 Grade Level Math %', 'Gr6-8 Grade Level Read %', 
           'Gr6-8 Keep Pace Math%', 'Gr6-8 Keep Pace Read %',
           'Gr-8 Explore Math %', 'Gr-8 Explore Read %', 
           'Students Taking  Algebra %', 'Students Passing  Algebra %',
           '9th Grade EXPLORE (2009) ', '9th Grade EXPLORE (2010) ',
           'Net Change EXPLORE and PLAN', '11th Grade Average ACT (2011) ',
           'Net Change PLAN and ACT', 'College Eligibility %',
           'Graduation Rate %', 'College Enrollment Rate %',
           'Freshman on Track Rate %']#with NaN in it 
percents = ['Average Student Attendance', 'Average Teacher Attendance', 
            'Individualized Education Program Compliance Rate ']

def preprocessing(df=CPS, object_columns=objects, percent_columns=percents):
    X = df.copy()
    for column in object_columns:
        X[column] = X[column].apply(pd.to_numeric, errors='coerce')
    for column in percent_columns:    
        X[column] = X[column].apply(lambda x: (x[:-1]))
        X[column] = X[column].apply(pd.to_numeric, downcast='float', errors='coerce')
    return X
CPS = preprocessing(df=CPS, object_columns=objects, percent_columns=percents)
CPS_test = preprocessing(df=CPS_test, object_columns=objects, percent_columns=percents)

def Get_Columns(data=None, columns=floats + objects + percents):
    #return data[:,1].reshape(-1, 1)
    return data[columns]
columns_extractor = FunctionTransformer(Get_Columns, validate=False)
imputer = Imputer(missing_values=np.nan, strategy='mean', axis=1)

# pipes for (floats + objects + percents) columns as first feature union step
pipes = make_pipeline(columns_extractor, imputer) 
feature_union_steps = [('floats+objects+percents', pipes)] 

class FeatureExtractor(TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x[self.column].values.reshape(-1,1)

binomial_lb = LabelBinarizer()
binomial_pipe = make_pipeline(FeatureExtractor('Healthy Schools Certified?'), binomial_lb)
feature_union_steps.append(('binomial', binomial_pipe))

def DropFirst(df=None): 
    return pd.DataFrame(df).iloc[:,1:]    
firstcolumn_dropper = FunctionTransformer(DropFirst, validate=False)    

# adding pipes for multinomial columns to feature union steps
LabelBinarizers = [LabelBinarizer() for item in multinomial]
for i, column in enumerate(multinomial):
    pipe = make_pipeline(FeatureExtractor(column), 
                         LabelBinarizers[i], 
                         firstcolumn_dropper)
    feature_union_steps.append((column, pipe))
fu = FeatureUnion(feature_union_steps)

fu_pipe = make_pipeline(fu, StandardScaler()) 
X = fu_pipe.fit_transform(CPS)
X_test = fu_pipe.transform(CPS_test)

sgd_cls_params = {
    'loss':['log'], #, 'squared_loss'],
    'penalty':['l1','l2', 'elasticnet'],
    'alpha':np.logspace(-5,2,50),
    'l1_ratio':[i/10.0 for i in range(11)]
    }
sgd_cls = SGDClassifier()
sgd_cls_gs = GridSearchCV(sgd_cls, sgd_cls_params, cv=5, n_jobs=-1, verbose=1, scoring='accuracy')

sgd_cls_gs.fit(X, y)
print(sgd_cls_gs.best_params_)
print(sgd_cls_gs.best_score_)
y_test = sgd_cls_gs.predict(X_test)
#{'alpha': 0.13894954943731361, 'penalty': 'elasticnet', 'loss': 'log', 'l1_ratio': 0.3}0.87922705314

# final_pipe = make_pipeline(fu, StandardScaler(), sgd_cls_gs)
# print(final_pipe.named_steps)
# final_pipe.fit(CPS, y)
# y_test = final_pipe.predict(CPS_test)

def csv_spitter(y_test=None, test=CPS_test):
    prediction = pd.DataFrame(y_test, columns=['Prediction'])
    Id = test[['id_number']]
    Id.join(prediction).to_csv('class_predictions.csv', index=False)
#csv_spitter(y_test, CPS_test)

# def LogisticRegressionGSCV(predictors=Xs, target=y):
#     lr = LogisticRegression()
#     params_grid = {
#         'penalty': ['l1', 'l2'],
#         'C': [0.01, 0.50, 0.10, 1.0, 10, 50, 100]
#         }
#     lr_gs = GridSearchCV(lr, param_grid=params_grid,
#                           n_jobs=-1, verbose=0)
#     lr_gs.fit(predictors, target)
#     print(lr_gs.best_params_)
#     print(lr_gs.best_score_)
#     lr_gs = lr_gs.best_estimator_