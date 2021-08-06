# import neccessary packages and libraries
import os
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# set a theme for seaborn
sns.set_theme()

from sklearn.base import (
    BaseEstimator, 
    TransformerMixin,
)

from sklearn.impute import (
    KNNImputer,
    SimpleImputer,
)
from sklearn.preprocessing import (
    OneHotEncoder, 
    OrdinalEncoder, 
    LabelEncoder,
    StandardScaler,
)

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
)
from sklearn.metrics import (
    r2_score, 
    mean_squared_error,
    auc,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from sklearn.pipeline import (
    FeatureUnion, 
    Pipeline 
)

import local_maps as lm

# numerical, statistical and machine learning packages and libraries
import xgboost as xgb
from scipy import stats

from sklearn import (
    ensemble,
    tree,
)
from sklearn.base import (
    BaseEstimator, 
    TransformerMixin,
)
from sklearn.pipeline import (
    make_pipeline,
    FeatureUnion, 
    Pipeline,
)
from sklearn.feature_selection import (
    SelectKBest, 
    chi2, 
    mutual_info_classif,
)
from sklearn.impute import (
    KNNImputer,
    SimpleImputer,
)
from sklearn.preprocessing import (
    OneHotEncoder, 
    OrdinalEncoder, 
    LabelEncoder,
    StandardScaler,
    MultiLabelBinarizer,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold,
    cross_val_score,
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import (
    SGDClassifier,
    LogisticRegression,
) 

from sklearn.metrics import (
    classification_report,
    r2_score, 
    mean_squared_error,
    auc,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    log_loss,
    roc_auc_score
)


import utils_functions as uf 
import utils_classes as uc
import local_maps as lm

class ParseMultiColumns(BaseEstimator, TransformerMixin):
    """Custom transformer that that changes a list of strings to a set 
    in a column of a dataframe, and assigns the empty set to missing entries.
    """
    #class constructor method 
    def __init__(self, multi_cols=['PlatformWorkedWith']):
            self.multi_cols = multi_cols
            
    # return self nothing else to do here
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col in self.multi_cols:
            X[col] = X[col].str.split(';').apply(lambda x: {} if x is np.nan else set(x))
        return X

    
class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    The constructor extracts and returns the pandas dataset 
    with only those columns whose names were passed to it 
    as an argument during its initialization. 
    It contains two methods: fit and transform.
    """
    
    # class constructor 
    def __init__(self, feature_names):
        self._feature_names = feature_names 
    
    # return self nothing else to do here    
    def fit(self, X, y = None):
        return self 
    
    # method that describes what we need this transformer to do
    def transform(self, X, y = None):
        return X[ self._feature_names ] 
    
    
class MultiColumnsEncoder(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer to convert a feature column of a list in 
    to multiple binary feature columns"""
    def __init__(self, feature_names=None):
            self.feature_names = feature_names

    def fit(self, X, y=None):
        self.encoder_dict_ = {}
        
        for col in self.feature_names:
            mlb = MultiLabelBinarizer()
            mlb.fit(X[col])
            self.encoder_dict_[col] = mlb
        return self

    def transform(self, X):
        for col in self.feature_names:
            col_encoded = pd.DataFrame(
                self.encoder_dict_[col].transform(X[col]),
                columns=self.encoder_dict_[col].classes_,
                index=X.index)
            cols_keep = list(col_encoded.sum().sort_values(ascending=False).head(3).index)

            X = pd.concat([X, col_encoded[cols_keep]], axis=1).drop(columns=[col])

        return X
    
    
    
    # source: https://github.com/Chancylin/StackOverflow_Survey

from sklearn.preprocessing import MultiLabelBinarizer

# custom transformer that that changes a list of strings to a set
# in a column of a dataframe
class ParseMultiColumns(BaseEstimator, TransformerMixin):
    """Custom transformer that that changes a list of strings to a set in a column of a dataframe, and assigns the empty set to missing entries.
    """
    #class constructor method 
    def __init__(self, multi_cols):
            self.multi_cols = multi_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for entry in self.multi_cols:
            X[entry] = X[entry].str.split(';').apply(lambda x: {} if x is np.nan else set(x))
            return X
    
    
    # example on how it performs on the data

df2 = df.copy()
multiple_response = ['LanguageWorkedWith']

str_to_list = StringtoListTranformer(variables=multiple_response)
df_tmp = str_to_list.fit_transform(df2[multiple_response])

list_encoder = ListColumnsEncoder(variables=multiple_response)
df_tmp = list_encoder.fit_transform(df_tmp)

class MultiColumnsEncoder(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer to convert a feature column of a list in 
    to multiple binary feature columns"""
    def __init__(self, multi_cols):
            self.multi_cols = muli_cols

    def fit(self, X, y=None):
        # persist mode in a dictionary
        self.encoder_dict_ = {}
        
        for col in self.multi_cols:
            mlb = MultiLabelBinarizer()
            mlb.fit(X[col])
            self.encoder_dict_[col] = mlb

        return self
    

    def transform(self, X, y=None, drop_dict):

        X = X.copy()
        for col in self.multi_cols:
            col_encoded = pd.DataFrame(
                self.encoder_dict_[col].transform(X[col]),
                columns=self.encoder_dict_[col].classes_,
                index=X.index)

            X = pd.concat([X, col_encoded], axis=1).drop(columns=[col])
            X = X.drop(columns=drop_dict[col], inplace=True)
            
        return X
    
    
    
    # upload the datafile as pandas dataframe
df = pd.read_csv(mypath+'/data/survey20_updated.csv', index_col=[0])

# create a fresh copy of the dataset
#dfp = df.copy()

# all data cleaning and preprocessing steps
#dfp = uf.remove_clean_data(dft)              DO THIS
# create a copy of the pre-processed dataframe
df2 = df1.copy()

# create the predictors dataframe
X = df2.drop(columns = 'JobSat')

# create the labels
y = df2['JobSat']

# check for success
X.shape, len(y)

# split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# summarize the data
print('Train', X_train.shape, y_train.shape)
print('Test', X_test.shape, y_test.shape)

## refactor code: processing data

# the steps in the categorical pipeline for columns of low cardinality
uni_cat_pipeline = Pipeline( steps = [( 'unicat_selector', uc.FeatureSelector(uni_cols) ),
                                  ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                  ( 'ordinal_encoder', OrdinalEncoder() ) ] )

# the steps in the categorical pipeline for columns of high cardinality
multi_cat_pipeline = Pipeline( steps = [( 'multicat_selector', uc.FeatureSelector(multi_cols) ),
                                  ( 'multi_encoder', uc.MultiColumnsEncoder(multi_cols) ) ] )

# the steps in the numerical pipeline     
num_pipeline = Pipeline( steps = [ ('num_selector', uc.FeatureSelector(num_cols) ),
                                  ('imputer', KNNImputer(n_neighbors=5) ),
                                  ( 'std_scaler', StandardScaler() ) ] )

# combine the numerical and the categorical pipelines
full_pipeline = FeatureUnion( transformer_list = [ ( 'unicat_pipeline', uni_cat_pipeline ), 
                                                  ( 'multicat_pipeline', multi_cat_pipeline ) ,
                                                 ( 'numerical_pipeline', num_pipeline )] )


