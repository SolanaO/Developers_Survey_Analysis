# general packages and libraries
import os
import sys
from collections import defaultdict
import importlib

import numpy as np
import pandas as pd

# numerical, statistical and machine learning packages and libraries
import xgboost as xgb
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


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

# import local modules
import local_maps as lm
import utils_functions as uf 


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

## processing data pipeline

# the steps in the categorical pipeline for columns of low cardinality
uni_cat_pipeline = Pipeline( steps = [( 'unicat_selector', FeatureSelector(lm.uni_cols) ),
                                  ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                  ( 'ordinal_encoder', OrdinalEncoder() ) ] )

# the steps in the categorical pipeline for columns of high cardinality
multi_cat_pipeline = Pipeline( steps = [( 'multicat_selector', FeatureSelector(lm.multi_cols) ),
                                  ( 'multi_encoder', MultiColumnsEncoder(lm.multi_cols) ) ] )

# the steps in the numerical pipeline     
num_pipeline = Pipeline( steps = [ ('num_selector', FeatureSelector(lm.num_cols) ),
                                  ('imputer', KNNImputer(n_neighbors=5) ),
                                  ( 'std_scaler', StandardScaler() ) ] )

# combine the numerical and the categorical pipelines
full_pipeline = FeatureUnion( transformer_list = [ ( 'unicat_pipeline', uni_cat_pipeline ), 
                                                  ( 'multicat_pipeline', multi_cat_pipeline ) ,
                                                 ( 'numerical_pipeline', num_pipeline )] )
    

