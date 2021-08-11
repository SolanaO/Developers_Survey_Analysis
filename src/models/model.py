
"""
The module:
    - uploads the raw data,
    - performs all the preprocessing and the processing steps,
    - performs feature engineering,
    - samples the data,
    - applies the tuned algorithm,
    - prints the performance metrics.
"""

########################################################################

# import general packages and libraries
import sys
import importlib

# data manipulation packages
import numpy as np
import pandas as pd

# numerical, statistical and machine learning packages and libraries

from sklearn.base import (
    BaseEstimator, 
    TransformerMixin,
)
from sklearn.pipeline import (
    make_pipeline,
    Pipeline,
    FeatureUnion,
)
from sklearn.impute import (
    KNNImputer,
    SimpleImputer,
)
from sklearn.preprocessing import (
    OrdinalEncoder, 
    StandardScaler,
    MultiLabelBinarizer,
)
from sklearn.model_selection import (
    train_test_split,
)

from sklearn.ensemble import (
    RandomForestClassifier,
)
    
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score
)
 
#################################################################################

# create a string for the working directory
mypath = '/home/silvia/Documents/udacityND/ml_dsnd/proj1_dsnd/'

# add src folder to sys.path to use the local modules
sys.path.insert(1, mypath + 'src')

##################################################################################

# import local modules 
import utils_functions as uf 
import utils_classes as uc
import local_maps as lm     

#################################################################################

# upload the datafile as pandas dataframe
df = pd.read_csv(mypath+'/data/raw/survey20_results_public.csv', index_col=[0])

# create a copy of the dataframe
df1  = df.copy()

# preprocess data: change types, columns, remove features
df_proc = (df1.
                pipe(uf.data_prep).
                pipe(uf.parse_dev_type).
                pipe(uf.remove_clean_data))

# create the predictors dataframe
X = df_proc.drop(columns = 'JobSat')

# create the labels
y = df_proc['JobSat']

# split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# create an instance of the classifier, using the optimizing parameters
RF_clf = RandomForestClassifier(max_depth=90, min_samples_split=5, 
                                n_estimators=1400, random_state=42)

# process the train data
X_train_proc = uc.full_pipeline.fit_transform(X_train)

# fit and transform the train data
RF_clf.fit(X_train_proc, y_train)

# process the test data
X_test_proc = uc.full_pipeline.transform(X_test)

# predict labels on test set
y_pred = RF_clf.predict(X_test_proc)

# evaluate performance metrics on the train set
perf_train_RF = pd.Series(uf.get_perf_metrics(RF_clf.fit(X_train_proc,y_train),
                                              X_train_proc, y_train), 
                       index = lm.metrics_list)

# evaluate performance metrics on the test set
perf_test_RF = pd.Series(uf.get_perf_metrics(RF_clf.fit(X_train_proc,y_train),
                                             X_test_proc, y_test), 
                         index = lm.metrics_list)

# combine performance metrics for the baseline model
perf_model_RF = pd.DataFrame.from_dict({'train': perf_train_RF,
                                        'test': perf_test_RF}).round(3)

# print evaluation metrics and results

print('Performance metrics comparison for RandomForestClassifier:\n', perf_model_RF)

result1_RF = confusion_matrix(y_test, y_pred)
print('\nRandomForestClassifier Confusion Matrix for Test Set:')
print(result1_RF)

result2_RF = classification_report(y_test, y_pred)
print('\nRandomForestClassifier Classification Report for Test Set:')
print (result2_RF)

