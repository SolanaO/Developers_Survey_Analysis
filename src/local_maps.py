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

import local_maps as lm
import utils_functions as uf 
import utils_classes as uc
import local_maps as lm



## Contains: replacement dictionaries and useful lists used in the data processing. 

# dictionary with shorter strings for education levels
new_edLevel = {'Master’s degree (M.A., M.S., M.Eng., MBA, etc.)': 'Master’s degree',
 'Bachelor’s degree (B.A., B.S., B.Eng., etc.)': 'Bachelor’s degree',
 'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)': 'Secondary school',
 'Professional degree (JD, MD, etc.)': 'Professional degree',
 'Some college/university study without earning a degree': 'College study/no degree',
 'Associate degree (A.A., A.S., etc.)' : 'Associate degree',
 'Other doctoral degree (Ph.D., Ed.D., etc.)': 'Other doctoral degree',
 'I never completed any formal education' : 'No formal education'}
 
# dictionary with shorter descriptions for the undegraduate majors
new_UndergradMajor = {'Computer science, computer engineering, or software engineering':
                           'Computer science',
       'Another engineering discipline (such as civil, electrical, mechanical, etc.)':'Engineering other',
       'A humanities discipline (such as literature, history, philosophy, etc.)': 'Humanities',
       'A health science (such as nursing, pharmacy, radiology, etc.)': 'Health science',
       'Information systems, information technology, or system administration' : 'Information system',
       'Web development or web design': 'Web dev/design',
        'Mathematics or statistics': 'Math or stats',
       'A natural science (such as biology, chemistry, physics, etc.)': 'Natural science',
       'Fine arts or performing arts (such as graphic design, music, studio art, etc.)': 'Arts',
       'I never declared a major': 'No major',
       'A social science (such as anthropology, psychology, political science, etc.)': 'Social science',
       'A business discipline (such as accounting, finance, marketing, etc.)': 'Business'}
       
# replace some strings in the EdImpt column
new_EdImpt = {'Not at all important/not necessary': 'Not important'}

# encoding map for job satisfaction
JobSat_dict =  {'Very dissatisfied': 1, 'Slightly dissatisfied': 2,
               'Neither satisfied nor dissatisfied': 3, 
               'Slightly satisfied': 4, 'Very satisfied': 5}

# list of columns to be removed
cols_del = [
    # personal, demographics  information
    #'Respondent', 
    'MainBranch', 'Employment', 'Hobbyist', 
    'Country',
    'Ethnicity', 'Gender', 'Sexuality', 'Trans', 'Age',                                
    
    # related to ConvertedComp
    'CompFreq', 'CompTotal', 'CurrencyDesc', 'CurrencySymbol',                 
    
    # questions regarding future activities
    'DatabaseDesireNextYear', 'MiscTechDesireNextYear',                    
    'CollabToolsDesireNextYear', 'PlatformDesireNextYear',
    'LanguageDesireNextYear', 'WebframeDesireNextYear',
    
    # questions regarding this survey
    'SurveyEase', 'SurveyLength', 'WelcomeChange',                           
    
    # question regarding participation is StackOverflow
    'SOSites', 'SOComm', 'SOPartFreq',
    'SOVisitFreq', 'SOAccount',                                               

    # columns related to other columns
    'Age1stCode', 'YearsCodePro', 'DevClass', 
    
    # high cardinality, multiple choices columns, add noise 
    'DatabaseWorkedWith','MiscTechWorkedWith','LanguageWorkedWith',
    'WebframeWorkedWith', #'CollabToolsWorkedWith',                                                 

    # other questions not directly related to our goal
    'JobHunt', 
    'JobHuntResearch', 'Stuck',
    'PurchaseResearch', 
     #'PurchaseWhat', 
    'Stuck', 'PurpleLink',
    'OffTopic', 'OtherComms',
    'JobFactors', #'JobSeek',
    'DevType']                                                            


# the columns grouped by types in the predictors matrix

# numerical columns
num_cols = ['ConvertedComp', 'WorkWeekHrs', 'YearsCode']
# the list of discrete columns with many levels 
multi_cols = ['PlatformWorkedWith', 'CollabToolsWorkedWith']
# the list of discrete columns with several levels
uni_cols = ['EdLevel', 'EdImpt', 'OnboardGood', 'JobSeek', 
            'Overtime', 'DevOps', 'Learn', 'UndergradMajor', 'OpSys', 
            'DevOpsImpt', 'OrgSize', 'PurchaseWhat']

# the list of performance metrics associated to confusion matrix
metrics_list = ['accuracy','precision','recall', 'f1']


