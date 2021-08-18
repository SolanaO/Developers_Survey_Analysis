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
    recall_score, f1_score,
    log_loss,
    roc_auc_score
)

# import local modules
import local_maps as lm
import utils_classes as uc


# print to a file a list of packages and their versions used in this jupyter notebook
def package_requirements():
    "Creates a text file with the packages list and their versions."
    with open('packageinfo.txt', 'w') as f:
        print('Python {}'.format(sys.version), file=f)
        print('Packages:', file=f)
        print('\n'.join(f'{m.__name__} {m.__version__}' 
                    for m in globals().values() if getattr(m, '__version__', None)),file=f)
    return()

#############################################################################################

#### GENERAL FUNCTIONS

def groupby_to_df(df, groupby_col, count_col, method):
    """
    Counts the values of a column grouped by the categories of
    a second column and converts the answer into a dataframe whose columns
    correspond to the caegories of the groupby column.
    INPUT:
        df = dataframe
        groupy_col = column used in groupby 
        count_col = column whose values are counted
        method = True (if output is normalized), False (for counts only).
    OUTPUT:
        df = dataframe whose columns are categories in groupby_col
    """
    # use groupby() to get the percentage of count_col in each level
    temp_df = df.groupby(groupby_col)[count_col].value_counts(normalize=method)
    # convert the series into a dataframe object
    temp_df = temp_df.reset_index(name='result')
    # put the columns in array form 
    temp_df = temp_df.pivot(index=count_col, columns=groupby_col)
    temp_df.columns = temp_df.columns.to_flat_index()
    return temp_df

# prepare a column with multiple choice answers by replicating the rows
def explode_col(df, col):
    """
    Takes a column whose entries are multiple strings, originating
    in a multiple choice and multiple answers question, and 
    creates rows where each entry in the specified column has
    one string option.
    INPUT:
        df = dataframe
        col = column to transform
    OUTPUT:
        df = dataframe whose entries in col are individual strings
    """
    # transform each element of col into a list
    df[col] = df[col].str.split(';')
    # transform each element of a list-like to a row, replicating index values
    df = df.explode(col)
    return df
    
# function that will count the type of strings in a column
def counts_strings(strings_list, dframe, incol):
    """
    Counts the number of occurences of a given string among
    the text entries of a column in a pandas dataframe.
    Renames the columns in the output dataframe.
    INPUT: 
        strings_list = list of strings to search for
        dframe = pandas dataframe, must contain incol
        incol = the column where to search for strings
    OUTPUT:
        new_df = dataframe with two columns
    """
    my_counts = defaultdict(int)
    for entry in strings_list:
        my_counts[entry] = dframe[incol].str.contains(entry).sum()
    new_df = pd.DataFrame.from_dict(my_counts, orient = 'index').reset_index()
    new_df.rename(columns = {'index':incol, 0:'counts'}, inplace=True)
    return new_df

# find possible individual answers from a multiple choice answers question
def possible_choices(df,col):
    """
    Lists all individual strings from entries of a column that contains 
    lists of strings, originating from questions that have multiple answers.
    INPUT:
        df = dataframe
        col = column whose entries are split in individual strings
    OUTPUT:
        choice_list = list of strings, that consist of unique possible choices
    """
    temp_list = df[col].dropna().str.strip('()').str.split(';').to_list()
    flat_list = [item for sublist in temp_list for item in sublist]
    choice_list = list(set(flat_list))
    return choice_list

# simple imputer to fill missing values and return a dataframe
def df_simple_imputer(df):
    """
    Inputs the missing values in a pandas dataframe, by filling them with 'missing'.
    INPUT:
        df = dataframe with missing values
    OUTPUT:
        df_cat_imp = dataframe with no missing entries
    """
    #create an instance of the imputer
    cat_imputer = SimpleImputer(strategy='constant', fill_value='missing')
    # impute the missing categorical values and return a dataframe
    df_cat_imp = pd.DataFrame(cat_imputer.fit_transform(df), columns=df.columns)
    return df_cat_imp

# replaces a numerical column with a categorical one, by binning the values
def binarize_col(df, old_col, new_col, cut_labels, cut_bins):
    """
    Discretizes the values in a column to a number of
    custom made bins. Creates a new column, drops the old 
    column. 
    INPUT: 
        df = dataframe
        old_col = column to be binnarized
        cut_labels = the labels of the new column
        new_col = the column to hold the binnarized data
    OUTPUT:
        new_df = dataframe with two columns
    """
    df[new_col] = pd.cut(df[old_col], bins=cut_bins, labels=cut_labels)
    df[new_col] = df[new_col].astype('object')
    df.drop(columns = old_col, inplace=True)
    return df

# parse columns with multiple strings as entries
def parse_multi_columns(df, multi_cols):
    """
    Replaces the list of entries with a set, missing values with the empty set.
    INPUT: 
       df = dataframe
       multi_cols = list of columns to be parsed
    OUTPUT = transformed column
    """
    for col in multi_cols:
        df[col] = df[col].str.split(';').apply(lambda x: {} if x is np.nan else set(x))
    return df

# for each categorical column, print possible row values and their counts
def list_answers(df, cat_cols):
    """
    Simple function to print the counts on categories of discrete features.
    INPUT: 
        df = dataframe
        cat_cols = list of names of discrete features
    OUTPUT:
        printed table with column names and the class counts for each category
    """
    for col in cat_cols:
        print(col)
        print(' ')
        print(df1[col].value_counts())
        print(' ')
        
# compute performance metrics related to the confusion matrix        
def get_perf_metrics(model, X, y_comp):
    """
    Calculate and print performance metrics for the model evaluation.
    Metrics evaluated are: roc_auc, accuracy, precision, recall
    INPUT:
        model = an instance of the classifier
        X = the dataframe of predictors, have to be processed for the model
        y_comp = vector of labels
    OUTPUT:
        model scores = table of scores (accuracy, precision, recall, f1),
                        for each of the labels to be classified
    """
    
    y_pred = model.predict(X)
    
    accuracy = accuracy_score(y_comp, y_pred)
    precision = precision_score(y_comp, y_pred, average = 'macro')
    recall = recall_score(y_comp, y_pred, average = 'macro')
    f1 = f1_score(y_comp, y_pred, average = 'macro')
     
    model_scores = [accuracy, precision, recall, f1]
    
    return model_scores

######################################################################################

#### DATA SPECIFIC FUNCTIONS

# pre-processing steps, useful mostly to render better looking plots
def data_prep(df):
    """
    Pre-processing steps for StackOverflow data.
    INPUT:
        df = raw data as pandas dataframe
    OUTPUT:
        df1 = preprocessed data, as pandas dataframe
    """
    
    # create a copy of the data
    df1 = df.copy()
    # drop the NEW prefix in some of the columns' names
    df1.columns = [col.replace('NEW', '') for col in df1.columns]
    # create column DevClass, entry data_coder or other_coder
    df1['DevClass'] = np.where(df1["DevType"].str.contains("Data ", 
                                                           na = False),'data_coder','other_coder')
    # replace longer strings with shorter expressions in several of the columns
    df1.replace(lm.new_EdLevel, inplace=True)
    df1.replace(lm.new_UndergradMajor, inplace=True)
    df1.replace(lm.new_EdImpt, inplace=True)
    # rename the data engineer string 
    df1['DevType'] = df1['DevType'].str.replace('Engineer, data', 'Data engineer')
    # drop duplicates if any
    df1.drop_duplicates()
    return df1

# parse the 'DevType' column
def parse_dev_type(dft):
    """
    Rewrite entries in 'DevType' column as strings to replicate rows.
    INPUT:
        dft = pandas dataframe
    OUTPUT:
        dft = dataframe with parsed `DevType` column
    """
    
    # transform each element of col into a list
    dft['DevType'] = dft['DevType'].str.split(';')
    
    # transform each element of a list-like to a row, replicating index values
    dft = dft.explode('DevType')
    
    # retain only those rows that contain data coders
    dft = dft.loc[dft.DevType.str.contains('Data ', na=False)]
    
    return dft

# steps for removing unnecessary data
def remove_clean_data(dft): 
    """
    Steps to remove unnecessary rows and columns.
    INPUT:
        dft = dataframe to be processed, based on StackOverflow data
    OUTPUT:
        dft = dataframe that contains only rows, columns relevant 
              for the classification
    """
    
    # retain only the employed data developers
    dft = dft[dft['Employment'] != 'Not employed, but looking for work']
    
    # retain only the respondents that code professionally
    # create a list of main branch choices
    main_choices = dft.MainBranch.value_counts().index.to_list()
    # retain rows where MainBranch contains the data professionals
    dft = dft[dft.MainBranch.isin(main_choices[:2])]
    
    # drop rows with missing JobSat
    dft.dropna(subset=['JobSat'], inplace=True)
    
    # drop all the columns in the specified list of columns
    dft.drop(columns=lm.cols_del, inplace=True)
    
    #  encode the 'JobSat' data to numerical values
    dft['JobSat'] = dft['JobSat'].replace(lm.JobSat_dict)
    
    # replace strings with numerical entries in YearsCode column
    replace_dict = {'Less than 1 year': '0', 'More than 50 years': '51'}
    dft.replace(replace_dict, inplace=True)
    # change dtype to numeric
    dft['YearsCode'] = pd.to_numeric(dft['YearsCode'])
    
    # parse the multi columns
    multi_cols = ['PlatformWorkedWith', 'CollabToolsWorkedWith']
    dft = parse_multi_columns(dft, multi_cols)
    
    return dft
    
    
###########################################################################    

#### PREVIOUS VERSIONS OF THE DATA PREPARATION STEPS

# older version of the data pre-processing steps    
def preprocess_data_old(df):
    """
    Steps to preprocess the dataframe that contains the data coders.
    Steps: drop columns, binarize sertain columns.
    INPUT:
        df = preprocessed dataframe based on StackOverflow data,
             must contain an indicator column DevClass, which marks
             if a devloper works with data or not
    OUTPUT:
        df = processed dataframe
    """
    # get the data coders only
    df = df[df.DevClass == 'data_coder']
    # keep only columns of interest
    df = df[['MainBranch', 'ConvertedComp', 'EdLevel', 'Employment', 
             'JobSat', 'EdImpt','Learn', 'Overtime', 'OpSys', 'OrgSize', 
             'UndergradMajor', 'WorkWeekHrs']]
    # drop duplicates
    df.drop_duplicates(subset=None, keep='first', inplace=True)
    
    # bin numerical columns 
    
    # create the labels for work week hours
    cut_labels_week = ['<10', '10-20', '20-30', '30-40', '40-50', '>50']
    # define the bins for work week hours
    m1 = df.WorkWeekHrs.max()
    cut_bins_week = [0, 10, 20, 30, 40, 50, m1]
    # create the binned column and drop the old one
    binarize_col(df, 'WorkWeekHrs', 'WorkWeek_Bins', cut_labels_week, cut_bins_week)
    
    # create the labels for converted compensation
    cut_labels_comp = ['<10K', '10K-30K', '30K-50K', '50K-100K', '100K-200K', '>200K']
    # define the bins 
    m2 = df.ConvertedComp.max()
    cut_bins_comp = [0, 10000, 30000, 50000, 100000, 200000, m2]
    # binn the column and drop the old one
    binarize_col(df, 'ConvertedComp', 'Comp_Bins', cut_labels_comp, cut_bins_comp)
    
    return df

# older version of the data processing steps
def process_data_old(df, y_col):
    """
    Performs the following data processing steps:
        - split into predictors and labels
        - encode the discrete columns 
        - split into train and test sets
    INPUT:
        df = pandas dataframe
        y_col = column of labels
    OUTPUT:
        X_train, X_test = train and test dataframes of predictors
        y_train, y_test = train and test sets of labels
    
    """
    # create label column
    y = df[y_col]
    # create predictors dataframe
    X = df.drop(columns=y_col)
    # create dummies for X
    X = pd.get_dummies(X)
    # split the data in train and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # create an instance of the imputer, dataset has 'object' columns only
    imputer = KNNImputer(n_neighbors=5)
    # fit the imputer on the dataset
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns = X_train.columns)
    X_test = pd.DataFrame(imputer.fit_transform(X_test), columns = X_test.columns)
    return X_train, y_train, X_test, y_test

# imput the predictors, fit on train set and transform both sets
def impute_predictors_old(X_train, X_test):
    """
    Imputes the predictors, by creating an instance of the SimpleImputer.
    Fit the imputer on the train set, transform the train and the test sets.
    INPUT: 
        X_train, X_test = train and test dataframes
    OUTPUT:
        X_train_trans, X_test_trans = transformed train and test dataframes
    """
    imputer = SimpleImputer(strategy='constant', fill_value='missing')
    # fit the imputer on the train set only
    imputer.fit(X_train)
    X_train_trans = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns)
    X_test_trans = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    return X_train_trans, X_test_trans


    
