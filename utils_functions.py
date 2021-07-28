# import neccessary packages and libraries
import os
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# set a theme for seaborn
sns.set_theme()

from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer

from sklearn import (
    ensemble,
    preprocessing,
    tree,
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



# print to a file a list of packages and their versions used in this jupyter notebook
def package_requirements():
    "Creates a text file with the packages list and their versions."
    with open('packageinfo.txt', 'w') as f:
        print('Python {}'.format(sys.version), file=f)
        print('Packages:', file=f)
        print('\n'.join(f'{m.__name__} {m.__version__}' 
                    for m in globals().values() if getattr(m, '__version__', None)),file=f)
    return()

# create a function that will count the type of strings 
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
    
def preprocess_data(df):
    # get the data coders only
    df = df[df.DevClass == 'data_coder']
    # keep only columns of interest
    #df = df[['MainBranch', 'ConvertedComp', 'EdLevel', 'Employment', 'JobSat', 'EdImpt','Learn', 'Overtime', 'OpSys', 'OrgSize', 'UndergradMajor', 'WorkWeekHrs']]
   # df = df[['ConvertedComp','Country', 'EdLevel', 'JobSat', 'EdImpt','Overtime', 'OrgSize', #'UndergradMajor', 'WorkWeekHrs']]
    # drop duplicates
    df.drop_duplicates(subset=None, keep='first', inplace=True)
    
    # binarize numerical columns work wek hours
    
    # create the labels for work week hours
    cut_labels_week = ['<10', '10-20', '20-30', '30-40', '40-50', '>50']
    # define the bins for work wek hours
    m1 = df.WorkWeekHrs.max()
    cut_bins_week = [0, 10, 20, 30, 40, 50, m1]
    # create the binnarized column and drop the old one
    binarize_col(df, 'WorkWeekHrs', 'WorkWeek_Bins', cut_labels_week, cut_bins_week)
    
    # create the labels for converted compensation
    cut_labels_comp = ['<10K', '10K-30K', '30K-50K', '50K-100K', '100K-200K', '>200K']
    # define the bins 
    m2 = df.ConvertedComp.max()
    cut_bins_comp = [0, 10000, 30000, 50000, 100000, 200000, m2]
    # binnarize the column and drop the old one
    binarize_col(df, 'ConvertedComp', 'Comp_Bins', cut_labels_comp, cut_bins_comp)
    return df
    
def process_data(df, y_col):
    # create label column
    y = df[y_col]
    # create predictors dataframe
    X = df.drop(columns=y_col)
    # create dummies for X
    X = pd.get_dummies(X)
    # split the data in train and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # create an instance of the imputer
    imputer = KNNImputer(n_neighbors=5)
    # fit the imputer on the dataset
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns = X_train.columns)
    X_test = pd.DataFrame(imputer.fit_transform(X_test), columns = X_test.columns)
    return X_train, y_train, X_test, y_test


    
