# import neccessary packages and libraries
from collections import defaultdict
import numpy as np
import pandas as pd

# create a function that will count the type of strings 
def counts_strings(strings_list, dframe, incol):
    """
    Counts the number of occurences of a given string among
    the text entries of a column in a pandas dataframe.
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
    return new_df

def binarize_col(df, old_col, new_col, cut_labels, cut_bins):
    """
    Discretizes the values in a column to a number
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
    
    
def preprocess_data(df1):
    # get the data coders only
    df1 = df1[df1.DevClass == 'data_coder']
    # keep only columns of interest
    df1 = df1['MainBranch', 'ConvertedComp', 'EdLevel', 'Employment',
       'JobSat', 'EdImpt','Learn', 'Overtime', 'OpSys', 'OrgSize', 
       'UndergradMajor', 'WorkWeekHrs']
    # drop duplicates
    df1.drop_duplicates(subset=None, keep='first', inplace=True)
    
    # binarize numerical columns work wek hours
    
    # create the labels for work week hours
    cut_labels_week = ['<10', '10-20', '20-30', '30-40', '40-50', '>50']
    # define the bins for work wek hours
    m1 = df1.WorkWeekHrs.max()
    cut_bins_week = [0, 10, 20, 30, 40, 50, m1]
    # create the binnarized column and drop the old one
    binarize_col(df1, 'WorkWeekHrs', 'WorkWeek_Bins', cut_labels_week, cut_bins_week)
    
    # create the labels for converted compensation
    cut_labels_comp = ['<10K', '10K-30K', '30K-50K', '50K-100K', '100K-200K', '>200K']
    # define the bins 
    m2 = df1.ConvertedComp.max()
    cut_bins_comp = [0, 10000, 30000, 50000, 100000, 200000, m]
    # binnarize the column and drop the old one
    binarize_col(df1, 'ConvertedComp', 'Comp_Bins', cut_labels_comp, cut_bins_comp)
    returnd df1
    
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
    X_train_trans = pd.DataFrame(imputer.fit_transform(X_train), columns = X_train.columns)
    return X_train_trans, y_train, X_test, y_test


    
