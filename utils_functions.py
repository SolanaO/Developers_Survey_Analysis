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

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline 

# print to a file a list of packages and their versions used in this jupyter notebook
def package_requirements():
    "Creates a text file with the packages list and their versions."
    with open('packageinfo.txt', 'w') as f:
        print('Python {}'.format(sys.version), file=f)
        print('Packages:', file=f)
        print('\n'.join(f'{m.__name__} {m.__version__}' 
                    for m in globals().values() if getattr(m, '__version__', None)),file=f)
    return()


# create a dictionary with shorter strings for education levels
new_edLevel = {'Master’s degree (M.A., M.S., M.Eng., MBA, etc.)': 'Master’s degree',
 'Bachelor’s degree (B.A., B.S., B.Eng., etc.)': 'Bachelor’s degree',
 'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)':  'Secondary school',
 'Professional degree (JD, MD, etc.)': 'Professional degree',
 'Some college/university study without earning a degree': 'College study/no degree',
 'Associate degree (A.A., A.S., etc.)' : 'Associate degree',
 'Other doctoral degree (Ph.D., Ed.D., etc.)': 'Other doctoral degree',
 'I never completed any formal education' : 'No formal education'}
 
# create a dictionary with shorter descriptions for the undegraduate majors
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
    'WebframeWorkedWith',  #'CollabToolsWorkedWith',                                                 

    # questions not relevant to our goal
    'JobHunt', 'JobHuntResearch', 'Stuck',
    'PurchaseResearch', 'PurchaseWhat', 
    'Stuck', 'PurpleLink',
    'OffTopic', 'OtherComms',
    'JobFactors', 'JobSeek']                                                            

JobSat_dict =  {'Very dissatisfied': 1, 'Slightly dissatisfied': 2,
               'Neither satisfied nor dissatisfied': 3, 
               'Slightly satisfied': 4, 'Very satisfied': 5}

num_cols = [ 'ConvertedComp','WorkWeekHrs', 'YearsCode']

multi_cols = ['CollabToolsWorkedWith', 'PlatformWorkedWith']

cat_cols = [ 'DevType','EdLevel', 'DevOps',  'OnboardGood','OpSys', 'UndergradMajor']

ordinal_cols = ['DevOpsImpt', 'EdImpt', 'Learn', 'Overtime','OrgSize']




####
# pre-processing steps, useful mostly to render better looking plots
def data_prep(df):
    # create a copy of the data
    df1 = df.copy()
    # drop the NEW prefix in some of the columns' names
    df1.columns = [col.replace('NEW', '') for col in df1.columns]
    # create column DevClass, entry data_coder or other_coder, based on DevType contains data or not
    df1['DevClass'] = np.where(df1["DevType"].str.contains("Data ", na = False), 'data_coder','other_coder')
    # replace strings in several columns
    df1.replace(new_EdLevel, inplace=True)
    df1.replace(new_UndergradMajor, inplace=True)
    df1.replace(new_EdImpt, inplace=True)
    return df1

####
# function to prepare a column with multiple choice answers
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

# find individual answers from a multiple choice aswers question
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

# steps for removing unnecessary data
def remove_clean_data(dft):
    """
    Steps to remove unnecessary rows and columns.
    """
    
    # rewrite entries in 'DevType' column as strings to replicate rows
    dft = explode_col(dft, 'DevType')
    # retain only those rows that contain data coders
    dft = dft.loc[dft.DevType.str.contains('Data ', na=False)]
    # retain only the employed data developers
    dft = dft[dft['Employment'] != 'Not employed, but looking for work']
    # create a list of main branch choices
    main_choices = dft.MainBranch.value_counts().index.to_list()
    # retain rows where MainBranch contains the data professionals
    dft = dft[dft.MainBranch.isin(main_choices[:2])]
    # drop all the columns in the list
    dft.drop(columns=cols_del, inplace=True)
    
    # drop rows with missing JobSat
    dft.dropna(subset=['JobSat'], inplace=True)
    #  encode the 'JobSat' data to numerical values
    #dft['JobSat'] = dft['JobSat'].replace(JobSat_dict)
    
    # replace strings with numerical entries
    replace_dict = {'Less than 1 year': '0', 'More than 50 years': '51'}
    dft.replace(replace_dict, inplace=True)
    # change dtype to numeric
    dft['YearsCode'] = pd.to_numeric(dft['YearsCode'])
    
    # rewrite entries in multi_cols as strings and replicate rows 
    for col in multi_cols:
        dft = explode_col(dft, col)
    
    # drop duplicate rows
    dft.drop_duplicates(subset=None, keep='first', inplace=True)

    return dft


# from https://towardsdatascience.com/custom-transformers-and-ml-data-pipelines-with-python-20ea2a7adb65
# custom transformer that extracts columns passed as argument to its constructor 

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
    
    

    
    
def DataFrameSimpleImputer(df):
    #create an instance of the imputer
    cat_imputer = SimpleImputer(strategy='constant', fill_value='missing')
    # impute the missing categorical values
    df_cat_imp = pd.DataFrame(cat_imputer.fit_transform(df), columns=df.columns)
    return df_cat_imp

    
############################################################

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
    
def preprocess_data_old(df):
    # get the data coders only
    df = df[df.DevClass == 'data_coder']
    # keep only columns of interest
    df = df[['MainBranch', 'ConvertedComp', 'EdLevel', 'Employment', 'JobSat', 'EdImpt','Learn', 'Overtime', 'OpSys', 'OrgSize', 'UndergradMajor', 'WorkWeekHrs']]
    # drop duplicates
    df.drop_duplicates(subset=None, keep='first', inplace=True)
    
    # binarize numerical columns work week hours
    
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
    
def process_data_old(df, y_col):
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


# feature selection
def select_features(X_train, y_train, X_test):
	fs = SelectKBest(score_func=mutual_info_classif, k=20)
	fs.fit(X_train, y_train)
	X_train_fs = fs.transform(X_train)
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs
# plot the scores
#plt.bar([i for i in range(len(fs.scores_))], fs.scores_)

def impute_predictors(X_train, X_test):
    imputer = SimpleImputer(strategy='constant', fill_value='missing')
    # fit the imputer on the train set only
    imputer.fit(X_train)
    X_train_trans = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns)
    X_test_trans = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    return X_train_trans, X_test_trans


    
