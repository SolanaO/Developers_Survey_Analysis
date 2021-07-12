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

# merge two dataframes and add a clean list of entries 
def merge_df(df1, df2, old_col):
    """
    Merges two dataframes on a common column. 
    INPUT: 
        df1, df2 = the two pandas datframes to merge
        old_col = column shared by the two dataframes
    OUTPUT:
        df_temp = new dataframe that contains the common column and all the other 
                  columns in df1 and df2
    """
    
    df_temp = pd.merge(df1, df2, on = old_col)
    #df_temp[new_col] = col_list
    return df_temp