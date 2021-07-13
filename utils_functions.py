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

# what other DevTypes do the data coders choose
data_coders_groups = uf.counts_strings(dev_choice_short,data_coders,'DevType')
# rename the columns
data_coders_groups.rename(columns = {'index':'DevTypes', 0:'counts_data'}, inplace=True)
# add a column with percentages computed with respect to all data coders
data_coders_groups['perc_data'] = (data_coders_groups.counts_data/data_coders.shape[0] * 100).round(2)
# show the results ordered by counts and percentages
data_coders_groups.sort_values('counts_data', ascending=False).head(2)