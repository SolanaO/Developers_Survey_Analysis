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
