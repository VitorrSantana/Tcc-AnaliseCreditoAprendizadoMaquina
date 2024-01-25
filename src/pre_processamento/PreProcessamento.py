import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class PreProcessamento:
    def __init__(self,data,columns=None,option='OneHotEncoder'):
        self.data  = data
        self.option = option
        if columns == None:
            self.columns = self.get_columns_objetct()
        self.method = None
        if option=='OneHotEncoder':
            self.modelo = OneHotEncoder(handle_unknown='ignore')

    def apply_method_encoding(self):
        dataframe_encode = pd.DataFrame(self.modelo.fit_transform(self.data[self.columns]).toarray(),columns = self.modelo.get_feature_names_out())

        return pd.concat([self.data,dataframe_encode],axis=1)

    def get_columns_objetct(self):
        return list(self.data.select_dtypes(include=[object]))