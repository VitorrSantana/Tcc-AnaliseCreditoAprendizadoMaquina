import pandas as pd
import numpy as np

def remove_null_over(data,over_porc=0.5):
    for col in data.columns:
        porc_null = data[col].isna().sum()/data.shape[0]
        if porc_null > over_porc:
            print(f'Eliminando coluna {col} -> {porc_null} null')
            data = data.drop(col,axis=1) 
    return data

def calculate_iv(df, feature, target, epsilon=0.0001):
    table = pd.crosstab(df[feature], df[target])
    table['Prop_0'] = (table[0] + epsilon) / (table[0].sum() + epsilon)
    table['Prop_1'] = (table[1] + epsilon) / (table[1].sum() + epsilon)
    table['WOE'] = np.log(table['Prop_0'] / table['Prop_1'])
    table['IV'] = (table['Prop_0'] - table['Prop_1']) * table['WOE']
    return table['IV'].sum()

def show_iv(df,column_target='TARGET'):
    variables = list(df.drop(column_target,axis=1).columns)
    iv_s = []
    for variable in variables:
        iv = calculate_iv(df.fillna(-1), variable, column_target)
        iv_s.append(iv)
        print(f"{variable}: {iv}")
    
    iv_resultados = pd.DataFrame({'variavel':variables,'iv':iv_s})
    
    return iv_resultados

def create_vars_numeric(df,cols_drop = ['SK_ID_CURR','SK_ID_BUREAU'],col_identifier = 'SK_ID_CURR'):
    
    df = remove_null_over(df,0.16)
    base_final = pd.DataFrame({col_identifier:df[col_identifier].unique(),'qtd_vezes':df.groupby([col_identifier])[[col_identifier]].count()[col_identifier].values})

    columns  = list(df.drop(cols_drop,axis=1).select_dtypes(include=['int64','float64']))
    
    for col in columns:
        col_rename = [col_identifier,f'{col}_mean',f'{col}_max',f'{col}_min']
        variaveis = df.groupby([col_identifier])[col].agg(['mean','max','min']).reset_index()
        variaveis.columns = col_rename
        base_final = base_final.merge(variaveis,how='left',on=[col_identifier])    
        del col_rename
        del variaveis
    
    return base_final