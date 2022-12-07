import pandas as pd

def separate(df:pd.core.frame.DataFrame,start_train:str,end_train:str,end_test:str='end'):
    """
    Sépare notre DataFrame initial en deux DataFrame
    df : DataFrame
    Contient en index les dates étudiées
    start_train : str
    date de début pour notre DataFrame d'entrainement
    end_train : str
    date de fin pour notre DataFrame d'entrainement
    end_test : None
    date de fin pour notre DataFrame d'entrainement
    """
    if end_test == 'end':
        df_test = df[(df.index > end_train)]
    else:
        df_test = df[(df.index > end_train) & (df.index <= end_test)]
    df_train = df[(df.index > start_train) & (df.index <= end_train)]
    return df_train,df_test
