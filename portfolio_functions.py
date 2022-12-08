import pandas as pd
import numpy as np
from scipy.optimize import minimize

def separate(df:pd.DataFrame,start_train:str,end_train:str,end_test:str='end'):
    """
    Sépare notre DataFrame initial en deux DataFrame\n
    df : pd.DataFrame\n
    Contient en index les dates étudiées\n
    start_train : str\n
    Date de début pour notre DataFrame d'entrainement\n
    end_train : str\n
    Date de fin pour notre DataFrame d'entrainement\n
    end_test : None\n
    Date de fin pour notre DataFrame d'entrainement
    """
    if end_test == 'end':
        df_test = df[(df.index > end_train)]
    else:
        df_test = df[(df.index > end_train) & (df.index <= end_test)]
    df_train = df[(df.index > start_train) & (df.index <= end_train)]
    return df_train,df_test

def supr_assets(df:pd.DataFrame,show:bool=False):
    """
    Supprime les colonnes entièrement composées de 0. et renvoie un DataFrame sans ces colonnes
    df : pd.DataFrame
    DataFrame contenant les rendements.
    """
    supr = []
    for column in df.columns:
        if (df[column] == 0.).all() or annu_rend(df[column])<0:
            supr.append(column)
    if show:
        print(f'Les colonnes supprimées sont :')
        print(*supr)
    return df.drop(supr,axis=1)

def annu_rend(portfolio:pd.Series,period_per_year:int=252,show:bool=False):
    """
    Calcule le rendement annualisé de la série considérée\n
    portfolio : pd.Series\n
    Series des rendements pris en compte\n
    period_per_year : int\n
    Période considérée pour calculer le rendement annualisé\n
    show : bool\n
    Pour afficher ou non le résultat
    """
    nb_days = portfolio.shape[0]
    final_return = (portfolio+1).prod()
    annu_return = final_return**(period_per_year/nb_days) - 1
    if show:
        print(f'On obtient un rendement annualisé sur la période de {round(annu_return*100,3)}%.')
    return annu_return

def annu_rend_df(df:pd.DataFrame,period_per_year:int=252,arr:bool=False,show:bool=False):
    """
    Calcule le rendement annualisé pour chaque asset considéré dans le DataFrame et retourne un dictionnaire\n
    df : pd.DataFrame\n
    DataFrame avec les séries de chaque asset\n
    period_per_year : int\n
    Période considérée pour calculer le rendement annualisé\n
    show : bool\n
    Pour afficher ou non le résultat\n
    arr : bool\n
    Si faux la fonction retourne un dictionnaire sinon un array
    """
    rendements = {}
    for i in df.columns:
        rendements[i] = round(annu_rend(df[i]),5)
    if not arr:
        return rendements
    else:
        return np.array(list(rendements.values()))

def cov(df:pd.DataFrame):
    """
    Calcule la matrice de covariance de nos assets sélectionnés\n
    df : pd.DataFrame\n
    DataFrame composé des rendements de nos assets\n
    """
    # Ici il n'y a pas de racine puisqu'on travaille avec la variance et non l'écart-type
    return df.cov()

def annu_vol(portfolio:pd.Series,period_per_year:int=252,show=False):
    """
    Calcule la volatilité annualisée de la série\n
    portfolio : pd.Series\n
    Series des rendements pris en compte\n
    period_per_year : int\n
    Période considérée pour calculer la volatilité annualisée\n
    show : bool\n
    Pour afficher ou non le résultat
    """
    nb_days = portfolio.shape[0]
    final_vol = portfolio.std()
    annu_vol = final_vol*(period_per_year**0.5)
    if show:
        print(f'On obtient une volatilité annualisée sur la période de {round(annu_vol*100,3)}%.')
    return annu_vol

def portfolio_var(weights,cov:pd.DataFrame,period_per_year:int=252,show:bool=False):
    """
    Calcule la variance à partir d'un matrice de covariance et des poids\n
    cov : pd.DataFrame\n
    Matrice de covariance\n
    weights\n
    Poids considérés pour chaque assets\n
    period_per_year : int\n
    Période considérée pour calculer la volatilité annualisée\n
    show : bool\n
    Pour afficher ou non le résultat
    """
    annu_var = (weights@cov@weights)*period_per_year
    if show:
        print(f'On obtient une variance sur la période de {round(annu_var*100,3)}%.')
    return annu_var

def portfolio_vol(weights,cov:pd.DataFrame,period_per_year:int=252,show:bool=False):
    """
    Calcule la volatilité à partir d'un matrice de covariance annualisée et des poids\n
    cov : pd.DataFrame\n
    Matrice de covariance\n
    weights\n
    Poids considérés pour chaque assets\n
    period_per_year : int\n
    Période considérée pour calculer la volatilité annualisée\n
    show : bool\n
    Pour afficher ou non le résultat
    """
    annu_vol = (weights@(cov*252)@weights)**0.5
    if show:
        print(f'On obtient une volatilité annualisée sur la période de {round(annu_vol*100,3)}%.')
    return annu_vol

def portfolio_rend(weights,rends:pd.DataFrame,show:bool=False):
    """
    Renvoie les rendements du portefeuille selon les assets considérés et les poids associés\n
    annu_rends\n
    DataFrame contenant les rendements annualisés de chaque asset\n
    weights\n
    Liste des poids pour chaque asset\n
    show : bool\n
    Pour afficher ou non le résultat
    """
    rendements = pd.Series((rends*weights).sum(axis=1),name="Portfolio")
    rendement = annu_rend(rendements,show=show)
    return rendement

def opt_mean_variance(rends:pd.DataFrame,cov:pd.DataFrame,obj_rend:float,show:bool=False):
    """
    Renvoie un dictionnaire des poids associés à chaque asset\n
    rends : pd.DataFrame\n
    DataFrame contenant les rendements de chaque assets\n
    cov : pd.DataFrame\n
    Matrice de covariance annualisé des prix\n
    obj_rend : float\n
    Représente le rendement annualisé souhaité 
    """
    n_assets = rends.shape[1]
    init_weights = np.repeat(1/n_assets,n_assets)

    # Limites pour les poids
    bounds = ((0.0, 1.0),) * n_assets

    # Définition des contraintes
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1}

    return_is_target = {'type': 'eq',
                        'args': (rends,),
                        'fun': lambda weights,rends: obj_rend - portfolio_rend(weights,rends)}
    
    # Fonction d'optimisation
    weights = minimize(portfolio_var,init_weights,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)

    W = {rends.columns[i]:round(weights.x[i],3) for i in range(rends.shape[1])}
    r = portfolio_rend(weights.x,rends,show=show)
    v = portfolio_vol(weights.x,cov,show=show)
    return W,r,v