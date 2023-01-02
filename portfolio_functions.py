import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def separate(df:pd.DataFrame, start_train:str,end_train:str,end_test:str='end'):
    """
    Divide the DataFrame into two DataFrame, one for training and one for testing.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to divide in two.
    start_train: str
        Date of the beginning of the training DataFrame.
    end_train: str
        Date of the end of the training DataFrame.
    end_test: str
        Date of the end of the testing DataFrame.
    
    Returns
    -------
    two pd.DataFrame
        Two DataFrame, one for training and one for testing.
    """

    if end_test == 'end':
        df_test = df[(df.index > end_train)]
    else:
        df_test = df[(df.index > end_train) & (df.index <= end_test)]
    df_train = df[(df.index > start_train) & (df.index <= end_train)]

    return df_train,df_test

def supr_assets(df:pd.DataFrame,show:bool=False):
    """
    Delete the columns entirely composed of 0.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to clean.
    show : bool
        To show or not the assets that will be suppressed.
    
    Returns
    -------
    pd.DataFrame
        DataFrame without the columns entirely composed of 0.
    """

    supr = []
    for column in df.columns:
        if (df[column] == 0.).all() or annu_rend(df[column])<0:
            supr.append(column)
    if show:
        print(f'The following assets will be suppressed:')
        print(*supr)

    return df.drop(supr,axis=1)

def dynamic_df(df:pd.DataFrame,period_per_year:int=4):
    """ 
    Divide the DataFrame in several DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to divide.
    period_per_year : int
        Number of period per year to create from the DataFrame.
    
    Returns
    -------
    dict
        Dictionary of DataFrame with the key being the period_per_year number.
    """

    nb_years = df.shape[0]//252
    period_per_year = nb_years*period_per_year

    df_dict = {}
    for i in range(period_per_year):
        start = i*df.shape[0]//period_per_year
        end = (i+1)*df.shape[0]//period_per_year
        df_slice = supr_assets(df.iloc[start:end,:])
        df_dict[i+1] = df_slice

    return df_dict

def absolute_weights(weights:dict):
    """
    Calculate the absolute weights of the portfolio.

    Parameters
    ----------
    weights : dict
        Dictionary of the weights of the portfolio.
    
    Returns
    -------
    dict
        Dictionary of the absolute weights of the portfolio.
    """

    abs_weights = {}
    for i in weights.keys():
        abs_weights[i] = abs(weights[i])
    return abs_weights

def plot_weights(weights:dict):
    """
    Plot the weights of the portfolio.

    Parameters
    ----------
    weights : dict
        Dictionary of the weights of the portfolio.
    """

    plt.figure(figsize=(10,10))
    abs_weights = absolute_weights(weights)
    sort = {k: v for k, v in sorted(abs_weights.items(), key=lambda item: item[1],reverse=True)}
    
    def func(pct, values):
        if pct < 2:
            return ""
        return "{:.1f}%".format(pct)

    plt.pie(sort.values(),autopct=lambda pct: func(pct,sort.values()))
    plt.axis('equal')
    plt.title('Weights of the portfolio')
    plt.legend(labels=sort.keys(),borderpad=1,fancybox=True,framealpha=1,prop={'size': 9})
    plt.show()

def annu_rend(portfolio:pd.Series,period_per_year:int=252,show:bool=False):
    """
    Calculate the annualized return of the series considered.

    Parameters
    ----------
    portfolio : pd.Series
        Series of the returns considered.
    period_per_year : int
        Period considered to calculate the annualized return.
    show : bool
        To show or not the result.

    Returns
    -------
    float
        Annualized return of the series considered.
    """

    nb_days = portfolio.shape[0]
    final_return = (portfolio+1).prod()
    annu_return = final_return**(period_per_year/nb_days) - 1
    if show:
        print(f'Annualized return on the period: {round(annu_return*100,3)}%.')
    
    return annu_return

def annu_rend_df(df:pd.DataFrame,period_per_year:int=252,arr:bool=False,show:bool=False):
    """
    Calculated annualized return for each asset considered in the DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame composed of the returns of the assets.
    period_per_year : int
        Period considered to calculate the annualized return.
    arr : bool
        If false the function returns a dictionary otherwise an array.
    show : bool
        To show or not the result of each annualized return.

    Returns
    -------
    dict or array
        Dictionary of annualized returns of the assets or array of annualized returns of the assets.
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
    Calculate the covariance matrix of the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to calculate the covariance matrix.
    
    Returns
    -------
    pd.DataFrame
        Covariance matrix of the DataFrame.
    """

    # So useless
    return df.cov()

def annu_vol(portfolio:pd.Series,period_per_year:int=252,show=False):
    """
    Calculate the annualized volatility of the series considered.

    Parameters
    ----------
    portfolio : pd.Series
        Series of the returns considered.
    period_per_year : int
        Period considered to calculate the annualized return.
    show : bool
        To show or not the result.
    
    Returns
    -------
    int
        Annualized volatility of the series considered.
    """

    nb_days = portfolio.shape[0]
    final_vol = portfolio.std()
    annu_vol = final_vol*(period_per_year**0.5)
    if show:
        print(f'Annualized volatility on the period: {round(annu_vol*100,3)}%.')
    return annu_vol

def portfolio_var(weights,cov:pd.DataFrame,period_per_year:int=252,show:bool=False):
    """
    Calculate the variance of the portfolio from the covariance matrix and the weights.
    
    Parameters
    ----------
    cov : pd.DataFrame
        Covariance matrix of the DataFrame.
    weights :
        Weights considered for each assets.
    period_per_year : int
        Period considered to calculate the annualized variance.
    show : bool
        To show or not the result.
    
    Returns
    -------
    int
        Annualized variance of the portfolio.
    """

    annu_var = (weights@cov@weights)*period_per_year
    if show:
        print(f'Annualized variance on the period: {round(annu_var*100,3)}%.')
    return annu_var

def portfolio_vol(weights,cov:pd.DataFrame,period_per_year:int=252,show:bool=False):
    """
    Calculate the annualized volatility of the portfolio from the covariance matrix and the weights.
    
    Parameters
    ----------
    cov : pd.DataFrame
        Covariance matrix of the DataFrame.
    weights :
        Weights considered for each assets.
    period_per_year : int
        Period considered to calculate the annualized volatility.
    show : bool
        To show or not the result.
    
    Returns
    -------
    int
        Annualized volatility of the portfolio.
    """

    annu_vol = (weights@(cov*252)@weights)**0.5
    if show:
        print(f'Annualized volatility on the period: {round(annu_vol*100,3)}%.')
    return annu_vol

def portfolio_rend(weights,rends:pd.DataFrame,period_per_year:int=252,show:bool=False):
    """
    Calculate the annualized return of the portfolio from the returns and the weights.
    
    Parameters
    ----------
    rend : pd.DataFrame
        Returns of the DataFrame.
    weights :
        Weights considered for each assets.
    period_per_year : int
        Period considered to calculate the annualized return.
    show : bool
        To show or not the result.
    
    Returns
    -------
    int
        Annualized return of the portfolio.
    """

    rendements = pd.Series((rends*weights).sum(axis=1),name="Portfolio")
    rendement = annu_rend(rendements,period_per_year=period_per_year,show=show)
    return rendement

def gmv_portfolio(rends:pd.DataFrame,cov:pd.DataFrame,show:bool=False):
    """
    Calculate the annualized return and volatility of the global minimum variance portfolio from the returns and the covariance matrix.
    
    Parameters
    ----------
    rends : pd.DataFrame
        Returns of the DataFrame.
    cov : pd.DataFrame
        Covariance matrix of the DataFrame.
    show : bool
        To show or not the result.
    
    Returns
    -------
    array, float, float
        Weights of the assets, annualized return of the portfolio, annualized volatility of the portfolio.
    """

    weights = np.dot(np.linalg.inv(cov*252),np.ones(cov.shape[0]))/(np.dot(np.ones(cov.shape[0]),np.dot(np.linalg.inv(cov*252),np.ones(cov.shape[0]))))
    annual_return = portfolio_rend(weights,rends,show=show) 
    annual_volatility = portfolio_vol(weights,cov,show=show)
    return weights,annual_return,annual_volatility

def opt_mean_variance(rends:pd.DataFrame,cov:pd.DataFrame,obj_rend:float,risk_free=None,period_per_year:int=252,show:bool=False):
    """
    Calculate the weights of the portfolio for a given objective return.
    
    Parameters
    ----------
    rends : pd.DataFrame
        Returns of the DataFrame.
    cov : pd.DataFrame
        Covariance matrix of the DataFrame.
    obj_rend : float
        Annualized return of the portfolio.
    risk_free : float or None
        Annualized risk free return.
    period_per_year : int
        Period considered to calculate the annualized return.
    show : bool
        To show or not the result.
    
    Returns
    -------
    array, float, float
        Weights of the assets, annualized return of the portfolio, annualized volatility of the portfolio.
    """
    if risk_free != None:
        rends = rends.copy()
        rends["Risk Free"] = (1+risk_free)**(1/period_per_year)-1
        cov = rends.cov()
    n_assets = rends.shape[1]
    init_weights = np.repeat(1/n_assets, n_assets)

    # Limites pour les poids
    bounds = ((-1., 1.),) * n_assets

    # Définition des contraintes
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1}

    if risk_free == None:
        return_is_target = {'type': 'eq',
                            'args': (rends,),
                            'fun': lambda weights,rends: obj_rend - portfolio_rend(weights,rends)}
    else :
        return_is_target = {'type': 'eq',
                            'args': (rends,risk_free,),
                            'fun': lambda weights,rends,risk_free: obj_rend - portfolio_rend(weights,rends) - (1-np.sum(weights))*risk_free}
    
    # Fonction d'optimisation
    weights = minimize(portfolio_vol,init_weights,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)

    w = {rends.columns[i]:round(weights.x[i],3) for i in range(rends.shape[1])}
    r = portfolio_rend(weights.x,rends,show=show)
    v = portfolio_vol(weights.x,cov,show=show)
    return w,r,v

def transactions_costs(initial,final,prices:pd.DataFrame):
    """
    Calculate the transactions costs of a portfolio.
    
    Parameters
    ----------
    initial : list or array
        Initial weights of the assets.
    final : list or array
        Final weights of the assets.
    prices : pd.DataFrame
        Prices of the assets.
    
    Returns
    -------
    float
        Transactions costs.
    """
    tc = 0
    initial,final = np.array(initial),np.array(final)
    for i in range(initial.shape[0]):
        tc += (final[i]-initial[i])*prices.iloc[i]
    return tc

def opt_mean_variance_tc(rends:pd.DataFrame,last_prices:pd.DataFrame,cov:pd.DataFrame,obj_rend:float,last_weights=None,tc:float=0.001,period_per_year:int=252,show:bool=False):
    """
    Calculate the weights of the portfolio for a given objective return with transactions costs.
    
    Parameters
    ----------
    rends : pd.DataFrame
        Returns of the DataFrame.
    prices : pd.DataFrame
        Prices of the assets.
    last_weights : dict or None
        Weights of the assets at the previous period.
    tc : float
        Transaction costs.
    cov : pd.DataFrame
        Covariance matrix of the DataFrame.
    obj_rend : float
        Annualized return of the portfolio.
    period_per_year : int
        Period considered to calculate the annualized return.
    show : bool
        To show or not the result.
    
    Returns
    -------
    array, float, float
        Weights of the assets, annualized return of the portfolio, annualized volatility of the portfolio.
    """

    n_assets = rends.shape[1]
    if last_weights == None:
        # We have to buy the assets
        init_weights = np.repeat(0, n_assets)
    else:
        init_weights = []
        for asset in rends.columns:
            if asset in last_weights.keys():
                init_weights.append(last_weights[asset])
            else:
                init_weights.append(0)
        init_weights = np.array(init_weights)

    # Limites pour les poids
    bounds = ((-1., 1.),) * n_assets

    # Définition des contraintes
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1}

    return_is_target = {'type': 'eq',
                        'args': (rends,last_prices,init_weights,tc,),
                        'fun': lambda weights,rends,last_prices,init_weights,tc: obj_rend - portfolio_rend(weights,rends) - tc*transactions_costs(init_weights,weights,last_prices)}
    
    # Fonction d'optimisation
    weights = minimize(portfolio_vol,init_weights,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)

    w = {rends.columns[i]:round(weights.x[i],3) for i in range(rends.shape[1])}
    r = portfolio_rend(weights.x,rends,show=show)
    v = portfolio_vol(weights.x,cov,show=show)
    print(f"Transactions costs for the rebalancing: {round(transactions_costs(init_weights,weights.x,last_prices),3)}")
    return w,r,v

def efficient_frontier(rends:pd.DataFrame,cov:pd.DataFrame,min:float,max:float,number:int=25,risk_free=None,plot:bool=False):
    """
    Calculate the efficient frontier of a portfolio.
    
    Parameters
    ----------
    rends : pd.DataFrame
        Returns of the DataFrame.
    cov : pd.DataFrame
        Covariance matrix of the DataFrame.
    min : float
        Start of the range of returns.
    max : float
        End of the range of returns.
    number : int
        Number of points to consider.
    risk_free : float or None
        Risk free rate.
    plot : bool
        To plot or not the efficient frontier.
    
    Returns
    -------
    array
        Efficient frontier of the portfolio without the risk free rate or both of them.
    """
    gmv_w,gmv_r,gmv_vol = gmv_portfolio(rends,cov)
    r_eff,vol_eff = np.linspace(gmv_r,max,number),[]
    eff = []
    if risk_free != None:
        r_eff_rf,vol_eff_rf = np.linspace(risk_free,max,number),[]
        eff_rf = []

    if plot:
        r_eff_low,vol_eff_low = np.linspace(min,gmv_r,number),[]
    for i in range(r_eff.shape[0]):
        w,r,v = opt_mean_variance(rends,cov,r_eff[i])
        vol_eff.append(v)
        eff.append([r_eff[i],v])
        if risk_free != None:
            w,r,v = opt_mean_variance(rends,cov,r_eff_rf[i],risk_free=risk_free)
            vol_eff_rf.append(v)
            eff_rf.append([r_eff_rf[i],v])
        if plot:
            w,r,v = opt_mean_variance(rends,cov,r_eff_low[i])
            vol_eff_low.append(v)
    
    
    if plot:
        plt.plot(vol_eff,r_eff,"black",label="Efficient Frontier")
        if risk_free != None:
            plt.plot(vol_eff_rf,r_eff_rf,"black",label="Capital Market Line",linestyle='dotted')
        plt.plot(vol_eff_low,r_eff_low,"black",label="Low Frontier",linestyle="-.")
        plt.scatter(gmv_vol,gmv_r,color="black",label="GMV Portfolio",marker="o",s=50)
        plt.legend()
        plt.xlabel("Annualized volatility")
        plt.ylabel("Annualized return")
        plt.show()
    
    if risk_free != None:
        return eff,eff_rf
    else:
        return eff

