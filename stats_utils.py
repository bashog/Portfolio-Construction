import pandas as pd
import numpy as np
import scipy.stats as stats

### normality tests ###

def skewness(r):
    return r.skew()


def kurtosis(r):
    return r.kurtosis()


def is_normal(r, level=0.05):
    return stats.jarque_bera(r)[1] > level


### risk measures ###

def annualize_rets(r, periods_per_year=252):
    ''' annualize a set of returns with a given number of periods per year '''
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods) - 1


def annualize_vol(r, periods_per_year=252):
    ''' annualize the vol of a set of returns with a given number of periods per year '''
    return r.std()*(periods_per_year**0.5)


def sharpe_ratio(r, riskfree_rate, periods_per_year=252):
    ''' calculate the annualized sharpe ratio of a set of returns '''
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year) - 1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol


def max_drawdown(r):
    ''' calculate the max drawdown of a set of returns '''
    wealth_index = 1000*(1+r).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return drawdowns.min()


def calmar_ratio(r, periods_per_year=252):
    ''' calculate the calmar ratio of a set of returns '''
    ann_ret = annualize_rets(r, periods_per_year)
    dd = max_drawdown(r)
    return ann_ret/dd

def semi_deviation(r):
    ''' calculate the semi-deviation of negative returns '''
    is_negative = r < 0
    return r[is_negative].std(ddof=0)

def var_historic(r, level=5):
    ''' calculate the historic Value at Risk at a given level '''
    return -np.percentile(r, level)

def cvar_historic(r, level=5):
    ''' calculate the historic Conditional Value at Risk at a given level '''
    is_beyond = r <= -var_historic(r, level=level)
    return -r[is_beyond].mean()

def var_cornish_fisher(r, level=5, modified=True):
    ''' calculate the Parametric Gaussian VaR of a Series or DataFrame '''
    # if the modified version is desired, scale the z score appropriately
    z = stats.norm.ppf(level/100)
    # modify the z score based on observed skewness and kurtosis
    s = skewness(r)
    k = kurtosis(r)
    z = (z +
            (z**2 - 1)*s/6 +
            (z**3 - 3*z)*(k-3)/24 -
            (2*z**3 - 5*z)*(s**2)/36
        )
    return -(r.mean() + z*r.std(ddof=0))

def summary_stats(r, riskfree_rate=0.02, periods_per_year=252):
    ''' summarize a set of returns '''
    skew = skewness(r)
    kurt = kurtosis(r)
    normal = is_normal(r)

    ann_ret = annualize_rets(r, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    ann_sr = sharpe_ratio(r, riskfree_rate, periods_per_year)
    dd = max_drawdown(r)
    calmar = calmar_ratio(r, periods_per_year)
    semi_dev = semi_deviation(r)
    var_hist = var_historic(r)
    cvar_hist = cvar_historic(r)
    var_cornish = var_cornish_fisher(r)

    col = ['Skewness', 'Kurtosis', 'Normal', 'Annualized Return', 
    'Annualized Vol', 'Sharpe Ratio', 'Max Drawdown', 'Calmar Ratio', 
    'Semi Deviation', 'Historic VaR', 'Historic CVaR', 'Cornish-Fisher VaR']
    
    return pd.DataFrame(col = col, data = [skew, kurt, normal, ann_ret, ann_vol, ann_sr, dd, calmar, semi_dev, var_hist, cvar_hist, var_cornish])

def summary_portfolio(returns, weights, riskfree_rate=0.02, periods_per_year=252):
    ''' summarize stats of a portfolio of returns '''
    summary = None
    for col in returns.columns:
        summary_col = summary_stats(returns[col], riskfree_rate, periods_per_year)
        summary = pd.concat(summary_col, axis=1) if summary is None else pd.concat([summary, summary_col], axis=0)