import pandas as pd
import numpy as np
import scipy.stats as stats

## returns ##

def get_returns_from_prices(prices, method='log'):
    ''' calculate returns from prices '''
    if method == 'log':
        return np.log(prices/prices.shift(1)).dropna()
    elif method == 'simple':
        return (prices/prices.shift(1) - 1).dropna()
    else:
        raise ValueError('method must be either "log" or "simple"')

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

def summary_stats(name_col, r, riskfree_rate=0.02, periods_per_year=252):
    ''' summarize a set of returns '''
    pd_stats = pd.DataFrame(index=[name_col])
    r.astype('float64')
    pd_stats['skewness'] = skewness(r)
    pd_stats['kurtosis'] = kurtosis(r)
    pd_stats['is_normal'] = is_normal(r)
    pd_stats['annualized_return'] = annualize_rets(r, periods_per_year)
    pd_stats['annualized_vol'] = annualize_vol(r, periods_per_year)
    pd_stats['sharpe_ratio'] = sharpe_ratio(r, riskfree_rate, periods_per_year)
    pd_stats['max_drawdown'] = max_drawdown(r)
    pd_stats['calmar_ratio'] = calmar_ratio(r, periods_per_year)
    pd_stats['var_historic'] = var_historic(r)
    pd_stats['cvar_historic'] = cvar_historic(r)
    pd_stats['var_cornish_fisher'] = var_cornish_fisher(r)
    return pd_stats

