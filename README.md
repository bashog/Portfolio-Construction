# Portfolio Construction based on ESG Factors

## Context and Purpose of the project

In this project, we will build a portfolio based on ESG factors. We will use the MSCI Europe index as a benchmark. All the assets are part of the MSCI Europe index.

The goal is to build a portfolio with the best ESG score possible and to compare it to the MSCI Europe index.

The period of the study is from 2013 until now. The portfolio will be rebalanced every quarter. This means that we will build a new portfolio every quarter and we will compare the performance of the portfolio with the performance of the MSCI Europe index.

## Table of Contents

- [Portfolio Construction based on ESG Factors](#portfolio-construction-based-on-esg-factors)
  - [Context and Purpose of the project](#context-and-purpose-of-the-project)
  - [Table of Contents](#table-of-contents)
  - [Structure of the repository](#structure-of-the-repository)
  - [Methodology](#methodology)
    - [What are ESG factors?](#what-are-esg-factors)
    - [Why MSCI Europe index as a benchmark?](#why-msci-europe-index-as-a-benchmark)
    - [How to build the portfolio ?](#how-to-build-the-portfolio-)
  - [Results](#results)
    - [About the comparison of returns](#about-the-comparison-of-returns)
    - [About the comparison of volatility](#about-the-comparison-of-volatility)

## Structure of the repository

The structure of the repository is the following :

| Folders | Description |
| --- | --- |
| `datas` | datas used for the construction of the portfolio |
| `main folder` | notebooks and python files used for the construction of the portfolio |

Description of python files :

| Files | Description |
| --- | --- |
| `portfolio_functions.py` | Main functions used for the construction of the portfolio |
| `portfolio_utils.py` | Functions to get the returns and the volatility of the portfolio |
| `stats_utils.py` | Functions to get the statistics on the portfolio and assets |
| `tickers_utils.py` | All the tickers used for the construction of the portfolio after the cleaning |

Notebooks used for the construction of the portfolio :

| Notebooks | Description |
| --- | --- |
| `0_data_extraction.ipynb` | Extraction of the datas from the MSCI Europe index and from yahoo finance |
| `1_data_engineering.ipynb` | Cleaning of the datas : remove uncomplete datas, convert in euros, etc. |
| `2_basic_exploration.ipynb` | Basic exploration of the datas used for the construction of the portfolio |
| `3_portfolio_construction.ipynb` | Construction of the portfolio based on ESG factors and some strategies |

## Methodology

### What are ESG factors?

**Environmental, social, and governance (ESG) factors** are non-financial factors that can affect the performance of a company. It tells investors the level of risk exposure a company has in these three areas. Therefore, a high ESG score means that the company has a high risk exposure in these three areas.

**Environmental factors** are related to the impact of a company on the environment. For example, the use of renewable energy, the use of recycled materials, carbon footprint, etc.

**Social factors** are related to the impact of a company on the society. For example, the use of child labor, the use of forced labor, the use of human rights, etc. Also linked with the labor management, health and safety, etc.

**Governance factors** are related to the management of the company. For example, the management of the board of directors, the management of the executive committee, the management of the shareholders, employee pay and benefits, etc.

### Why MSCI Europe index as a benchmark?

The client target are young people form Europe that want to invest in a sustainable way. The MSCI Europe index is a good benchmark because it is a free float-adjusted market capitalization index that is designed to measure the equity market performance of developed markets in Europe. With an addition of ESG factors, we can build a portfolio that is more sustainable than the MSCI Europe index.

### How to build the portfolio ?

First, we will select the assets from the MSCI index based on their ESG score. We will select the assets with the lowest ESG score with a focus on the environmental factors.
So, we keep the assets that are not in the sectors of Energie, Materiaux, Services Publics. Then, for the Total score, Governance Score and Social Score, we keep the assets with a score inferior to their median. For the Environmental Score, we keep the assets with a score inferior to the 25th percentile.

Then we will build a dynamic portfolio based on different strategies :

- equal weight
- mean variance

We keep the mean variance portfolio but we add the transaction costs. We will compare the performance of the portfolio with the performance of the MSCI Europe index.

The portfolio will be rebalanced every quarter.

## Results

With the mean variance portfolio with transaction cost :

1. The porfolio has **60%** of the time positive returns compared to **70%** of the time for the MSCI EUROPE (it's a difference of 5 quarters).
2. The maximum drawdown of our portfolio and the MSCI Europe. We found a drawdown of **-25.72%** (annualized) during Q3-2015 and **-23.79%** for the MSCI (annualized) during Q1-2021.
3. The higest return is **88.43%** (annualized) during Q1-2015 and **110.42%** for the MSCI (annualized) during Q1-2015.
4. The average return is **6.01%** (annualized) for the portfolio and **14.86%** for the MSCI (annualized).
5. The average volatility is **10.49%** (annualized) for the portfolio and **15.16%** for the MSCI (annualized).
6. The annualized return (not the average) is **5.13%** for the portfolio and **11.72%** for the MSCI.
7. The annualized volatility (not the average) is **11.88%** for the portfolio and **15.63%** for the MSCI.
8. The sharpe ratio is **4.184** for the portfolio and **9.865** for the MSCI.
9. 1000€ in 2013 is **1668€** in 2022 for our portfolio, **2617€** for the MSCI and **1224€** for the livret A.

### About the comparison of returns

!['ok'](_attachments/comp%20ret.png)  
!['ok'](_attachments/diff%20returns.png)  

### About the comparison of volatility

!['ok'](_attachments/comp%20vol.png)  
!['ok'](_attachments/diff%20vol.png)