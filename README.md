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
  - [How to evaluate the biodiversity impact of our portfolio ?](#how-to-evaluate-the-biodiversity-impact-of-our-portfolio-)
    - [First method : Environment score](#first-method--environment-score)
    - [The second method : Sector analysis with Encore Biodiversity](#the-second-method--sector-analysis-with-encore-biodiversity)

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

![risk esg](_attachments/risk%20esg.png)  

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

1. The porfolio and the MSCI EUROPE have **70%** of the time positive returns.
2. About the maximum drawdown of our portfolio and the MSCI Europe. We found a drawdown of **-27.91%** (annualized) during Q3-2020 and **-23.79%** for the MSCI (annualized) during Q1-2021.
3. The higest return is **102.57%** (annualized) and **110.42%** for the MSCI (annualized) in Q1-2015.
4. The **annualized return** over 10-years periods is **8.27%** for the portfolio and **11.72%** for the MSCI.
5. The **annualized volatility** over 10-years periods is **14.04%** for the portfolio and **15.63%** for the MSCI.
6. The sharpe ratio is **0.447** for the portfolio over 10-years periods and **0.621** for the MSCI.
7. 1000€ in 2013 is **2253.65** in 2022 for our portfolio, **2617€** for the MSCI and **1224€** for the livret A.

### About the comparison of returns

!['ok'](_attachments/comp%20returns.png)

!['ok'](_attachments/diff%20returns.png)  

### About the comparison of volatility

!['ok'](_attachments/comp%20vol.png)

!['ok'](_attachments/diff%20vol.png)

## How to evaluate the biodiversity impact of our portfolio ?

### First method : Environment score

 The first method is to approximate the biodiversity  by using the environment score of yahoo finance. This approximation is based on the fact that the biodiversity is the diversity of species in a the world or a particular habitat and if a company has a good environment score this implies that it will help to have a better biodiversity (the company doesn't degrade the environment).

 ![comp score](_attachments/comp%20env%20score.png)  

 The result is clear, the environment score of our portfolio is far way lower than the MSCI. This result is logical because we applied filters on environment score to choose our assets.

### The second method : Sector analysis with Encore Biodiversity

The second method is to focus on sectors of our porfolio and the impact of each sector on biodiversity.

![comp score](_attachments/sector%20biodiversity.png)  

We used the website Encore Biodiversity to evaluate each sector from our portfolio according 
to five criterias

![encore bio](_attachments/encore%20bio.png)  

- Impact score:
  - 1: Very Low
  - 2: Low
  - 3: Medium
  - 4: High
  - 5: Very High

We  can  see  on  this  analysis  that  our  sectors  are  not  the  best  in  terms  of  solid  waste,  soil pollutants, water pollutants, GHG emissions and Ecosystem use. All these factors are taken into account to calculate the biodiversity impact. To conclude we can say that the two methods are contradictory because our sectors are not well rated but according to environmental score they are classified as companies with a negligible impact on the environment.
