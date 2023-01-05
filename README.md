# Portfolio Construction based on ESG Factors

## Context and Purpose of the project

In this project, we will build a portfolio based on ESG factors. We will use the MSCI Europe index as a benchmark. All the assets are part of the MSCI Europe index.

The goal is to build a portfolio with the best ESG score possible and to compare it to the MSCI Europe index.

The period of the study is from 2013 until now. The portfolio will be rebalanced every quarter. This means that we will build a new portfolio every quarter and we will compare the performance of the portfolio with the performance of the MSCI Europe index.

## Table of Contents

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua

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

### How to build a portfolio (basic approach)?

First, we will select the assets from the MSCI index based on their ESG score. We will select the assets with the lowest ESG score with a focus on the environmental factors.

Then we will build a dynamic portfolio based on two strategies :



### How to build a portfolio (advanced approach)?