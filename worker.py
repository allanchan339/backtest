import bt
import backtest
import matplotlib
import pandas as pd
import platform
import matplotlib.pyplot as plt
import datetime
import random
import numpy as np
from geneticalgorithm import geneticalgorithm as ga

# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)

if platform.system() != 'Linux':  # to show plt
    matplotlib.use('tkagg')


class GeniticAlgo:  # nothing to inheritance
    # multi processing inside the class?
    def __init__(self, tickers_GA, prices_pool):
        self.tickers_GA = tickers_GA
        self.prices_pool = prices_pool

    def extractPrices(self, prices_pool):
        pass


def pricesFilter(prices_pool, lowerIndex, upperIndex, bool_dropna):
    lowerIndex = datetime.datetime.strftime(lowerIndex, '%Y-%m-%d')
    upperIndex = datetime.datetime.strftime(upperIndex, '%Y-%m-%d')
    prices_pool = prices_pool.loc[prices_pool.index > lowerIndex]
    prices_pool = prices_pool.loc[prices_pool.index < upperIndex]
    #TODO: identify ticker who is empty within these year, and drop it, and update tickers_pool
    if bool_dropna:
        prices_pool = prices_pool.dropna()
    return prices_pool


startTime = datetime.datetime.now()
tickers_pool = backtest.SPXlist()
tickers_pool += backtest.ETFlist()
extra = ['TMF', 'SOXL', 'ARKW', 'ARKK', 'SMH', 'SOXX']
tickers_pool += extra
tickers_pool = list(set(tickers_pool))
tickers_pool.remove('OTIS')  # the backtest period is too short
tickers_pool.remove('CARR')  # same problem
tickers_pool.remove('TT')
beginDate = datetime.date(2010, 1, 1)
endDate = datetime.date.today()
lowerIndex = datetime.date(2015, 1, 1)
upperIndex = datetime.date(2018, 1, 1)
prices_pool, tickers_pool = backtest.datafeedMysql(tickers_pool, lowerIndex, upperIndex, clean_tickers = False, common_dates =
False)
prices_pool = pricesFilter(prices_pool, lowerIndex, upperIndex, False)
print(prices_pool)


# start GA
# workflow
# random get ticker, flexible number, to 1000 father??
# GA to optim, step store into mysql algo, mutant gen with prob.
# will i use Microbial GA?
# use calmer ratio as my obj function
# 5% from max - min , stop ?
# alert then show the top5 outcome, and display result?

class ApplyLeverage(bt.Algo):

    def __init__(self, leverage):
        super(ApplyLeverage, self).__init__()
        self.leverage = leverage

    def __call__(self, target):
        target.temp['weights'] = target.temp['weights'] * self.leverage
        return True


def checkDuplicate(X):
    temp = len(X)
    X = list(set(X))
    while len(X) != temp:
        X.append(random.randint(min(X), max(X)))
        X = list(set(X))
    return X


def f(X):
    X = checkDuplicate(X)
    # X is a list, storing my genes
    # use it and calculate calmar ratio, then -ve it
    report_df, tickers = showResult(X)
    CAGR = report_df.loc['CAGR'].values.tolist()[0]
    CAGR = float(str(CAGR).strip('%'))
    calmar = report_df.loc['Calmar Ratio'].values
    calmar = float(calmar)
    print(tickers)
    print(report_df.head(10))
    if report_df.loc['Win 12m %'].values == '-':
        return 0
    # elif calmar > 3:
    #     return -0.1 * calmar
    return -0.4*CAGR + 0.6*(-calmar + max(calmar - 3, 0)) # well i need to max.


# def g(X):
#     # X is a weight
#     500 * abs(np.sum(X) - 1) - calmer
#     pass


def geneticAlgo(tickers_size):
    # varbound = np.array([[0.5, 1.5], [1, 100], [0, 1]])
    varbound = np.array([[0, len(tickers_pool) - 1]] * tickers_size)
    # remember to count down 1
    # vartype = np.array([['real'], ['int'], ['int']])
    vartype = np.array([['int'] * tickers_size])

    algorithm_param = {'max_num_iteration': None,
                       'population_size': 50,
                       'mutation_probability': 0.1,
                       'elit_ratio': 0.01,
                       'crossover_probability': 0.5,
                       'parents_portion': 0.3,
                       'crossover_type': 'uniform',
                       'max_iteration_without_improv': 500}
    model = ga(function = f,
               dimension = tickers_size,
               variable_type = 'int',
               # variable_type_mixed = vartype
               variable_boundaries = varbound,
               algorithm_parameters = algorithm_param)
    model.run()
    solution = model.output_dict
    df = pd.DataFrame.from_dict(solution)
    root_list = df['variable'].values.tolist()
    return root_list


def showResult(root_list):
    tickers = [tickers_pool[int(i)] for i in root_list]
    prices = prices_pool[tickers].dropna()  # TODO: to change to local var, without speed slowdown
    equal = bt.Strategy('Equal',
                        algos = [
                                bt.algos.RunOnce(),
                                bt.algos.SelectAll(),
                                bt.algos.WeighEqually(),
                                bt.algos.Rebalance(),
                                ]
                        )
    inverseVol = bt.Strategy('InverseVol',
                             algos = [
                                     bt.algos.RunMonthly(),
                                     bt.algos.SelectAll(),
                                     bt.algos.WeighInvVol(),
                                     ApplyLeverage(1.),
                                     bt.algos.Rebalance(),
                                     ]
                             )
    backtest_equal = bt.Backtest(equal, prices)
    backtest_inverese = bt.Backtest(inverseVol, prices)
    # report = bt.run(backtest_equal)
    report = bt.run(backtest_inverese)
    report_df = backtest.readReportCSV(report)
    return report_df, tickers

root_list = geneticAlgo(10)

report_df, tickers = showResult(root_list)
print(tickers)
print(report_df)
finishTime = datetime.datetime.now()
print(f'We need {finishTime - startTime} to calculate')

# _sum_weight = 100%
