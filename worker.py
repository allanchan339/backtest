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
import os
from varname import nameof

startTime = datetime.datetime.now()

# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# pd.set_option('display.max_colwidth', None)

# if platform.system() != 'Linux':  # to show plt
#     matplotlib.use('tkagg')


class ApplyLeverage(bt.Algo):

    def __init__(self, leverage):
        super(ApplyLeverage, self).__init__()
        self.leverage = leverage

    def __call__(self, target):
        target.temp['weights'] = target.temp['weights'] * self.leverage
        return True


class GenAlgo(object):
    def __init__(self, tickers_size, algorithm_param, tickers_pool, beginDate, endDate):
        self.tickers_size = tickers_size
        self.algorithm_param = algorithm_param
        self.tickers_pool = tickers_pool
        self.beginDate = beginDate
        self.endDate = endDate
        self.root_list = []


def showResult(root_list, savefig = False):
    if all(isinstance(root, str) for root in root_list): #test if ticker is converted
        tickers = root_list
    else:
        tickers = [tickers_pool[int(i)] for i in root_list]

    prices = prices_pool[tickers].dropna()
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
    if savefig:
        plot_return = report.plot()
        plt.savefig(nameof(plot_return))
        plot_weights = report.plot_security_weights()
        plt.savefig(nameof(plot_weights))
        # plot_scatter_matrix = report.plot_scatter_matrix()
        # plt.savefig(nameof(plot_scatter_matrix))
        plot_correlation = report.plot_correlation()
        plt.savefig(nameof(plot_correlation))
        plot_histrograms = report.plot_histograms()
        plt.savefig(nameof(plot_histrograms))
        plot_prices = prices.plot()
        plt.savefig(nameof(plot_prices))

    return report_df, tickers


def create_algorithm_param(max_num_iteration = None, population_size = 50, max_iteration_without_improv = 100,
                           multiprocessing_ncpus = os.cpu_count()):
    algorithm_param = {'max_num_iteration': max_num_iteration,
                       'population_size': population_size,
                       'mutation_probability': 0.1,
                       'elit_ratio': 0.01,
                       'crossover_probability': 0.5,
                       'parents_portion': 0.3,
                       'crossover_type': 'uniform',
                       'max_iteration_without_improv': max_iteration_without_improv,
                       'multiprocessing_ncpus': multiprocessing_ncpus,
                       'multiprocessing_engine': None}
    return algorithm_param


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
    report_df, tickers = showResult(X)  # is a really bad writing method
    CAGR = report_df.loc['CAGR'].values.tolist()[0]
    CAGR = float(str(CAGR).strip('%'))
    calmar = report_df.loc['Calmar Ratio'].values
    calmar = float(calmar)
    # print(tickers)
    # print(report_df.head(10))
    if report_df.loc['Win 12m %'].values == '-':
        return 0
    # elif calmar > 3:
    #     return -0.1 * calmar
    return -0.4 * CAGR + 0.6 * (-calmar + max(calmar - 3, 0))  # well i need to max.


def createTickerpool(bool_ALL = False, bool_SPX = True, bool_ETF = True, engine = None, extra_tickers = None):
    tickers_pool = backtest.code_list(bool_ALL, bool_SPX, bool_ETF, engine, extra_tickers)
    tickers_pool = list(set(tickers_pool))
    tickers_pool.remove('TT')  # the data in yahoo is problematic
    return tickers_pool


def start(tickers_size, tickers_pool, algorithm_param):
    # varbound = np.array([[0.5, 1.5], [1, 100], [0, 1]])
    varbound = np.array([[0, len(tickers_pool) - 1]] * tickers_size)
    # remember to count down 1
    # vartype = np.array([['real'], ['int'], ['int']])
    vartype = np.array([['int'] * tickers_size])

    model = ga(function = f,  # i cant submit a self.f to ga lib
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


def main(root_list = None):
    global prices_pool, tickers_pool
    extra = ['TMF', 'SOXL', 'ARKW', 'ARKK', 'SMH', 'SOXX']
    tickers_pool = createTickerpool(extra_tickers = extra)
    algorithm_param = create_algorithm_param(max_num_iteration = None, population_size = 500)
    beginDate = datetime.date(2015, 7, 31)
    endDate = datetime.date(2018, 7, 31)
    tickers_size = 12
    prices_pool, tickers_pool = backtest.datafeedMysql(tickers_pool, beginDate, endDate,
                                                       clean_tickers = False,
                                                       common_dates = True)
    if root_list is None:
        root_list = start(tickers_size, tickers_pool, algorithm_param)

    report_df, tickers = showResult(root_list, savefig = True)
    print(tickers)
    print(report_df)

    if (datetime.date.today() - endDate) > datetime.timedelta(days = 7):  # if the end date is bigger than current week
        prices_pool, tickers_pool = backtest.datafeedMysql(tickers, endDate, datetime.datetime.now(), False, True)
        report_df, tickers = showResult(root_list, savefig = True)
        print(tickers)
        print(report_df)

    endTime = datetime.datetime.now()
    print(f'{endTime - startTime} is used')


if __name__ == '__main__':
    global prices_pool, tickers_pool
    root_list_manual = None
    root_list_manual = ['LRCX', 'NOW', 'SCO', 'TQQQ', 'ALGN', 'MKTX', 'NVDA', 'ADSK', 'PAYC', 'AMD', 'NUGT', 'SOXL',
                        'DRIP']
    main(root_list_manual)
