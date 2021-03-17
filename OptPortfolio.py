import random
import numpy as np
import backtest
import bt
from varname import nameof
import matplotlib.pyplot as plt
import pandas as pd
def create_algorithm_param(max_num_iteration = None, population_size = 500, max_iteration_without_improv = 100,
                           mutation_probability = 0.2, elit_ratio = 0.02,
                           crossover_probability = 0.5, parents_portion = 0.3, multiprocessing_ncpus = os.cpu_count(),
                           function_timeout = 20):
    algorithm_param = {'max_num_iteration': max_num_iteration,
                       'population_size': population_size,
                       'mutation_probability': mutation_probability,
                       'elit_ratio': elit_ratio,
                       'crossover_probability': crossover_probability,
                       'parents_portion': parents_portion,
                       'crossover_type': 'uniform',
                       'max_iteration_without_improv': max_iteration_without_improv,
                       'multiprocessing_ncpus': multiprocessing_ncpus,
                       'multiprocessing_engine': None,
                       "function_timeout": function_timeout}
    return algorithm_param


def checkDuplicate(X):
    temp = len(X)
    X = list(set(X))
    while len(X) != temp:
        X.append(random.randint(min(X), max(X)))
        X = list(set(X))
    return X

def f(X, kwargs):
    tickers_pool = kwargs.get('tickers_pool')
    prices_pool = kwargs.get('prices_pool')
    X = checkDuplicate(X)
    # X is a list, storing my genes
    # use it and calculate calmar ratio, then -ve it
    report_df, tickers = showResult(X, tickers_pool, prices_pool)  # is a really bad writing method
    print(tickers)
    try:
        CAGR = report_df.loc['CAGR'].values.tolist()[0]
        CAGR = float(str(CAGR).strip('%'))
        calmar = report_df.loc['Calmar Ratio'].values
        calmar = float(calmar)
    except ValueError:
        print('The following set of ticker have some problem, please CHECK ')
        print(tickers)
        print(prices_pool[tickers].dropna())
        CAGR = 0
        calmar = 0
    # if report_df.loc['Win 12m %'].values == '-':
    #     return 0
    # print()
    # print(CAGR)
    # print(calmar)
    y = -0.2 * CAGR + 0.8 * (-calmar + max(calmar - 5, 0))
    print('\n', y)
    return np.round(y, 5)  # well i need to max.

def showResult(root_list, tickers_pool, prices_pool, savefig = False, prefix = None, train = True):
    if all(isinstance(root, str) for root in root_list):  # test if ticker is converted
        tickers = root_list
        tickers_size = len(tickers)
    else:
        tickers = [tickers_pool[int(i)] for i in root_list]
        tickers_size = len(tickers)

    prices = prices_pool[tickers].dropna()
    equal = bt.Strategy('EqualWeight',
                        algos = [
                                bt.algos.RunMonthly(),
                                bt.algos.SelectAll(),
                                bt.algos.WeighEqually(),
                                bt.algos.Rebalance(),
                                ]
                        )
    # inverseVol = bt.Strategy('InverseVol',
    #                          algos = [
    #                                  bt.algos.RunMonthly(),
    #                                  bt.algos.SelectAll(),
    #                                  bt.algos.WeighInvVol(),
    #                                  ApplyLeverage(1.),
    #                                  bt.algos.Rebalance(),
    #                                  ]
    #                          )
    try:
        backtest_equal = bt.Backtest(equal, prices)
        report = bt.run(backtest_equal)

        # backtest_inverese = bt.Backtest(inverseVol, prices)
        # report = bt.run(backtest_inverese)

        report_df = backtest.readReportCSV(report)

        if savefig:

            # prefix = f'{algorithm_param["max_num_iteration"]}_{algorithm_param["population_size"]}_' \
            #          f'{algorithm_param["max_iteration_without_improv"]}_{algorithm_param["mutation_probability"]}_' \
            #          f'{algorithm_param["elit_ratio"]}_{algorithm_param["crossover_probability"]}_{algorithm_param["parents_portion"]}_N'
            prefix = f'./{prefix}_{tickers_size}/'
            if train:
                suffix = '_train.jpg'
            else:
                suffix = '_test.jpg'
            plot_return = report.plot()
            plt.savefig(prefix + nameof(plot_return) + suffix)
            plot_weights = report.plot_security_weights()
            plt.savefig(prefix + nameof(plot_weights) + suffix)
            # plot_scatter_matrix = report.plot_scatter_matrix()
            # plt.savefig(nameof(plot_scatter_matrix))
            plot_correlation = report.plot_correlation()
            plt.savefig(prefix + nameof(plot_correlation) + suffix)
            plot_histrograms = report.plot_histograms()
            plt.savefig(prefix + nameof(plot_histrograms) + suffix)
            plot_prices = prices.plot()
            plt.savefig(prefix + nameof(plot_prices) + suffix)
            plt.close('all')

    except Exception as e:
        print(e)
        print('The following calculation is wrong, please CHECK')
        print(tickers)
        print(prices)
        report_df = pd.DataFrame({
                'CAGR': [0],
                'Calmar Ratio': 0,
                'Win 12m %': '-'
                })
        report_df = report_df.transpose()

    return report_df, tickers
