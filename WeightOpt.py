from backtest import *
from yahoo import *
from worker import *
import concurrent.futures
import itertools
from matplotlib import pyplot as plt
import pandas as pd

class GAWeightingOpt(bt.Algo):
    def __init__(self, lookback=pd.DateOffset(months=3),
                 lag=pd.DateOffset(days=0)):
        super(GAWeightingOpt, self).__init__()
        self.lookback = lookback
        self.lag = lag

    def __call__(self, target):
        selected = target.temp['selected']

        if len(selected) == 0:
            target.temp['weights'] = {}
            return True

        if len(selected) == 1:
            target.temp['weights'] = {selected[0]: 1.}
            return True

        t0 = target.now - self.lag
        prc = target.universe[selected].loc[t0 - self.lookback:t0]

    def startGA(self, tickers_size, tickers_pool, prices_pool, algorithm_param):
        varbound = np.array([[0, 1]] * tickers_size)
        model = ga(function = g,
                   dimension = tickers_size,
                   variable_type = 'real',
                   variable_boundaries = varbound,
                   algorithm_parameters = algorithm_param,
                   tickers_pool = tickers_pool, prices_pool = prices_pool, function_timeout = 30)
        model.run()
        solution = model.output_dict
        # df = pd.DataFrame.from_dict(solution)
        df = pd.DataFrame.from_dict(solution)
        weighting_list = df['variable'].values.tolist()
        return weighting_list
    # how to write a whole GA weigh in a bt package????

def backtestWeigh(X, tickers_pool, prices_pool, savefig = False, folder = None, train = True, i = None):
    prices = prices_pool.dropna()
    weighting = Int2Weigh(X, tickers_pool)
    strategyGA = bt.Strategy('GA',
                     algos = [
                             bt.algos.RunOnce(),
                             bt.algos.SelectAll(),
                             bt.algos.WeighSpecified(**weighting),
                             bt.algos.Rebalance(),
                             ]
                     )
    try:
        backtest_equal = bt.Backtest(strategyGA, prices)
        report = bt.run(backtest_equal)

        report_df = backtest.readReportCSV(report)

        if savefig:
            tickers_size = len(X)
            prefix = folder
            if train:
                if i is not None:
                    suffix = f'_train_opt_{i}.jpg'
                else:
                    suffix = '_train_opt.jpg'
            else:
                if i is not None:
                    suffix = f'_test_opt_{i}.jpg'
                else:
                    suffix = '_test_opt.jpg'
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
        print(prices)
        report_df = pd.DataFrame({
                'CAGR': [0],
                'Calmar Ratio': 0,
                'Win 12m %': '-'
                })
        report_df = report_df.transpose()

    return report_df, weighting

def g(X, kwargs):
    tickers_pool = kwargs.get('tickers_pool')
    prices_pool = kwargs.get('prices_pool')
    report_df, weighting = backtestWeigh(X, tickers_pool, prices_pool)
    CAGR = report_df.loc['CAGR'].values.tolist()[0]
    calmar = report_df.loc['Calmar Ratio'].values

    try:
        CAGR = float(str(CAGR).strip('%'))
        calmar = float(calmar)
        y = -(0.2*CAGR+0.8*calmar)
        # y = -0.2 * CAGR + 0.8 * (-calmar + max(calmar - 5, 0))  # well i need to max.
    except:
        return 0
    return y


def startWeighOpt(tickers_size, tickers_pool, prices_pool, algorithm_param):
    varbound = np.array([[0, 1]] * tickers_size)
    model = ga(function = g,
               dimension = tickers_size,
               variable_type = 'real',
               variable_boundaries = varbound,
               algorithm_parameters = algorithm_param,
               tickers_pool = tickers_pool, prices_pool = prices_pool, function_timeout = 30)
    model.run()
    solution = model.output_dict
    df = pd.DataFrame.from_dict(solution)
    weighting_list = df['variable'].values.tolist()
    return weighting_list

def Int2Weigh(X, tickers_pool):
    # can be used to ensure the sum = 1
    norm = np.linalg.norm(X)
    X = X/norm
    X = [0 if i < 0 else i for i in X] #to scan if negative number exists
    weighting = dict(zip(tickers_pool, X))
    return weighting

def GAWeighOpt(tickers_list, beginDate, endDate, algorithm_param, prefix = None, testEndDate = None, i = None):
    start_time = datetime.datetime.now()
    tickers_pool = tickers_list
    tickers_size = len(tickers_pool)
    prices_pool, tickers_pool = backtest.datafeedMysql(tickers_pool, beginDate, endDate,
                                                       clean_tickers = False,
                                                       common_dates = True)

    weighting_list = startWeighOpt(tickers_size,tickers_pool,prices_pool,algorithm_param)
    if prefix is None:
        folder_name = f'GA_weighting_opt_{tickers_size}_{tickers_list}'
        folder_location = createFolder(folder_name)
    else:
        folder_location = f'./{prefix}/'

    report_df_train, weighting = backtestWeigh(weighting_list, tickers_pool, prices_pool, True, folder = folder_location,
                                               train = True, i = i)
    report_df_test = []
    if testEndDate is None:
        prices_pool, tickers_pool = backtest.datafeedMysql(tickers_pool, endDate, datetime.datetime.now(), False, True)
        report_df_test, weighting = backtestWeigh(weighting_list, tickers_pool, prices_pool, savefig = True,
                                                folder = folder_location,
                                         train = False, i = i)
    else:
        prices_pool, tickers_pool = backtest.datafeedMysql(tickers_pool, endDate, testEndDate, False, True)
        report_df_test, weighting = backtestWeigh(weighting_list, tickers_pool, prices_pool, savefig = True,
                                              folder = folder_location,
                                              train = False, i = i)

    usedTime = datetime.datetime.now() - start_time
    return weighting, report_df_train, report_df_test, usedTime

if __name__ == '__main__':
    beginDate = datetime.date(2015, 9, 3)
    endDate = datetime.date(2020, 9, 3)
    tickers_list = ['PAYC','GBTC','AMZN','ETSY','ADBE','NOW','NVDA','AMD']
    algorithm_param = create_algorithm_param(population_size = 500, multiprocessing_ncpus = 24)
    weighting, report_df_train, report_df_test, usedTime = GAWeighOpt(tickers_list, beginDate, endDate,
                                                                      algorithm_param = algorithm_param)
    print(weighting)
    print(report_df_train)
    print(report_df_test)
    print(usedTime)
