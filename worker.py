import bt
import backtest
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import random
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import os
from varname import nameof


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

def showResult(root_list, tickers_pool, prices_pool, savefig = False, prefix = None, train = True):
    if all(isinstance(root, str) for root in root_list):  # test if ticker is converted
        tickers = root_list
        tickers_size = len(tickers)
    else:
        tickers = [tickers_pool[int(i)] for i in root_list]
        tickers_size = len(tickers)
        tickers.append('GLD')
        tickers.append('GBTC')

    prices = prices_pool[tickers].dropna()
    equal = bt.Strategy('EqualWeight',
                        algos = [
                                bt.algos.RunMonthly(),
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
            plt.savefig(prefix+nameof(plot_return)+suffix)
            plot_weights = report.plot_security_weights()
            plt.savefig(prefix+nameof(plot_weights)+suffix)
            # plot_scatter_matrix = report.plot_scatter_matrix()
            # plt.savefig(nameof(plot_scatter_matrix))
            plot_correlation = report.plot_correlation()
            plt.savefig(prefix+nameof(plot_correlation)+suffix)
            plot_histrograms = report.plot_histograms()
            plt.savefig(prefix+nameof(plot_histrograms)+suffix)
            plot_prices = prices.plot()
            plt.savefig(prefix+nameof(plot_prices)+suffix)
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


def create_algorithm_param(max_num_iteration = None, population_size = 500, max_iteration_without_improv = 100,
                            mutation_probability = 0.1, elit_ratio = 0.01,
                           crossover_probability = 0.5, parents_portion = 0.3, multiprocessing_ncpus = os.cpu_count()):
    algorithm_param = {'max_num_iteration': max_num_iteration,
                       'population_size': population_size,
                       'mutation_probability': mutation_probability,
                       'elit_ratio': elit_ratio,
                       'crossover_probability': crossover_probability,
                       'parents_portion': parents_portion,
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


def f(X, kwargs):
    tickers_pool = kwargs.get('tickers_pool')
    prices_pool = kwargs.get('prices_pool')
    X = checkDuplicate(X)
    # X is a list, storing my genes
    # use it and calculate calmar ratio, then -ve it
    report_df, tickers = showResult(X, tickers_pool, prices_pool)  # is a really bad writing method
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
    # print(tickers)
    # print(report_df.head(10))
    if report_df.loc['Win 12m %'].values == '-':
        return 0
    # elif calmar > 3:
    #     return -0.1 * calmar
    return -0.2 * CAGR + 0.8 * (-calmar + max(calmar - 5, 0))  # well i need to max.


def createTickerpool(bool_ALL = False, bool_SPX = True, bool_ETF = True, bool_levETF = True, engine = None,
                     extra_tickers = None):
    tickers_pool = backtest.code_list(bool_ALL, bool_SPX, bool_ETF, bool_levETF, engine, extra_tickers)
    tickers_pool.remove('TT')  # the data in yahoo is problematic
    tickers_pool.remove('LUMN') #the ticker is too new

    if bool_ALL:
        tickers_pool.remove('^DJI')
        tickers_pool.remove('^GSPC')
        tickers_pool.remove('^HSI')
        tickers_pool.remove('^IXIC')
        tickers_pool.remove('^RUT')
        tickers_pool.remove('^VIX')
    return tickers_pool


def start(tickers_size, tickers_pool, prices_pool, algorithm_param):
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
               algorithm_parameters = algorithm_param,
               tickers_pool = tickers_pool, prices_pool = prices_pool
               )
    model.run()
    solution = model.output_dict
    df = pd.DataFrame.from_dict(solution)
    root_list = df['variable'].values.tolist()
    return root_list


def removeUnwantedTickers_pool(tickers_pool, unwanted_tickers_list = None):
    for ticker in unwanted_tickers_list:
        if ticker in tickers_pool:
            tickers_pool.remove(ticker)
    return tickers_pool


def runGA(tickers_size, beginDate, endDate, algorithm_param, prefix, root_list = None, extra = []):
    startTime = datetime.datetime.now()

    tickers_pool = createTickerpool(bool_ALL = False, bool_SPX = True, bool_ETF = False, bool_levETF = False,
                                    extra_tickers = extra)
    unwanted_tickers_list = []
    tickers_pool = removeUnwantedTickers_pool(tickers_pool, unwanted_tickers_list)
    prices_pool, tickers_pool = backtest.datafeedMysql(tickers_pool, beginDate, endDate,
                                                       clean_tickers = False,
                                                       common_dates = True)

    if root_list is None:
        root_list = start(tickers_size, tickers_pool, prices_pool, algorithm_param)
        root_list = list(set(root_list))
    report_df_train, tickers = showResult(root_list, tickers_pool, prices_pool, savefig = True, prefix = prefix,
                                          train = True)

    report_df_test = None
    #In case the test didnt run
    if (datetime.date.today() - endDate) > datetime.timedelta(days = 7):  # if the end date is bigger than current week
        #used to check the validation of the GA result
        prices_pool, tickers_pool = backtest.datafeedMysql(tickers_pool, endDate, datetime.datetime.now(), False, True)
        report_df_test, tickers = showResult(tickers, tickers_pool, prices_pool, savefig = True,  prefix = prefix,
                                             train = False)

    endTime = datetime.datetime.now()
    usedTime = endTime - startTime
    return tickers, report_df_train, report_df_test, usedTime

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)


# Example
# createFolder('./data/')
# Creates a folder in the current directory called data

if __name__ == '__main__':
    root_list_manual = None
    extra = ['TMF', 'SOXX', 'MSFT', 'GLD', 'GBTC','SPY','QQQ']
    algorithm_param = create_algorithm_param(max_num_iteration = None, population_size = 500, multiprocessing_ncpus =
    24)
    beginDate = datetime.date(2015, 9, 3)
    endDate = datetime.date(2020, 9, 3)
    tickers_size = 1

    # endDate = datetime.date.today()

    # root_list_manual = [84,334,382,315,166,314,340,36]
    # root_list_manual = list(set(root_list_manual))
    # root_list_manual = ['LRCX', 'NOW', 'SCO', 'TQQQ', 'ALGN', 'MKTX', 'NVDA', 'ADSK', 'PAYC', 'AMD', 'NUGT', 'SOXL',
    #                     'DRIP']
    root_list_manual = ['QQQ']
    # root_list_manual = ['PAYC', 'GBTC','AMZN','ETSY','ADBE','NOW','NVDA','AMD']
    prefix = f'{algorithm_param["max_num_iteration"]}_{algorithm_param["population_size"]}_' \
             f'{algorithm_param["max_iteration_without_improv"]}_{algorithm_param["mutation_probability"]}_' \
             f'{algorithm_param["elit_ratio"]}_{algorithm_param["crossover_probability"]}_' \
             f'{algorithm_param["parents_portion"]}_N'
    createFolder(f'./{prefix}_{tickers_size}/')

    tickers, report_df_train, report_df_test, usedTime = runGA(tickers_size, beginDate, endDate, algorithm_param, prefix,
                                                               root_list_manual, extra)
    tickers_df = pd.DataFrame(tickers).transpose()
    usedTime = pd.DataFrame({"UsedTime": usedTime}.items())
    usedTime['UsedTime_str'] = usedTime[1].astype(str)
    usedTime = usedTime.drop(0, axis = 1).drop(1, axis = 1)
    print(tickers_df)
    print(report_df_train)
    print(usedTime)
    with pd.ExcelWriter(f'./{prefix}_{tickers_size}/stage1.xlsx', mode = 'A', datetime_format= 'hh:mm:ss.000') as writer:
        tickers_df.to_excel(writer, sheet_name='train', index=False)
        tickers_df.to_excel(writer, sheet_name='test', index=False)
        usedTime.to_excel(writer, sheet_name='train', index=False, startrow = 3)
        usedTime.to_excel(writer, sheet_name='test', index=False,startrow = 3)
        report_df_train.to_excel(writer, sheet_name = 'train', startrow = 7)
        report_df_test.to_excel(writer, sheet_name = 'test', startrow = 7)
