import kMean
import WeightOpt
import worker
import backtest
import datetime
import pandas as pd


def filterKMeansdf(df_kMeanResult, centroids):
    location_max = centroids.argmax(0)
    df_kMeanResult = df_kMeanResult[df_kMeanResult['Class'] == f'C{location_max[1]}']
    return df_kMeanResult.index.tolist()


def stage2(tickers_pool, beginDate, endDate, kMeans = True, clusterSize = 5, GAPortfolio = True, tickerSize = 8,
           GAWeighOpt = True):
    # report_df_train_opt, report_df_test_opt, report_df_train, report_df_test = []
    # weighting, usedTime, usedTime_opt = []
    if kMeans:
        df_kMeanResult, centroids = kMean.runKMeanClustering(tickers_pool, beginDate, endDate, clusterSize,
                                                             showFigure = True)
        tickers_pool = filterKMeansdf(df_kMeanResult, centroids)
        print(tickers_pool)
    if GAPortfolio:
        algorithm_param = worker.create_algorithm_param(max_num_iteration = None, population_size = 500,
                                                        multiprocessing_ncpus =
                                                        24)
        prefix = f'{algorithm_param["max_num_iteration"]}_{algorithm_param["population_size"]}_' \
                 f'{algorithm_param["max_iteration_without_improv"]}_{algorithm_param["mutation_probability"]}_' \
                 f'{algorithm_param["elit_ratio"]}_{algorithm_param["crossover_probability"]}_' \
                 f'{algorithm_param["parents_portion"]}'
        root_list_manual = None
        extra = []
        tickers, report_df_train, report_df_test, usedTime = worker.runGA(tickerSize, tickers_pool, beginDate, endDate,
                                                                          algorithm_param,
                                                                          prefix,
                                                                          root_list_manual, extra, saveFigure = False)
        tickers_pool = sorted(tickers)

    if GAWeighOpt:
        algorithm_param = worker.create_algorithm_param(population_size = 500, multiprocessing_ncpus = 24)
        weighting, report_df_train_opt, report_df_test_opt, usedTime_opt = WeightOpt.GAWeighOpt(tickers_pool, beginDate,
                                                                                                endDate, algorithm_param)
    return tickers_pool, weighting, report_df_train, report_df_train_opt, report_df_test, report_df_test_opt, \
           usedTime, usedTime_opt


if __name__ == '__main__':
    tickers_pool_root = worker.createTickerpool(bool_ALL = False, bool_ETF = True, bool_SPX = True, bool_levETF = False,
                                                extra_tickers =\
        [])
    tickers_pool_root = worker.removeUnwantedTickers_pool(tickers_pool_root, unwanted_tickers_list = None)
    beginDate = datetime.date(2015, 9, 3)
    endDate = datetime.date(2020, 9, 3)
    tickers_pool, weighting, report_df_train, report_df_train_opt, report_df_test, report_df_test_opt, \
    usedTime, usedTime_opt = stage2(tickers_pool_root, beginDate, endDate, kMeans = True, clusterSize = 4, GAPortfolio =
    True, tickerSize = 8, GAWeighOpt = True)
    print(tickers_pool)
    print(report_df_train)
    print(report_df_test)
    print(usedTime)
    print(weighting)
    print(report_df_train_opt)
    print(report_df_test_opt)
    print(usedTime_opt)

