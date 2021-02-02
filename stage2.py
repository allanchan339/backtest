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
        tickers = []
        while(len(tickers) != tickerSize):
            algorithm_param = worker.create_algorithm_param(population_size = 500,
                                                            multiprocessing_ncpus =
                                                            23)
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
        algorithm_param = worker.create_algorithm_param(population_size = 500, multiprocessing_ncpus = 23)
        weighting, report_df_train_opt, report_df_test_opt, usedTime_opt = WeightOpt.GAWeighOpt(tickers_pool, beginDate,
                                                                                                endDate, algorithm_param)
    return tickers_pool, weighting, report_df_train, report_df_train_opt, report_df_test, report_df_test_opt, \
           usedTime, usedTime_opt

def resultSaver(tickers_pool, weighting, report_df_train, report_df_train_opt, report_df_test, report_df_test_opt, \
                usedTime, usedTime_opt, print_result = False, saveResult = True):

    if print_result:
        print() # a blank line
        print(tickers_pool)
        print(report_df_train)
        print(report_df_test)
        print(usedTime)
        print(weighting)
        print(report_df_train_opt)
        print(report_df_test_opt)
        print(usedTime_opt)


    if saveResult:
        tickers_df = pd.DataFrame(tickers_pool).transpose()
        usedTime = pd.DataFrame({"UsedTime": usedTime}.items())
        usedTime['UsedTime_str'] = usedTime[1].astype(str)
        usedTime = usedTime.drop(0, axis = 1).drop(1, axis = 1)
        tickersSize = len(tickers_pool)

        if report_df_train_opt.empty:
            prefix = f'GA_{tickersSize}_{tickers_pool}'
            with pd.ExcelWriter(f'./{prefix}/stage2.xlsx', mode = 'A',
                                datetime_format = 'hh:mm:ss.000') as writer:
                tickers_df.to_excel(writer, sheet_name = 'train', index = False)
                tickers_df.to_excel(writer, sheet_name = 'test', index = False)
                usedTime.to_excel(writer, sheet_name = 'train', index = False, startrow = 4)
                usedTime.to_excel(writer, sheet_name = 'test', index = False, startrow = 4)
                report_df_train.to_excel(writer, sheet_name = 'train', startrow = 8)
                report_df_test.to_excel(writer, sheet_name = 'test', startrow = 8)
        else:
            weighting_df = pd.DataFrame.from_dict(weighting.items()).transpose()
            usedTime_opt = pd.DataFrame({"UsedTime": usedTime_opt}.items())
            usedTime_opt['UsedTime_str'] = usedTime_opt[1].astype(str)
            usedTime_opt = usedTime_opt.drop(0, axis = 1).drop(1, axis = 1)
            prefix = f'GA_weighting_opt_{tickersSize}_{tickers_pool}'
            with pd.ExcelWriter(f'./{prefix}/stage2.xlsx', mode = 'A',
                                datetime_format = 'hh:mm:ss.000') as writer:
                tickers_df.to_excel(writer, sheet_name = 'train', index = False)
                tickers_df.to_excel(writer, sheet_name = 'test', index = False)
                usedTime.to_excel(writer, sheet_name = 'train', index = False, startrow = 4)
                usedTime.to_excel(writer, sheet_name = 'test', index = False, startrow = 4)
                report_df_train.to_excel(writer, sheet_name = 'train', startrow = 8)
                report_df_test.to_excel(writer, sheet_name = 'test', startrow = 8)
                weighting_df.to_excel(writer, sheet_name = 'train_opt', index = False)
                weighting_df.to_excel(writer, sheet_name = 'test_opt', index = False)
                usedTime_opt.to_excel(writer, sheet_name = 'train_opt', index = False, startrow = 4)
                usedTime_opt.to_excel(writer, sheet_name = 'test_opt', index = False, startrow = 4)
                report_df_train_opt.to_excel(writer, sheet_name = 'train_opt', startrow = 8)
                report_df_test_opt.to_excel(writer, sheet_name = 'test_opt', startrow = 8)




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
    resultSaver(tickers_pool, weighting, report_df_train, report_df_train_opt, report_df_test, report_df_test_opt, \
                usedTime, usedTime_opt, print_result = True, saveResult = True)



