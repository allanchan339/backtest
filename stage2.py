import kMean
import WeightOpt
import worker
import datetime
import pandas as pd
import numpy as np
import itertools
import backtest


def filterKMeansdf(df_kMeanResult, centroids):
    location_max = centroids.argmax(0)
    df_kMeanResult = df_kMeanResult[df_kMeanResult['Class'] == f'C{location_max[1]}']
    return df_kMeanResult.index.tolist()


def indexResultSaver(index, beginDate, endDate, testEndDate, prefix, saveFigure=False, i=None):
    ticker_list = [index]
    prices, ticker = backtest.datafeedMysql(ticker_list, beginDate, endDate, clean_tickers=False, common_dates=True)
    report_df_train, tickers = worker.showResult(ticker, ticker, prices, saveFigure, prefix, train=True,
                                                 i=f'{index}_{i}')
    prices, ticker = backtest.datafeedMysql(ticker_list, endDate, testEndDate, clean_tickers=False, common_dates=True)
    report_df_test, tickers = worker.showResult(ticker, ticker, prices, saveFigure, prefix, train=False,
                                                i=f'{index}_{i}')
    return report_df_train, report_df_test, tickers


def stage2(tickers_pool, beginDate, endDate, kMeans=True, clusterSize=5, GAPortfolio=True, tickerSize=8,
           GAWeighOpt=True, testEndDate=None, prefix=None, kfold_no=None):
    # report_df_train_opt, report_df_test_opt, report_df_train, report_df_test = []
    # weighting, usedTime, usedTime_opt = []

    if kMeans and clusterSize > 1:
        df_kMeanResult, centroids = kMean.runKMeanClustering(tickers_pool, beginDate, endDate, clusterSize,
                                                             saveFigure=True, prefix=prefix, i=kfold_no)
        tickers_pool = filterKMeansdf(df_kMeanResult, centroids)
        print(tickers_pool)

    if len(tickers_pool) < tickerSize:
        raise Exception("Sorry, the ticker pool has lower number than required ticker size")
    if GAPortfolio:
        tickers = []
        count = 0
        while (len(tickers) != tickerSize and count<5):
            count += 1
            algorithm_param = worker.create_algorithm_param(population_size=500,
                                                            multiprocessing_ncpus=
                                                            24)
            # prefix = f'{algorithm_param["max_num_iteration"]}_{algorithm_param["population_size"]}_' \
            #          f'{algorithm_param["max_iteration_without_improv"]}_{algorithm_param[
            #          "mutation_probability"]}_' \
            #          f'{algorithm_param["elit_ratio"]}_{algorithm_param["crossover_probability"]}_' \
            #          f'{algorithm_param["parents_portion"]}'
            root_list_manual = None
            extra = []
            tickers, report_df_train, report_df_test, usedTime = worker.runGA(tickerSize, tickers_pool, beginDate,
                                                                              endDate,
                                                                              algorithm_param,
                                                                              prefix,
                                                                              root_list_manual, extra, saveFigure=
                                                                              True, testEndDate=testEndDate, i=kfold_no)
        tickers_pool = sorted(tickers)

    if GAWeighOpt:
        algorithm_param = worker.create_algorithm_param(population_size=500, multiprocessing_ncpus=24)
        weighting, report_df_train_opt, report_df_test_opt, usedTime_opt = WeightOpt.GAWeighOpt(tickers_pool, beginDate,
                                                                                                endDate,
                                                                                                algorithm_param,
                                                                                                testEndDate=
                                                                                                testEndDate,
                                                                                                prefix=prefix,
                                                                                                i=kfold_no)
    return tickers_pool, weighting, report_df_train, report_df_train_opt, report_df_test, report_df_test_opt,\
           usedTime, usedTime_opt


def resultSaver(tickers_pool, weighting, report_df_train, report_df_train_opt, report_df_test, report_df_test_opt,\
                usedTime, usedTime_opt, print_result=False, saveResult=True, i=None, prefix=None, Index_train=
                None, Index_test=None, Index=None):
    if print_result:
        print()  # a blank line
        print(tickers_pool)
        print(report_df_train)
        print(report_df_test)
        print(usedTime)
        print(weighting)
        print(report_df_train_opt)
        print(report_df_test_opt)
        print(usedTime_opt)

    if saveResult:
        if i is None:
            suffix = ''
        else:
            suffix = f'{i+1}'
        tickers_df = pd.DataFrame(tickers_pool).transpose()
        usedTime = pd.DataFrame({"UsedTime":usedTime}.items())
        usedTime['UsedTime_str'] = usedTime[1].astype(str)
        usedTime = usedTime.drop(0, axis=1).drop(1, axis=1)
        tickersSize = len(tickers_pool)

        if report_df_train_opt.empty:
            if prefix is None:
                prefix = f'GA_{tickersSize}_{tickers_pool}'
            with pd.ExcelWriter(f'./{prefix}/stage2_{i}.xlsx', mode='w',
                                datetime_format='hh:mm:ss.000') as writer:
                tickers_df.to_excel(writer, sheet_name=f'train_{suffix}', index=False)
                tickers_df.to_excel(writer, sheet_name=f'test_{suffix}', index=False)
                usedTime.to_excel(writer, sheet_name=f'train_{suffix}', index=False, startrow=4)
                usedTime.to_excel(writer, sheet_name=f'test_{suffix}', index=False, startrow=4)
                report_df_train.to_excel(writer, sheet_name=f'train_{suffix}', startrow=8)
                report_df_test.to_excel(writer, sheet_name=f'test_{suffix}', startrow=8)
        else:
            weighting_df = pd.DataFrame.from_dict(weighting.items()).transpose()
            usedTime_opt = pd.DataFrame({"UsedTime":usedTime_opt}.items())
            usedTime_opt['UsedTime_str'] = usedTime_opt[1].astype(str)
            usedTime_opt = usedTime_opt.drop(0, axis=1).drop(1, axis=1)
            if prefix is None:
                prefix = f'GA_{i}_weighting_opt_{tickersSize}_{tickers_pool}'
            with pd.ExcelWriter(f'./{prefix}/stage2_{i}.xlsx', mode='w',
                                datetime_format='hh:mm:ss.000') as writer:
                tickers_df.to_excel(writer, sheet_name=f'train_{suffix}', index=False)
                tickers_df.to_excel(writer, sheet_name=f'test_{suffix}', index=False)
                usedTime.to_excel(writer, sheet_name=f'train_{suffix}', index=False, startrow=4)
                usedTime.to_excel(writer, sheet_name=f'test_{suffix}', index=False, startrow=4)
                report_df_train.to_excel(writer, sheet_name=f'train_{suffix}', startrow=8)
                report_df_test.to_excel(writer, sheet_name=f'test_{suffix}', startrow=8)
                weighting_df.to_excel(writer, sheet_name=f'train_opt_{suffix}', index=False)
                weighting_df.to_excel(writer, sheet_name=f'test_opt_{suffix}', index=False)
                usedTime_opt.to_excel(writer, sheet_name=f'train_opt_{suffix}', index=False, startrow=4)
                usedTime_opt.to_excel(writer, sheet_name=f'test_opt_{suffix}', index=False, startrow=4)
                report_df_train_opt.to_excel(writer, sheet_name=f'train_opt_{suffix}', startrow=8)
                report_df_test_opt.to_excel(writer, sheet_name=f'test_opt_{suffix}', startrow=8)
                if Index_train is not None:
                    Index_df = pd.DataFrame(Index).transpose()
                    Index_df.to_excel(writer, sheet_name=f'Index_train_{suffix}', index=False)
                    Index_df.to_excel(writer, sheet_name=f'Index_test_{suffix}', index=False)
                    Index_train.to_excel(writer, sheet_name=f'Index_train_{suffix}', startrow=8)
                    Index_test.to_excel(writer, sheet_name=f'Index_test_{suffix}', startrow=8)


def timeSplit(beginDate, testEndDate, kFold):
    list = pd.date_range(start=beginDate, end=testEndDate, freq='D')
    list = np.array_split(list, kFold+1)
    return list


def stage2_cross_valid(tickers_pool_root, beginDate, testEndDate=None, kMeans=True, clusterSize=4, GAPortfolio=
True, tickerSize=8, GAWeighOpt=True, kFold=5):
    prefix = f'CrossValid_GA_WeighOpt_{tickerSize}'
    folderLocation = worker.createFolder(prefix)

    if testEndDate is None:
        testEndDate = datetime.datetime.now()

    from sklearn.model_selection import TimeSeriesSplit
    time = timeSplit(beginDate, testEndDate, kFold)
    tscv = TimeSeriesSplit(n_splits=kFold)
    for i, (train_index, test_index) in enumerate(tscv.split(time)):

        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = np.array(time)[train_index], np.array(time)[test_index]
        X_train = list(itertools.chain.from_iterable(X_train))
        X_test = list(itertools.chain.from_iterable(X_test))
        beginDate_temp = X_train[0]
        endDate_temp = X_test[0]
        testEndDate_temp = X_test[-1]
        print(beginDate_temp, endDate_temp, testEndDate_temp)
        tickers_pool, weighting, report_df_train, report_df_train_opt, report_df_test, report_df_test_opt,\
        usedTime, usedTime_opt = stage2(tickers_pool_root, beginDate_temp, endDate_temp, kMeans=kMeans, clusterSize=
        clusterSize,
                                        GAPortfolio=GAPortfolio, tickerSize=tickerSize, GAWeighOpt=GAWeighOpt,
                                        testEndDate=testEndDate_temp, prefix=prefix, kfold_no=i)
        SPY_train, SPY_test, SPY_index = indexResultSaver('SPY', beginDate_temp, endDate_temp, testEndDate_temp,
                                                          prefix, saveFigure=True, i=i)

        resultSaver(tickers_pool, weighting, report_df_train, report_df_train_opt, report_df_test, report_df_test_opt,
                    usedTime, usedTime_opt, print_result=True, saveResult=True, i=i, prefix=prefix, Index_train=
                    SPY_train, Index_test=SPY_test, Index=SPY_index)


if __name__ == '__main__':
    tickers_pool_root = worker.createTickerpool(bool_ALL=False, bool_ETF=True, bool_SPX=True, bool_levETF=False,
                                                extra_tickers=\
                                                    [])
    tickers_pool_root = worker.removeUnwantedTickers_pool(tickers_pool_root, unwanted_tickers_list=None)
    beginDate = datetime.date(2015, 1, 1)
    endDate = datetime.date(2021, 2, 12)
    for i in range(8, 15, 2):
        # if i <= 12:
        #     continue
        stage2_cross_valid(tickers_pool_root, beginDate, testEndDate=endDate, kFold=20, clusterSize=3,
                           tickerSize=i, kMeans=True, GAPortfolio=True, GAWeighOpt=True)
# tickers_pool, weighting, report_df_train, report_df_train_opt, report_df_test, report_df_test_opt, \
# usedTime, usedTime_opt = stage2(tickers_pool_root, beginDate, endDate, kMeans = True, clusterSize = 4,
# GAPortfolio =
# True, tickerSize = 8, GAWeighOpt = True)
# resultSaver(tickers_pool, weighting, report_df_train, report_df_train_opt, report_df_test, report_df_test_opt, \
#             usedTime, usedTime_opt, print_result = True, saveResult = True)
