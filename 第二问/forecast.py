import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
from lightgbm import LGBMRegressor
from chinese_calendar import is_workday
from 第二问.工具类.optimal_bins import optimal_bins
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib
warnings.filterwarnings("ignore")

#数据处理函数，将2015年到2019年3月的数据进行处理
def dataProcessing():
    #读取官方原始数据
    train1DF = pd.read_csv('../数据/官方数据_完整版/order_train1.csv', encoding='utf-8')
    #构造天数排序范围
    #Days对应的天数,2015年9月1日定义为Days=1，也就是第一天
    dateList = pd.date_range(start="20150901", end="20181220", freq="D")
    daysList = {}
    i = 1
    for date in dateList:
        daysList[str(date).split(' ')[0]] = i
        i += 1
    train1DF['days_range'] = train1DF['order_date'].map(daysList)
    #构造促销列表
    #如果是促销日，对应的值为1，否则为0
    train1DF['order_date'] = pd.to_datetime(train1DF['order_date'])
    salesPromotionDate = ['1-1', '2-14', '3-8', '5-1', '6-1', '6-18','10-1', '11-11', '12-12']
    salesPromotionList = []
    for index,row in train1DF.iterrows():
        str_date = str(row['order_date'].month) + '-' + str(row['order_date'].day)
        if str_date in salesPromotionDate:
            salesPromotionList.append(1)
        else:
            salesPromotionList.append(0)
    train1DF['promotion'] = salesPromotionList
    # 构建年份列表
    yearList = []
    for index, row in train1DF.iterrows():
        yearList.append(row['order_date'].year)
    train1DF['year'] = yearList
    # 构建月份列表
    monthList = []
    for index, row in train1DF.iterrows():
        monthList.append(row['order_date'].month)
    train1DF['month'] = monthList
    # 构建周列表
    weekList = []
    for index, row in train1DF.iterrows():
        weekList.append(row['order_date'].week)
    train1DF['week_of_year'] = weekList
    #构建一年中的天数列表
    train1DF['day_of_year'] =train1DF.order_date.dt.dayofyear
    #构造一个月中的天数列表
    dayList = []
    for index, row in train1DF.iterrows():
        dayList.append(row['order_date'].day)
    train1DF['day_of_month'] = dayList
    #构造价格区间列表
    cut_bins = optimal_bins(train1DF.ord_qty, train1DF.item_price, n=10)
    train1DF['price_range'] = pd.cut(train1DF['item_price'], cut_bins, labels=[x for x in range(len(cut_bins) - 1)])
    # 销售渠道列表
    channelDict = {'offline': 0, 'online': 1}
    train1DF.sales_chan_name = train1DF.sales_chan_name.map(channelDict)
    #构造月段列表
    #'月初': 0, '月中': 1, '月末': 2
    monthPeriodList = []
    for index, row in train1DF.iterrows():
        if row['order_date'].day <= 10:
            monthPeriodList.append('0')
        elif row['order_date'].day > 10 and row['order_date'].day <= 20:
            monthPeriodList.append('1')
        elif row['order_date'].day > 20 :
            monthPeriodList.append('2')
    train1DF['monthe_period'] = monthPeriodList
    # 构造季节列表
    #'春': 0, '夏': 1, '秋': 2, '冬': 3
    seasonList=[]
    for index, row in train1DF.iterrows():
        if row['order_date'].month==3 or row['order_date'].month==4 or row['order_date'].month==5:
            seasonList.append('0')
        elif row['order_date'].month==6 or row['order_date'].month==7 or row['order_date'].month==8:
            seasonList.append('1')
        elif row['order_date'].month==9 or row['order_date'].month==10 or row['order_date'].month==11:
            seasonList.append('2')
        elif row['order_date'].month==12 or row['order_date'].month==1 or row['order_date'].month==2:
            seasonList.append('3')
    train1DF['seaon'] = seasonList
    #构造星期列表
    #星期天-星期一分别是：0-6
    train1DF['day_of_week'] = train1DF.order_date.dt.weekday
    #构造工作日列表
    #工作日：1，非工作日：0
    train1DF['workday'] =train1DF['order_date'].map(lambda x: is_workday(x))
    workdayDict= {False: 0, True: 1}
    train1DF['workday'] = train1DF.workday.map(workdayDict)
    #保存数据到EncodedTrainData_2015_2018文件
    train1DF.to_csv('../数据/过程中数据/EncodedTrainData_2015_2018.csv', index=False, encoding='utf-8')

#构建模型为训练集和验证集检验模型的效果,训练2015年1月到2017年12月的数据
#来预测2018年1月到2018年3月的数据，并进行验证
def vaildModel():
    #读取编码以及特征工程后的2015年1月到2018年月的数据
    Train15_1901DF = pd.read_csv('../数据/过程中数据/FeaturedEncodedTrainData_2015_201901.csv', encoding='utf-8')
    #划分验证集和测试集
    valid = Train15_1901DF[(Train15_1901DF['days_range'] >= 854) & (Train15_1901DF['days_range'] <= 943)][['item_code', 'days_range', 'ord_qty']]
    #测试集训练预测和验证集预测结果
    validResult = valid['ord_qty']  # 这是已有真实标签需求量1175到1206，31天间隔的真实数据
    # 对五个销售区域分别建模并进行训练
    regionList = [102]
    for region in regionList:
           regionedTrain15_1901DF = Train15_1901DF[Train15_1901DF['sales_region_code'] == region]
           #划分训练集和验证集
           X_train, y_train = regionedTrain15_1901DF[regionedTrain15_1901DF['days_range'] < 854].drop('ord_qty', axis=1), regionedTrain15_1901DF[regionedTrain15_1901DF['days_range'] < 854]['ord_qty']
           X_valid, y_valid = regionedTrain15_1901DF[(regionedTrain15_1901DF['days_range'] >= 854) & (regionedTrain15_1901DF['days_range'] <= 943)].drop('ord_qty', axis=1), \
                              regionedTrain15_1901DF[(regionedTrain15_1901DF['days_range'] >= 854) & (regionedTrain15_1901DF['days_range'] <= 943)]['ord_qty']
           #构建模型并模型
           model = LGBMRegressor(
               n_estimators=1000,
               learning_rate=0.1,
               subsample=0.8,
               colsample_bytree=0.8,
               max_depth=3,
               num_leaves=8,
               min_child_weight=300
           )
           print('*****Prediction for 销售区域: {}*****'.format(region))
           model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)],
                     eval_metric='rmse', verbose=20, early_stopping_rounds=100)
           validResult[X_valid.index] = model.predict(X_valid)
           # #保存模型文件数据
           joblib.dump(model, "../模型/" + "model_" + str(region) + ".pkl")
           del model, X_train, y_train, X_valid, y_valid
           validModel101(validResult)
import joblib
def validModel101(df):
    data = pd.read_csv('../数据/过程中数据/FeaturedEncodedTrainData_2015_201901.csv')
    model = joblib.load("../模型/model_102.pkl")
    newdf = data[data['sales_region_code'] == 102]
    X_valid, y_valid = newdf[(newdf['days_range'] >= 854) & (newdf['days_range'] <= 943)].drop('ord_qty', axis=1), \
                       newdf[(newdf['days_range'] >= 854) & (newdf['days_range'] <= 943)]['ord_qty']
    s = model.predict(X_valid)
    x_axis = np.linspace(1, len(y_valid), len(y_valid))
    plt.title('region_code_104')
    plt.ylabel('ord_qty')
    plt.plot(x_axis[:500], y_valid[:500])
    plt.plot(x_axis[:500], s[:500])
    plt.legend(['true', 'prediction'])
    plt.show()
def searchBestParam():
    # 读取编码以及特征工程后的2015年1月到2018年月的数据
    Train15_1901DF = pd.read_csv('../数据/过程中数据/FeaturedEncodedTrainData_2015_201901.csv', encoding='utf-8')
    regionedTrain15_1901DF = Train15_1901DF[Train15_1901DF['sales_region_code'] == 101]
    # 划分训练集和验证集
    X_train, y_train = regionedTrain15_1901DF[regionedTrain15_1901DF['days_range'] < 1208].drop('ord_qty', axis=1), \
                       regionedTrain15_1901DF[regionedTrain15_1901DF['days_range'] < 1208]['ord_qty']
    parameters = {
        'learning_rate': [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1],
        'n_estimators': [1000,1500, 2000,2500,3000],
    }
    # 'n_estimators': [100, 300, 500, 1000, 1500, 2000],
    # 'max_depth': [3, 4, 5, 6, 7, 8],
    # 'num_leaves': [5, 7, 8, 9, 10, 13, 15, 17, 19, 21, 23, 27, 31, 63],
    # 'subsample': [0.6, 0.7, 0.8, 1.0],
    # 'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
    # 'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.1]
    model = LGBMRegressor(
        #n_estimators=1000,
        #learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        max_depth=8,
        num_leaves=50,
        min_child_weight=300


    )
    # 有了gridsearch我们便不需要fit函数


    gsearch = GridSearchCV(model, param_grid=parameters,scoring='r2',cv= 3, verbose=1, n_jobs=-1)
    gs_results=gsearch.fit(X_train, y_train)
    print("BEST PARAMETERS: " + str(gs_results.best_params_))
    print("BEST CV SCORE: " + str(gs_results.best_score_))


#查看模型特征重要性
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:20].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (averaged over store predictions)')
    plt.show()
#查看模型特征重要性函数
def model_feature_importance():
    import os
    data = pd.read_pickle('../数据/过程中数据/FeaturedEncodedTrainData_2015_201901.pkl')
    feature_importance_df = pd.DataFrame()
    features = [f for f in data.columns if f != 'ord_qty']
    for filename in os.listdir('../模型/'):
        print(filename)
        if 'model' in filename:
            # load model
            model = joblib.load("../模型/" + filename)
            store_importance_df = pd.DataFrame()
            store_importance_df["feature"] = features
            store_importance_df["importance"] = model.feature_importances_
            store_importance_df["store"] = filename[5:9]
            feature_importance_df = pd.concat([feature_importance_df, store_importance_df], axis=0)
    display_importances(feature_importance_df)

#构造出2019年1月的数据，并且将数据编码后合合并保存
def addDataOf201901():
    #读取官方文件的2019年1月数据predict_sku1.csv
    sku1DF = pd.read_csv('../数据/官方数据_完整版/predict_sku1.csv', encoding='utf-8')
    sku1DF['combination'] = sku1DF['sales_region_code'].astype(str) + '_' + sku1DF[
        'first_cate_code'].astype(str) + '_' + sku1DF['second_cate_code'].astype(str) + '_' + sku1DF[
                                      'item_code'].astype(str)
    # 读取2015年到2018年编码处理后的数据
    train15_18DF = pd.read_csv('../数据/过程中数据/EncodedTrainData_2015_2018.csv', encoding='utf-8')
    train15_18DF['combination'] = train15_18DF['sales_region_code'].astype(str) + '_' + train15_18DF['first_cate_code'].astype(str) + '_' + train15_18DF['second_cate_code'].astype(str) + '_' + train15_18DF[
        'item_code'].astype(str)
    #将两张数据表进行合并，这样得到的2019年1月的待预测数据就没有缺失的商品了
    Completedtrain15_18DF=pd.concat([train15_18DF,sku1DF],axis=0)
    dateList = pd.date_range(start="20181221", end="20190131", freq="D")
    dates = []
    demandList = []
    combinationList = []
    for d in dateList:
        date = str(d).split(' ')[0]
        dates = dates + [date] * len(set(Completedtrain15_18DF['combination']))
        demandList = demandList + [0] * len(set(Completedtrain15_18DF['combination']))
        combinationList = combinationList + list(set(Completedtrain15_18DF['combination']))
    #构造2019年1月的数据（订单需求量设置为0），并进行编码
    train201901DF = pd.DataFrame()
    train201901DF['order_date'] = dates
    train201901DF['combination'] = combinationList
    train201901DF['ord_qty'] = demandList
    train201901DF['sales_region_code'] = train201901DF['combination'].str.split('_', expand=True)[0]
    train201901DF['first_cate_code'] = train201901DF['combination'].str.split('_', expand=True)[1]
    train201901DF['second_cate_code'] = train201901DF['combination'].str.split('_', expand=True)[2]
    train201901DF['item_code'] = train201901DF['combination'].str.split('_', expand=True)[3]
    train201901DF['order_date'] = pd.to_datetime(train201901DF['order_date'])
    # 构造2019年1月数据的促销列
    train201901DF['order_date'] = pd.to_datetime(train201901DF['order_date'])
    salesPromotionDate = ['1-1', '2-14', '3-8', '5-1', '6-1', '6-18','10-1', '11-11', '12-12']
    salesPromotionList = []
    for index, row in train201901DF.iterrows():
        str_date = str(row['order_date'].month) + '-' + str(row['order_date'].day)
        if str_date in salesPromotionDate:
            salesPromotionList.append(1)
        else:
            salesPromotionList.append(0)
    train201901DF['promotion'] = salesPromotionList
    # 构造2019年1月数据的价格区间列
    priceDF = Completedtrain15_18DF[['combination', 'price_range']]
    priceDF.drop_duplicates(['combination'], keep='last', inplace=True)
    priceList = {}
    for i in range(len(priceDF)):
        priceList[list(priceDF['combination'])[i]] = list(priceDF['price_range'])[i]
    prices = []
    for i in train201901DF['combination']:
        prices.append(priceList[i])
    train201901DF['price_range'] = prices
    # 构造2019年1月数据的销售渠道列
    channelDF = Completedtrain15_18DF[['combination','sales_chan_name']]
    channelDF.drop_duplicates(['combination'], keep='last', inplace=True)
    channelList = {}
    for i in range(len(channelDF)):
        channelList[list(channelDF['combination'])[i]] = list(channelDF['sales_chan_name'])[i]
    channel = []
    for i in train201901DF['combination']:
        channel.append(channelList[i])
    train201901DF['sales_chan_name'] = channel
    # 构造2019年1月数据的月段列
    #'月初': 0, '月中': 1, '月末': 2
    monthList = []
    for index, row in train201901DF.iterrows():
        if row['order_date'].day <= 10:
            monthList.append('0')
        elif row['order_date'].day > 10 and row['order_date'].day <= 20:
            monthList.append('1')
        elif row['order_date'].day > 20:
            monthList.append('2')
    train201901DF['monthe_period'] = monthList
    # 构造2019年1月数据的季节列
    #'春': 0, '夏': 1, '秋': 2, '冬': 3
    seasonList = []
    for index, row in train201901DF.iterrows():
        if row['order_date'].month == 3 or row['order_date'].month == 4 or row['order_date'].month == 5:
            seasonList.append('0')
        elif row['order_date'].month == 6 or row['order_date'].month == 7 or row['order_date'].month == 8:
            seasonList.append('1')
        elif row['order_date'].month == 9 or row['order_date'].month == 10 or row['order_date'].month == 11:
            seasonList.append('2')
        elif row['order_date'].month == 12 or row['order_date'].month == 1 or row['order_date'].month == 2:
            seasonList.append('3')
    train201901DF['seaon'] = seasonList
    # 构造2019年1月数据的星期列
    # 0-6  {0, 1, 2, 3, 4, 5, 6}
    train201901DF['day_of_week'] = train201901DF.order_date.dt.weekday
    # 构造2019年1月数据的是否工作日列
    train201901DF['workday'] = train201901DF['order_date'].map(lambda x: is_workday(x))
    dict_is_workday = {False: 0, True: 1}
    train201901DF['workday'] =train201901DF.workday.map(dict_is_workday)
    # 构建年份列表
    yearList = []
    for index, row in train201901DF.iterrows():
        yearList.append(row['order_date'].year)
    train201901DF['year'] = yearList
    # 构建月份列表
    monthList = []
    for index, row in train201901DF.iterrows():
        monthList.append(row['order_date'].month)
    train201901DF['month'] = monthList
    # 构建周列表
    weekList = []
    for index, row in train201901DF.iterrows():
        weekList.append(row['order_date'].week)
    train201901DF['week_of_year'] = weekList
    # 构建一年中的天数列表
    train201901DF['day_of_year'] = train201901DF.order_date.dt.dayofyear
    # 构造一个月中的天数列表
    dayList = []
    for index, row in train201901DF.iterrows():
        dayList.append(row['order_date'].day)
    train201901DF['day_of_month'] = dayList
    #构造2019年1月数据的天数列
    dateList = pd.date_range(start="20150901", end="20190131", freq="D")
    daysList = {}
    i = 1
    for date in dateList:
        daysList[str(date).split(' ')[0]] = i
        i += 1
    days = []
    for dat in train201901DF['order_date']:
        days.append(daysList[str(dat).split(' ')[0]])
    train201901DF['days_range'] = days
    #删除辅助2019年1月数据构造的组合编码列
    train201901DF.drop(['combination'], axis=1, inplace=True)
    # 保存数据到train201901文件
    train201901DF.to_csv('../数据/过程中数据/EncodedTrainData_201901.csv', index=False, encoding='utf-8')
    # train15_18DF.drop(['combination'], axis=1, inplace=True)
    # train15_18DF.to_csv('../数据/过程中数据/EncodedTrainData_2015_2018.csv', index=False, encoding='utf-8')
    #拼接2015年到2018年的真实数据和201901年自己构造的待预测数据
    train2015_201901DF=pd.concat([train15_18DF,train201901DF],axis=0)
    train2015_201901DF.drop(['combination'], axis=1, inplace=True)
    train2015_201901DF.to_csv('../数据/过程中数据/EncodedTrainData_2015_201901.csv', index=False, encoding='utf-8')

#对2015年到2019年01月编码处理后的数据进行特征工程
def featureEngineeringOf2015_201901():
    # 读取2015年到2019年01月编码处理后的数据
    train15_1901DF = pd.read_csv('../数据/过程中数据/EncodedTrainData_2015_201901.csv', encoding='utf-8')

    #均值编码，计算的是2015年到2018年的均值
    s = train15_1901DF.groupby('item_code')['ord_qty'].transform('mean')
    train15_1901DF['item_avg'] = train15_1901DF.groupby('item_code')['ord_qty'].transform('mean').astype(np.float16)
    train15_1901DF['sales_region_code_need_avg'] = train15_1901DF.groupby('sales_region_code')['ord_qty'].transform('mean').astype(np.float16)
    train15_1901DF['first_cate_code_need_avg'] = train15_1901DF.groupby('first_cate_code')['ord_qty'].transform('mean').astype(np.float16)
    train15_1901DF['second_cate_code_need_avg'] = train15_1901DF.groupby('second_cate_code')['ord_qty'].transform('mean').astype(np.float16)
    train15_1901DF['sales_chan_name_need_avg'] = train15_1901DF.groupby('sales_chan_name')['ord_qty'].transform('mean').astype(np.float16)
    train15_1901DF['sales_region_code_item_code_need_avg'] = train15_1901DF.groupby(['item_code', 'sales_region_code'])['ord_qty'].transform('mean').astype(np.float16)
    train15_1901DF['first_cate_code_item_code_need_avg'] = train15_1901DF.groupby(['first_cate_code', 'item_code'])['ord_qty'].transform('mean').astype(np.float16)
    train15_1901DF['second_cate_code_item_code_need_avg'] = train15_1901DF.groupby(['second_cate_code', 'item_code'])['ord_qty'].transform('mean').astype(np.float16)
    train15_1901DF['sales_chan_name_item_code_need_avg'] =train15_1901DF.groupby(['sales_chan_name', 'item_code'])['ord_qty'].transform('mean').astype(np.float16)
    train15_1901DF['sales_region_code_first_cate_code_need_avg'] = train15_1901DF.groupby(['sales_region_code', 'first_cate_code'])['ord_qty'].transform('mean').astype(np.float16)
    train15_1901DF['first_cate_code_second_cate_code_need_avg'] = train15_1901DF.groupby(['first_cate_code', 'second_cate_code'])['ord_qty'].transform('mean').astype(np.float16)
    train15_1901DF['sales_chan_name_second_cate_code_need_avg'] = train15_1901DF.groupby(['sales_chan_name', 'second_cate_code'])['ord_qty'].transform('mean').astype(np.float16)
    #将订单需求列空值填充为0
    train15_1901DF['ord_qty'].fillna(0, inplace=True)

    #滑动窗口统计
    train15_1901DF['rolling_need_mean'] = train15_1901DF.groupby(['sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code'])['ord_qty'].transform(
        lambda x:x.rolling(window=7).mean()).astype(np.float16)
    #开窗数据统计
    train15_1901DF['expanding_need_mean'] = train15_1901DF.groupby(['sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code'])['ord_qty'].transform(
        lambda x: x.expanding(2).mean()).astype(np.float16)
    #需求量趋势构建
    train15_1901DF['daily_avg_need'] = train15_1901DF.groupby(['sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code','days_range'])['ord_qty'].transform('mean').astype(
        np.float16)
    train15_1901DF['avg_need'] = train15_1901DF.groupby(['sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code'])['ord_qty'].transform('mean').astype(np.float16)
    train15_1901DF['need_trend'] = (train15_1901DF['daily_avg_need'] - train15_1901DF['avg_need']).astype(np.float16)
    train15_1901DF.drop(['daily_avg_need', 'avg_need'], axis=1, inplace=True)
    train15_1901DF.drop('order_date', axis=1, inplace=True)
    # 滞后特征
    lags = [1, 2, 3, 6, 12, 24, 36, 48, 60]
    for lag in lags:
        train15_1901DF['need_lag_' + str(lag)] = \
        train15_1901DF.groupby(['sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code', ],
                               as_index=False)[
            'ord_qty'].shift(lag).astype(np.float16)
    #保存数据
    featureedTrain15_1901DF = train15_1901DF[train15_1901DF['days_range']>=60]
    featureedTrain15_1901DF.to_pickle('../数据/过程中数据/FeaturedEncodedTrainData_2015_201901.pkl')
    featureedTrain15_1901DF.to_csv('../数据/过程中数据/FeaturedEncodedTrainData_2015_201901.csv', index=False, encoding='utf-8')


#构建模型为训练集和测试集用来预测2019年1月份的销售量
#读取编码以及特征工程后的2015年1月到2018年月的数据
def ModelPredict201901():
    # 读取编码以及特征工程后的2015年1月到2018年月的数据
    Train15_1901DF = pd.read_csv('../数据/过程中数据/FeaturedEncodedTrainData_2015_201901.csv', encoding='utf-8')
    Train15_1901DF.drop('item_price', axis=1, inplace=True)
    # 划分验证集和测试集
    valid = Train15_1901DF[(Train15_1901DF['days_range'] >= 1177) & (Train15_1901DF['days_range'] < 1208)][['item_code', 'days_range', 'ord_qty']]
    test = Train15_1901DF[(Train15_1901DF['days_range'] >= 1208)][['item_code', 'days_range', 'ord_qty']]
    # 测试集训练预测和验证集预测结果
    testResult = test['ord_qty']  # 这是已有真实标签需求量1175到1206，31天间隔的真实数据
    validResult = valid['ord_qty']  # 这是已有真实标签需求量1207到1276，70天间隔的真实数据
    # 对五个销售区域分别建模并进行训练
    regionList = [101, 102, 103, 104, 105]
    for region in regionList:
        try:
            regionedTrain15_1901DF = Train15_1901DF[Train15_1901DF['sales_region_code'] == region]
            # 划分训练集和验证集
            X_train, y_train = regionedTrain15_1901DF[regionedTrain15_1901DF['days_range'] <= 1207].drop('ord_qty', axis=1), \
                               regionedTrain15_1901DF[regionedTrain15_1901DF['days_range'] <= 1207]['ord_qty']
            X_valid, y_valid = regionedTrain15_1901DF[(regionedTrain15_1901DF['days_range'] < 1208)&(regionedTrain15_1901DF['days_range'] >=1177)].drop('ord_qty', axis=1), \
                               regionedTrain15_1901DF[(regionedTrain15_1901DF['days_range'] < 1208)&(regionedTrain15_1901DF['days_range'] >=1177)]['ord_qty']
            X_test= regionedTrain15_1901DF[regionedTrain15_1901DF['days_range'] >= 1208].drop('ord_qty', axis=1)
            # 构建模型并模型
            model = LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                max_depth=3,
                num_leaves=8,
                min_child_weight=300
            )
            print('*****Prediction for 销售区域: {}*****'.format(region))
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)],
                      eval_metric='rmse', verbose=20, early_stopping_rounds=100)
            validResult[X_valid.index] = model.predict(X_valid)
            testResult[X_test.index] = model.predict(X_test)
            # 保存模型文件数据
            joblib.dump(model, "../模型/" + "model_" + str(region) + ".pkl")
            del model, X_train, y_train, X_valid, y_valid
        except:
            del model, X_train, y_train, X_valid, y_valid
            continue
    #预测完成2019年1月份的订单需求量后，将结果保存到文件中
    result = []
    for i in testResult:
        if i < 0:
            result.append(0)
        else:
            result.append(round(i, 0))
    #读取之前构造好的19年1月数据,将预测的订单需求量替换
    train201901DF = pd.read_csv('../数据/过程中数据/EncodedTrainData_201901.csv')
    train201901DF['ord_qty'] = result  # 把预测的订单需求量替换
    train201901DF.to_csv('../数据/结果数据/ForecastedEncodedTrainData_201901.csv', index=False)
    #将预测后的2019年1月的数据合并到官方存在的的2015年1月到2018年12月的数据中
    tain2015_2018DF = pd.read_csv('../数据/过程中数据/EncodedTrainData_2015_2018.csv')
    PredictedEncodedTrainData_2015_201901DF = pd.concat([tain2015_2018DF, train201901DF], axis=0)
    PredictedEncodedTrainData_2015_201901DF.to_csv('../数据/过程中数据/PredictedEncodedTrainData_2015_201901.csv', index=False)
    #计算出一月总销售额
    forecasted201901DF = train201901DF.loc[train201901DF['days_range'] >= 1219]  # 定位到19年1月的数据
    forecasted201901DF = forecasted201901DF[['order_date', 'sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code', 'ord_qty']]  # 筛选出需要的列
    resultDF = pd.pivot_table(forecasted201901DF, index=['sales_region_code', 'first_cate_code', 'second_cate_code', 'item_code'], columns='order_date',
                               values='ord_qty', aggfunc=np.sum, fill_value=0).reset_index()
    resultDF['January_demand'] = resultDF.iloc[:, 4:].sum(axis=1)  # 对0，1列按行求和，生成新列
    resultDF = resultDF[['sales_region_code', 'first_cate_code', 'second_cate_code', 'item_code', 'January_demand']]
    #读取预测的sku1的数据
    predictDF = pd.read_csv("../数据/官方数据_完整版/predict_sku1.csv")
    predictDF.columns = ['sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code']
    s = pd.merge(predictDF, resultDF, how='left')
    s.to_csv('../数据/结果数据/ForecastedData_201901.csv', index=False)


#构造出2019年2月的数据，并且将数据编码后合合并保存
def addDataOf201902():
    # 读取官方文件的2019年1月数据predict_sku1.csv
    sku1DF = pd.read_csv('../数据/官方数据_完整版/predict_sku1.csv', encoding='utf-8')
    sku1DF['combination'] = sku1DF['sales_region_code'].astype(str) + '_' + sku1DF[
        'first_cate_code'].astype(str) + '_' + sku1DF['second_cate_code'].astype(str) + '_' + sku1DF[
                                'item_code'].astype(str)
    # 读取2015年到2019年1月预测后并且编码处理后的数据
    train15_1901DF = pd.read_csv('../数据/过程中数据/PredictedEncodedTrainData_2015_201901.csv', encoding='utf-8')
    train15_1901DF['combination'] = train15_1901DF['sales_region_code'].astype(str) + '_' + train15_1901DF['first_cate_code'].astype(str) + '_' + train15_1901DF['second_cate_code'].astype(str) + '_' + train15_1901DF[
        'item_code'].astype(str)
    Completedtrain15_1901DF= pd.concat([train15_1901DF, sku1DF], axis=0)
    dateList = pd.date_range(start="20190201", end="20190228", freq="D")
    dates = []
    demandList = []
    combinationList = []
    for d in dateList:
        date = str(d).split(' ')[0]
        dates = dates + [date] * len(set(Completedtrain15_1901DF['combination']))
        demandList = demandList + [0] * len(set(Completedtrain15_1901DF['combination']))
        combinationList = combinationList + list(set(Completedtrain15_1901DF['combination']))
    #构造2019年2月的数据（订单需求量设置为0），并进行编码
    train201902DF = pd.DataFrame()
    train201902DF['order_date'] = dates
    train201902DF['combination'] = combinationList
    train201902DF['ord_qty'] = demandList
    train201902DF['sales_region_code'] = train201902DF['combination'].str.split('_', expand=True)[0]
    train201902DF['first_cate_code'] = train201902DF['combination'].str.split('_', expand=True)[1]
    train201902DF['second_cate_code'] = train201902DF['combination'].str.split('_', expand=True)[2]
    train201902DF['item_code'] = train201902DF['combination'].str.split('_', expand=True)[3]
    train201902DF['order_date'] = pd.to_datetime(train201902DF['order_date'])  # 对日期列进行日期格式转换
    # 构造2019年1月数据的促销列
    train201902DF['order_date'] = pd.to_datetime(train201902DF['order_date'])
    salesPromotionDate = ['1-1', '2-14', '3-8', '5-1', '6-1', '6-18','10-1', '11-11', '12-12']
    salesPromotionList = []
    for index, row in train201902DF.iterrows():
        str_date = str(row['order_date'].month) + '-' + str(row['order_date'].day)
        if str_date in salesPromotionDate:
            salesPromotionList.append(1)
        else:
            salesPromotionList.append(0)
    train201902DF['promotion'] = salesPromotionList
    # 构造2019年1月数据的价格区间列
    priceDF = Completedtrain15_1901DF[['combination', 'price_range']]
    priceDF.drop_duplicates(['combination'], keep='last', inplace=True)
    priceList = {}
    for i in range(len(priceDF)):
        priceList[list(priceDF['combination'])[i]] = list(priceDF['price_range'])[i]
    prices = []
    for i in train201902DF['combination']:
        prices.append(priceList[i])
    train201902DF['price_range'] = prices
    # 构造2019年1月数据的销售渠道列
    channelDF = Completedtrain15_1901DF[['combination','sales_chan_name']]
    channelDF.drop_duplicates(['combination'], keep='last', inplace=True)
    channelList = {}
    for i in range(len(channelDF)):
        channelList[list(channelDF['combination'])[i]] = list(channelDF['sales_chan_name'])[i]
    channel = []
    for i in train201902DF['combination']:
        channel.append(channelList[i])
    train201902DF['sales_chan_name'] = channel
    # 构造2019年1月数据的月段列
    #'月初': 0, '月中': 1, '月末': 2
    monthList = []
    for index, row in train201902DF.iterrows():
        if row['order_date'].day <= 10:
            monthList.append('0')
        elif row['order_date'].day > 10 and row['order_date'].day <= 20:
            monthList.append('1')
        elif row['order_date'].day > 20:
            monthList.append('2')
    train201902DF['monthe_period'] = monthList
    # 构造2019年1月数据的季节列
    #'春': 0, '夏': 1, '秋': 2, '冬': 3
    seasonList = []
    for index, row in train201902DF.iterrows():
        if row['order_date'].month == 3 or row['order_date'].month == 4 or row['order_date'].month == 5:
            seasonList.append('0')
        elif row['order_date'].month == 6 or row['order_date'].month == 7 or row['order_date'].month == 8:
            seasonList.append('1')
        elif row['order_date'].month == 9 or row['order_date'].month == 10 or row['order_date'].month == 11:
            seasonList.append('2')
        elif row['order_date'].month == 12 or row['order_date'].month == 1 or row['order_date'].month == 2:
            seasonList.append('3')
    train201902DF['seaon'] = seasonList
    # 构造2019年1月数据的星期列
    # 0-6  {0, 1, 2, 3, 4, 5, 6}
    train201902DF['day_of_week'] = train201902DF.order_date.dt.weekday
    # 构造2019年1月数据的是否工作日列
    train201902DF['workday'] = train201902DF['order_date'].map(lambda x: is_workday(x))
    dict_is_workday = {False: 0, True: 1}
    train201902DF['workday'] =train201902DF.workday.map(dict_is_workday)
    # 构建年份列表
    yearList = []
    for index, row in train201902DF.iterrows():
        yearList.append(row['order_date'].year)
    train201902DF['year'] = yearList
    # 构建月份列表
    monthList = []
    for index, row in train201902DF.iterrows():
        monthList.append(row['order_date'].month)
    train201902DF['month'] = monthList
    # 构建周列表
    weekList = []
    for index, row in train201902DF.iterrows():
        weekList.append(row['order_date'].week)
    train201902DF['week_of_year'] = weekList
    # 构建一年中的天数列表
    train201902DF['day_of_year'] = train201902DF.order_date.dt.dayofyear
    # 构造一个月中的天数列表
    dayList = []
    for index, row in train201902DF.iterrows():
        dayList.append(row['order_date'].day)
    train201902DF['day_of_month'] = dayList
    #构造2019年2月数据的天数列
    dateList = pd.date_range(start="20150901", end="20190228", freq="D")
    daysList = {}
    i = 1
    for date in dateList:
        daysList[str(date).split(' ')[0]] = i
        i += 1
    days = []
    for dat in train201902DF['order_date']:
        days.append(daysList[str(dat).split(' ')[0]])
    train201902DF['days_range'] = days
    #删除辅助2019年1月数据构造的组合编码列
    train201902DF.drop(['combination'], axis=1, inplace=True)
    # 保存数据到train201901文件
    train201902DF.to_csv('../数据/过程中数据/EncodedTrainData_201902.csv', index=False, encoding='utf-8')
    #拼接2015年到2018年的真实数据和201901年自己构造的待预测数据
    train2015_201901DF=pd.concat([train15_1901DF,train201902DF],axis=0)
    train2015_201901DF.drop(['combination'], axis=1, inplace=True)
    train2015_201901DF.to_csv('../数据/过程中数据/EncodedTrainData_2015_201902.csv', index=False, encoding='utf-8')
#对2015年到2019年02月编码处理后的数据进行特征工程
def featureEngineeringOf2015_201902():
    # 读取2015年到2019年01月编码处理后的数据
    train15_1902DF = pd.read_csv('../数据/过程中数据/EncodedTrainData_2015_201902.csv', encoding='utf-8')

    #均值编码
    s = train15_1902DF.groupby('item_code')['ord_qty'].transform('mean')
    train15_1902DF['item_avg'] = train15_1902DF.groupby('item_code')['ord_qty'].transform('mean').astype(np.float16)
    train15_1902DF['sales_region_code_need_avg'] = train15_1902DF.groupby('sales_region_code')['ord_qty'].transform('mean').astype(np.float16)
    train15_1902DF['first_cate_code_need_avg'] = train15_1902DF.groupby('first_cate_code')['ord_qty'].transform('mean').astype(np.float16)
    train15_1902DF['second_cate_code_need_avg'] = train15_1902DF.groupby('second_cate_code')['ord_qty'].transform('mean').astype(np.float16)
    train15_1902DF['sales_chan_name_need_avg'] = train15_1902DF.groupby('sales_chan_name')['ord_qty'].transform('mean').astype(np.float16)
    train15_1902DF['sales_region_code_item_code_need_avg'] =train15_1902DF.groupby(['item_code', 'sales_region_code'])['ord_qty'].transform('mean').astype(np.float16)
    train15_1902DF['first_cate_code_item_code_need_avg'] = train15_1902DF.groupby(['first_cate_code', 'item_code'])['ord_qty'].transform('mean').astype(np.float16)
    train15_1902DF['second_cate_code_item_code_need_avg'] = train15_1902DF.groupby(['second_cate_code', 'item_code'])['ord_qty'].transform('mean').astype(np.float16)
    train15_1902DF['sales_chan_name_item_code_need_avg'] =train15_1902DF.groupby(['sales_chan_name', 'item_code'])['ord_qty'].transform('mean').astype(np.float16)
    train15_1902DF['sales_region_code_first_cate_code_need_avg'] = train15_1902DF.groupby(['sales_region_code', 'first_cate_code'])['ord_qty'].transform('mean').astype(np.float16)
    train15_1902DF['first_cate_code_second_cate_code_need_avg'] = train15_1902DF.groupby(['first_cate_code', 'second_cate_code'])['ord_qty'].transform('mean').astype(np.float16)
    train15_1902DF['sales_chan_name_second_cate_code_need_avg'] = train15_1902DF.groupby(['sales_chan_name', 'second_cate_code'])['ord_qty'].transform('mean').astype(np.float16)
    # 将订单需求列空值填充为0
    train15_1902DF['ord_qty'].fillna(0, inplace=True)
    # 滞后特征
    lags = [1, 2, 3, 6, 12, 24, 36, 48, 60]
    for lag in lags:
        train15_1902DF['need_lag_' + str(lag)] = \
        train15_1902DF.groupby(['sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code', ],
                               as_index=False)[
            'ord_qty'].shift(lag).astype(np.float16)
    #滑动窗口统计
    train15_1902DF['rolling_need_mean'] = train15_1902DF.groupby(['sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code'])['ord_qty'].transform(
        lambda x: x.rolling(window=7).mean()).astype(np.float16)
    #开窗数据统计
    train15_1902DF['expanding_need_mean'] = train15_1902DF.groupby(['sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code'])['ord_qty'].transform(
        lambda x: x.expanding(2).mean()).astype(np.float16)
    #需求量趋势构建
    train15_1902DF['daily_avg_need'] = train15_1902DF.groupby(['sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code','days_range'])['ord_qty'].transform('mean').astype(np.float16)
    train15_1902DF['avg_need'] = train15_1902DF.groupby(['sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code'])['ord_qty'].transform('mean').astype(np.float16)
    train15_1902DF['need_trend'] = (train15_1902DF['daily_avg_need'] - train15_1902DF['avg_need']).astype(np.float16)
    train15_1902DF.drop(['daily_avg_need', 'avg_need'], axis=1, inplace=True)
    train15_1902DF.drop('order_date', axis=1, inplace=True)
    #保存数据
    featureedTrain15_1902DF = train15_1902DF[train15_1902DF['days_range']>=60]
    featureedTrain15_1902DF.to_pickle('../数据/过程中数据/FeaturedEncodedTrainData_2015_201902.pkl')
    featureedTrain15_1902DF.to_csv('../数据/过程中数据/FeaturedEncodedTrainData_2015_201902.csv', index=False, encoding='utf-8')

#读取编码以及特征工程后的2015年1月到2018年月的数据
#为什么这一块的数据，从1207开始，就没有了，然后突然到1250才出现结果？
def ModelPredict201902():
    # 读取编码以及特征工程后的2015年1月到2018年月的数据
    Train15_1902DF = pd.read_csv('../数据/过程中数据/FeaturedEncodedTrainData_2015_201902.csv', encoding='utf-8')
    Train15_1902DF.drop('item_price', axis=1, inplace=True)
    # 划分验证集和测试集
    valid=Train15_1902DF[(Train15_1902DF['days_range']<1250)&(Train15_1902DF['days_range'] >=1208)][['item_code', 'days_range', 'ord_qty']]
    test = Train15_1902DF[(Train15_1902DF['days_range'] >= 1250)][['item_code', 'days_range', 'ord_qty']]
    # 测试集训练预测和验证集预测结果
    testResult = test['ord_qty']  # 这是已有真实标签需求量1175到1206，31天间隔的真实数据
    validResult = valid['ord_qty']  # 这是已有真实标签需求量1207到1237，31天间隔的真实数据
    # 对五个销售区域分别建模并进行训练
    regionList = [101, 102, 103, 104, 105]
    for region in regionList:
        #try:
            regionedTrain15_1902DF = Train15_1902DF[Train15_1902DF['sales_region_code'] == region]
            # 划分训练集和验证集
            X_train, y_train = regionedTrain15_1902DF[regionedTrain15_1902DF['days_range'] < 1208].drop('ord_qty', axis=1), \
                               regionedTrain15_1902DF[regionedTrain15_1902DF['days_range'] < 1208]['ord_qty']
            X_valid, y_valid = regionedTrain15_1902DF[(regionedTrain15_1902DF['days_range'] < 1250)&(regionedTrain15_1902DF['days_range'] >= 1208)].drop('ord_qty', axis=1), \
                               regionedTrain15_1902DF[(regionedTrain15_1902DF['days_range'] < 1250)&(regionedTrain15_1902DF['days_range'] >= 1208)]['ord_qty']
            X_test = regionedTrain15_1902DF[regionedTrain15_1902DF['days_range'] >= 1250].drop('ord_qty', axis=1)
            # 构建模型并模型
            model = LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                max_depth=3,
                num_leaves=8,
                min_child_weight=300
            )
            print('*****Prediction for 销售区域: {}*****'.format(region))
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)],
                      eval_metric='rmse', verbose=20, early_stopping_rounds=100)
            validResult[X_valid.index] = model.predict(X_valid)
            testResult[X_test.index] = model.predict(X_test)
            # 保存模型文件数据
            joblib.dump(model, "../模型/" + "model_" + str(region) + ".pkl")
            del model, X_train, y_train, X_valid, y_valid
        # except:
        #     del model, X_train, y_train, X_valid, y_valid
        #     continue
    #预测完成2019年2月份的订单需求量后，将结果保存到文件中
    result = []
    for i in testResult:
        if i < 0:
            result.append(0)
        else:
            result.append(round(i, 0))
    #读取之前构造好的19年1月数据,将预测的订单需求量替换
    train201902DF = pd.read_csv('../数据/过程中数据/EncodedTrainData_201902.csv')
    train201902DF['ord_qty'] = result  # 把预测的订单需求量替换
    train201902DF.to_csv('../数据/结果数据/ForecastedEncodedTrainData_201902.csv', index=False)
    #将预测后的2019年1月的数据合并到官方存在的的2015年1月到2018年12月的数据中
    tain2015_2018DF = pd.read_csv('../数据/过程中数据/EncodedTrainData_2015_2018.csv')
    PredictedEncodedTrainData_2015_201901DF = pd.concat([tain2015_2018DF, train201902DF], axis=0)
    PredictedEncodedTrainData_2015_201901DF.to_csv('../数据/过程中数据/PredictedEncodedTrainData_2015_201902.csv', index=False)
    #计算出而月总销售额
    forecasted201902DF = train201902DF.loc[train201902DF['days_range'] >= 1250]  # 定位到19年2月的数据
    forecasted201902DF = forecasted201902DF[['order_date', 'sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code', 'ord_qty']]  # 筛选出需要的列
    resultDF = pd.pivot_table(forecasted201902DF, index=['sales_region_code', 'first_cate_code', 'second_cate_code', 'item_code'], columns='order_date',
                               values='ord_qty', aggfunc=np.sum, fill_value=0).reset_index()
    resultDF['February_demand'] = resultDF.iloc[:, 4:].sum(axis=1)  # 对0，1列按行求和，生成新列
    resultDF = resultDF[['sales_region_code', 'first_cate_code', 'second_cate_code', 'item_code', 'February_demand']]
    #读取预测的sku1的数据
    predictDF = pd.read_csv("../数据/官方数据_完整版/predict_sku1.csv")
    predictDF.columns = ['sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code']
    s = pd.merge(predictDF, resultDF, how='inner')
    s.to_csv('../数据/结果数据/ForecastedData_201902.csv', index=False)

#构造出2019年3月的数据，并且将数据编码后合合并保存
def addDataOf201903():
    # 读取官方文件的2019年1月数据predict_sku1.csv
    sku1DF = pd.read_csv('../数据/官方数据_完整版/predict_sku1.csv', encoding='utf-8')
    sku1DF['combination'] = sku1DF['sales_region_code'].astype(str) + '_' + sku1DF[
        'first_cate_code'].astype(str) + '_' + sku1DF['second_cate_code'].astype(str) + '_' + sku1DF[
                                'item_code'].astype(str)
    # 读取2015年到2018 年编码处理后的数据
    train15_1903DF = pd.read_csv('../数据/过程中数据/PredictedEncodedTrainData_2015_201902.csv', encoding='utf-8')
    train15_1903DF['combination'] = train15_1903DF['sales_region_code'].astype(str) + '_' + train15_1903DF['first_cate_code'].astype(str) + '_' + train15_1903DF['second_cate_code'].astype(str) + '_' + train15_1903DF[
        'item_code'].astype(str)
    Completedtrain15_1902DF = pd.concat([train15_1903DF, sku1DF], axis=0)
    dateList = pd.date_range(start="20190301", end="20190331", freq="D")
    dates = []
    demandList = []
    combinationList = []
    for d in dateList:
        date = str(d).split(' ')[0]
        dates = dates + [date] * len(set(Completedtrain15_1902DF['combination']))
        demandList = demandList + [0] * len(set(Completedtrain15_1902DF['combination']))
        combinationList = combinationList + list(set(Completedtrain15_1902DF['combination']))
    #构造2019年3月的数据（订单需求量设置为0），并进行编码
    train201903DF = pd.DataFrame()
    train201903DF['order_date'] = dates
    train201903DF['combination'] = combinationList
    train201903DF['ord_qty'] = demandList
    train201903DF['sales_region_code'] = train201903DF['combination'].str.split('_', expand=True)[0]
    train201903DF['first_cate_code'] = train201903DF['combination'].str.split('_', expand=True)[1]
    train201903DF['second_cate_code'] = train201903DF['combination'].str.split('_', expand=True)[2]
    train201903DF['item_code'] = train201903DF['combination'].str.split('_', expand=True)[3]
    train201903DF['order_date'] = pd.to_datetime(train201903DF['order_date'])  # 对日期列进行日期格式转换
    # 构造2019年1月数据的促销列
    train201903DF['order_date'] = pd.to_datetime(train201903DF['order_date'])
    salesPromotionDate = ['1-1', '2-14', '3-8', '5-1', '6-1', '6-18','10-1', '11-11', '12-12']
    salesPromotionList = []
    for index, row in train201903DF.iterrows():
        str_date = str(row['order_date'].month) + '-' + str(row['order_date'].day)
        if str_date in salesPromotionDate:
            salesPromotionList.append(1)
        else:
            salesPromotionList.append(0)
    train201903DF['promotion'] = salesPromotionList
    # 构造2019年3月数据的价格区间列
    priceDF = Completedtrain15_1902DF[['combination', 'price_range']]
    priceDF.drop_duplicates(['combination'], keep='last', inplace=True)
    priceList = {}
    for i in range(len(priceDF)):
        priceList[list(priceDF['combination'])[i]] = list(priceDF['price_range'])[i]
    prices = []
    for i in train201903DF['combination']:
        prices.append(priceList[i])
    train201903DF['price_range'] = prices
    # 构造2019年1月数据的销售渠道列
    channelDF = Completedtrain15_1902DF[['combination','sales_chan_name']]
    channelDF.drop_duplicates(['combination'], keep='last', inplace=True)
    channelList = {}
    for i in range(len(channelDF)):
        channelList[list(channelDF['combination'])[i]] = list(channelDF['sales_chan_name'])[i]
    channel = []
    for i in train201903DF['combination']:
        channel.append(channelList[i])
    train201903DF['sales_chan_name'] = channel
    # 构造2019年1月数据的月段列
    #'月初': 0, '月中': 1, '月末': 2
    monthList = []
    for index, row in train201903DF.iterrows():
        if row['order_date'].day <= 10:
            monthList.append('0')
        elif row['order_date'].day > 10 and row['order_date'].day <= 20:
            monthList.append('1')
        elif row['order_date'].day > 20:
            monthList.append('2')
    train201903DF['monthe_period'] = monthList
    # 构造2019年1月数据的季节列
    #'春': 0, '夏': 1, '秋': 2, '冬': 3
    seasonList = []
    for index, row in train201903DF.iterrows():
        if row['order_date'].month == 3 or row['order_date'].month == 4 or row['order_date'].month == 5:
            seasonList.append('0')
        elif row['order_date'].month == 6 or row['order_date'].month == 7 or row['order_date'].month == 8:
            seasonList.append('1')
        elif row['order_date'].month == 9 or row['order_date'].month == 10 or row['order_date'].month == 11:
            seasonList.append('2')
        elif row['order_date'].month == 12 or row['order_date'].month == 1 or row['order_date'].month == 2:
            seasonList.append('3')
    train201903DF['seaon'] = seasonList
    # 构造2019年1月数据的星期列
    # 0-6  {0, 1, 2, 3, 4, 5, 6}
    train201903DF['day_of_week'] = train201903DF.order_date.dt.weekday
    # 构造2019年1月数据的是否工作日列
    train201903DF['workday'] = train201903DF['order_date'].map(lambda x: is_workday(x))
    dict_is_workday = {False: 0, True: 1}
    train201903DF['workday'] =train201903DF.workday.map(dict_is_workday)
    # 构建年份列表
    yearList = []
    for index, row in train201903DF.iterrows():
        yearList.append(row['order_date'].year)
    train201903DF['year'] = yearList
    # 构建月份列表
    monthList = []
    for index, row in train201903DF.iterrows():
        monthList.append(row['order_date'].month)
    train201903DF['month'] = monthList
    # 构建周列表
    weekList = []
    for index, row in train201903DF.iterrows():
        weekList.append(row['order_date'].week)
    train201903DF['week_of_year'] = weekList
    # 构建一年中的天数列表
    train201903DF['day_of_year'] = train201903DF.order_date.dt.dayofyear
    # 构造一个月中的天数列表
    dayList = []
    for index, row in train201903DF.iterrows():
        dayList.append(row['order_date'].day)
    train201903DF['day_of_month'] = dayList
    #构造2019年3月数据的天数列
    dateList = pd.date_range(start="20150901", end="20190331", freq="D")
    daysList = {}
    i = 1
    for date in dateList:
        daysList[str(date).split(' ')[0]] = i
        i += 1
    days = []
    for dat in train201903DF['order_date']:
        days.append(daysList[str(dat).split(' ')[0]])
    train201903DF['days_range'] = days
    #删除辅助2019年1月数据构造的组合编码列
    train201903DF.drop(['combination'], axis=1, inplace=True)
    # 保存数据到train201901文件
    train201903DF.to_csv('../数据/过程中数据/EncodedTrainData_201903.csv', index=False, encoding='utf-8')
    #拼接2015年到2018年的真实数据和201901年自己构造的待预测数据
    train2015_201901DF=pd.concat([train15_1903DF,train201903DF],axis=0)
    train2015_201901DF.drop(['combination'], axis=1, inplace=True)
    train2015_201901DF.to_csv('../数据/过程中数据/EncodedTrainData_2015_201903.csv', index=False, encoding='utf-8')
#对2015年到2019年03月编码处理后的数据进行特征工程
def featureEngineeringOf2015_201903():
    # 读取2015年到2019年01月编码处理后的数据
    train15_1903DF = pd.read_csv('../数据/过程中数据/EncodedTrainData_2015_201903.csv', encoding='utf-8')

    #均值编码
    s = train15_1903DF.groupby('item_code')['ord_qty'].transform('mean')
    train15_1903DF['item_avg'] = train15_1903DF.groupby('item_code')['ord_qty'].transform('mean').astype(np.float16)
    train15_1903DF['sales_region_code_need_avg'] = train15_1903DF.groupby('sales_region_code')['ord_qty'].transform('mean').astype(np.float16)
    train15_1903DF['first_cate_code_need_avg'] = train15_1903DF.groupby('first_cate_code')['ord_qty'].transform('mean').astype(np.float16)
    train15_1903DF['second_cate_code_need_avg'] = train15_1903DF.groupby('second_cate_code')['ord_qty'].transform('mean').astype(np.float16)
    train15_1903DF['sales_chan_name_need_avg'] = train15_1903DF.groupby('sales_chan_name')['ord_qty'].transform('mean').astype(np.float16)
    train15_1903DF['sales_region_code_item_code_need_avg'] =train15_1903DF.groupby(['item_code', 'sales_region_code'])['ord_qty'].transform('mean').astype(np.float16)
    train15_1903DF['first_cate_code_item_code_need_avg'] = train15_1903DF.groupby(['first_cate_code', 'item_code'])['ord_qty'].transform('mean').astype(np.float16)
    train15_1903DF['second_cate_code_item_code_need_avg'] = train15_1903DF.groupby(['second_cate_code', 'item_code'])['ord_qty'].transform('mean').astype(np.float16)
    train15_1903DF['sales_chan_name_item_code_need_avg'] =train15_1903DF.groupby(['sales_chan_name', 'item_code'])['ord_qty'].transform('mean').astype(np.float16)
    train15_1903DF['sales_region_code_first_cate_code_need_avg'] = train15_1903DF.groupby(['sales_region_code', 'first_cate_code'])['ord_qty'].transform('mean').astype(np.float16)
    train15_1903DF['first_cate_code_second_cate_code_need_avg'] = train15_1903DF.groupby(['first_cate_code', 'second_cate_code'])['ord_qty'].transform('mean').astype(np.float16)
    train15_1903DF['sales_chan_name_second_cate_code_need_avg'] = train15_1903DF.groupby(['sales_chan_name', 'second_cate_code'])['ord_qty'].transform('mean').astype(np.float16)
    # 将订单需求列空值填充为0
    train15_1903DF['ord_qty'].fillna(0, inplace=True)
    # 滞后特征
    lags = [1, 2, 3, 6, 12, 24, 36, 48, 60]
    for lag in lags:
        train15_1903DF['need_lag_' + str(lag)] = \
        train15_1903DF.groupby(['sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code', ],
                               as_index=False)[
            'ord_qty'].shift(lag).astype(np.float16)

    #滑动窗口统计
    train15_1903DF['rolling_need_mean'] = train15_1903DF.groupby(['sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code'])['ord_qty'].transform(
        lambda x: x.rolling(window=7).mean()).astype(np.float16)
    #开窗数据统计
    train15_1903DF['expanding_need_mean'] = train15_1903DF.groupby(['sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code'])['ord_qty'].transform(
        lambda x: x.expanding(2).mean()).astype(np.float16)
    #需求量趋势构建
    train15_1903DF['daily_avg_need'] = train15_1903DF.groupby(['sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code','days_range'])['ord_qty'].transform('mean').astype(
        np.float16)
    train15_1903DF['avg_need'] = train15_1903DF.groupby(['sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code'])['ord_qty'].transform('mean').astype(np.float16)
    train15_1903DF['need_trend'] = (train15_1903DF['daily_avg_need'] - train15_1903DF['avg_need']).astype(np.float16)
    train15_1903DF.drop(['daily_avg_need', 'avg_need'], axis=1, inplace=True)
    train15_1903DF.drop('order_date', axis=1, inplace=True)
    #保存数据
    featureedTrain15_1903DF = train15_1903DF[train15_1903DF['days_range']>=60]
    featureedTrain15_1903DF.to_pickle('../数据/过程中数据/FeaturedEncodedTrainData_2015_201903.pkl')
    featureedTrain15_1903DF.to_csv('../数据/过程中数据/FeaturedEncodedTrainData_2015_201903.csv', index=False, encoding='utf-8')


#读取编码以及特征工程后的2015年1月到2018年月的数据
def ModelPredict201903():
    # 读取编码以及特征工程后的2015年1月到2018年月的数据
    Train15_1903DF = pd.read_csv('../数据/过程中数据/FeaturedEncodedTrainData_2015_201903.csv', encoding='utf-8')
    Train15_1903DF.drop('item_price', axis=1, inplace=True)
    # 划分验证集和测试集
    valid = Train15_1903DF[(Train15_1903DF['days_range'] >= 1219) & (Train15_1903DF['days_range'] < 1278)][['item_code', 'days_range', 'ord_qty']]
    test = Train15_1903DF[(Train15_1903DF['days_range'] >= 1278)][['item_code', 'days_range', 'ord_qty']]
    # 测试集训练预测和验证集预测结果
    validResult = valid['ord_qty']  # 这是已有真实标签需求量1218到1277，60天间隔的真实数据
    testResult = test['ord_qty']  # 这是已有真实标签需求量1175到1206，31天间隔的真实数据
    # 对五个销售区域分别建模并进行训练
    regionList = [101, 102, 103, 104, 105]
    for region in regionList:
        try:
            regionedTrain15_1903DF = Train15_1903DF[Train15_1903DF['sales_region_code'] == region]
            # 划分训练集和验证集
            X_train, y_train = regionedTrain15_1903DF[regionedTrain15_1903DF['days_range'] < 1250].drop('ord_qty', axis=1), \
                               regionedTrain15_1903DF[regionedTrain15_1903DF['days_range'] < 1250]['ord_qty']
            X_valid, y_valid = regionedTrain15_1903DF[(regionedTrain15_1903DF['days_range'] < 1278)&(regionedTrain15_1903DF['days_range'] >= 1250)].drop('ord_qty', axis=1), \
                               regionedTrain15_1903DF[(regionedTrain15_1903DF['days_range'] < 1278)&(regionedTrain15_1903DF['days_range'] >= 1250)]['ord_qty']
            X_test = regionedTrain15_1903DF[(regionedTrain15_1903DF['days_range'] >= 1278)].drop('ord_qty', axis=1)
            # 构建模型并模型
            model = LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                max_depth=3,
                num_leaves=8,
                min_child_weight=300
            )
            print('*****Prediction for 销售区域: {}*****'.format(region))
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)],
                      eval_metric='rmse', verbose=20, early_stopping_rounds=100)
            validResult[X_valid.index] = model.predict(X_valid)
            testResult[X_test.index] = model.predict(X_test)
            # 保存模型文件数据
            joblib.dump(model, "../模型/" + "model_" + str(region) + ".pkl")
            del model, X_train, y_train, X_valid, y_valid
        except:
            del model, X_train, y_train, X_valid, y_valid
            continue
    #预测完成2019年3月份的订单需求量后，将结果保存到文件中
    result = []
    for i in testResult:
        if i < 0:
            result.append(0)
        else:
            result.append(round(i, 0))
    #读取之前构造好的19年1月数据,将预测的订单需求量替换
    train201903DF = pd.read_csv('../数据/过程中数据/EncodedTrainData_201903.csv')
    train201903DF['ord_qty'] = result  # 把预测的订单需求量替换
    train201903DF.to_csv('../数据/结果数据/ForecastedEncodedTrainData_201903.csv', index=False)
    #将预测后的2019年1月的数据合并到官方存在的的2015年1月到2018年12月的数据中
    tain2015_2018DF = pd.read_csv('../数据/过程中数据/EncodedTrainData_2015_2018.csv')
    PredictedEncodedTrainData_2015_201901DF = pd.concat([tain2015_2018DF, train201903DF], axis=0)
    PredictedEncodedTrainData_2015_201901DF.to_csv('../数据/过程中数据/PredictedEncodedTrainData_2015_201903.csv', index=False)
    #计算出而月总销售额
    forecasted201903DF = train201903DF.loc[train201903DF['days_range'] >= 1278]  # 定位到19年3月的数据
    forecasted201903DF = forecasted201903DF[['order_date', 'sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code', 'ord_qty']]  # 筛选出需要的列
    resultDF = pd.pivot_table(forecasted201903DF, index=['sales_region_code', 'first_cate_code', 'second_cate_code', 'item_code'], columns='order_date',
                               values='ord_qty', aggfunc=np.sum, fill_value=0).reset_index()
    resultDF['March_demand'] = resultDF.iloc[:, 4:].sum(axis=1)  # 对0，1列按行求和，生成新列
    resultDF = resultDF[['sales_region_code', 'first_cate_code', 'second_cate_code', 'item_code', 'March_demand']]
    #读取预测的sku1的数据
    predictDF = pd.read_csv("../数据/官方数据_完整版/predict_sku1.csv")
    predictDF.columns = ['sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code']
    s = pd.merge(predictDF, resultDF, how='inner')
    s.to_csv('../数据/结果数据/ForecastedData_201903.csv', index=False)
#合并1月2月3月的数据
def Merge():
    Precited201901DF = pd.read_csv('../数据/结果数据/ForecastedData_201901.csv')
    Precited201902DF = pd.read_csv('../数据/结果数据/ForecastedData_201902.csv')
    Precited201903DF = pd.read_csv('../数据/结果数据/ForecastedData_201903.csv')
    PrecitedDF = pd.merge(Precited201901DF, Precited201902DF)
    PrecitedDF = pd.merge(PrecitedDF, Precited201903DF)
    submit_df = PrecitedDF[['sales_region_code', 'item_code', 'January_demand', 'February_demand', 'March_demand']]
    submit_df.columns = ['sales_region_code', 'item_code', '2019年1月预测需求量', '2019年2月预测需求量', '2019年3月预测需求量']
    # 保存
    submit_df.to_excel('../数据/结果数据/result1.xlsx', index=False)

if __name__ == '__main__':
    #vaildModel()
    #dataProcessing()
    #addDataOf201901()
    #featureEngineeringOf2015_201901()
    ModelPredict201901()
    # addDataOf201902()
    # featureEngineeringOf2015_201902()
    # ModelPredict201902()
    # addDataOf201903()
    # featureEngineeringOf2015_201903()
    # ModelPredict201903()
    # Merge()
    #model_feature_importance()
    #searchBestParam()
    #ModelPredict201901()
    #vaildModel()
    pass

