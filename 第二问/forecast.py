import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
from lightgbm import LGBMRegressor
from chinese_calendar import is_workday
from 第二问.工具类.optimal_bins import optimal_bins
import joblib
warnings.filterwarnings("ignore")
#数据读取
df = pd.read_csv('../数据/order_train4.csv', encoding='utf-8')


#数据处理函数，将2015年到2018年的数据进行处理
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
    train1DF['DaysRange'] = train1DF['order_date'].map(daysList)
    #构造促销列表
    #如果是促销日，对应的值为1，否则为0
    train1DF['order_date'] = pd.to_datetime(train1DF['order_date'])
    salesPromotionDate = ['1-1', '4-1', '2-14', '3-1', '3-8', '4-11', '5-1', '6-1', '6-18', '11-11', '12-12']
    salesPromotionList = []
    for index,row in train1DF.iterrows():
        str_date = str(row['order_date'].month) + '-' + str(row['order_date'].day)
        if str_date in salesPromotionDate:
            salesPromotionList.append(1)
        else:
            salesPromotionList.append(0)
    train1DF['promotion'] = salesPromotionList
    #构造价格区间列表
    cut_bins = optimal_bins(train1DF.ord_qty, train1DF.item_price, n=10)
    train1DF['price_range'] = pd.cut(train1DF['item_price'], cut_bins, labels=[x for x in range(len(cut_bins) - 1)])
    # 销售渠道列表
    channelDict = {'offline': 0, 'online': 1}
    train1DF.sales_chan_name = train1DF.sales_chan_name.map(channelDict)
    #构造月段列表
    #'月初': 0, '月中': 1, '月末': 2
    monthList = []
    for index, row in train1DF.iterrows():
        if row['order_date'].day <= 10:
            monthList.append('0')
        elif row['order_date'].day > 10 and row['order_date'].day <= 20:
            monthList.append('1')
        elif row['order_date'].day > 20 :
            monthList.append('2')
    train1DF['monthe_period'] = monthList
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
    train1DF['workday'] = train1DF.is_workday.map(workdayDict)
    #保存数据到EncodedTrainData_2015_2018文件
    train1DF.to_csv('数据/过程中数据/EncodedTrainData_2015_2018.csv', index=False, encoding='utf-8')

#构造2019年1月的数据
def Add201901():
    df['combination'] = df['sales_region_code'].astype(str) + '_' + df['first_cate_code'].astype(str) + '_' + df['second_cate_code'].astype(str) + '_' + df[
        'item_code'].astype(str)
    dt1 = pd.date_range(start="20181221", end="20190131", freq="D")
    dates = []
    need = []
    ids = []
    for d in dt1:
        date = str(d).split(' ')[0]  # 得到日期xx-xx-xx
        dates = dates + [date] * len(set(df['combination']))
        need = need + [0] * len(set(df['combination']))
        ids = ids + list(set(df['combination']))
    tempdf = pd.DataFrame()
    tempdf['order_date'] = dates
    tempdf['combination'] = ids
    tempdf['ord_qty'] = need
    tempdf['sales_region_code'] = tempdf['combination'].str.split('_', expand=True)[0]
    tempdf['first_cate_code'] = tempdf['combination'].str.split('_', expand=True)[1]
    tempdf['second_cate_code'] = tempdf['combination'].str.split('_', expand=True)[2]
    tempdf['item_code'] = tempdf['combination'].str.split('_', expand=True)[3]
    tempdf['order_date'] = pd.to_datetime(tempdf['order_date'])  # 对日期列进行日期格式转换
    # 构造是否为促销的列表
    tempdf['order_date'] = pd.to_datetime(tempdf['order_date'])
    list_sales_promotion = ['1-1', '4-1', '2-14', '3-1', '3-8', '4-11', '5-1', '6-1', '6-18', '11-11', '12-12']
    sales_promotion = []
    for index, row in tempdf.iterrows():
        str_date = str(row['order_date'].month) + '-' + str(row['order_date'].day)
        if str_date in list_sales_promotion:
            sales_promotion.append(1)
        else:
            sales_promotion.append(0)
    tempdf['promotion'] = sales_promotion
    # 对价格进行分箱
    pricedf = df[['combination', 'price_range', 'sales_chan_name']]
    pricedf.drop_duplicates(['combination'], keep='last', inplace=True)
    dict_jg = {}
    qudao_dict = {}
    for i in range(len(pricedf)):
        dict_jg[list(pricedf['combination'])[i]] = list(pricedf['price_range'])[i]
        qudao_dict[list(pricedf['combination'])[i]] = list(pricedf['sales_chan_name'])[i]
    # 价格区间：
    jiage = []
    qudao = []
    for v in tempdf['combination']:
        jiage.append(dict_jg[v])
        qudao.append(qudao_dict[v])
    tempdf['price_range'] = jiage
    tempdf['sales_chan_name'] = qudao
    # 对月初，月中、月末进行编码处理
    dict_moth = {'月初': 0, '月中': 1, '月末': 2}
    month = []
    for index, row in tempdf.iterrows():
        if row['order_date'].day <= 10:
            month.append('0')
        elif row['order_date'].day > 10 and row['order_date'].day <= 20:
            month.append('1')
        elif row['order_date'].day > 20:
            month.append('2')
    tempdf['month_time_period'] = sales_promotion
    # 对季节进行编码处理
    dict_season = {'春': 0, '夏': 1, '秋': 2, '冬': 3}
    season = []
    for index, row in tempdf.iterrows():
        if row['order_date'].month == 3 or row['order_date'].month == 4 or row['order_date'].month == 5:
            season.append('0')
        elif row['order_date'].month == 6 or row['order_date'].month == 7 or row['order_date'].month == 8:
            season.append('1')
        elif row['order_date'].month == 9 or row['order_date'].month == 10 or row['order_date'].month == 11:
            season.append('2')
        elif row['order_date'].month == 12 or row['order_date'].month == 1 or row['order_date'].month == 2:
            season.append('3')
    tempdf['seaon'] = season
    # 对星期几进行编码处理
    tempdf['day_of_week'] = tempdf.order_date.dt.weekday  # 0-6  {0, 1, 2, 3, 4, 5, 6}
    # 对是否工作日进行编码处理
    tempdf['is_workday'] = tempdf['order_date'].map(lambda x: is_workday(x))
    dict_is_workday = {False: 0, True: 1}
    tempdf['is_workday'] =tempdf.is_workday.map(dict_is_workday)
    #添加D列表示天数
    dt1 = pd.date_range(start="20150901", end="20190131", freq="D")
    dicts = {}
    i = 1
    for date in dt1:
        dicts[str(date).split(' ')[0]] = i
        i += 1
    Ds = []
    for dat in tempdf['order_date']:
        Ds.append(dicts[str(dat).split(' ')[0]])
    tempdf['D'] = Ds
    tempdf.drop(['combination'], axis=1, inplace=True)
    # 保存数据到tran3文件
    tempdf.to_csv('数据/order_train3.csv', index=False, encoding='utf-8')
    #拼接2019年以前的真实数据和2019年自己构造的数据
    alldf=pd.concat([df,tempdf],axis=0)
    alldf.drop(['combination'], axis=1, inplace=True)
    alldf.to_csv('数据/order_train4.csv', index=False, encoding='utf-8')
def feature_engineering():
    #滞后特征
    lags = [1, 2, 3, 6, 12, 24, 36, 48, 60]
    for lag in lags:
        df['need_lag_' + str(lag)] = df.groupby(['sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code', ], as_index=False)[
            'ord_qty'].shift(lag).astype(np.float16)
    #均值编码
    s = df.groupby('item_code')['ord_qty'].transform('mean')
    df['item_avg'] = df.groupby('item_code')['ord_qty'].transform('mean').astype(np.float16)
    df['sales_region_code_need_avg'] = df.groupby('sales_region_code')['ord_qty'].transform('mean').astype(np.float16)
    df['first_cate_code_need_avg'] = df.groupby('first_cate_code')['ord_qty'].transform('mean').astype(np.float16)
    df['second_cate_code_need_avg'] = df.groupby('second_cate_code')['ord_qty'].transform('mean').astype(np.float16)
    df['sales_chan_name_need_avg'] = df.groupby('sales_chan_name')['ord_qty'].transform('mean').astype(np.float16)
    df['sales_region_code_item_code_need_avg'] = df.groupby(['item_code', 'sales_region_code'])['ord_qty'].transform('mean').astype(np.float16)
    df['first_cate_code_item_code_need_avg'] = df.groupby(['first_cate_code', 'item_code'])['ord_qty'].transform('mean').astype(np.float16)
    df['second_cate_code_item_code_need_avg'] = df.groupby(['second_cate_code', 'item_code'])['ord_qty'].transform('mean').astype(np.float16)
    df['sales_chan_name_item_code_need_avg'] = df.groupby(['sales_chan_name', 'item_code'])['ord_qty'].transform('mean').astype(np.float16)
    df['sales_region_code_first_cate_code_need_avg'] = df.groupby(['sales_region_code', 'first_cate_code'])['ord_qty'].transform('mean').astype(np.float16)
    df['first_cate_code_second_cate_code_need_avg'] = df.groupby(['first_cate_code', 'second_cate_code'])['ord_qty'].transform('mean').astype(np.float16)
    df['sales_chan_name_second_cate_code_need_avg'] = df.groupby(['sales_chan_name', 'second_cate_code'])['ord_qty'].transform('mean').astype(np.float16)
    #滑动窗口统计
    df['rolling_need_mean'] = df.groupby(['sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code'])['ord_qty'].transform(
        lambda x: x.rolling(window=7).mean()).astype(np.float16)
    #开窗数据统计
    df['expanding_need_mean'] = df.groupby(['sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code'])['ord_qty'].transform(
        lambda x: x.expanding(2).mean()).astype(np.float16)
    #需求量趋势构建
    df['daily_avg_need'] = df.groupby(['sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code'])['ord_qty'].transform('mean').astype(
        np.float16)
    df['avg_need'] = df.groupby(['sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code'])['ord_qty'].transform('mean').astype(np.float16)
    df['need_trend'] = (df['daily_avg_need'] - df['avg_need']).astype(np.float16)
    df.drop(['daily_avg_need', 'avg_need'], axis=1, inplace=True)
    df.drop('order_date', axis=1, inplace=True)
    #保存数据
    newdf = df[df['D']>=60]
    df.to_pickle('数据/train_data.pkl')
    df.to_csv('数据/train_data.csv', index=False, encoding='utf-8')


def model():
    # 构建模型并训练
    data = pd.read_pickle('数据/train_data.pkl')
    valid = data[(data['D'] >= 1176) & (data['D'] <= 1207)][['item_code', 'D', 'ord_qty']]
    test = data[data['D'] > 1207][['item_code', 'D', 'ord_qty']]  # 作为待预测的数据
    eval_preds = test['ord_qty']  ## 待预测 目前全为0
    valid_preds = valid['ord_qty']  # 这是已有真实标签需求量1175到1206，31天间隔的真实数据


    states = [101, 102, 103, 104, 105]
    # 对五个销售区域分别建模并进行训练
    for store in states:
       try:
           df = data[data['sales_region_code'] == store]
           # Split the data
           X_train, y_train = df[df['D'] < 1176].drop('ord_qty', axis=1), df[df['D'] < 1176]['ord_qty']
           X_valid, y_valid = df[(df['D'] >= 1176) & (df['D'] <= 1207)].drop('ord_qty', axis=1), \
                              df[(df['D'] >= 1176) & (df['D'] <= 1207)]['ord_qty']
           X_test = df[df['D'] > 1207].drop('ord_qty', axis=1)
           # Train and validate
           model = LGBMRegressor(
               n_estimators=1000,
               learning_rate=0.3,
               subsample=0.8,
               colsample_bytree=0.8,
               max_depth=8,
               num_leaves=50,
               min_child_weight=300
           )
           print('*****Prediction for 销售区域: {}*****'.format(store))
           model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)],
                     eval_metric='rmse', verbose=20, early_stopping_rounds=20)
           valid_preds[X_valid.index] = model.predict(X_valid)
           eval_preds[X_test.index] = model.predict(X_test)
           filename = "模型/" + "model_" + str(store) + ".pkl"
           # save model
           joblib.dump(model, filename)
           del model, X_train, y_train, X_valid, y_valid
       except:
        del model, X_train, y_train, X_valid, y_valid
        continue
def validModel():
    data = pd.read_pickle('数据/train_data.pkl')
    model = joblib.load("../模型/model_101.pkl")
    newdf = data[data['sales_region_code'] == 101]
    X_valid, y_valid = newdf[(df['D'] >= 1176) & (newdf['D'] <= 1207)].drop('ord_qty', axis=1), \
                       newdf[(df['D'] >= 1176) & (newdf['D'] <= 1207)]['ord_qty']
    s = model.predict(X_valid)
    x_axis = np.linspace(1, len(y_valid), len(y_valid))
    plt.plot(x_axis[:200], y_valid[:200])
    plt.plot(x_axis[:200], s[:200])
    plt.legend(['true', 'prediction'])
    plt.show()
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
    data = pd.read_pickle('数据/train_data.pkl')
    feature_importance_df = pd.DataFrame()
    features = [f for f in data.columns if f != 'ord_qty']
    for filename in os.listdir('../模型/'):
        print(filename)
        if 'model' in filename:
            # load model
            model = joblib.load("模型/" + filename)
            store_importance_df = pd.DataFrame()
            store_importance_df["feature"] = features
            store_importance_df["importance"] = model.feature_importances_
            store_importance_df["store"] = filename[5:9]
            feature_importance_df = pd.concat([feature_importance_df, store_importance_df], axis=0)
    display_importances(feature_importance_df)
#预测2019年1月份的销售量并记录
def forecast20191():

    result = []
    # for v in eval_preds:
    #     if v < 0:
    #         result.append(0)  # 预测值里有负值，这是不可能的，所有我们用0来代表异常值，产生这负数多半是因为训练时，对应产品的时间序列数据很少
    #     else:
    #         result.append(round(v, 0))
    # #读取之前构造好的19年1月数据：
    # df_sub = pd.read_csv('../数据/order_train3.csv')
    # df_sub['ord_qty'] = result  # 把预测的订单需求量替换
    # df_sub.to_csv('数据/result1.csv', index=False)
    #
    # result_df1 = df_sub.loc[df_sub['D'] >= 1219]  # 定位到19年1月的数据
    # result_df = result_df1[['order_date', 'sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code', 'ord_qty']]  # 筛选出需要的列
    #
    # #计算出一月总销售额
    # # 制作透视表得到所有产品19年1月的需求量
    # result_df = pd.pivot_table(result_df, index=['sales_region_code', 'first_cate_code', 'second_cate_code', 'item_code'], columns='order_date',
    #                            values='ord_qty', aggfunc=np.sum, fill_value=0).reset_index()
    # result_df.iloc[:, 4:]
    # result_df['1monthly_need'] = result_df.iloc[:, 4:].sum(axis=1)  # 对0，1列按行求和，生成新列
    # result_df = result_df[['sales_region_code', 'first_cate_code', 'second_cate_code', 'item_code', '1monthly_need']]
    # predict_df = pd.read_csv("../数据/官方数据_完整版/predict_sku1.csv")
    # predict_df.columns = ['sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code']
    # s = pd.merge(predict_df, result_df, how='inner')
    # s.to_csv('数据/1_monthly_predict_sku1.csv', index=False)
#预测2019年2月份的销售量并记录
def forecast20192():
    df = pd.read_csv('../数据/result1.csv', encoding='utf-8')
    df['combination'] = df['sales_region_code'].astype(str) + '_' + df['first_cate_code'].astype(str) + '_' + df['second_cate_code'].astype(str) + '_' + df[
        'item_code'].astype(str)
    # 从2019-2-1到2019-2-28日的数据
    dt1 = pd.date_range(start="20190201", end="20190228", freq="D")
    dates = []
    need = []
    ids = []
    for d in dt1:
        date = str(d).split(' ')[0]  # 得到日期xx-xx-xx
        dates = dates + [date] * len(set(df['combination']))
        need = need + [0] * len(set(df['combination']))
        ids = ids + list(set(df['combination']))
    tempdf = pd.DataFrame()
    tempdf['order_date'] = dates
    tempdf['combination'] = ids
    tempdf['ord_qty'] = need
    tempdf['sales_region_code'] = tempdf['combination'].str.split('_', expand=True)[0]
    tempdf['first_cate_code'] = tempdf['combination'].str.split('_', expand=True)[1]
    tempdf['second_cate_code'] = tempdf['combination'].str.split('_', expand=True)[2]
    tempdf['item_code'] = tempdf['combination'].str.split('_', expand=True)[3]
    tempdf['order_date'] = pd.to_datetime(tempdf['order_date'])  # 对日期列进行日期格式转换
    # 构造是否为促销的列表
    tempdf['order_date'] = pd.to_datetime(tempdf['order_date'])
    list_sales_promotion = ['1-1', '4-1', '2-14', '3-1', '3-8', '4-11', '5-1', '6-1', '6-18', '11-11', '12-12']
    sales_promotion = []
    for index, row in tempdf.iterrows():
        str_date = str(row['order_date'].month) + '-' + str(row['order_date'].day)
        if str_date in list_sales_promotion:
            sales_promotion.append(1)
        else:
            sales_promotion.append(0)
    tempdf['promotion'] = sales_promotion
    # 对价格进行分箱
    pricedf = df[['combination', 'price_range', 'sales_chan_name']]
    pricedf.drop_duplicates(['combination'], keep='last', inplace=True)
    dict_jg = {}
    qudao_dict = {}
    for i in range(len(pricedf)):
        dict_jg[list(pricedf['combination'])[i]] = list(pricedf['price_range'])[i]
        qudao_dict[list(pricedf['combination'])[i]] = list(pricedf['sales_chan_name'])[i]
    # 价格区间：
    jiage = []
    qudao = []
    for v in tempdf['combination']:
        jiage.append(dict_jg[v])
        qudao.append(qudao_dict[v])
    tempdf['price_range'] = jiage
    tempdf['sales_chan_name'] = qudao
    # 对月初，月中、月末进行编码处理
    dict_moth = {'月初': 0, '月中': 1, '月末': 2}
    month = []
    for index, row in tempdf.iterrows():
        if row['order_date'].day <= 10:
            month.append('0')
        elif row['order_date'].day > 10 and row['order_date'].day <= 20:
            month.append('1')
        elif row['order_date'].day > 20:
            month.append('2')
    tempdf['month_time_period'] = sales_promotion
    # 对季节进行编码处理
    dict_season = {'春': 0, '夏': 1, '秋': 2, '冬': 3}
    season = []
    for index, row in tempdf.iterrows():
        if row['order_date'].month == 3 or row['order_date'].month == 4 or row['order_date'].month == 5:
            season.append('0')
        elif row['order_date'].month == 6 or row['order_date'].month == 7 or row['order_date'].month == 8:
            season.append('1')
        elif row['order_date'].month == 9 or row['order_date'].month == 10 or row['order_date'].month == 11:
            season.append('2')
        elif row['order_date'].month == 12 or row['order_date'].month == 1 or row['order_date'].month == 2:
            season.append('3')
    tempdf['seaon'] = season
    # 对星期几进行编码处理
    tempdf['day_of_week'] = tempdf.order_date.dt.weekday  # 0-6  {0, 1, 2, 3, 4, 5, 6}
    # 对是否工作日进行编码处理
    tempdf['is_workday'] = tempdf['order_date'].map(lambda x: is_workday(x))
    dict_is_workday = {False: 0, True: 1}
    tempdf['is_workday'] = tempdf.is_workday.map(dict_is_workday)
    # 添加D列表示天数
    dt1 = pd.date_range(start="20150901", end="20190131", freq="D")
    dicts = {}
    i = 1
    for date in dt1:
        dicts[str(date).split(' ')[0]] = i
        i += 1
    Ds = []
    for dat in tempdf['order_date']:
        Ds.append(dicts[str(dat).split(' ')[0]])
    tempdf['D'] = Ds
    tempdf.drop(['combination'], axis=1, inplace=True)
    # 保存数据到tran3文件
    tempdf.to_csv('数据/order_train3.csv', index=False, encoding='utf-8')
    # 拼接2019年以前的真实数据和2019年自己构造的数据
    alldf = pd.concat([df, tempdf], axis=0)
    alldf.drop(['combination'], axis=1, inplace=True)
    alldf.to_csv('数据/order_train5.csv', index=False, encoding='utf-8')







if __name__ == '__main__':
    dataProcessing()
    pass

