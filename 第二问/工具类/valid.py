import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
from lightgbm import LGBMRegressor
from chinese_calendar import is_workday
from 第二问.工具类.optimal_bins import optimal_bins

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
    pass
    # # 读取编码以及特征工程后的2015年1月到2018年月的数据
    # Train15_1901DF = pd.read_csv('../数据/过程中数据/FeaturedEncodedTrainData_2015_201901.csv', encoding='utf-8')
    # regionedTrain15_1901DF = Train15_1901DF[Train15_1901DF['sales_region_code'] == 101]
    # # 划分训练集和验证集
    # X_train, y_train = regionedTrain15_1901DF[regionedTrain15_1901DF['days_range'] < 1208].drop('ord_qty', axis=1), \
    #                    regionedTrain15_1901DF[regionedTrain15_1901DF['days_range'] < 1208]['ord_qty']
    # parameters = {
    #     'learning_rate': [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1],
    #     'n_estimators': [1000,1500, 2000,2500,3000],
    # }
    # # 'n_estimators': [100, 300, 500, 1000, 1500, 2000],
    # # 'max_depth': [3, 4, 5, 6, 7, 8],
    # # 'num_leaves': [5, 7, 8, 9, 10, 13, 15, 17, 19, 21, 23, 27, 31, 63],
    # # 'subsample': [0.6, 0.7, 0.8, 1.0],
    # # 'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
    # # 'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.1]
    # model = LGBMRegressor(
    #     #n_estimators=1000,
    #     #learning_rate=0.05,
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     max_depth=8,
    #     num_leaves=50,
    #     min_child_weight=300
    #
    #
    # )
    # # 有了gridsearch我们便不需要fit函数
    #
    #
    # gsearch = GridSearchCV(model, param_grid=parameters,scoring='r2',cv= 3, verbose=1, n_jobs=-1)
    # gs_results=gsearch.fit(X_train, y_train)
    # print("BEST PARAMETERS: " + str(gs_results.best_params_))
    # print("BEST CV SCORE: " + str(gs_results.best_score_))
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
