import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
from lightgbm import LGBMRegressor
from chinese_calendar import is_workday
from 第二问.工具类.optimal_bins import optimal_bins

import joblib
def validModel(df):
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