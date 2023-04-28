import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
from lightgbm import LGBMRegressor
from chinese_calendar import is_workday
from 第二问.工具类.optimal_bins import optimal_bins
from sklearn.model_selection import GridSearchCV  # Perforing grid search
from sklearn.model_selection import train_test_split
#from 第二问.工具类.valid import validModel, model_feature_importance, display_importances
from sklearn.metrics import r2_score
import joblib
warnings.filterwarnings("ignore")

# 读取数据
Train15_1901DF = pd.read_csv('../数据/过程中数据/tes.csv', encoding='utf-8')

min=Train15_1901DF.groupby('type')['res'].transform('mean').astype(np.float16)
print(min)
