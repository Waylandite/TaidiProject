import random

import numpy as np
import  pandas as pd
import numpy
import  matplotlib.pyplot as plt
import  matplotlib
import seaborn as sns
from sklearn.linear_model import LinearRegression

def count_first_cate_prices():
    data_train = pd.read_csv("数据/order_train1_clear.csv")
    first_cate_code = data_train["first_cate_code"].unique()
    map={}
    for x in first_cate_code:
        map[x]=data_train[data_train["first_cate_code"]==x]["ord_qty"].sum()
    data=pd.DataFrame(map.items(),columns=['first_cate_code','ord_qty'])
    data.to_csv("数据/cate_order/first_cate_ord_qty.csv",index=False)
    plt.bar(map.keys(),map.values())
    plt.xlabel("first_cate_code")
    plt.ylabel("ord_qty")
    plt.show()
    plt.pie(map.values(),labels=map.keys(),autopct="%1.1f%%")
    plt.title("analyze first_cate_code with ord_qty_sum by pie")
    plt.xlabel("first_cate_code")
    plt.ylabel("ord_qty_sum")
    plt.show()

def count_second_cate_prices():
    data_train = pd.read_csv("数据/order_train1_clear.csv")
    first_cate_code = data_train["second_cate_code"].unique()
    print(first_cate_code)
    map={}
    for x in first_cate_code:
        map[x]=data_train[data_train["second_cate_code"]==x]["ord_qty"].sum()
    data=pd.DataFrame(map.items(),columns=['second_cate_code','ord_qty'])
    data.to_csv("数据/cate_order/second_cate_ord_qty.csv",index=False)
    plt.bar(map.keys(),map.values())
    plt.xlabel("second_cate_code")
    plt.ylabel("ord_qty")
    plt.show()
    plt.pie(map.values(),labels=map.keys(),autopct="%1.1f%%")
    plt.title("analyze second_cate_code_code with ord_qty_sum by pie")
    plt.xlabel("second_cate_code")
    plt.ylabel("ord_qty_sum")
    plt.show()


if __name__=="__main__":
    # count_first_cate_prices()
    # count_second_cate_prices()
    pass