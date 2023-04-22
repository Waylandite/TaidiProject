import numpy as np
import  pandas as pd
import numpy
import  matplotlib.pyplot as plt
import  matplotlib
import seaborn as sns
#所有产品的线上线下销售情况对比
def analyse_bar():
    data = pd.read_csv("数据/order_train1.csv")
    region_list = data.groupby("sales_chan_name")
    map = {}
    for value, group in region_list:
        map[value] = group["ord_qty"].sum()
    # 首先分析各个销售区域的订单量加和（柱状图）
    plt.bar(map.keys(), map.values())
    plt.title("analyze sales_chan_name with ord_qty_sum by bar")
    plt.xlabel("sales_chan_name")
    plt.ylabel("ord_qty_sum")
    plt.show()
    multiple=map["offline"]/map["online"]
    print(multiple)
def analyse_density():
    # 读取数据
    data = pd.read_csv('数据/order_train1.csv')
    # 提取线上和线下订单需求量
    online_ord_qty = data[data["sales_chan_name"] == "online"]["ord_qty"]
    offline_ord_qty = data[data["sales_chan_name"] == "offline"]["ord_qty"]

    # 绘制线上和线下订单需求量核密度图
    sns.kdeplot(online_ord_qty, fill=True, label="Online")
    sns.kdeplot(offline_ord_qty, fill=True, label="Offline")
    plt.legend(loc="upper right")
    plt.title("Distribution of Order Quantity by Sales Channel")
    plt.xlabel("Order Quantity")
    plt.ylabel("Density")
    plt.show()

def analyse_scatter():
    train_data = pd.read_csv('数据/order_train1.csv')
    # 绘制散点图
    sns.scatterplot(data=train_data, x="item_price", y="ord_qty", hue="sales_chan_name")
    plt.show()
#不同产品大类下产品的线上线下销售情况对比
def analyse_bar_by_first_cate_code():
    data = pd.read_csv("数据/order_train1.csv")
    chan_list = data.groupby("sales_chan_name")
    map1 = {}
    map2 = {}
    map3 = {}
    map4 = {}
    map5 = {}
    map6 = {}
    map7 = {}
    map8 = {}
    for value, group in chan_list:
        tmp=group.groupby("first_cate_code")
        for value1, group1 in tmp:
            if value1 == 301:
                map1[value] = group1["ord_qty"].sum()
            if value1 == 302:
                map2[value] = group1["ord_qty"].sum()
            if value1 == 303:
                map3[value] = group1["ord_qty"].sum()
            if value1 == 304:
                map4[value] = group1["ord_qty"].sum()
            if value1 == 305:
                map5[value] = group1["ord_qty"].sum()
            if value1 == 306:
                map6[value] = group1["ord_qty"].sum()
            if value1 == 307:
                map7[value] = group1["ord_qty"].sum()
            if value1 == 308:
                map8[value] = group1["ord_qty"].sum()

    width = 0.1
    labels=["offline","online"]
    x = np.arange(2)
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 0.35 , map1.values(), width, label='301')
    rects2 = ax.bar(x - 0.25 , map2.values(), width, label='302')
    rects3 = ax.bar(x - 0.15, map3.values(), width, label='303')
    rects4 = ax.bar(x - 0.05, map4.values(), width, label='304')
    rects5 = ax.bar(x + 0.05, map5.values(), width, label='305')
    rects6 = ax.bar(x + 0.15, map6.values(), width, label='306')
    rects7 = ax.bar(x + 0.25, map7.values(), width, label='307')
    rects8 = ax.bar(x + 0.35, map8.values(), width, label='309')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.title("analyze sales_chan_name with ord_qty_sum by bar")
    ax.set_ylabel('ord_qty_sum')
    plt.show()

if __name__=="__main__":
  #analyse_bar()
  #analyse_bar_by_first_cate_code()
  #analyse_density()
  analyse_scatter()
  pass