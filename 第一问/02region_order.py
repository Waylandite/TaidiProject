import  pandas as pd
import numpy
import  matplotlib.pyplot as plt
import  matplotlib
def analyse_region_order():
    data = pd.read_csv("../数据/官方数据_完整版/order_train1.csv")
    region_list=data.groupby("sales_region_code")
    map = {}
    for value,group in region_list:
        map[value]=group["ord_qty"].sum()
    #首先分析各个销售区域的订单量加和（柱状图）
    plt.bar(map.keys(),map.values())
    plt.title("analyze sales_region_code with ord_qty_sum by bar")
    plt.xlabel("sales_region_code")
    plt.ylabel("ord_qty_sum")
    plt.show()
    #分析各个销售区域的订单量加和（饼状图）
    plt.pie(map.values(),labels=map.keys(),autopct="%1.1f%%")
    plt.title("analyze sales_region_code with ord_qty_sum by pie")
    plt.xlabel("sales_region_code")
    plt.ylabel("ord_qty_sum")
    plt.show()
    #分析各个销售区域的订单量加和（折线图）
    category101_map=  {}
    category102_map = {}
    category103_map = {}
    category104_map = {}
    category105_map = {}
    for value,group in region_list:
        tmp=group.groupby("first_cate_code")
        for value1,group1 in tmp:
            if value==101:
                category101_map[value1]=group1["ord_qty"].sum()
            if value==102:
                category102_map[value1]=group1["ord_qty"].sum()
            if value==103:
                category103_map[value1]=group1["ord_qty"].sum()
            if value==104:
                category104_map[value1]=group1["ord_qty"].sum()
            if value==105:
                category105_map[value1]=group1["ord_qty"].sum()
    plt.plot(category101_map.keys(),category101_map.values(),label="101")
    plt.plot(category102_map.keys(), category102_map.values(), label="102")
    plt.plot(category103_map.keys(), category103_map.values(), label="103")
    plt.plot(category104_map.keys(), category104_map.values(), label="104")
    plt.plot(category105_map.keys(), category105_map.values(), label="105")
    plt.legend()
    plt.title("analyze first_cate_code with ord_qty_sum by line")
    plt.xlabel("sales_region_code")
    plt.ylabel("ord_qty_sum")
    plt.show()
if __name__=="__main__":
  analyse_region_order()
  pass
