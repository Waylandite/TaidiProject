import  pandas as pd
import numpy
import  matplotlib.pyplot as plt
import  matplotlib



def cout_prices():
    data_train = pd.read_csv("数据/order_train1.csv")
    item_code_list = data_train.item_code.unique()
    print(len(item_code_list))
    map={}
    for row in data_train.itertuples():
        if row.item_code not in map.keys():
            map[row.item_code]=[]
        if row.item_price not in map[row.item_code]:
            map[row.item_code].append([row.item_price,row.ord_qty])
        else:
            map[row.item_code][map[row.item_code].index(row.item_price)][1]+=row.ord_qty
    data=pd.DataFrame(map.items(),columns=['item_code','priceorder_list'])
    data.to_csv("数据/price_order.csv",index=False)
    # for key in map.keys():
    #     map[key]=len(map[key])
    # a = sorted(map.items(), key=lambda x: x[1])
    # print(a)
    # data = pd.DataFrame(map.items(), columns=['item_code', 'price_count'])
    # data.to_csv("数据/price_count.csv", index=False)
    # plt.plot(map.values())
    # plt.show()
# cout_prices()
def analyse_price_order():
    data=pd.read_csv("数据/price_order.csv")
    map={}
    mapindex={}
    for row in data.itertuples():
        # print(row.item_code)
        # print(row.priceorder_list)
        priceoder_list=eval(row.priceorder_list)
        map[row.item_code]=len(priceoder_list)

    ##map中存储的是
    mapcopy=map.copy()
    for i in mapcopy.keys():
        if map[i]<100:
            map.pop(i)
    
    # print(map)
    # plt.plot(map.values())
    # plt.show()
def analyse_region_order():
    data = pd.read_csv("数据/order_train1.csv")
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