import  pandas
import numpy
import  matplotlib.pyplot as plt
import  matplotlib

data_train=pandas.read_csv("数据/order_train1.csv")
# print(data.head())
print(data_train.info())
# data2=data[(data["item_code"]==20011)]
# data3=data[(data["item_code"]==20028)&(data["sales_region_code"]==104)]

def sale_region_code_analy():
    print(data_train.sales_region_code.value_counts())
    data_train.sales_region_code.value_counts().plot(kind="bar")
    plt.title("sales_region_code")
    plt.show()

def item_code_analy():
    print(data_train.item_code.value_counts())
    data_train.item_code.value_counts().plot(kind="bar")
    plt.title("item_code")
    plt.show()
def first_cate_code_analy():
    print(data_train.first_cate_code.value_counts())
    data_train.first_cate_code.value_counts().plot(kind="bar")
    plt.title("first_cate_code")
    plt.show()
def second_cate_code_analy():
    print(data_train.second_cate_code.value_counts())
    data_train.second_cate_code.value_counts().plot(kind="bar")
    plt.title("second_cate_code")
    plt.show()
def sales_chan_name_analy():
    print(data_train.sales_chan_name.value_counts())
    data_train.sales_chan_name.value_counts().plot(kind="bar")
    plt.title("sales_chan_name")
    plt.show()
def item_price_analy():
    print(data_train.item_price.value_counts())
    data_train.item_price.value_counts().plot(kind="bar")
    plt.title("item_price")
    plt.show()
def order_qty_analy():
    # print(data_train.ord_qty.value_counts())
    # data_train.ord_qty.value_counts().plot(kind="bar")
    print(data_train.ord_qty.describe())
    # plt.title("ord_qty")
    # plt.show()
def count_unique(data):
    for col in data.columns:
        print(col,":",data[col].nunique())
if __name__=="__main__":
    # sale_region_code_analy()
    # item_code_analy()
    # first_cate_code_analy()
    # second_cate_code_analy()
    # sales_chan_name_analy()
    # item_price_analy()
    order_qty_analy()
    pass