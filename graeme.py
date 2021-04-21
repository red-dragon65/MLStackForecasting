
# Graeme's Ideas

# Like the assignment says, first look at the total growth percentages and price differences
# comparing the start and end dates to figure out what the best stocks are.
# We can look at trends to see what could be important to look for/compare

# create a sliding window to find the rolling averages of the close number
# or do a linear regression to find the trendline

# same idea but with P/E


import pandas as pd
import DataPreProcessor as dp
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# df = pd.read_csv('stocks/Nasdaq-Friday-April-20-2018.csv')
# print(df)

def graemeStuff():
    nasdaq_df, nyse_df, scap_df, all_stock_df = dp.initializeDf()
    apple_stock = dp.getStockDfByName('Apple', all_stock_df)
    # print(apple_stock.head())
    # rollingAverage(apple_stock)
    rawBest(all_stock_df)

# find the rolling average of a single stock df
def rollingAverage(stock_df):
    averageList = []
    low = stock_df['High']
    high = stock_df['Low']
    for x in range(len(stock_df)):
        avg = (float(high.iloc[x]) + float(low.iloc[x])) / 2
        averageList.append(avg)
    Date = stock_df['Date']
    dict = {'Average': averageList, 'Date': Date}
    average_df = pd.DataFrame(dict)

    xpoints = stock_df['Date'].to_numpy()
    ypoints = averageList
    rollingAvg5 = average_df.rolling(5).mean()
    rollingAvg10 = average_df.rolling(10).mean()
    rollingAvg50 = average_df.rolling(50).mean()

    plt.plot(xpoints, ypoints, label="Average")
    plt.plot(xpoints, rollingAvg5, label="Roll avg 5")
    plt.plot(xpoints, rollingAvg10, label="Roll avg 10")
    plt.plot(xpoints, rollingAvg50, label="Roll avg 50")

    # plt.plot(xpoints, high, label="High")
    # plt.plot(xpoints, low, label="Low")
    plt.legend()
    plt.show()
    # print(averageList)
    # print(len(averageList))
    # print(len(stock_df))

def rawBest(big_df):
    # make a set of the names of the companies
    # field all the names into the get stock by name
    # take the first open and the last close date to see raw total difference
    # make a dict of the company and its overall change
    # sort this dict

    # set of all the company names
    compNameSet = set()

    # dict of the company and its overall stock diff
    compDiff = {}

    # list to hold the dicts so that it can be sorted by stock diff
    compDiffList = []

    print(dp.getStockDfByName("Eyenovia", big_df))
    # for stock in big_df:
    #     print(stock)


def helloWorld():

    print("hello people")

    value = "testy"
    print("hello other people")

    return value


graemeStuff()