



def initializeDf():
    nyse_df = [nyse_wed, nyse_thursday, df, ]

    for files in directory:
        nyse_df.append = open('files' + '.csv')

    return nasdaq, nyse, spac





# return df using group by for one stock
def getStockDF(apple):
    apple_df = big_ass_df['stock'].groupby('apple')
    return apple_df




# calculate new column for all data frames
def findFeature():


    # find best stocks

    for df in nyse_df:
        if df['open'] < df['close']
            df['newColumn'] = 1
        else:
            df['newColumn'] = 0

    df.drop(df['newColumn' == 0)




    for stockName in nyse_df[0]:

        new_df = getStockDF(stockname)

        model.fit(new_df)

        rmse = model.calculate()

        df['rmse'] = rmse



    nyse_df[0].sort(rmse)

    name = nyse_df[0].iloc[:1]



    return name




print("this is a test")