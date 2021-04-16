'''
3580: Stock Prediction / Forecasting
Author: Mujtaba Ashfaq
Date: 4/15/21

This class picks one stock that seems the best
based on arbitrary values.
'''

# Used for reading in and processing data
import DataPreProcessor as dp

# Library for handling dataframes
import pandas as pd



# Make data readable during testing
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)




# calculate new column for all data frames
'''
Creates new features to find the single best stock.
'''
def findBestStock(nasdaq_df):

    # Use this to get training and test data
    #X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size=0.2, random_state=0)

    '''
    # Create new feature
    for df in nyse_df:
        if df['open'] < df['close']
            df['newColumn'] = 1
        else:
            df['newColumn'] = 0

    df.drop(df['newColumn' == 0)



    # Run model on new feature
    for stockName in nyse_df[0]:

        new_df = getStockDF(stockname)

        model.fit(new_df)

        rmse = model.calculate()

        df['rmse'] = rmse



    nyse_df[0].sort(rmse)

    name = nyse_df[0].iloc[:1]
    '''

    name = ''


    # Create new feature column


    return name


