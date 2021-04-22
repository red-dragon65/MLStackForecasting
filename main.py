

# Used for reading in and processing data
import DataPreProcessor as dp

# Library for handling dataframes
import pandas as pd

# Custom classes/libraries
import graeme as g
import mushy as m
# import paxton as p



'''
Run predictions for 'best' stocks
'''
def runForecasting():
    # holt winters

    # Get data frames
    nasdaq_df, nyse_df, spac_df, allDf = dp.initializeDf()


    # Get top stocks
    mushyStock = m.findBestStock(nasdaq_df)


    # TODO: Remove this example
    # Show stock graph
    dp.showStockGraph('Apple', nasdaq_df)



    #TODO: Calculate stock future prediction
    #holtWinters.fit(m.getStockDF(stock))
    #hltWinters.predict




runForecasting()