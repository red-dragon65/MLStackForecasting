'''
3580: Stock Prediction / Forecasting
Author: Mujtaba Ashfaq
Date: 4/15/21

This class reads data in and processes it
for usability.
'''

# Library for handling dataframes
import pandas as pd

# Library for reading in files
import os

# Used for handling dates
import datetime as dt

# Used for plotting
import matplotlib.pyplot as plt



'''
DON'T USE THIS. PRIVATE USE ONLY
Build data by reading in multiple csv files

Puts all files for one stock index into a single
file and calculates the date time using the files
name
'''
def __readInStockData(directory, fileStartingValue):

    # Hold all dfs
    frames = []

    # Read in nasdaq into one dataframe
    for filename in os.listdir(directory):

        # Only get files that match 'startswith' criteria
        if filename.startswith(fileStartingValue):

            # Read in data frame
            df = pd.read_csv(directory + "/" + filename)

            # Extract date info
            fileSplit = filename.replace('.', '-').split("-")

            # Get day of the week
            dayOfWeek = fileSplit[1]

            # Hold dates as strings
            month = fileSplit[2]
            day = fileSplit[3]
            year = fileSplit[4]

            # Convert month to int
            monthInt = dt.datetime.strptime(month, "%B").month

            # Create date time object
            date = dt.datetime(int(year), monthInt, int(day))

            # Add column for date
            df['Day'] = dayOfWeek
            df['Date'] = date

            # Add data frame to array
            frames.append(df)

    # Combine all frames in array
    result_df = pd.concat(frames, ignore_index=True)

    # Sort data by date
    result_df = result_df.sort_values('Date')

    # Return df result
    return result_df



'''
Build all necessary dataframes
'''
def initializeDf():

    # Hold directory for stock data
    directory = 'stocks'

    # Hold final dfs
    nasdaq_df = __readInStockData(directory, 'Nasdaq')
    nyse_df = __readInStockData(directory, 'NYSE')
    scap_df = __readInStockData(directory, 'SCAP')

    # Hold all dfs as a single df
    allDfs = []
    allDfs.append(nasdaq_df)
    allDfs.append(nyse_df)
    allDfs.append(scap_df)
    big_ass_df = pd.concat(allDfs, ignore_index=True)

    return nasdaq_df, nyse_df, scap_df, big_ass_df



'''
Return df containing data for one company
Create df by name column
'''
def getStockDfByName(compName, df):

    comp_df = df.loc[(df['Name'] == compName)]

    return comp_df



'''
Return df containing data for one company
Create df by symbol column
'''
def getStockDfBySymbol(compSymbol, df):

    comp_df = df.loc[(df['Name'] == compSymbol)]

    return comp_df



'''
Shows a graph for any given stock
Enter the stocks name to display stock chart
Pass the data frame for nasdaq, nyse, spac
'''
def showStockGraph(stockName, stockIndexDf):

    # Get data frame for apple stock
    compData = stockIndexDf.loc[(stockIndexDf['Name'] == stockName)]

    # Create a plot chart for stock data
    xpoints = compData['Date'].to_numpy()
    ypoints = compData['Close'].to_numpy()

    plt.title("Stock Price Over Time: " + stockName)
    plt.xlabel("Date")
    plt.ylabel("Price")

    plt.plot(xpoints, ypoints)
    plt.show()




