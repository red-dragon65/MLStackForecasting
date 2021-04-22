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


# Use for logistical regression modeling
#--------------------
# working with arrays
import numpy as np

# splitting the data
from sklearn.model_selection import train_test_split

# model algorithm
from sklearn.linear_model import LinearRegression

# data normalization
from sklearn.preprocessing import StandardScaler

# Metrics library
from sklearn.metrics import mean_squared_error

# For RFE
from sklearn.feature_selection import RFE
#--------------------


# Used for holt winters
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Used for plotting
import matplotlib.pyplot as plt

# Make sklearn shutup about errors
pd.options.mode.chained_assignment = None  # default='warn'


# Make data readable during testing
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)



'''
Creates new features to find the single best stock.

Figuring out best stock
------------------------

RFE: (day trading)
Loop through each stock company
# Get tomorrows change percent
Split up stock data into training and test data
Use RFE to find best feature for each company
Put top 1 feature name in new column for each company
# Put accuracy in new column

Forecasting: (Find long term stock)
Pull out top 50 stocks
Use holt winters for each stock
List highest percentage return for prediction

Get top 3 stock
Predict future outside of algorithm

Return top stock (good accuracy, and best return)
'''
def findBestStock(nasdaq_df):

    # Remove comma found in some numbers
    nasdaq_df.replace(',', '', regex=True, inplace=True)

    # Get a list of all unique companies
    companies = nasdaq_df['Name'].unique()

    # Create empty column for top company feature
    nasdaq_df['TopFeature'] = 'null'

    # Create empty column for company prediction accuracy
    nasdaq_df['ModelAccuracy'] = 0.0

    # Track how many companies have been looped through
    # Max for nasdaq is about 2000
    comp_count = 0

    # Loop through each company
    for company in companies:

        # Limit number of stocks looped
        #if comp_count == 51:
            #break

        # Get data frame for company
        comp_df = nasdaq_df.loc[(nasdaq_df['Name'] == company)]

        # Get tomorrow change percent in new column
        comp_df['TomorrowPercentChange'] = comp_df.groupby('Name')['% Chg'].shift(-1)

        # Remove last row (TomorrowPercentChange is null on last row)
        comp_df.drop(comp_df.tail(1).index, inplace=True)


        # Get available columns for RFE
        columns = list(comp_df.columns)

        # Remove columns we don't care about
        columns.remove('Name')
        columns.remove('Symbol')
        columns.remove('Net Chg')
        columns.remove('% Chg')
        columns.remove('Day')
        columns.remove('Date')
        columns.remove('TopFeature')
        columns.remove('ModelAccuracy')

        # Convert list to array
        features = np.array(columns)

        # Set correct data type (for columns we care about)
        for f in features:
            comp_df[f] = pd.to_numeric(comp_df[f], downcast="float", errors='coerce')

        # Clean tomorrow percentage before cleaning rest of data
        # (Prevents column from getting entirely deleted)
        comp_df = comp_df[comp_df['TomorrowPercentChange'].notna()]

        # Drop columns with missing values from data frame
        comp_df = comp_df.dropna(axis='columns')

        # Drop any rows missing data
        comp_df = comp_df.dropna()

        # Reset index (causes problems with array conversion)
        comp_df.reset_index(drop=True, inplace=True)


        # Get the remaining columns we care about
        columns = list(comp_df.columns)

        # Remove columns we don't care about
        columns.remove('Name')
        columns.remove('Symbol')
        columns.remove('Net Chg')
        columns.remove('% Chg')
        columns.remove('Day')
        columns.remove('Date')
        columns.remove('TopFeature')
        columns.remove('ModelAccuracy')
        columns.remove('TomorrowPercentChange')

        # Convert list to array
        features = np.array(columns)






        # Hold top feature and accuracy
        topFeature = 'null'
        accuracy = 0.0


        # Define dependent variable
        y_var = comp_df['TomorrowPercentChange']

        # Track number of features left
        feat_rem = 0

        #TODO Multithread this

        # Keep making models while reducing the number of features
        for num_components in range(len(features), 0, -1):

            # Skip empty dataframe
            if len(comp_df.index) < 10:
                break

            # Skip if less than one feature
            if len(features) < 2:
                break

            # Show data frame
            #if feat_rem == 0:
                #print(comp_df)


            # Hold model
            rfe_model = LinearRegression()

            # Reset independent variable
            X_var = comp_df.loc[:, features]

            # Scale data to remove bias
            X_var = StandardScaler().fit(X_var).transform(X_var)

            # Create rfe model (test with specified number of features)
            rfe = RFE(rfe_model, n_features_to_select=num_components)
            rfe = rfe.fit(X_var, y_var)

            # Get top features
            top_features = features[rfe.support_]

            # Update features X var with top features
            X_var = comp_df.loc[:, top_features]

            # Scale data to remove bias
            X_var = StandardScaler().fit(X_var).transform(X_var)

            # Split data into training and testing data sets (train using top features)
            X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size=0.2, random_state=0)

            # Train logistic regression model
            rfe_model.fit(X_train, y_train)

            # Use logsitic regression for predictions
            yhat = rfe_model.predict(X_test)

            # Display feature ranking
            #print(rfe.support_)
            #print(rfe.ranking_)

            # Display top features
            #print(top_features)

            # List accuracy of model
            #print(mean_squared_error(y_test, yhat))


            # Track features remaining
            feat_rem += 1

            # Hold result data
            if feat_rem == len(features):

                topFeature = top_features[0]
                accuracy = mean_squared_error(y_test, yhat)

                # Track number of companies parsed
                comp_count += 1
                print("Calculated company... " + str(comp_count) + " of " + "1996")

        #print(topFeature)
        #print(accuracy)

        # Save top feature selected for stock in new column
        nasdaq_df.loc[nasdaq_df['Name'] == company, 'TopFeature'] = topFeature

        # Save prediction accuracy of company
        nasdaq_df.loc[nasdaq_df['Name'] == company, 'ModelAccuracy'] = accuracy

        #print(company)
        #print(nasdaq_df)







    # Shrink the data frame to get accuracy of each company
    bastardized_df = nasdaq_df.drop_duplicates('ModelAccuracy')

    # Sort df by accuracy
    bastardized_df = bastardized_df.sort_values(by='ModelAccuracy', ascending=True)

    print(bastardized_df)

    # Get a list of all unique companies (now sorted by accuracy)
    companies = bastardized_df['Name'].unique()

    # Create empty column for company prediction accuracy
    nasdaq_df['LongTermReturn'] = 0.0

    #TODO Multithread this

    # Loop through top 50 stocks
    for i in range(0, 50):

        # Get data frame for company
        comp_df = nasdaq_df.loc[(nasdaq_df['Name'] == companies[i])]


        try:

            # Convert everything to float
            comp_df = comp_df.apply(pd.to_numeric, errors='coerce')

            comp_df.index.freq = 'MS'

            # Calculate train cutoff
            length = len(comp_df.index)
            cutoff = length * 0.8
            cutoff = int(cutoff)

            # Split into train and test data frames
            train, test = comp_df.iloc[:cutoff], comp_df.iloc[cutoff:]

            # Train the model
            model = ExponentialSmoothing(train['Close'], seasonal='mul', seasonal_periods=12).fit()
            pred = model.forecast(24)

            # TODO Display plot with holt winters
            # Display the plot
            #plt.plot(train['Date'], train['Close'], label='Train')
            #plt.plot(test['Date'], test['Close'], label='Test')
            #plt.plot(pred.index, pred, label='Holt-Winters')

            #plt.legend(loc='best')
            #plt.show()



            # Assumes money was invested at end of train data
            # and sold at all time high
            highestPredValue = pred.max()
            returnPercentage = (highestPredValue / train['Close'].iloc[-1]) - 1


            #TODO: Calculate accuracy of holt winters
            #longTermPred = 0
            #longTermActl = 0
            #predAccuracy = longTermPred - longTermActl

            nasdaq_df.loc[nasdaq_df['Name'] == companies[i], 'LongTermReturn'] = returnPercentage
        except:
            continue








    # Shrink the data frame to get accuracy of each company
    bastardized_df = nasdaq_df.drop_duplicates('LongTermReturn')
    bastardized_df.reset_index(inplace=True)

    # Sort df by long term return
    bastardized_df = bastardized_df.sort_values(by='LongTermReturn', ascending=False)

    print(bastardized_df)

    # Hold top 3 stocks
    stocks = []

    # Get top 3 stocks
    stocks.append(bastardized_df['Name'].iloc[0])
    stocks.append(bastardized_df['Name'].iloc[1])
    stocks.append(bastardized_df['Name'].iloc[2])

    # Show top 3 stocks
    print(bastardized_df.iloc[:3])


    # Show
    for s in stocks:

        comp_df = nasdaq_df.loc[(nasdaq_df['Name'] == s)]

        # Convert everything to float
        comp_df = comp_df.apply(pd.to_numeric, errors='coerce')

        comp_df.index.freq = 'MS'

        # Calculate train cutoff
        length = len(comp_df.index)
        cutoff = length * 0.8
        cutoff = int(cutoff)

        # Split into train and test data frames
        train, test = comp_df.iloc[:cutoff], comp_df.iloc[cutoff:]

        # Display the plot
        plt.plot(train['Date'], train['Close'], label='Train')
        plt.plot(test['Date'], test['Close'], label='Test')

        plt.legend(loc='best')

    # Show plots
    plt.show()



    name = ''


    return name









# Get data frames
nasdaq_df, nyse_df, spac_df, allDf = dp.initializeDf()

# Get top stock
mushyStock = findBestStock(nasdaq_df)

# Show top stock
print(mushyStock)









