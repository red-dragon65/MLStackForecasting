
# Graeme's Ideas

# Like the assignment says, first look at the total growth percentages and price differences
# comparing the start and end dates to figure out what the best stocks are.
# We can look at trends to see what could be important to look for/compare

# create a sliding window to find the rolling averages of the close number
# or do a linear regression to find the trendline

# same idea but with P/E


import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv('stocks/Nasdaq-Friday-April-20-2018.csv')
print(df)

def helloWorld():

    print("hello people")

    value = "testy"
    print("hello other people")

    return value


