# MLStockForecasting

This is the final project for data science 3580

This project attempts to predict which stocks are the best for trading. Specifically, which stocks have an accurate model
and have a high return rate.

### Theory:
The idea is to find a stock that has a consistent repeating short term pattern and an exponential long term pattern. Combined,
this means that we can trade short term, and if lack of accuracy means we make a bad trade, we can still recover losses in the long term.

### Outcome:
The problem with the current algorithm is that it leans too much towards accuracy. This means the algoritm is biased towards
'safe' stocks that have little or no return. In addition, stocks are looked at on a day by day basis for short term trading. In
reality, we want to look for stocks that have a repeating pattern in the short term. This could be a pattern ranging from a couple
hours, to a week. Finally,
multiple parameters need to be added to allow a trader to decide how risky they want to be with their trades. A risk factor, and
risk/return metrics for each stock, can allow a mathematical approach to which combination of stocks create a balanced portfolio
with high returns.

## Method for figuring out best stock

### Day trading
- RFE: Recursive feature elimination to find the best feature for day trading
- The target is `tommorows percent change` for a given stock
- The top feature for each company is stored in a new column
- The feature accuracy is stored in a new column
- *Note: This method lacks in that it only finds the stocks with the best accuracy, not the best return*

### Forecasting: Long term stock trading
- Pull out top 50 stocks for day trading
- Use holt winters for each stock to predict long term gain
- List highest percentage return for prediction
- Get the top 3 stocks
- Predict the future of the stock
- Return top stock (good accuracy, and best return)
- *Note: This method is flawed as the most accurate day trading stocks are usually not stocks with high returns*


