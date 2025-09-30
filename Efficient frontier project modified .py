import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBRegressor

simulations = 100000


tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "PLTR"] #can use any tickers you want, just make sure they are valid. The more tickers you use the longer it will take to run tho
numberofassets = len(tickers)
data = yf.download(tickers, start="2023-01-01", interval="1d")


if isinstance(data.columns, pd.MultiIndex):
    if "Adj Close" in data.columns.levels[0]:
        adjclose = data["Adj Close"]
    else:
        adjclose = data["Close"]
else:
    if "Adj Close" in data.columns:
        adjclose = data["Adj Close"]
    else:
        adjclose = data["Close"]

predictedvolatility = {}

for ticker in tickers:
    df = pd.DataFrame()
    df["Close"] = adjclose[ticker]
    df["Return"] = df["Close"].pct_change()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA100"] = df["Close"].rolling(window=100).mean()
    df["Volatility100"] = df["Return"].rolling(window=100).std()
    df["Volatility30"] = df["Return"].rolling(window=30).std()
    df["Target"] = df["Volatility30"].shift(-30) #want to predict standtard deviation of next 30 days for the next day
    df = df.dropna()

    X = df[["MA10","MA20","MA100","Volatility100","Volatility30"]]
    Y = df["Target"]

    tscv = TimeSeriesSplit(n_splits=5)
    paramgrid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1], #i chose smaller values because higher learning rates can lead to overfitting which is not what i want.
        "subsample": [0.6, 0.8, 1.0],
    }

    model = GridSearchCV(XGBRegressor(objective="reg:squarederror", random_state=42),
                         paramgrid, cv=tscv, n_jobs=-1, scoring="neg_mean_squared_error")
    model.fit(X, Y)

    bestmodel = model.best_estimator_
    latest_features = X.iloc[[-1]]
    predictedvol = bestmodel.predict(latest_features)[0]

    # annualize using forecast_vol better than using average volatility from daily returns over 2 years
    predictedvolatility[ticker] = predictedvol * np.sqrt(252) 

dailyreturns = adjclose.pct_change().dropna()
meanreturn = dailyreturns.mean()
annualisedmeanreturn = (1+meanreturn)**252 - 1


corrmatrix = dailyreturns.corr() #this is a panda df
volatility = np.array([predictedvolatility[ticker] for ticker in tickers]) #we need to turn the dictionary values into a numpy array so we can deal with then values
Diagonal = np.diag(volatility)
covmatrix = Diagonal @ corrmatrix.values @ Diagonal 
covmatrix = pd.DataFrame(covmatrix, index=tickers, columns=tickers)

randomweights = np.random.dirichlet(np.ones(numberofassets), simulations)
portfolioexpectedreturn = np.zeros(simulations) #using the same method i used in the reinsurance treaty pricing simulation
portfoliovolatility = np.zeros(simulations)
portfolioskewness = np.zeros(simulations) 
portfolioexcesskurtosis = np.zeros(simulations)


for i in range(simulations):
    portfolioexpectedreturn[i] = np.dot(randomweights[i], annualisedmeanreturn.values)
    portfoliovolatility[i] = np.sqrt(np.dot(randomweights[i].T, np.dot(covmatrix, randomweights[i])))
    
    portfolioreturns = np.dot(dailyreturns.to_numpy(), randomweights[i])
    dailymean = np.mean(portfolioreturns)
    dailystd = np.std(portfolioreturns)

    portfolioskewness[i] = np.mean((portfolioreturns - dailymean)**3) / (dailystd ** 3) #skewness is dimensionless, thus if distribution is negatively skewed at a daily level it will still be skewed at the annual level.
    portfolioexcesskurtosis[i] = (np.mean((portfolioreturns - dailymean)**4) / (dailystd ** 2)**2) - 3 #kurtsosis is also dimensionless.



riskfreerate = 0 #this is for simnplicity had to search formula online. This is inaccurate, usually would follow a treasury bond yield.
sharperatio = (portfolioexpectedreturn - riskfreerate) / portfoliovolatility

minvarianceindex = np.argmin(portfoliovolatility)
minvariancereturn = portfolioexpectedreturn[minvarianceindex]
minvariancevolatility = portfoliovolatility[minvarianceindex]

maxsharpeidx = np.argmax(sharperatio)
maxsharpereturn = portfolioexpectedreturn[maxsharpeidx]
maxsharpevol = portfoliovolatility[maxsharpeidx]

 
def weightconstraint(weights): #this is a constraint to ensure sum of weights = 1
    return np.sum(weights) - 1

def returnconstraint(weights, meanreturn, target): #ensures portfolio return = target return
    return np.dot(weights, meanreturn) - target

def objectivefunction(weights, covmatrix, skewnesspenalty, kurtosispenalty, dailyreturns):
    portfolioreturns = np.dot(dailyreturns.to_numpy(), weights)
    mean = np.mean(portfolioreturns)
    std = np.std(portfolioreturns)

    skew = np.mean((portfolioreturns - mean)**3) / (std**3)
    excesskurtosis = np.mean((portfolioreturns - mean)**4) / (std**4) - 3

    annualvol = np.sqrt(weights.T @ covmatrix @ weights)
    skewpen = skewnesspenalty * max(0, -skew) #ensures only negative skew is penalised (we want postive skew)
    kurtosispen = kurtosispenalty * (max(0, excesskurtosis)) #this ensures we punish excess kurtosis (we dont want a postive kutosis value)

    return annualvol + skewpen + kurtosispen

targetreturns = np.linspace(min(portfolioexpectedreturn), max(portfolioexpectedreturn), 200)
frontiervalues = []

for target in targetreturns:
    guess = np.ones(numberofassets) / numberofassets
    constraints = [
        {"type": "eq", "fun": weightconstraint},
        {"type": "eq", "fun": lambda w: returnconstraint(w, annualisedmeanreturn.values, target)}
    ]
    bounds = []
    for i in range(numberofassets):
        bounds.append((0,1)) #ensures for each asset the weight is bounded between 0 and 1 ie no leveraging or short selling

    result = minimize(objectivefunction, guess, args= (covmatrix, 0.1, 0.1, dailyreturns), method="SLSQP", bounds=bounds, constraints=constraints)
    
    if result.success:
        vol = np.sqrt(result.x.T @ covmatrix @ result.x)
        frontiervalues.append(vol)
    else:
        frontiervalues.append(np.nan)


print(maxsharpereturn*100)
print(maxsharpevol*100)

maxsharpeweights = randomweights[maxsharpeidx]
optimalportfolio = pd.DataFrame({"Ticker": tickers, "Weight": maxsharpeweights})
print(optimalportfolio)

plt.figure(figsize=(10,6))
plt.scatter(portfoliovolatility, portfolioexpectedreturn, c='blue', alpha=0.5)
plt.plot(frontiervalues, targetreturns, 'r-', linewidth=2, label='Efficient Frontier')
plt.scatter(minvariancevolatility, minvariancereturn, c='red', marker='*', s=200, label='Min Variance')
plt.scatter(maxsharpevol, maxsharpereturn, c='green', marker='*', s=200, label='Max Sharpe')
plt.xlabel("Portfolio Volatility")
plt.ylabel("Expected Return")
plt.title("Random Portfolio Simulation with Efficient Frontier")
plt.legend()
plt.show(),
