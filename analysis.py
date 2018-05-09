from sklearn import linear_model
from sklearn import metrics
import pandas as pd
import numpy as np
from scipy import stats

def prepare_data(df):
    mapping = {
        "Ex": 5, # Excellent
        "Gd": 4, # Good
        "TA": 3, # Average/Typical
        "Fa": 2, # Fair
        "Po": 1 # Poor
    }
    df["ExterQual"] = df["ExterQual"].map(mapping)
    df["ExterCond"] = df["ExterCond"].map(mapping)

    df["log_SalePrice"] = np.log(df["SalePrice"])
    df["log_LotArea"] = np.log(df["LotArea"])
    df["log_BedroomAbvGr"] = np.log(df["BedroomAbvGr"])
    df["log_ExterQual"] = np.log(df["ExterQual"])
    df["log_ExterCond"] = np.log(df["ExterCond"])
    return df

# credit: https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
def summary(lm, X, y):
    params = np.append(lm.intercept_,lm.coef_)
    predictions = lm.predict(X)
    
    newX = pd.DataFrame({"Constant":np.ones(len(X))}).join(pd.DataFrame(X))
    MSE = (sum((y-predictions)**2))/(len(newX)-len(newX.columns))

    # Note if you don't want to use a DataFrame replace the two lines above with
    # newX = np.append(np.ones((len(X),1)), X, axis=1)
    # MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))

    var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params/ sd_b

    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in ts_b]

    sd_b = np.round(sd_b,3)
    ts_b = np.round(ts_b,3)
    p_values = np.round(p_values,3)
    params = np.round(params,4)

    myDF3 = pd.DataFrame()
    myDF3["Var Name"], myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["Probabilites"] = [newX.columns.tolist(), params,sd_b,ts_b,p_values]
    print(myDF3)
    
train = pd.read_csv("train.csv")
train = prepare_data(train)

sale_price = train["log_SalePrice"]
sale_price = [[elem] for elem in sale_price]
X = train[["LotArea", "BedroomAbvGr", "ExterQual","ExterCond"]]

y_true = train["log_SalePrice"]

ols = linear_model.LinearRegression()
ols.fit(X, sale_price)
y_pred = ols.predict(X)

print("R^2", metrics.r2_score(y_true, y_pred))
summary(ols, X, sale_price)
