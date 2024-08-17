from regmodel import predictor_sig, target_sig
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
from statsmodels import api
import pickle

X_sig_train, X_sig_test, y_sig_train, y_sig_test = train_test_split(
    predictor_sig, target_sig, test_size=0.2, random_state=42
)

lr_sig = LinearRegression()
lr_sig_model = lr_sig.fit(X_sig_train, y_sig_train)

X_sig_train = api.add_constant(X_sig_train)
model_ols_sig = api.OLS(y_sig_train, X_sig_train).fit()

with open('regression_model_api.pkl', 'wb') as file:
    pickle.dump(lr_sig, file)

# print(model_ols_sig.summary())


