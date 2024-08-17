from db_connection import Student_performance_df
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
from statsmodels import api


Student_performance_df_cols_dtype = ['Gender', 'Tutoring', 'Extracurricular',
                                     'Sports', 'Music', 'Volunteering']


Student_performance_df[Student_performance_df_cols_dtype] = Student_performance_df[Student_performance_df_cols_dtype].astype(int)

"""
predictor_cols_df = Student_performance_df.iloc[
    :, Student_performance_df.columns != [output_col_df,'StudentID']
]
"""
output_col_df = "GPA"

predictor_cols_df = Student_performance_df.drop(columns=[output_col_df, 'StudentID'])

target_df = Student_performance_df.loc[:, output_col_df]

X_train, X_test, y_train, y_test = train_test_split(
    predictor_cols_df, target_df, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr_model_df = lr.fit(X_train, y_train)

# print(f"Intercept: {lr.intercept_}\nCoef: {lr.coef_}\nR^2:{lr.score(X_train, y_train)}")
# 96.61% of the variation in the response variable can be explained by the two predictor variables in the model.

X_train = api.add_constant(X_train)
model_ols_df = api.OLS(y_train, X_train).fit()

predictors_df = ['Gender', 'StudyTimeWeekly', 'Absences', 'Tutoring',
                 'ParentalSupport', 'Extracurricular', 
                 'Sports', 'Music', 'GradeClass']

predictor_sig = Student_performance_df[predictors_df]
target_sig = Student_performance_df.loc[:, output_col_df]


# print(model_ols_df.summary())

# print(Student_performance_df.dtypes)
# print(target_sig.head())
# print(predictor_sig.dtypes)
# print(predictor_sig.dtypes)






"""
import scipy.stats

#find p-value
scipy.stats.norm.sf(abs(-0.77))

0.22064994634264962
"""

"""
import scipy.stats

#find p-value
scipy.stats.norm.sf(abs(1.87))

0.030741908929465954
"""
