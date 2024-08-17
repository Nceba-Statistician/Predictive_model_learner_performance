from db_connection import Student_performance_df
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.impute import KNNImputer


# Student_performance_df_drop_na = Student_performance_df.dropna()


"""
plt.figure(figsize=(6,6))
plt.scatter(
    Student_performance_df_drop_na.index, Student_performance_df_drop_na['GPA'],
    color = 'blue', label = 'GPA Scatterplot'
    )
plt.ylabel('GPA')
plt.xlabel('Index')
plt.title('Just a view')
plt.savefig('GPA_Scatterplot.png')
"""
"""
knn_imputer = KNNImputer(n_neighbors=3)
Student_performance_df_imputed = pd.DataFrame(
    knn_imputer.fit_transform(
        Student_performance_df
    ), columns=Student_performance_df.columns
)"""
"""
plt.figure(figsize=(6,6))
plt.scatter(
    Student_performance_df_imputed.index, Student_performance_df_imputed['GPA'],
    color = 'Red', label = 'GPA Scatterplot'
    )
plt.ylabel('GPA')
plt.xlabel('Index')
plt.title('Just a view')
plt.savefig('GPA_Scatterplot_imputed.png')
"""










