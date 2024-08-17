import requests
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
response = requests.get("http://127.0.0.1:8000/items_get").json()

Student_performance_df = pd.DataFrame(response)


# print(Student_performance_df.dtypes)










