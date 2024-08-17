from db_connection import Student_performance_df
from matplotlib import pyplot as plt
import seaborn as sn



plt.figure(figsize=(6,6))
sn.heatmap(Student_performance_df.corr() > 0.7, annot=True, cbar=False)
plt.title('Correlation > 70%')
plt.savefig('heatmap_df_corr.png')



