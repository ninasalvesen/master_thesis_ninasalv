import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')


df1 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Endelig/Total begge.csv',
                  sep=';', dtype={"Initial start": "str", "Start": "str", "Stop": "str"})
df1['Datetime'] = pd.to_datetime(df1['Datetime'], format='%Y-%m-%d %H:%M:%S')

df1.dropna(subset=['Datetime'], inplace=True)
df1.reset_index(inplace=True, drop=True)

df1 = df1.drop(columns=['Datetime', 'Lat', 'Lon', 'Data set', 'uniq.log',
                        'race', 'besetning'])


print(df1)
print(df1.columns)
print(df1.describe())

#print(df2)


#plt.figure()
#corr = df1.corr()
#sns.heatmap(corr)
#plt.show()

plt.figure(figsize=(16, 16))
pd.plotting.scatter_matrix(df1)
plt.tight_layout()
plt.show()


