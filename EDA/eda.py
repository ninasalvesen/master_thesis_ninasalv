import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')


df1 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Endelig/Total.csv',
                  sep=';', dtype={"Initial start": "str", "Start": "str", "Stop": "str"})
df1['Datetime'] = pd.to_datetime(df1['Datetime'], format='%Y-%m-%d %H:%M:%S')

df1.dropna(subset=['Datetime'], inplace=True)
df1.reset_index(inplace=True, drop=True)

#df1 = df1.drop(columns=['Datetime', 'Lat', 'Lon', 'Data set', 'uniq.log', 'race', 'besetning'])

#df1.rename(columns={'Haversine':'Velocity'}, inplace=True)
print(df1.columns)


plt.figure(figsize=(10, 5))
corr = df1.corr()
sns.heatmap(corr, annot=True, linewidths=0.5, vmin=-1, vmax=1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('Heatmap of feature correlation', fontsize=25, pad=15)
plt.tight_layout()
#plt.savefig("/Users/ninasalvesen/Documents/Sauedata/Bilder/Master/01 Total/heatmap.png", dpi=500)
plt.show()


plt.figure()
sns.pairplot(df1, diag_kind='hist')
plt.tight_layout()
#plt.savefig("/Users/ninasalvesen/Documents/Sauedata/Bilder/Master/01 Total/pairplot.png", dpi=500)
plt.show()


