import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter

sns.set_style("darkgrid")

df1 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Endelig/Total.csv',
                 delimiter=';', low_memory=False)

df1['Datetime'] = pd.to_datetime(df1['Datetime'], format='%Y-%m-%d %H:%M:%S')


df2 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Endelig/noise_new.csv', delimiter=';', low_memory=False)
df2 = df2.drop(columns=['sin_time', 'cos_time', 'age', 'n_lambs'])

df3 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Endelig/noise_old.csv', delimiter=';', low_memory=False)
df3 = df3.drop(columns=['sin_time', 'cos_time', 'age', 'n_lambs'])

df4 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Endelig/noise_early.csv', delimiter=';', low_memory=False)
df4 = df4.drop(columns=['sin_time', 'cos_time', 'age', 'n_lambs'])

df5 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Endelig/noise_late.csv', delimiter=';', low_memory=False)
df5 = df5.drop(columns=['sin_time', 'cos_time', 'age', 'n_lambs'])

df6 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Endelig/noise_day.csv', delimiter=';', low_memory=False)
df6 = df6.drop(columns=['sin_time', 'cos_time', 'age', 'n_lambs'])

df7 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Endelig/noise_night.csv', delimiter=';', low_memory=False)
df7 = df7.drop(columns=['sin_time', 'cos_time', 'age', 'n_lambs'])


def StatisticalSignificance(df_one, df_two, feature):
    t_crit = 1.645
    print('len 1:', len(df_one[feature]))
    print('len 2:', len(df_two[feature]))

    SE = np.sqrt( (df_one[feature].std()**2) / len(df_one[feature]) + (df_two[feature].std()**2) / len(df_two[feature]) )
    print('standard error: ', SE)

    t_test =  np.abs( (np.mean(df_one[feature]) - np.mean(df_two[feature])) / SE)

    print('t_test value:', t_test)
    if t_test > t_crit:
        print('Statistical significant, reject H0')
    else:
        print('False, cannot reject H0')


#StatisticalSignificance(df2, df3, 'Velocity')

df_mother = df1[df1['uniq.log'] == '1803011758_2019']
df_lamb = df1[df1['uniq.log'] == '1632004928_2019']
#print(df_mother)
#print(df_lamb)

# Plot of activity per date in Tingvoll
fig1, ax1 = plt.subplots(figsize=(16, 8))
sns.lineplot(data=df_mother, x='Datetime', y='Temp', label='Mother', lw=2, color='orange')
sns.lineplot(data=df_lamb, x='Datetime', y='Temp', label='Lamb', color='steelblue', lw=2)
plt.axhline(y=20, label='All data threshold', color='firebrick', lw=3)
#plt.axhline(y=313, label='Heavier breed treshold', color='lightcoral', lw=3)
ax1.set_xlabel(' ', fontsize=1)
ax1.set_ylabel('Temperature, degrees Celsius', fontsize=35, labelpad=30)
ax1.set_title('Temperature, with threshold value', fontsize=40, pad=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
date_form = DateFormatter("%d.%m")
ax1.xaxis.set_major_formatter(date_form)
plt.legend()
ax1.legend(loc='upper left', frameon=True, prop={'size': 20})
plt.tight_layout()
#plt.savefig("/Users/ninasalvesen/Documents/Sauedata/Bilder/Master/01 Total/mother_temp.png", dpi=500)

plt.show()


