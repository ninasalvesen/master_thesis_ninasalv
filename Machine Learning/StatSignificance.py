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


def StatisticalSignificance_ttest(df_one, df_two, feature):
    t_crit = 1.96
    print('len 1:', len(df_one[feature]))
    print('len 2:', len(df_two[feature]))

    SE = np.sqrt( (df_one[feature].std()**2) / len(df_one[feature]) + (df_two[feature].std()**2) / len(df_two[feature]) )

    t_test =  np.abs( (np.mean(df_one[feature]) - np.mean(df_two[feature])) / SE)

    print('t_test value:', t_test)
    if t_test > t_crit:
        print('Statistical significant, reject H0')
    else:
        print('False, cannot reject H0')


StatisticalSignificance_ttest(df2, df3, 'Temp')

# Find percentile of threshold values
df_percentile = df1[df1['Altitude'] < 350]
print((len(df_percentile)/len(df1))*100)

df_mother = df1[df1['uniq.log'] == '1803011758_2019']
df_lamb = df1[df1['uniq.log'] == '1632004928_2019']

# get one day instead for sow and lamb, to plot the angle

df_mother.reset_index(inplace=True, drop=True)
df_lamb.reset_index(inplace=True, drop=True)

for i in range(len(df_mother)):
    df_mother.at[i, 'hour'] = df_mother.at[i, 'Datetime'].hour
df_mother['Datetime'] = pd.to_datetime(df_mother['Datetime'].dt.date.astype(str))
df_mother = df_mother[df_mother['Datetime'] == '2019-07-20']

for i in range(len(df_lamb)):
    df_lamb.at[i, 'hour'] = df_lamb.at[i, 'Datetime'].hour
df_lamb['Datetime'] = pd.to_datetime(df_lamb['Datetime'].dt.date.astype(str))
df_lamb = df_lamb[df_lamb['Datetime'] == '2019-07-20']


# Plot of features against threshold values
fig1, ax1 = plt.subplots(figsize=(16, 8))
sns.lineplot(data=df_mother, x='Datetime', y='Altitude', label='Mother', lw=3, color='orange')
sns.lineplot(data=df_lamb, x='Datetime', y='Altitude', label='Lamb', color='steelblue', lw=3)
plt.axhline(y=350, label='All data threshold', color='firebrick', lw=3)
#plt.axhline(y=313, label='Heavier breed treshold', color='lightcoral', lw=3)
ax1.set_xlabel(' ', fontsize=1)
ax1.set_ylabel('Altitude, mamsl', fontsize=35, labelpad=30)
ax1.set_title('Altitude of mother sow & lamb, with threshold value', fontsize=40, pad=30)
#labels = ['00:00', '06:00', '12:00', '18:00']
#plt.xticks(fontsize=25, ticks=[0, 6, 12, 18], labels=labels)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
date_form = DateFormatter("%d.%m")
ax1.xaxis.set_major_formatter(date_form)
plt.legend()
ax1.legend(loc='lower right', frameon=True, prop={'size': 20})
plt.tight_layout()
#plt.savefig("/Users/ninasalvesen/Documents/Sauedata/Bilder/Master/01 Total/mother_altitude.png", dpi=500)

plt.show()


