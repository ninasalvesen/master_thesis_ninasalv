import pandas as pd
import numpy as np


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



#print(df6.describe())
#print(df7.describe())
#StatisticalSignificance(df2, df3, 'Velocity')


