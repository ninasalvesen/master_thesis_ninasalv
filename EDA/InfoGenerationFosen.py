import pandas as pd
import numpy as np

df = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Total Fosen med angle.csv', delimiter=';')
print(df.head())
df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S')

df2 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/NIBIO_Telespor_lamsøye.csv', delimiter=';', encoding='latin-1')

df2['uniq.log'] = df2['Telespor_id'].astype(str) + '_' + df2['yr'].astype(str)
df2['uniq.log_mor'] = df2['M_Telespor_id'].astype(str) + '_' + df2['yr'].astype(str)

df3 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Informasjon datasett Fosen mal.csv', delimiter=';')

#df3['i, slutt'] = np.nan


sett = 0
i = 0
while i < len(df):
    if df.at[i, 'Data set'] == 'new_sheep':
        df3.at[sett, 'Sett'] = sett + 1
        df3.at[sett, 'Start'] = df.at[i + 1, 'Datetime']
        df3.at[sett, 'uniq.log'] = df.at[i + 1, 'uniq.log']
        df3.at[sett, 'besetning'] = df.at[i + 1, 'besetning']
        df3['exists1'] = df3['uniq.log'].isin(df2['uniq.log'])
        df3['exists2'] = df3['uniq.log'].isin(df2['uniq.log_mor'])

        if df3.at[df3[df3['uniq.log'] == df3.at[sett, 'uniq.log']].index.values[0], 'exists1']:
            index = df2[df2['uniq.log'] == df3.at[sett, 'uniq.log']].index.values[0]
            df3.at[sett, 'rase'] = df2.at[index, 'Race']
        if df3.at[df3[df3['uniq.log'] == df3.at[sett, 'uniq.log']].index.values[0], 'exists2']:
            index = df2[df2['uniq.log_mor'] == df3.at[sett, 'uniq.log']].index.values[0]
            df3.at[sett, 'rase'] = df2.at[index, 'Race']

        if sett > 0:
            df3.at[sett - 1, 'Slutt'] = df.at[i - 1, 'Datetime']
            df3.at[sett - 1, 'i, slutt'] = i - 1
            if sett > 2:
                df3.at[sett - 1, 'Size'] = df3.at[sett - 1, 'i, slutt'] - df3.at[sett - 2, 'i, slutt'] - 1
                df3.at[0, 'Size'] = df3.at[0, 'i, slutt']
                df3.at[1, 'Size'] = df3.at[1, 'i, slutt'] - df3.at[0, 'i, slutt']


        sett += 1
    i += 1
    if i == len(df) - 1:
        df3.at[sett - 1, 'Slutt'] = df.at[i, 'Datetime']
        df3.at[sett - 1, 'i, slutt'] = i
        df3.at[sett - 1, 'Size'] = df3.at[sett - 1, 'i, slutt'] - df3.at[sett - 2, 'i, slutt'] - 1
    if (i % 10000) == 0:
        # check progress against number of iterations
        print("Reached number: ", i)


i = 0
count_true = 0
count_false = 0
df3['alderstatus'] = ''

while i < len(df3):
    if df3.at[i, 'exists1']:
        if df3.at[i, 'exists2']:
            print('begge true', i + 1, df3.at[i, 'uniq.log'])
            df3.at[i, 'alderstatus'] = 'lam'  # bekreftet av Jennifer at dupliserte sett er in fact lam
            count_true += 1
        else:
            df3.at[i, 'alderstatus'] = 'lam'
    if not df3.at[i, 'exists1']:
        if not df3.at[i, 'exists2']:
            print('begge false', i + 1, df3.at[i, 'uniq.log'])
            df3.at[i, 'alderstatus'] = 'ubestemt'
            count_false += 1
        else:
            df3.at[i, 'alderstatus'] = 'voksen'
    i += 1

print(count_true)
print(count_false)

df3.to_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Informasjon datasett Fosen after cut 6.0.csv', index=False, sep=';')



