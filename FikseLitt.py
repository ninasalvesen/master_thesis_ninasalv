import pandas as pd
import numpy as np

df = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen uten timeclean.csv', delimiter=';')

df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S')
#df['Data set'] = ""
print(df.head())

df2 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/NIBIO_Telespor_lamsøye.csv', delimiter=';',
                  encoding='latin-1')

df2['uniq.log'] = df2['Telespor_id'].astype(str) + '_' + df2['yr'].astype(str)
df2['uniq.log_mor'] = df2['M_Telespor_id'].astype(str) + '_' + df2['yr'].astype(str)
print(df2.head())

df3 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Informasjon datasett Fosen.csv', delimiter=';')
df3['i, slutt'] = np.nan


sett = 0
i = 0
while i < len(df):
    if df.at[i, 'Data set'] == 'new_sheep':
        df3.at[sett, 'Sett'] = sett + 1
        df3.at[sett, 'Start'] = df.at[i + 1, 'Datetime']
        df3.at[sett, 'uniq.log'] = df.at[i + 1, 'uniq.log']
        df3['exists1'] = df3['uniq.log'].isin(df2['uniq.log'])
        df3['exists2'] = df3['uniq.log'].isin(df2['uniq.log_mor'])

        df3["exists1"] = df3['uniq.log'].isin(df2['uniq.log'])
        df3["exists2"] = df3['uniq.log'].isin(df2['uniq.log_mor'])
        if df3.at[df3[df3['uniq.log'] == df3.at[sett, 'uniq.log']].index.values[0], 'exists1']:
            index = df2[df2['uniq.log'] == df3.at[sett, 'uniq.log']].index.values[0]
            df3.at[sett, 'rase'] = df2.at[index, 'Rase']
            sjekk2 = len(df2[df2['uniq.log'] == df3.at[sett, 'uniq.log']].index.values)
            if sjekk2 > 1:
                print('wopsidops', i, sett)
        if df3.at[df3[df3['uniq.log'] == df3.at[sett, 'uniq.log']].index.values[0], 'exists2']:
            index = df2[df2['uniq.log_mor'] == df3.at[sett, 'uniq.log']].index.values[0]
            df3.at[sett, 'rase'] = df2.at[index, 'Rase']
            sjekk3 = len(df2[df2['uniq.log_mor'] == df3.at[sett, 'uniq.log']].index.values)
            if sjekk3 > 1:
                print('wopsidops2', i, sett)

        if sett > 1:
            df3.at[sett - 1, 'Slutt'] = df.at[i - 1, 'Datetime']
            df3.at[sett - 1, 'i, slutt'] = i-1
        sett += 1

    i += 1
    if (i % 10000) == 0:
        # check progress against number of iterations
        print("Reached number: ", i)

print(df3)
print(df3['rase'])

def count(df):
    sett = []
    i = 0
    while i < len(df):
        if df.at[i, 'uniq.log'] not in sett:
            sett.append(df.at[i, 'uniq.log'])
        i += 1
    return sett
"""

sett1 = count(df)
sett2 = count(df2)
i = 0
while i < len(df2):
    if df2.at[i, 'uniq.log_mor'] not in sett2:
        sett2.append(df2.at[i, 'uniq.log_mor'])
    i += 1

print(len(sett1))
print(len(sett2))
count1 = 0
count2 = 0
print('------')
for i in range(len(sett1)):
    if sett1[i] not in sett2:
        count2 += 1
        print(sett1[i])

for i in range(len(sett2)):
    if sett2[i] not in sett1:
        count1 += 1
print(count2)
print(count1)
"""
# sett 1 = Samlet data Fosen med alle faktiske datasettene med punkter
# sett 2 er infosiden med info om hver sau og hvert sett

#df.to_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen uten timeclean.csv', index=False, sep=';')