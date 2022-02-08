import pandas as pd
import numpy as np

df = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen ubehandlet.csv', delimiter=';')

df['Datetime'] = pd.to_datetime(df['Date.Time'], format='%d/%m/%Y %H:%M')
df['Data set'] = ""
print(df.head())

# df2 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/NIBIO_Telespor_lamsøye.csv', delimiter=';',
#                  encoding='latin-1')

# df2['uniq.log'] = df2['Telespor_id'].astype(str) + '_' + df2['yr'].astype(str)
# df2['uniq.log_mor'] = df2['M_Telespor_id'].astype(str) + '_' + df2['yr'].astype(str)


sett = []
i = 0
while i < len(df):
    if df.at[i, 'uniq.log'] not in sett:
        sett.append(df.at[i, 'uniq.log'])
        new_line = pd.DataFrame({'insert': np.nan}, index=[i])
        df = pd.concat([df.iloc[:i], new_line, df.iloc[i:]]).reset_index(drop=True)
        df.at[i, 'Data set'] = 'new_sheep'
        df = df.drop(columns='insert')
    i += 1
    if (i % 10000) == 0:
        # check progress against number of iterations
        print("Reached number: ", i)


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