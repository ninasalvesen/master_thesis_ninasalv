import pandas as pd

df1 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V4 after cut 2.0.csv', delimiter=';',
                 dtype={"Initial start": "str", "Start": "str", "Stop": "str"})

df1['Datetime'] = pd.to_datetime(df1['Datetime'], format='%d/%m/%Y %H:%M')

df2 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen V5 after cut 2.0.csv', delimiter=';')

df2['Datetime'] = pd.to_datetime(df2['Datetime'], format='%d/%m/%Y %H:%M')


df1 = df1.drop(columns=['TTF', '2D3D', 'Alt', 'DOP', 'SVs', 'FOM', 'Temp', 'Bat', 'Status', 'SCap', 'GPS',
                        'GSM', 'Initial start', 'Start', 'Stop'])

df2 = df2.drop(columns=['Temp', 'M1', 'M2', 'yr', 'mon', 'day'])

print(df1)
print(df2)

df1.to_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V5 updated format.csv', index=False, sep=';')
df2.to_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen V6 updated format.csv', index=False, sep=';')
