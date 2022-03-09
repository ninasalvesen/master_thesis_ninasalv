import pandas as pd

df1 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V3 med Haversine.csv', delimiter=';',
                 dtype={"Initial start": "str", "Start": "str", "Stop": "str"})

df1['Datetime'] = pd.to_datetime(df1['Datetime'], format='%Y-%m-%d %H:%M:%S')


df2 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Informasjon datasett Tingvoll.csv', delimiter=';')

df1['uniq.log'] = ''
df1['Farm'] = ''

print(df1)
print(df2)


dataset = 0
i = 0
while i < len(df1):
    if pd.isnull(df1.at[i, 'Datetime']):
        i += 1
        dataset += 1

    df1.at[i, 'uniq.log'] = df2.at[dataset - 1, 'datasett']
    df1.at[i, 'Farm'] = df2.at[dataset - 1, 'Farm']
    i += 1

print(df1)

df1.to_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V3 med Haversine.csv', index=False, sep=';')
