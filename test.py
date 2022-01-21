import pandas as pd

df = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll test.csv', delimiter=';',
                 dtype={"Initial start": "str", "Start": "str", "Stop": "str"})

df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d/%m/%Y %H:%M')

print(df.head())

# 62.91054
#df.loc[1, ('Lat')] = 4
#print(df.loc[1, ('Lat')])

print((df['Datetime'][2]-df['Datetime'][1]).components)
print(df.loc[172000, 'Datetime'].month - df.loc[1, 'Datetime'].month)

print('------')
print(df['Datetime'][172000])
df.loc[172000, 'Datetime'] = df.loc[172000, 'Datetime'].replace(year=2002)
print(df['Datetime'][172000])
print('------')






#df.to_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll test.csv')
