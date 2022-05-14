import pandas as pd

df = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Total Tingvoll med angle.csv',
                 delimiter=';', dtype={"Initial start": "str", "Start": "str", "Stop": "str"})
print(df.head())
df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S')

df2 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Informasjon datasett Tingvoll.csv', delimiter=';')
df3 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Informasjon datasett Tingvoll after cut.csv', delimiter=';')
print(df2.head())

sett = 0
i = 0
while i < len(df):
    if df.at[i, 'Data set'] == 'new_sheep':
        df2.at[sett, 'Sett'] = sett + 1
        df2.at[sett, 'Start'] = df.at[i + 1, 'Datetime']
        df2.at[sett, 'datasett'] = df.at[i + 1, 'uniq.log']
        df2.at[sett, 'Farm'] = df.at[i + 1, 'Farm']
        index = df3[df3['datasett'] == df.at[i + 1, 'uniq.log']].index.values[0]
        df2.at[sett, 'rase'] = df3.at[index, 'rase']
        df2.at[sett, 'besetning'] = df.at[i + 1, 'besetning']

        if sett > 0:
            df2.at[sett - 1, 'Slutt'] = df.at[i - 1, 'Datetime']
            df2.at[sett - 1, 'i, slutt'] = i - 1
            df2.at[0, 'Size'] = df2.at[0, 'i, slutt']
            if sett > 1:
                df2.at[sett - 1, 'Size'] = df2.at[sett - 1, 'i, slutt'] - df2.at[sett - 2, 'i, slutt'] - 1
        sett += 1
    i += 1
    if i == len(df) - 1:
        df2.at[sett - 1, 'Slutt'] = df.at[i, 'Datetime']
        df2.at[sett - 1, 'i, slutt'] = i
        df2.at[sett - 1, 'Size'] = df2.at[sett - 1, 'i, slutt'] - df2.at[sett - 2, 'i, slutt'] - 1
        df2.at[0, 'Size'] = df2.at[0, 'i, slutt']
    if (i % 10000) == 0:
        # check progress against number of iterations
        print("Reached number: ", i)


df2.to_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Informasjon datasett Tingvoll final.csv', index=False, sep=';')
