import pandas as pd



df1 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V0 ubehandlet.csv',
                  sep='\t', dtype={"Initial start": "str", "Start": "str", "Stop": "str"})
#df1['Datetime'] = pd.to_datetime(df1['Datetime'], format='%Y-%m-%d %H:%M:%S')

print(df1)

df2 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen V0 ubehandlet.csv',
                  delimiter=';')
#df2['Datetime'] = pd.to_datetime(df2['Datetime'], format='%Y-%m-%d %H:%M:%S')

print(df2.columns)