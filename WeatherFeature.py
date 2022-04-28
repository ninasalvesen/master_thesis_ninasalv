import pandas as pd

df1 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V8 reduced.csv',
                  delimiter=';', low_memory=False)
df1['Datetime'] = pd.to_datetime(df1['Datetime'], format='%Y-%m-%d %H:%M:%S')


df2 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen V9 uten_feilsett.csv',
                  delimiter=';', low_memory=False)
df2['Datetime'] = pd.to_datetime(df2['Datetime'], format='%Y-%m-%d %H:%M:%S')

df3 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/weather Tingvoll 2012.csv', delimiter=';')
df3['Datetime'] = pd.to_datetime(df3['Tid(norsk normaltid)'], format='%d.%m.%Y %H:%M')

df4 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/weather Fosen.csv', delimiter=';')
df4['Datetime'] = pd.to_datetime(df4['Tid(norsk normaltid)'], format='%d.%m.%Y %H:%M')

#print(df1)
#print(df2)
df3.drop(columns=['Navn', 'Stasjon', 'Tid(norsk normaltid)'], inplace=True)
df4.drop(columns=['Navn', 'Stasjon', 'Nedbør (12 t)', 'Tid(norsk normaltid)'], inplace=True)
df1['Temp'] = None
df2['Temp'] = None

print(df1)
print(df3)
#print(df2.head())
#print(df4)


def initTime(df_temp):
    i = 0
    while i < len(df_temp):
        df_temp.at[i, 'Date'] = df_temp.at[i, 'Datetime'].date()
        df_temp.at[i, 'Hour'] = df_temp.at[i, 'Datetime'].hour
        i += 1
    return df_temp


def GetTemp(df, df_temp):
    i = 0
    while i < len(df):
        if (i % 10000) == 0:
            # check progress against number of iterations
            print("Reached number: ", i)

        if pd.isnull(df.at[i, 'Datetime']):
            i += 1  # start the next check on the next data set in the second point
            continue

        index = df_temp[(df_temp['Date'] == df.at[i, 'Datetime'].date()) &
                        (df_temp['Hour'] == df.at[i, 'Datetime'].hour)].index.values[0]
        df.at[i, 'Temp'] = df_temp.at[index, 'Lufttemperatur']

        i += 1

    return df


df3 = initTime(df3)
df4 = initTime(df4)
print(df3)
test = GetTemp(df1, df3)
print(test)
