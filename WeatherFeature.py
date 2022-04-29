import pandas as pd

df1 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V8 reduced.csv',
                  delimiter=';', low_memory=False)
df1['Datetime'] = pd.to_datetime(df1['Datetime'], format='%Y-%m-%d %H:%M:%S')


df2 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen V9 uten_feilsett.csv',
                  delimiter=';', low_memory=False)
df2['Datetime'] = pd.to_datetime(df2['Datetime'], format='%Y-%m-%d %H:%M:%S')

df3 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/weather Tingvoll.csv', delimiter=';')
df3['Datetime'] = pd.to_datetime(df3['Tid(norsk normaltid)'], format='%d.%m.%Y %H:%M')

df4 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/weather Fosen.csv', delimiter=';')
df4['Datetime'] = pd.to_datetime(df4['Tid(norsk normaltid)'], format='%d.%m.%Y %H:%M')


df3.drop(columns=['Navn', 'Stasjon', 'Tid(norsk normaltid)'], inplace=True)
df4.drop(columns=['Navn', 'Stasjon', 'Nedbør (12 t)', 'Tid(norsk normaltid)'], inplace=True)
#df1['Temp'] = None
#df2['Temp'] = None


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

        try:
            index = df_temp[(df_temp['Date'] == df.at[i, 'Datetime'].date()) &
                            (df_temp['Hour'] == df.at[i, 'Datetime'].hour)].index.values[0]
            df.at[i, 'Temp'] = df_temp.at[index, 'Lufttemperatur']
        except:
            print(df.at[i, 'Datetime'], i)

        i += 1

    return df


def ImputeTemp(df):
    df['Temp'] = df['Temp'].str.replace(',', '.').astype(float)
    i = 0
    while i < len(df):
        if (i % 10000) == 0:
            # check progress against number of iterations
            print("Reached number: ", i)

        if pd.isnull(df.at[i, 'Datetime']):
            i += 1  # start the next check on the next data set in the second point
            continue

        try:
            if pd.isnull(df.at[i, 'Temp']):
                df.at[i, 'Temp'] = (df.at[i - 1, 'Temp'] + df.at[i + 1, 'Temp']) / 2
        except:
            print(i, df.at[i, 'Datetime'])
            print(df.at[i - 1, 'Temp'])
            print(df.at[i + 1, 'Temp'])

        i += 1
    return df


#df3 = initTime(df3)
#temp1 = GetTemp(df1, df3)
#temp1 = ImputeTemp(temp1)
#temp1.to_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V9 med temp.csv', index=False, sep=';')

#df4 = initTime(df4)
#temp2 = GetTemp(df2, df4)
#temp2 = ImputeTemp(temp2)
#temp2.to_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen V10 med temp.csv', index=False, sep=';')
