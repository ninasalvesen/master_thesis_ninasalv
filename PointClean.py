import pandas as pd

df2 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen.csv', delimiter=';')
df2['Datetime'] = pd.to_datetime(df2['Datetime'], format='%Y-%m-%d %H:%M:%S')


def remove(df):
    temp = 0     # data set number
    i = 0        # have to check the first and two last lines for faulty points manually
    while i < (len(df)):
        if (i % 10000) == 0:
            # check progress against number of iterations
            print("Reached number: ", i)

        if pd.isnull(df.at[i, 'Datetime']) and temp != 0:
            df = df.drop(df.index[i + 1:i + 6])
            df = df.drop(df.index[i - 5:i])
            df.reset_index(inplace=True, drop=True)
        if pd.isnull(df.at[i, 'Datetime']) and temp == 0:
            df = df.drop(df.index[i + 1:i + 6])
            df.reset_index(inplace=True, drop=True)
        i += 1
    return df


def zeros(df):
    i = 0
    check = 0
    while i < (len(df)-2):
        if (i % 10000) == 0:
            # check progress against number of iterations
            print("Reached number: ", i)
        if pd.isnull(df.at[i, 'Datetime']):
            i += 1  # start the next check on the next data set
            continue
        if df.at[i, 'Lat'] == 0 and df.at[i + 1, 'Lat'] != 0:
            df.at[i, 'Lat'] = (df.at[i - 1, 'Lat'] + df.at[i + 1, 'Lat']) / 2
            df.at[i, 'Lon'] = (df.at[i - 1, 'Lon'] + df.at[i + 1, 'Lon']) / 2
            continue
        if df.at[i, 'Lat'] == 0 and df.at[i + 1, 'Lat'] == 0:
            if df.at[i + 2, 'Lat'] != 0:
                df.at[i, 'Lat'] = (df.at[i - 1, 'Lat'] + df.at[i + 2, 'Lat']) / 2
                df.at[i, 'Lon'] = (df.at[i - 1, 'Lon'] + df.at[i + 2, 'Lon']) / 2
                continue
        if df.at[i, 'Lat'] == 0 and df.at[i + 1, 'Lat'] == 0:
            if df.at[i + 2, 'Lat'] == 0 and df.at[i + 3, 'Lat'] != 0:
                df.at[i, 'Lat'] = (df.at[i - 1, 'Lat'] + df.at[i + 3, 'Lat']) / 2
                df.at[i, 'Lon'] = (df.at[i - 1, 'Lon'] + df.at[i + 3, 'Lon']) / 2
                continue

        if df.at[i, 'Lat'] == 0 and df.at[i + 1, 'Lat'] == 0:
            if df.at[i + 2, 'Lat'] == 0 and df.at[i + 3, 'Lat'] == 0:
                check += 1
                i += 4
                continue
        # heller prøve å se på å finne neste punkt som ikke er null med en ny while løkke, og iterere nedover
        # helt til alle null-punkter er oppfyllt
        i += 1
    return df, check

print(df2[2386:2393])
df2, test = zeros(df2)
print(df2[2386:2393])
print('sjekk:', test)
#df2 = remove(df2)
#print(df2.head())
#df2.to_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/test.csv', index=False, sep=';')