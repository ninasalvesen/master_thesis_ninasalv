import pandas as pd
import math

df2 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen uten pointclean.csv',
                  delimiter=';')
df2['Datetime'] = pd.to_datetime(df2['Datetime'], format='%Y-%m-%d %H:%M:%S')


def zeros(df):  # remover all zeros at beginning and end of sets
    i = 2
    while i < (len(df)):
        if (i % 10000) == 0:
            # check progress against number of iterations
            print("Reached number: ", i)
        temp = i
        n = 0
        while df.at[temp, 'Lat'] == 0:
            temp += 1
            n += 1
        if pd.isnull(df.at[temp, 'Datetime']) and n > 0:  # null på slutten av et sett
            df = df.drop(df.index[i:temp])
            df.reset_index(inplace=True, drop=True)
        if pd.isnull(df.at[i-1, 'Datetime']) and n > 0:  # null på starten av et sett
            df = df.drop(df.index[i:temp])
            df.reset_index(inplace=True, drop=True)
        i += 1
    return df


def remove(df):  # removes first and last 5 points in each set
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


def impute(df):  # imputes average points where there are zeros
    i = 0
    while i < (len(df)):
        if (i % 10000) == 0:
            # check progress against number of iterations
            print("Reached number: ", i)
        if pd.isnull(df.at[i, 'Datetime']):
            i += 1  # start the next check on the next data set
            continue
        temp = i
        n = 0
        while df.at[temp, 'Lat'] == 0:
            temp += 1
            n += 1
        if n % 2 == 0:
            if n >= 4:
                n = int(n / 2)
                df.at[i + n, 'Lat'] = (df.at[i - 1, 'Lat'] + df.at[temp, 'Lat']) / 2
                df.at[i + n, 'Lon'] = (df.at[i - 1, 'Lon'] + df.at[temp, 'Lon']) / 2
                while n > 1:
                    n -= 1
                    df.at[i + n, 'Lat'] = (df.at[i - 1, 'Lat'] + df.at[i + n + 1, 'Lat']) / 2
                    df.at[i + n, 'Lon'] = (df.at[i - 1, 'Lon'] + df.at[i + n + 1, 'Lon']) / 2
                    df.at[temp - n, 'Lat'] = (df.at[temp - n - 1, 'Lat'] + df.at[temp, 'Lat']) / 2
                    df.at[temp - n, 'Lon'] = (df.at[temp - n - 1, 'Lon'] + df.at[temp, 'Lon']) / 2
                df.at[i, 'Lat'] = (df.at[i - 1, 'Lat'] + df.at[i + 1, 'Lat']) / 2
                df.at[i, 'Lon'] = (df.at[i - 1, 'Lon'] + df.at[i + 1, 'Lon']) / 2
            if n == 2:
                df.at[i, 'Lat'] = (df.at[i - 1, 'Lat'] + df.at[temp + 1, 'Lat']) / 2
                df.at[i, 'Lon'] = (df.at[i - 1, 'Lon'] + df.at[temp + 1, 'Lon']) / 2
                df.at[i + 1, 'Lat'] = (df.at[i, 'Lat'] + df.at[temp + 1, 'Lat']) / 2
                df.at[i + 1, 'Lon'] = (df.at[i, 'Lon'] + df.at[temp + 1, 'Lon']) / 2
        if n % 2 != 0:
            if n >= 3:
                n = math.floor(n / 2)
                df.at[i + n, 'Lat'] = (df.at[i - 1, 'Lat'] + df.at[temp, 'Lat']) / 2
                df.at[i + n, 'Lon'] = (df.at[i - 1, 'Lon'] + df.at[temp, 'Lon']) / 2
                while n > 0:
                    n -= 1
                    df.at[i + n, 'Lat'] = (df.at[i - 1, 'Lat'] + df.at[i + n + 1, 'Lat']) / 2
                    df.at[i + n, 'Lon'] = (df.at[i - 1, 'Lon'] + df.at[i + n + 1, 'Lon']) / 2
                    df.at[temp - n - 1, 'Lat'] = (df.at[temp - n - 2, 'Lat'] + df.at[temp, 'Lat']) / 2
                    df.at[temp - n - 1, 'Lon'] = (df.at[temp - n - 2, 'Lon'] + df.at[temp, 'Lon']) / 2
            else:
                df.at[i, 'Lat'] = (df.at[i - 1, 'Lat'] + df.at[i + 1, 'Lat']) / 2
                df.at[i, 'Lon'] = (df.at[i - 1, 'Lon'] + df.at[i + 1, 'Lon']) / 2
        i += 1
    return df


df2 = zeros(df2)
df2 = remove(df2)
df2 = impute(df2)
df2.to_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen.csv', index=False, sep=';')
