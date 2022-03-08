import pandas as pd

df1 = pd.read_csv("/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V3 med Haversine.csv", delimiter=';',
                 dtype={"Initial start": "str", "Start": "str", "Stop": "str"})

df1['Datetime'] = pd.to_datetime(df1['Datetime'], format='%Y-%m-%d %H:%M:%S')

df2 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen V4 med Haversine.csv',
                  delimiter=';')
df2['Datetime'] = pd.to_datetime(df2['Datetime'], format='%Y-%m-%d %H:%M:%S')

print(df1)
print(df2)


def TimeIntervalFosen(df):
    dataset = 1
    k = 1  # mark the start of the dataset
    temp = dataset
    i = 1  # start at first line of data
    a = True
    while i < len(df):
        if pd.isnull(df.at[i, 'Datetime']):
            print(dataset)
            k = i + 1
            dataset += 1
            temp = dataset
            i += 1
            a = True
        if df.at[i, 'Datetime'].year == 2018:
            if df.at[i, 'Datetime'].month == 6 and df.at[i, 'Datetime'].day == 9:  # start date = 09.06.18
                if temp == dataset:
                    df = df.drop(df.index[k:i])
                    df.reset_index(inplace=True, drop=True)
                    temp += 1
            if df.at[i, 'Datetime'].month == 6 and df.at[i, 'Datetime'].day == 30:  # end date = 29.06.18
                end = i + 1
                while not pd.isnull(df.at[end, 'Datetime']):
                    end += 1
                df = df.drop(df.index[i:end])
                df.reset_index(inplace=True, drop=True)
                continue

        if df.at[i, 'Datetime'].year == 2019:
            if df.at[i, 'Datetime'].month == 6 and df.at[i, 'Datetime'].day == 7:  # start date = 07.06.19
                if temp == dataset:
                    df = df.drop(df.index[k:i])
                    df.reset_index(inplace=True, drop=True)
                    temp += 1
            if df.at[i, 'Datetime'].month == 7 and df.at[i, 'Datetime'].day == 4:  # end date = 03.07.19 or 31.08.19
                end = i + 1
                while not pd.isnull(df.at[end, 'Datetime']):
                    if df.at[end, 'Datetime'].month == 9 and df.at[end, 'Datetime'].day == 1:
                        a = False
                        q = end
                        while not pd.isnull(df.at[end, 'Datetime']):
                            end += 1
                        df = df.drop(df.index[q:end])
                        df.reset_index(inplace=True, drop=True)
                        print(df.at[i-1, 'Datetime'])
                        print(df.at[i, 'Datetime'])
                        print(df.at[i+1, 'Datetime'])
                    end += 1
                if a:
                    df = df.drop(df.index[i:end])
                    df.reset_index(inplace=True, drop=True)
                    continue
        i += 1
    return df


df2 = TimeIntervalFosen(df2)

df2.to_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen V5 kutt2018_test2.csv', index=False, sep=';')


"""

if df.at[i, 'Datetime'].year == 2020:
    if df.at[i, 'Datetime'].month == 6 and df.at[i, 'Datetime'].day == 16:  # start date = 16.06.20
        if temp == dataset:
            df = df.drop(df.index[k:i])
            df.reset_index(inplace=True, drop=True)
            temp += 1
    if df.at[i, 'Datetime'].month == 9 and df.at[i, 'Datetime'].day == 6:  # end date = 05.09.20
        end = i + 1
        while not pd.isnull(df.at[end, 'Datetime']) or end == len(df)-2:
            end += 1
        df = df.drop(df.index[i:end])
        df.reset_index(inplace=True, drop=True)
        continue
"""
