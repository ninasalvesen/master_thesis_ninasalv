import pandas as pd

df1 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V3 med Haversine.csv', delimiter=';',
                 dtype={"Initial start": "str", "Start": "str", "Stop": "str"})

df1['Datetime'] = pd.to_datetime(df1['Datetime'], format='%Y-%m-%d %H:%M:%S')

df2 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen V4 med Haversine.csv',
                  delimiter=';')
df2['Datetime'] = pd.to_datetime(df2['Datetime'], format='%Y-%m-%d %H:%M:%S')

print(df1)
print(df2)


def TimeIntervalFosen(df):
    dataset = 1
    k = 1  # initialize
    temp = dataset  # initialize
    i = 1  # start at first line of data
    a = True  # initialize
    while i < len(df):
        print('hei')
        if pd.isnull(df.at[i, 'Datetime']):
            print(dataset)  # keep track of progress
            k = i + 1  # k marks the beginning of each dataset and stores the row of the first line of data
            dataset += 1
            temp = dataset
            i += 1
            a = True

        if df.at[i, 'Datetime'].year == 2018:
            if df.at[i, 'Datetime'].month == 6 and df.at[i, 'Datetime'].day == 9:  # start date = 09.06.18
                if temp == dataset:  # fail safe as to not delete every point that includes 09.06
                    df = df.drop(df.index[k:i])  # deletes rows k-i including k but not including i
                    df.reset_index(inplace=True, drop=True)
                    i = k  # puts i back to be the next row, after i-k lines were deleted
                    temp += 1  # changes temp here so that it won't go into the if and delete every point of 09.06
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
                    i = k
                    temp += 1
            if df.at[i, 'Datetime'].month == 7 and df.at[i, 'Datetime'].day == 4:  # end date = 03.07.19 or 31.08.19
                end = i + 1
                while not pd.isnull(df.at[end, 'Datetime']):
                    #  depending on how large each dataset is, they may or may not reach 31.08
                    if df.at[end, 'Datetime'].month == 9 and df.at[end, 'Datetime'].day == 1:
                        a = False  # makes sure you don't double delete lines in the same set by making it here not go
                        # into the if below, should it go into the if here by reaching the date 31.08
                        i = end  # set i to be after the last valid value of the set (31.08), that is 01.09 00:00
                        while not pd.isnull(df.at[end, 'Datetime']):
                            end += 1
                        df = df.drop(df.index[i:end])
                        df.reset_index(inplace=True, drop=True)
                        end = i  # makes the loop exit the (while not)-condition above after continue
                        i -= 1
                        continue
                    end += 1
                if a:
                    df = df.drop(df.index[i:end])
                    df.reset_index(inplace=True, drop=True)
                    continue

        if df.at[i, 'Datetime'].year == 2020:
            if df.at[i, 'Datetime'].month == 6 and df.at[i, 'Datetime'].day == 16:  # start date = 16.06.18
                if temp == dataset:
                    df = df.drop(df.index[k:i])
                    df.reset_index(inplace=True, drop=True)
                    i = k
                    temp += 1
            if df.at[i, 'Datetime'].month == 9 and df.at[i, 'Datetime'].day == 6:  # end date = 05.09.18
                end = i + 1
                while not pd.isnull(df.at[end, 'Datetime']) and end != len(df)-1:
                    end += 1
                df = df.drop(df.index[i:end])
                df.reset_index(inplace=True, drop=True)
                continue

        i += 1
    print(dataset)
    df = df.drop(df.index[-1])
    return df


def TimeIntervalTingvoll(df):
    dataset = 1
    k = 1  # initialize
    temp = dataset  # initialize
    i = 1  # start at first line of data
    while i < len(df):
        if pd.isnull(df.at[i, 'Datetime']):
            print(dataset)  # keep track of progress
            k = i + 1  # k marks the beginning of each dataset and stores the row of the first line of data
            dataset += 1
            temp = dataset
            i += 1

        if df.at[i, 'Datetime'].year == 2012:
            if df.at[i, 'Datetime'].month == 6 and df.at[i, 'Datetime'].day == 10:  # start date = 10.06.12
                if temp == dataset:
                    df = df.drop(df.index[k:i])
                    df.reset_index(inplace=True, drop=True)
                    i = k
                    temp += 1
            if df.at[i, 'Datetime'].month == 9 and df.at[i, 'Datetime'].day == 8:  # end date = 07.09.12
                end = i + 1
                while not pd.isnull(df.at[end, 'Datetime']):
                    end += 1
                df = df.drop(df.index[i:end])
                df.reset_index(inplace=True, drop=True)
                continue

        if df.at[i, 'Datetime'].year == 2013:
            if df.at[i, 'Farm'] == 'koksvik':
                if df.at[i, 'Datetime'].month == 6 and df.at[i, 'Datetime'].day == 23:  # start date koksvik = 23.06.13
                    if temp == dataset:
                        df = df.drop(df.index[k:i])
                        df.reset_index(inplace=True, drop=True)
                        i = k
                        temp += 1
            if df.at[i, 'Farm'] == 'torjul':
                if df.at[i, 'Datetime'].month == 6 and df.at[i, 'Datetime'].day == 16:  # start date torjul = 16.06.13
                    if temp == dataset:
                        df = df.drop(df.index[k:i])
                        df.reset_index(inplace=True, drop=True)
                        i = k
                        temp += 1
            if df.at[i, 'Datetime'].month == 8 and df.at[i, 'Datetime'].day == 26:  # end date = 25.08.13
                end = i + 1
                while not pd.isnull(df.at[end, 'Datetime']):
                    end += 1
                df = df.drop(df.index[i:end])
                df.reset_index(inplace=True, drop=True)
                continue

        if df.at[i, 'Datetime'].year == 2014:
            if df.at[i, 'Farm'] == 'koksvik':
                if df.at[i, 'Datetime'].month == 6 and df.at[i, 'Datetime'].day == 13:  # start date koksvik = 13.06.14
                    if temp == dataset:
                        df = df.drop(df.index[k:i])
                        df.reset_index(inplace=True, drop=True)
                        i = k
                        temp += 1
            if df.at[i, 'Farm'] == 'torjul':
                if df.at[i, 'Datetime'].month == 6 and df.at[i, 'Datetime'].day == 25:  # start date torjul = 25.06.14
                    if temp == dataset:
                        df = df.drop(df.index[k:i])
                        df.reset_index(inplace=True, drop=True)
                        i = k
                        temp += 1
            if df.at[i, 'Datetime'].month == 9 and df.at[i, 'Datetime'].day == 11:  # end date = 10.09.14
                end = i + 1
                while not pd.isnull(df.at[end, 'Datetime']):
                    end += 1
                df = df.drop(df.index[i:end])
                df.reset_index(inplace=True, drop=True)
                continue

        if df.at[i, 'Datetime'].year == 2015:
            if df.at[i, 'Farm'] == 'koksvik':
                if df.at[i, 'Datetime'].month == 6 and df.at[i, 'Datetime'].day == 13:  # start date koksvik = 13.06.15
                    if temp == dataset:
                        df = df.drop(df.index[k:i])
                        df.reset_index(inplace=True, drop=True)
                        i = k
                        temp += 1
            if df.at[i, 'Farm'] == 'torjul':
                if df.at[i, 'Datetime'].month == 7 and df.at[i, 'Datetime'].day == 3:  # start date torjul = 03.07.15
                    if temp == dataset:
                        df = df.drop(df.index[k:i])
                        df.reset_index(inplace=True, drop=True)
                        i = k
                        temp += 1
            if df.at[i, 'Datetime'].month == 9 and df.at[i, 'Datetime'].day == 7:  # end date = 06.09.15
                end = i + 1
                while not pd.isnull(df.at[end, 'Datetime']):
                    end += 1
                df = df.drop(df.index[i:end])
                df.reset_index(inplace=True, drop=True)
                continue

        if df.at[i, 'Datetime'].year == 2016:
            if df.at[i, 'Datetime'].month == 6 and df.at[i, 'Datetime'].day == 18:  # start date = 18.06.16
                if temp == dataset:
                    df = df.drop(df.index[k:i])
                    df.reset_index(inplace=True, drop=True)
                    i = k
                    temp += 1
            if df.at[i, 'Datetime'].month == 7 and df.at[i, 'Datetime'].day == 21:  # end date = 20.07.16
                end = i + 1
                while not pd.isnull(df.at[end, 'Datetime']) and end != len(df) - 1:
                    end += 1
                df = df.drop(df.index[i:end])
                df.reset_index(inplace=True, drop=True)
                continue

        i += 1
    print(dataset, df.at[i-1, 'Farm'])
    df = df.drop(df.index[-1])
    return df


df1 = TimeIntervalTingvoll(df1)
df1.to_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V4 after cut.csv', index=False, sep=';')

df2 = TimeIntervalFosen(df2)
df2.to_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen V5 after cut.csv', index=False, sep=';')
