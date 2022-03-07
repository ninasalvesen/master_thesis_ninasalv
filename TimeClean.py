import pandas as pd

df1 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V1 uten timeclean.csv', delimiter=';',
                 dtype={"Initial start": "str", "Start": "str", "Stop": "str"})

df2 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen V1 uten timeclean.csv', delimiter=';')

df1['Datetime'] = df1['Date'] + " " + df1['Time']
df1['Datetime'] = pd.to_datetime(df1['Datetime'], format='%Y-%m-%d %H:%M:%S')

df2['Datetime'] = pd.to_datetime(df2['Datetime'], format='%d/%m/%Y %H:%M')
print(df1.head())
print(df2.head())


def TimeClean(df, delta):
    month_end = [30, 31]
    count = 0
    temp = 1     # data set number, skips the first row so must be initialized at 1
    i = 1        # have to check the first and two last lines for faulty points manually
    while i < (len(df)-2):
        if (i % 10000) == 0:
            # check progress against number of iterations
            print("Reached number: ", i)

        if pd.isnull(df.at[i + 1, 'Datetime']):
            temp += 1
            i += 2  # start the next check on the next data set
            continue

        year_diff = df.at[i+1, 'Datetime'].year - df.at[i, 'Datetime'].year
        month_diff = df.at[i+1, 'Datetime'].month - df.at[i, 'Datetime'].month
        day_diff = df.at[i+1, 'Datetime'].day - df.at[i, 'Datetime'].day

        if year_diff != 0:
            count += 1
            df.at[i + 1, 'Datetime'] = df.at[i + 1, 'Datetime'].replace(year=df.at[i, 'Datetime'].year)
            continue  # check again that it works before moving on

        if abs(month_diff) > 1:
            count += 1
            df.at[i + 1, 'Datetime'] = df.at[i + 1, 'Datetime'].replace(month=df.at[i, 'Datetime'].month)
            continue

        if month_diff == 1 and df['Datetime'][i].day not in month_end:
            if df['Datetime'][i+2].month == df['Datetime'][i].month:
                count += 1
                df.at[i + 1, 'Datetime'] = df.at[i + 1, 'Datetime'].replace(month=df.at[i, 'Datetime'].month)
            else:
                i += 1
            continue

        if month_diff == -1:
            count += 1
            df.at[i + 1, 'Datetime'] = df.at[i + 1, 'Datetime'].replace(month=df.at[i, 'Datetime'].month)
            continue

        if abs(day_diff) > 1 and df['Datetime'][i].day not in month_end:
            if df.at[i+2, 'Datetime'].day == df.at[i, 'Datetime'].day:
                count += 1
                df.at[i + 1, 'Datetime'] = df.at[i + 1, 'Datetime'].replace(day=df.at[i, 'Datetime'].day)
            else:
                i += 1
            continue

        if day_diff == 1 and df['Datetime'][i].time().hour < 22:
            if df['Datetime'][i + 2].day == df['Datetime'][i].day:
                count += 1
                df.at[i + 1, 'Datetime'] = df.at[i + 1, 'Datetime'].replace(day=df.at[i, 'Datetime'].day)
            else:
                i += 1
            continue

        if day_diff < 0 and df['Datetime'][i].day not in month_end:
            count += 1
            df.at[i + 1, 'Datetime'] = df.at[i + 1, 'Datetime'].replace(day=df.at[i, 'Datetime'].day)
            continue

        k = i
        while ((df.at[k + 1, 'Datetime'] - df.at[i, 'Datetime']).total_seconds() / (60 * 60)) < 0:
            k += 1
        if k - i > 1:
            df = df.drop(df.index[i + 1:k])
            df.reset_index(inplace=True, drop=True)
        elif k - i == 1:
            if not pd.isnull(df.at[i + 2, 'Datetime']):
                count += 1
                mid_diff = (df.at[i + 2, 'Datetime'] - df.at[i, 'Datetime']).total_seconds() / 2
                df.at[i + 1, 'Datetime'] = df.at[i, 'Datetime'] + pd.Timedelta(hours=mid_diff / (60 * 60))

        # delta enables the user to choose the time difference limit in hours
        if ((df.at[i + 1, 'Datetime'] - df.at[i, 'Datetime']).total_seconds() / (60*60)) > delta:
            if pd.isnull(df.at[i + 2, 'Datetime']) == False \
                    and ((df.at[i + 2, 'Datetime'] - df.at[i, 'Datetime']).total_seconds() / (60*60)) < delta:
                count += 1
                mid_diff = (df.at[i + 2, 'Datetime'] - df.at[i, 'Datetime']).total_seconds() / 2
                df.at[i + 1, 'Datetime'] = df.at[i, 'Datetime'] + pd.Timedelta(hours=mid_diff/(60*60))

        i += 1
    return df, count


def EqualTimeError(df):
    i = 0
    while i < len(df):
        if pd.isnull(df1.at[i, 'Datetime']):
            i += 2
            continue

        time = ((df.at[i, 'Datetime'] - df.at[i - 1, 'Datetime']).total_seconds() / (60 * 60))  # in hour(s)
        if time == 0:
            df = df.drop(df.index[i])
            df.reset_index(inplace=True, drop=True)
            continue
        i += 1
    return df


df1, count1 = TimeClean(df1, 2)
print(count1)
df1 = EqualTimeError(df1)
df1 = df1.drop(columns=['Date', 'Time'])  # deleting the (uncleaned) columns that we no longer need
df1.to_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V2 med timeclean.csv', index=False, sep=';')

df2, count2 = TimeClean(df2, 4)
print(count2)
df2 = df2.drop(columns=['Date.Time'])
df2.to_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen V2 uten pointclean.csv', index=False, sep=';')
