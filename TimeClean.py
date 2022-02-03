import pandas as pd

df = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll.csv', delimiter=';',
                 dtype={"Initial start": "str", "Start": "str", "Stop": "str"})

df['Datetime'] = df['Date'] + " " + df['Time']
df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y %m %d %H:%M:%S')
print(df.head())
#df['Datetime'].replace({pd.NaT: 'new_sheep'}, inplace=True)
print(df['Datetime'][20].day-df['Datetime'][21].day)

month_end = [30, 31]
temp = 1  # telle hvilket datasett det er
count = 0
check = []
i = 1
while i < (len(df)-1):
    if pd.isnull(df.loc[i + 1, 'Datetime']):
        temp += 1
        i += 2  # start the next check on the next data set
        continue

    year_diff = df.loc[i+1, 'Datetime'].year - df.loc[i, 'Datetime'].year
    month_diff = df.loc[i+1, 'Datetime'].month - df.loc[i, 'Datetime'].month
    day_diff = (df['Datetime'][i+1] - df['Datetime'][i]).components[0]
    #hour_diff = (df['Datetime'][i+1]-df['Datetime'][i]).components[1]
    #minute_diff = (df['Datetime'][i+1]-df['Datetime'][i]).components[2]

    # sjekke om dato/tidspunkt er helt like?

    if year_diff != 0:
        df.loc[i + 1, 'Datetime'] = df.loc[i + 1, 'Datetime'].replace(year=df.loc[i, 'Datetime'].year)
        continue  # check again that it works before moving on

    if abs(month_diff) > 1:
        df.loc[i + 1, 'Datetime'] = df.loc[i + 1, 'Datetime'].replace(month=df.loc[i, 'Datetime'].month)
        continue

    if month_diff == 1 and df['Datetime'][i].day not in month_end:
        df.loc[i + 1, 'Datetime'] = df.loc[i + 1, 'Datetime'].replace(month=df.loc[i, 'Datetime'].month)
        continue

    if month_diff == -1:
        df.loc[i + 1, 'Datetime'] = df.loc[i + 1, 'Datetime'].replace(month=df.loc[i, 'Datetime'].month)
        continue

    if abs(day_diff) > 1:
        df.loc[i + 1, 'Datetime'] = df.loc[i + 1, 'Datetime'].replace(day=df.loc[i, 'Datetime'].day)
        continue

    if day_diff == 1 and df['Datetime'][i].time().hour != 23:
        df.loc[i + 1, 'Datetime'] = df.loc[i + 1, 'Datetime'].replace(day=df.loc[i, 'Datetime'].day)
        continue

    if day_diff == -1:
        df.loc[i + 1, 'Datetime'] = df.loc[i + 1, 'Datetime'].replace(day=df.loc[i, 'Datetime'].day)
        continue

    if (i % 10000) == 0:
        # check progress against number of iterations
        print("Reached number: ", i)

    i += 1

print(check)
print(count)

# plotte datoene for å se om det stemmer?

#df = df.drop(columns=['Date', 'Time'])  # deleting the uncleaned columns that we no longer need
#print(df.head())
#df.to_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll dato.csv', index=False, sep=';')