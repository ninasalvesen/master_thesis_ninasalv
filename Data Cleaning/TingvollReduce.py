import pandas as pd
#from FeatureEngineering import Haversine

df1 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V7 info_features.csv',
                 delimiter=';', low_memory=False)

df1['Datetime'] = pd.to_datetime(df1['Datetime'], format='%Y-%m-%d %H:%M:%S')

print(df1)


def TimeReduce(df):
    i = 1  # start at first line
    df['less_than_hour'] = 'no'
    while i < (len(df)-1):
        if (i % 10000) == 0:
            # check progress against number of iterations
            print("Reached number: ", i)

        if pd.isnull(df.at[i + 1, 'Datetime']):
            i += 2  # start the next check on the next data set
            continue

        hour_diff = (df.at[i + 1, 'Datetime'] - df.at[i, 'Datetime']).total_seconds() / (60*60)
        if hour_diff <= 1 and df.at[i, 'less_than_hour'] == 'no':
            df.at[i + 1, 'less_than_hour'] = 'yes'

        i += 1
    df = df[df.less_than_hour != 'yes']
    df = df.drop(columns=['less_than_hour'])
    df.reset_index(inplace=True, drop=True)
    return df


df_reduced = TimeReduce(df1)

# update the Haversine
df_reduced['Haversine'] = 0
#df_reduced = Haversine.insert_speed(df_reduced)
#df_reduced = Haversine.dist_check(df_reduced, 15000)
#df_reduced = Haversine.insert_speed(df_reduced)

df_reduced.to_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V8 reduced.csv', index=False, sep=';')
