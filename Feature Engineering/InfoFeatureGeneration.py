import pandas as pd

df1 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V6 sinetime.csv',
                 delimiter=';', low_memory=False)

df1['Datetime'] = pd.to_datetime(df1['Datetime'], format='%Y-%m-%d %H:%M:%S')

df2 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen V8 uten_feilsett.csv',
                 delimiter=';', low_memory=False)

df2['Datetime'] = pd.to_datetime(df2['Datetime'], format='%Y-%m-%d %H:%M:%S')

df3 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/journal_GPS_sau.csv', delimiter=';')

df4 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/NIBIO_Telespor_lamsøye.csv', delimiter=';', encoding='latin-1')

df5 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Informasjon datasett Tingvoll after cut 4.0.csv', delimiter=';')

df6 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Informasjon datasett Fosen after cut 4.0.csv', delimiter=';')


#FOSEN:
#df2['age'] = 0
#df2['n_lambs'] = 0
#df2['race'] = ''
#df2['besetning'] = 0
#df4['uniqlog'] = df4['Telespor_id'].astype(str) + '_' + df4['yr'].astype(str)
#df4['uniqlog_mor'] = df4['M_Telespor_id'].astype(str) + '_' + df4['yr'].astype(str)

#TINGVOLL
#df1['age'] = 0
#df1['n_lambs'] = 0
#df1['race'] = ''
#df1['besetning'] = 0
# besetning koksvik: 1, besetning torjul:2
#df3['datasett'] = df3['år'].astype(str) + '_' + df3['lokalitet'].astype(str) + '_' + df3['sau_id'].astype(str) + '_' + df3['id_T5H'].astype(str)


def FosenGenerate(df, df_ext, df_info):
    i = 0
    count = 0
    while i < len(df):
        if pd.isnull(df.at[i, 'Datetime']):
            i += 1
            count += 1
            print(count)
            continue
        uniqlog = df.at[i, 'uniq.log']
        indx = df_info[df_info['uniq.log'] == uniqlog].index.values[0]

        df.at[i, 'race'] = df_info.at[indx, 'rase']

        if df_info.at[indx, 'alderstatus'] == 'voksen':
            indx2 = df_ext[df_ext['uniqlog_mor'] == uniqlog].index.values[0]
            df.at[i, 'age'] = df_ext.at[indx2, 'M_age']
            df.at[i, 'n_lambs'] = df_ext.uniqlog_mor.value_counts()[uniqlog]
            indx3 = df_ext[df_ext['uniqlog_mor'] == uniqlog].index.values[0]
            df.at[i, 'besetning'] = df_ext.at[indx3, 'Owner']

        if df_info.at[indx, 'alderstatus'] == 'lam':
            indx2 = df_ext[df_ext['uniqlog'] == uniqlog].index.values[0]
            df.at[i, 'age'] = df_ext.at[indx2, 'yr'] - df_ext.at[indx2, 'Birth year']
            indx3 = df_ext[df_ext['uniqlog'] == uniqlog].index.values[0]
            df.at[i, 'besetning'] = df_ext.at[indx3, 'Owner']

        i += 1


def TingvollGenerate(df, df_ext, df_info):
    i = 0
    count = 0
    while i < len(df):
        if pd.isnull(df.at[i, 'Datetime']):
            i += 1
            count += 1
            print(count)
            continue
        uniqlog = df.at[i, 'uniq.log']

        indx = df_info[df_info['datasett'] == uniqlog].index.values[0]

        df.at[i, 'race'] = df_info.at[indx, 'rase']

        if df_info.at[indx, 'Farm'] == 'koksvik':
            df.at[i, 'besetning'] = 1
        if df_info.at[indx, 'Farm'] == 'torjul':
            df.at[i, 'besetning'] = 2

        indx2 = df_ext[df_ext['datasett'] == uniqlog].index.values[0]

        df.at[i, 'age'] = df_ext.at[indx2, 'alder']

        lambs = 0
        if not pd.isnull(df_ext.at[indx2, 'lam_1_id']):
            lambs += 1
            if not pd.isnull(df_ext.at[indx2, 'lam_2_id']):
                lambs += 1
                if not pd.isnull(df_ext.at[indx2, 'lam_3_id']):
                    lambs += 1
        df.at[i, 'n_lambs'] = lambs

        i += 1


FosenGenerate(df2, df4, df6)
df2.to_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen V11 info_features.csv', index=False, sep=';')

TingvollGenerate(df1, df3, df5)
df1.to_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V7 info_features.csv', index=False, sep=';')
