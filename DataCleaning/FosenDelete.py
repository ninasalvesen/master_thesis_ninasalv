import pandas as pd

df2 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen V7 sinetime.csv',
                  delimiter=';', low_memory=False)

df2['Datetime'] = pd.to_datetime(df2['Datetime'], format='%Y-%m-%d %H:%M:%S')

set_delete = ['1801001638_2018', '1803009788_2018', '1803009898_2018', '1803011648_2018', '1804001948_2018', '1804002088_2018',
              '1801001638_2019', '1801002088_2019', '1801002528_2019', '1803009608_2019', '1803010868_2019', '1803011558_2019',
              '1804001198_2019', '1804001878_2019', '1804001928_2019', '1804002158_2019', '1804002168_2019', '1804002478_2019',
              '1804002928_2019', '1632006848_2019', '1804001948_2019', '1801003728_2020', '1803009888_2020', '1803010508_2020',
              '1803011758_2020', '1804001828_2020', '1804001918_2020', '1804002118_2020', '1804002158_2020', '1804002308_2020',
              '1804002628_2020', '1804002768_2020']


for i in range(len(set_delete)):
    df2 = df2[df2['uniq.log'] != set_delete[i]]

df2.reset_index(inplace=True, drop=True)

i = 0
while i < len(df2):
    k = i
    while df2.at[k, 'Data set'] == 'new_sheep':
        k += 1
    if (k-i) > 1:
        df2 = df2.drop(df2.index[i + 1:k])
        df2.reset_index(inplace=True, drop=True)
    i += 1

df2.to_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen V9 uten_feilsett.csv', index=False, sep=';')
