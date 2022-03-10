import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.dates import DateFormatter

sns.set_style('darkgrid')


df1 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V4 after cut.csv', delimiter=';',
                  dtype={"Initial start": "str", "Start": "str", "Stop": "str"})

df1['Datetime'] = pd.to_datetime(df1['Datetime'], format='%Y-%m-%d %H:%M:%S')

df2 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen V5 after cut.csv',
                  delimiter=';')
df2['Datetime'] = pd.to_datetime(df2['Datetime'], format='%Y-%m-%d %H:%M:%S')

print(df1.head())
print(df2.head())


def FindExtremeDates(df):
    df_copy = df.copy()  # make a copy so that the original dataframe is not altered
    df_copy['Date'] = df_copy['Datetime'].dt.date.astype('str').str[-5:]

    start = []
    end = []
    dataset = 0
    i = 0
    while i < len(df):
        if pd.isnull(df.at[i, 'Datetime']):
            dataset += 1
            if dataset == 1:
                start.append(df_copy.at[i + 1, 'Date'])
            else:
                start.append(df_copy.at[i + 1, 'Date'])
                end.append(df_copy.at[i - 1, 'Date'])
        i += 1
    end.append(df_copy.at[i - 1, 'Date'])
    #total_dates = pd.to_datetime(start + end, format='%m-%d')
    start = pd.to_datetime(start, format='%m-%d')
    end = pd.to_datetime(end, format='%m-%d')
    return start, end


start1, end1 = FindExtremeDates(df1)
start2, end2 = FindExtremeDates(df2)

# Plot of start and end dates in Tingvoll
fig1, ax1 = plt.subplots(figsize=(16, 8))
plt.hist(end1, bins=len(end1), color='pink', label='End dates')
plt.hist(start1, bins=len(start1), color='blue', label='Start dates')
ax1.set_xlabel(' ', fontsize=1)
ax1.set_ylabel('Frequency of dates', fontsize=35, labelpad=30)
ax1.set_title('Dataset start- and end dates in Tingvoll', fontsize=40, pad=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
ax1.legend(loc='upper right', frameon=False, prop={'size': 22})
plt.tight_layout()
date_form = DateFormatter("%d.%m")
ax1.xaxis.set_major_formatter(date_form)
plt.savefig("/Users/ninasalvesen/Documents/Sauedata/Bilder/Master/start_end_dates_Tingvoll_aftercut.png", dpi=500)

# Plot of start and end dates in Fosen
fig2, ax2 = plt.subplots(figsize=(16, 8))
plt.hist(end2, bins=50, color='pink', label='End dates')
#plt.hist(start2, bins=len(start2), color='blue', label='Start dates')
ax2.set_xlabel(' ', fontsize=1)
ax2.set_ylabel('Frequency of dates', fontsize=35, labelpad=30)
ax2.set_title('Dataset end dates in Fosen', fontsize=40, pad=30)
ax2.set_ylim(0, 100)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
ax2.legend(loc='upper right', frameon=False, prop={'size': 22})
plt.tight_layout()
ax2.xaxis.set_major_formatter(date_form)
plt.savefig("/Users/ninasalvesen/Documents/Sauedata/Bilder/Master/start_end_dates_Fosen_aftercut.png", dpi=500)

plt.show()
