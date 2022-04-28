import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_style('darkgrid')

df1 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V6 sinetime.csv',
                 delimiter=';')

df1['Datetime'] = pd.to_datetime(df1['Datetime'], format='%Y-%m-%d %H:%M:%S')

df2 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen V7 sinetime.csv',
                  delimiter=';')

df2['Datetime'] = pd.to_datetime(df2['Datetime'], format='%Y-%m-%d %H:%M:%S')


def TrigTime(df):
    # find minutes that has passed since midnight for each point
    i = 0
    df['minutes'] = 0
    while i < len(df):
        if pd.isnull(df.at[i, 'Datetime']):
            i += 1
            continue

        df.at[i, 'minutes'] = (df.at[i, 'Datetime'] - df.at[i, 'Datetime'].replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / (60)
        i += 1

    # transform into sine and cosine values
    minutes_in_day = 24 * 60
    df['sin_time'] = np.sin(2 * np.pi * df.minutes / minutes_in_day)
    df['cos_time'] = np.cos(2 * np.pi * df.minutes / minutes_in_day)
    return df


#df1 = TrigTime(df1)
#df2 = TrigTime(df2)

#df1.to_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V6 sinetime.csv', index=False, sep=';')
#df2.to_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen V7 sinetime.csv', index=False, sep=';')


fig1, ax1 = plt.subplots(figsize=(16, 8))
df1.iloc[1:50].plot(y='sin_time', ax=ax1)
df1.iloc[1:50].plot(y='cos_time', ax=ax1)
ax1.set_xlabel('Row', fontsize=30, labelpad=20)
ax1.set_ylabel('Trig. value', fontsize=30, labelpad=20)
ax1.set_title('Diurnal value in trigonometric representation', fontsize=40, pad=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend()
ax1.legend(loc='lower left', frameon=True, prop={'size': 20})
plt.tight_layout()
#plt.savefig("/Users/ninasalvesen/Documents/Sauedata/Bilder/Master/after cut 4.0/diurnal_values_sine.png", dpi=500)

fig2, ax2 = plt.subplots(figsize=(8, 8))
df1.sample(100).plot.scatter('sin_time', 'cos_time', ax=ax2).set_aspect('equal')
ax2.set_xlabel('sin_time', fontsize=20, labelpad=20)
ax2.set_ylabel('cos_time', fontsize=20, labelpad=20)
ax2.set_title('Sin and cos time plotted against each other', fontsize=22, pad=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
#plt.savefig("/Users/ninasalvesen/Documents/Sauedata/Bilder/Master/after cut 4.0/diurnal_values_circle.png", dpi=500)


fig3, ax3 = plt.subplots(figsize=(16, 8))
df1.iloc[1:100].plot(y='minutes', ax=ax3)
ax3.set_xlabel('Row', fontsize=30, labelpad=20)
ax3.set_ylabel('Linear value', fontsize=30, labelpad=20)
ax3.set_title('Diurnal value in linear representation', fontsize=40, pad=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend()
ax3.legend(loc='upper left', frameon=True, prop={'size': 20})
plt.tight_layout()
#plt.savefig("/Users/ninasalvesen/Documents/Sauedata/Bilder/Master/after cut 4.0/diurnal_values_linear.png", dpi=500)

plt.show()

