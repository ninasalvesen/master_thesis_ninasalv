import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import NullFormatter

sns.set_style("darkgrid")

df1 = pd.read_csv("/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Total Tingvoll med angle.csv", delimiter=';',
                 dtype={"Initial start": "str", "Start": "str", "Stop": "str"})

#df1['Datetime'] = pd.to_datetime(df1['Datetime'], format='%d/%m/%Y %H:%M')

df2 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Total Fosen med angle.csv',
                  delimiter=';')
#df2['Datetime'] = pd.to_datetime(df2['Datetime'], format='%d/%m/%Y %H:%M')


def formatter_thousands(x, pos):
    return str(round(x / 1000, 1)) + "k"


def SizeCounter(df):
    count = 0
    size = []
    i = 1
    while i < len(df):
        count += 1
        if pd.isnull(df.at[i, 'Datetime']):
            count -= 1
            size.append(count)
            count = 0
        i += 1
    size.append(count)  # adds the size of the last set
    cut_off = 0.1 * np.mean(size)
    const = [cut_off]*len(size)
    print('Cut-off value:', cut_off)
    average = sum(size) / len(size)
    print("Average data set size: ", average)
    return size, const

print('Tingvoll:')
size_tingvoll, const_tingvoll = SizeCounter(df1)
print('Fosen:')
size_fosen, const_fosen = SizeCounter(df2)

x_tingvoll = np.linspace(1, len(size_tingvoll), len(size_tingvoll))
x_fosen = np.linspace(1, len(size_fosen), len(size_fosen))

fig1, ax1 = plt.subplots(figsize=(16, 8))
sns.scatterplot(x=x_tingvoll, y=size_tingvoll, color='cornflowerblue', label='Data set size')
#sns.lineplot(x=x_tingvoll, y=const_tingvoll, linewidth=3, color='darkorange', label="Cut-off threshold")
plt.hlines(333, colors='darkorange', label="Cut-off threshold", xmin=0, xmax=85, linewidth=2)
ax1.set_xlabel('Data set number', fontsize=35, labelpad=30)
ax1.set_ylabel('Size', fontsize=35, labelpad=30)
ax1.set_title('Data set sizes for Tingvoll', fontsize=40, pad=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
ax1.set(ylim=(0, 2995))
ax1.yaxis.set_major_formatter(formatter_thousands)
ax1.yaxis.set_minor_formatter(NullFormatter())
plt.legend()
ax1.legend(loc='upper right', frameon=True, prop={'size': 20})
plt.tight_layout()
#plt.savefig("/Users/ninasalvesen/Documents/Sauedata/Bilder/Master/01 Total/sizecheck_tingvoll.png", dpi=500)

fig2, ax2 = plt.subplots(figsize=(16, 8))
sns.scatterplot(x=x_fosen, y=size_fosen, color='cornflowerblue', label='Data set size')
#sns.lineplot(x=x_fosen, y=const_fosen, linewidth=3, color='darkorange', label="Cut-off threshold")
plt.hlines(131, colors='darkorange', label="Cut-off threshold", xmin=0, xmax=315, linewidth=2)
ax2.set_xlabel('Data set number', fontsize=35, labelpad=30)
ax2.set_ylabel('Size', fontsize=35, labelpad=30)
ax2.set_title('Data set sizes for Fosen', fontsize=40, pad=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
ax2.set(ylim=(0, 2995))
ax2.yaxis.set_major_formatter(formatter_thousands)
ax2.yaxis.set_minor_formatter(NullFormatter())
plt.legend()
ax2.legend(loc='upper left', frameon=True, prop={'size': 20})
plt.tight_layout()
#plt.savefig("/Users/ninasalvesen/Documents/Sauedata/Bilder/Master/01 Total/sizecheck_fosen.png", dpi=500)

plt.show()
