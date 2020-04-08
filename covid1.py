import numpy as np
import datetime
from datetime import datetime
import pandas as pd
import config
from sklearn.model_selection import train_test_split
from sklearn.linear_model import *
from surprise.model_selection import train_test_split
from surprise import accuracy
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import calinski_harabasz_score
from sklearn.decomposition import PCA
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

covid_data = pd.read_csv('COVID_data.csv', index_col=0)
date_cols = covid_data.Date.unique()
covid_data2 = pd.read_csv('COVID_Data_Basic.csv', index_col=0)
covid_data2.groupby('Country').sum()
covid_data.columns

#condition1 = covid_data['Confirmed'] > 0
covid_data_sum = covid_data.groupby('Country').sum()#[condition1].head()
covid_data_sum[covid_data_sum['Death'] > 0]

condition2 = covid_data_sum['Death'] > 0
condition3 = covid_data_sum['Death'] = 0
covid_deaths = covid_data_sum[condition2]
#covid_data_sum[condition3]
covid_data['Death'].value_counts()

columns = ['Country', 'Confirmed', 'Death']
covid_totals = covid_data[columns].groupby('Country').max()
covid_totals.head()

fig, ax = plt.subplots(figsize=(50, 25))
sns.scatterplot(covid_totals['Confirmed'], covid_totals['Death'], s=20)

sns.distplot(covid_totals['Death'])

fig, ax = plt.subplots(figsize=(50, 25))
sns.lineplot(covid_data['Date'], covid_data['newConfirmed'], hue=covid_data['Country'])
plt.xticks(size=25, rotation=90)
plt.yticks(size=25)
plt.xlabel('Date', size=25)
plt.ylabel('Confirmed', size=25)
plt.show()

country_rows = covid_data.Country.unique()
country_index = list(country_rows)
date_columns = list(date_cols)
date_columns
new_col_list = ['Country', '2019-12-31', '2020-01-01', '2020-01-02', '2020-01-03',
       '2020-01-04', '2020-01-05', '2020-01-06', '2020-01-07', '2020-01-08',
       '2020-01-09', '2020-01-10', '2020-01-11', '2020-01-12', '2020-01-13',
       '2020-01-14', '2020-01-15', '2020-01-16', '2020-01-17', '2020-01-18',
       '2020-01-19', '2020-01-20', '2020-01-21', '2020-01-22', '2020-01-23',
       '2020-01-24', '2020-01-25', '2020-01-26', '2020-01-27', '2020-01-28',
       '2020-01-29', '2020-01-30', '2020-01-31', '2020-02-01', '2020-02-02',
       '2020-02-03', '2020-02-04', '2020-02-05', '2020-02-06', '2020-02-07',
       '2020-02-08', '2020-02-09', '2020-02-10', '2020-02-11', '2020-02-12',
       '2020-02-13', '2020-02-14', '2020-02-15', '2020-02-16', '2020-02-17',
       '2020-02-18', '2020-02-19', '2020-02-20', '2020-02-21', '2020-02-22',
       '2020-02-23', '2020-02-24', '2020-02-25', '2020-02-26', '2020-02-27',
       '2020-02-28', '2020-02-29', '2020-03-01', '2020-03-02', '2020-03-03',
       '2020-03-04', '2020-03-05', '2020-03-06', '2020-03-07', '2020-03-08',
       '2020-03-09', '2020-03-10', '2020-03-11', '2020-03-12', '2020-03-13',
       '2020-03-14', '2020-03-15', '2020-03-16', '2020-03-17', '2020-03-18',
       '2020-03-19', '2020-03-20', '2020-03-21', '2020-03-22', '2020-03-23',
       '2020-03-24', '2020-03-25', '2020-03-26', '2020-03-27', '2020-03-28',
       '2020-03-29', '2020-03-30']
covid_df = pd.DataFrame(index=country_index, columns=date_columns)
covid_df = covid_df.reset_index()
covid_df.columns
covid_data.shape
covid_df.head()

covid_df[date_columns[0]][covid_df.index == 'Italy']
covid_df.iloc[0,0] = (covid_data['Death'][(covid_data['Country'] == 'Italy') & (covid_data['Date'] == '2020-03-08')]).values[0]
covid_df.head()
covid_data = pd.read_csv('COVID_data.csv', index_col=0)
covid_data.head(10)
date_cols = covid_data.Date.unique()
date_columns = list(date_cols)
country_rows = covid_data.Country.unique()
country_index = list(country_rows)
country_index[0]
covid_df = pd.DataFrame(index=country_index, columns=date_columns)

for i in range(len(date_columns)):
    for j in range(len(country_index)):
        condition1 = covid_data['Country'] == j
        condition2 = covid_data['Date'] == i
        covid_df.iloc[j,i] = (covid_data['Death'][(condition1) & (condition2)]).values
### this did not work!! ###
covid_df['2020-03-22'].value_counts()
