import numpy as np
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
covid_data.columns
covid_data2 = pd.read_csv('COVID_Data_Basic.csv', index_col=0)
covid_data2.groupby('Country').sum()

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
