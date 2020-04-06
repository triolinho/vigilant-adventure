import numpy as np
import pandas as pd
import config
from sklearn.model_selection import train_test_split
from sklearn.linear_model import *
from surprise.model_selection import train_test_split
from surprise import accuracy
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
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
