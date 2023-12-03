import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.cluster import hierarchy
from code.graph_generation.initial_graphs import preprocessing
from code.graph_generation.distances_in_distributions import psi_distance, kullback_leibler_distance, normalize_values

# filepath = './data/Final data/all_data.csv'
# df = pd.read_csv(filepath,encoding='latin-1')
# df = preprocessing(df)

def plot_heatmap(df, file_name, label):    
    linkage = hierarchy.linkage(df, method='ward')
    dendrogram = hierarchy.dendrogram(linkage, labels=df.index, orientation='right')
    clustered_data = df.loc[dendrogram['ivl'], dendrogram['ivl']]
    print(dendrogram['ivl'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(clustered_data, fmt='.0f', cmap='Blues', robust=True, cbar_kws={'label': label}, xticklabels=True, yticklabels=True)
    plt.savefig(file_name, format="pdf", bbox_inches="tight")
    plt.show()

def calculate_distance(df, kl):    
    distances_kullback = {}
    excluded_countries = ['China', 'Chile', 'India', 'Japan', 'Taiwan']
    df = df[~df['Country'].isin(excluded_countries)]
    countries = df['Country'].unique()
    for country_1 in countries:
        for country_2 in countries:
            if country_1 not in distances_kullback:
                distances_kullback[country_1] = {}
                if kl:
                    distances_kullback[country_1][country_2] = kullback_leibler_distance(
                df[country_1], df[country_2])
                else: 
                    distances_kullback[country_1][country_2] = psi_distance(
                df[country_1], df[country_2])
            
    data = pd.DataFrame.from_dict(distances_kullback)
    return data

def calculate_songs_that_appear_in_each_country_pair(df):
    song_country_counts = df.groupby('Song')['Country'].unique().reset_index()
    pair_song_counts = {}
    unique_countries = df['Country'].unique()
    for i in range(len(unique_countries)):
        for j in range(len(unique_countries)):
            if(i==j):
                continue
            country1 = unique_countries[i]
            country2 = unique_countries[j]
            songs_in_both_countries = set(song_country_counts[song_country_counts['Country'].apply(lambda x: (country1 in x) and (country2 in x))]['Song'])
            if country1 not in pair_song_counts:
                pair_song_counts[country1] = {}
            pair_song_counts[country1][country2] = len(songs_in_both_countries)
    data = pd.DataFrame.from_dict(pair_song_counts)
    data = data.fillna(0)
    return data

def calculate_songs_that_appear_in_every_country(df):
    excluded_countries = ['China', 'India', 'Japan']
    df = df[~df['Country'].isin(excluded_countries)]
    songs_appeared_in_all_countries = df.groupby('Song').filter(appeared_in_all_countries)
    song_count = songs_appeared_in_all_countries['Song'].nunique()
    songs_in_all_countries = songs_appeared_in_all_countries['Song'].unique()
    
    return songs_in_all_countries


def appeared_in_all_countries(group):
    return group['Country'].nunique() == df['Country'].nunique()


# data = calculate_songs_that_appear_in_each_country_pair(df)
# print(len(data))
# print(data.head())
# plot_heatmap(data, "./diagrams/summary_statistics/common_song_in_countries_2.pdf", "Number of the same songs")    


