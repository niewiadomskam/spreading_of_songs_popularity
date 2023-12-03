import math
import statistics
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import colorcet as cc

from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
from code.graph_generation.initial_graphs import preprocessing

filepath = './data/Final data/all_data.csv'
df = pd.read_csv(filepath,encoding='latin-1')
df = preprocessing(df)

def calculate_bars_for_genre(df):
    res = df[df['Genre'].notna()]
    res = res.groupby(['Country','Genre']).size().reset_index(name='Counts')
    res = res[res['Counts'] > 100]
    return res

def plot_genre_violin_for_each_country(df):
    countries  = df['Country'].unique()
    res_top10 = df[df['Position'] <=10]
    df = calculate_bars_for_genre(df)
    res_top10 = calculate_bars_for_genre(res_top10)
    res_top10.rename(columns={'Counts': 'Counts_Top10'}, inplace=True)
    res_all = pd.merge(df, res_top10, on=['Country','Genre'], how='outer')
    all_genres = res_all['Genre'].unique()
    offset = 0.25
    fig, axs = plt.subplots(nrows=9, ncols=3, sharex=True, figsize=(40,15))
    for (i,c) in enumerate(countries):
        res = res_all[res_all['Country'] == c]
        j = i // 3 # row
        i = i % 3  # col
        axs[j,i].bar(res['Genre'], 
                     res['Counts'], 
                     offset)
        axs[j, i].set_title(c)
        axs[j, i].tick_params(axis='x', rotation=90)
             
    fig.tight_layout()
    plt.savefig("./diagrams/genre_analysis/genre_country.pdf", format="pdf")
    plt.show()

def plot_genre_clusters(df):
    df = df.groupby(['Country', 'Genre']).size().reset_index(name='Counts')
    pivot_df = df.pivot(index='Country', columns='Genre', values='Counts')
    pivot_df = pivot_df.fillna(0)

    linked = hierarchy.linkage(pivot_df, 'ward', metric='euclidean')
    fig = plt.figure(figsize=(20, 15))
    dendrogram = hierarchy.dendrogram(linked, orientation='top', labels=pivot_df.index, distance_sort='descending', show_leaf_counts=True)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.show()

    clustered_data = pivot_df.loc[dendrogram['ivl'], :]
    model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
    clusters = model.fit_predict(clustered_data)
    clustered_data['cluster'] = clusters

    fig.tight_layout()
    plt.xticks(rotation=90)    
    sns.scatterplot(x=clustered_data.index, y='cluster', hue='cluster', data=clustered_data, palette='Set1')
    plt.savefig("./diagrams/genre_analysis/genre_country_clusters_3.pdf", format="pdf", bbox_inches="tight")
    plt.show()

def calculate_fraction_of_genre_for_country(df):
    res = df[df['Genre'].notna()]
    res = res.groupby(['Country','Genre']).size().reset_index(name='Counts')
    groups_country = res.groupby(['Country'])['Counts']
    g_min, g_max, total_count = groups_country.transform("min"), groups_country.transform("max"), groups_country.transform('sum')
    res['Normalized'] = (res['Counts'] - g_min) / (g_max - g_min)
    res['Fraction'] = res['Counts'] / total_count
    return res

def plot_genre_all_countries_in_one(df):
    res = calculate_fraction_of_genre_for_country(df)
    res = res.sort_values(by=['Country','Fraction'], ascending=[True, False])
    res = res.groupby(['Country']).head(8).reset_index()
    fig, ax = plt.subplots()
    palette = sns.color_palette(cc.glasbey, n_colors=27)
    sns.scatterplot(x='Fraction', y='Genre', hue='Country', data=res, palette=palette)
    plt.show()

plot_genre_all_countries_in_one(df)