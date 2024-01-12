import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.cluster import hierarchy
from code.graph_generation.correlation import calculate_posistion_from_release_to_occurance
from code.graph_generation.initial_graphs import preprocessing

def kullback_leibler_distance(distribution1, distribution2):
    return sum((x * math.log(x / y)) if (abs(y) > 0.000005 and abs(x) > 0.000005) else 1 for x, y in zip(distribution1, distribution2))

def psi_distance(distribution1, distribution2):
    return sum((x-y) * math.log(x / y) for x, y in zip(distribution1, distribution2))

def calculate_kullback_leibler_distance(df):    
    # distributions_normalized = calculate_normalized_distributions(df)
    date_diff_release_occ = calculate_posistion_from_release_to_occurance(df)
    diff_normalized = normalize_values(date_diff_release_occ, 'Days_Difference')
    distances_kullback = {}
    excluded_countries = ['China', 'Chile', 'India', 'Japan', 'Taiwan']
    df = df[~df['Country'].isin(excluded_countries)]
    countries = df['Country'].unique()
    for country_1 in countries:
        for country_2 in countries:
            if country_1 not in distances_kullback:
                distances_kullback[country_1] = {}
            distances_kullback[country_1][country_2] = kullback_leibler_distance(
                diff_normalized[country_1], diff_normalized[country_2])
            
    data = pd.DataFrame.from_dict(distances_kullback)
    linkage = hierarchy.linkage(data, method='ward')
    dendrogram = hierarchy.dendrogram(linkage, labels=data.index, orientation='right')
    clustered_data = data.loc[dendrogram['ivl'], dendrogram['ivl']]

    plt.figure(figsize=(10, 8))
    sns.heatmap(clustered_data, fmt='.0f', cmap='Blues_r', robust=True, cbar_kws={'label': 'KL distance'}, xticklabels=True, yticklabels=True)
    plt.savefig("./diagrams/position_change_heatmaps/distances_distributions_release_occurance_kl.pdf", format="pdf", bbox_inches="tight")
    plt.show()

def calculate_normalized_distributions(df):
    excluded_countries = ['China', 'Chile', 'India', 'Japan', 'Taiwan']
    distributions_normalized = {}
    df = df[~df['Country'].isin(excluded_countries)]
    countries = df['Country'].unique()
    for (idx, c) in enumerate(countries):
        result = df[df['Country'] == c]
        result.sort_values(by=['Song title', 'Song author', 'Date'], inplace=True, ascending=[False, False, False])
        result['Position_change'] = result.groupby(['Song title', 'Song author'])['Position'].diff().fillna(0).astype(int)
        year_subset = result['Position_change']
        count_changes = year_subset.value_counts().sort_index()
        distributions_normalized[c] = count_changes / count_changes.sum()

    return distributions_normalized

def normalize_values(df, column_name):
    distributions_normalized = {}
    countries = df['Country'].unique()
    for (idx, c) in enumerate(countries):
        result = df[df['Country'] == c]
        year_subset = result[column_name]
        count_changes = year_subset.value_counts().sort_index()
        distributions_normalized[c] = count_changes / count_changes.sum()

    return distributions_normalized

# filepath = './data/Final data/all_data.csv'
# df = pd.read_csv(filepath,encoding='latin-1')
# df = preprocessing(df)
# calculate_kullback_leibler_distance(df)

