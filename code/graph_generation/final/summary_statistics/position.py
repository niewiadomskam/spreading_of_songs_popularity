import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.cluster import hierarchy
from code.graph_generation.distances_in_distributions import kullback_leibler_distance, normalize_values
from code.graph_generation.initial_graphs import preprocessing

filepath = './data/Final data/all_data.csv'
df = pd.read_csv(filepath,encoding='latin-1')
df = preprocessing(df)

def calculate_position_changes(df):
    position_changes = {}
    countries = df['Country'].unique()
    for (i, c) in enumerate(countries):
        result = df[df['Country'] == c] 
        result = result.sort_values(by=['Song title', 'Song author', 'Date'], inplace=False, ascending=[False, False, False])
        result['Position_change'] = result.groupby(['Song title', 'Song author'])['Position'].diff().fillna(0).astype(int)
        year_subset = result['Position_change']
        count_changes = year_subset.value_counts().sort_index()
        position_changes[c] = (count_changes / count_changes.sum()).to_dict()
    return position_changes

def calculate_kullback_leibler_distance(df):    
    distances_kullback = {}
    excluded_countries = ['China', 'Chile', 'India', 'Japan', 'Taiwan']
    df = df[~df['Country'].isin(excluded_countries)]
    countries = df['Country'].unique()
    diff_normalized = calculate_position_changes(df)
    for country_1 in countries:
        for country_2 in countries:
            dist1, dist2 = align_distributions(diff_normalized[country_1], diff_normalized[country_2])
            if country_1 not in distances_kullback:
                distances_kullback[country_1] = {}
            distances_kullback[country_1][country_2] = kullback_leibler_distance(
                dist1, dist2)
            
    data = pd.DataFrame.from_dict(distances_kullback)
    linkage = hierarchy.linkage(data, method='ward')
    dendrogram = hierarchy.dendrogram(linkage, labels=data.index, orientation='right')
    clustered_data = data.loc[dendrogram['ivl'], dendrogram['ivl']]

    plt.figure(figsize=(10, 8))
    sns.heatmap(clustered_data, fmt='.0f', cmap='Blues_r', robust=True, cbar_kws={'label': 'PSI distance'}, xticklabels=True, yticklabels=True)
    plt.savefig("./diagrams/final/heatmap_position_change.pdf", format="pdf", bbox_inches="tight")
    plt.show()

def align_distributions(distribution1, distribution2):
    # Identify the common set of events
    common_events = set(distribution1.keys()).union(set(distribution2.keys()))

    # Align the distributions by padding with zeros for missing events
    aligned_distribution1 = [distribution1.get(event, 0) for event in common_events]
    aligned_distribution2 = [distribution2.get(event, 0) for event in common_events]

    return np.array(aligned_distribution1), np.array(aligned_distribution2)

calculate_kullback_leibler_distance(df)