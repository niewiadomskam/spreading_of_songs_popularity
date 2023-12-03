import random
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.subplots as sp
import plotly.graph_objects as go
import seaborn as sns

from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
from plotly.subplots import make_subplots
from .initial_graphs import preprocessing
from ..data_scraping.get_artist_metadata import get_release_date
# 
# filepath = './data/Final data/all_data.csv'
# df = pd.read_csv(filepath,encoding='latin-1')


def calculate_first_occurance_of_song(df):
    songs_first_occurance = df.groupby('Song')['Date'].min().reset_index()
    songs_first_occurance['Week_number'] = songs_first_occurance['Date'].dt.strftime('%U').astype(int)
    songs_first_occurance['Year'] = songs_first_occurance['Date'].dt.year
    return songs_first_occurance

def calculate_first_occurance_of_song_in_country(df):
    songs_first_occurance_in_country = df.groupby(['Song', 'Country'])['Date'].min().reset_index()
    songs_first_occurance_in_country['Week_number'] = songs_first_occurance_in_country['Date'].dt.strftime('%U').astype(int)
    songs_first_occurance_in_country['Year'] = songs_first_occurance_in_country['Date'].dt.year
    return songs_first_occurance_in_country

def calculate_last_occurance_of_song_in_country(df):
    songs_last_occurance_in_country = df.groupby(['Song', 'Country'])['Date'].max().reset_index()
    songs_last_occurance_in_country['Week_number'] = songs_last_occurance_in_country['Date'].dt.strftime('%U').astype(int)
    songs_last_occurance_in_country['Year'] = songs_last_occurance_in_country['Date'].dt.year
    return songs_last_occurance_in_country

def calculate_peak_of_song_in_country(df):
    min_positions_idx = df.groupby(['Song', 'Country'])['Position'].idxmin()
    songs_peak_in_country = df.loc[min_positions_idx, ['Song', 'Country', 'Date', 'Position']]
    return songs_peak_in_country

def calculate_posistion_from_release_to_occurance(df):
    songs_first_occurance = calculate_first_occurance_of_song(df)
    songs_first_occurance_in_country = calculate_first_occurance_of_song_in_country(df)

    result_df = songs_first_occurance_in_country.merge(songs_first_occurance, on=['Song'], how='left', suffixes=('_Country', '_World'))
    result_df['Days_Difference'] = (result_df['Date_Country']-result_df['Date_World']).dt.days
    differences_result_df = []  
    unique_songs = df['Song'].unique()
    for song in unique_songs:
        song_df = result_df[result_df['Song'] == song]
        for i in range(len(song_df)):
            for j in range(i+1, len(song_df)):
                country_1 = song_df.iloc[i]['Country']
                country_2 = song_df.iloc[j]['Country']
                diff_in_days = song_df.iloc[i]['Days_Difference'] - song_df.iloc[j]['Days_Difference']
                
                # Add the calculated values to the result DataFrame
                differences_result_df.append({
                    'Country_1': country_1,
                    'Country_2': country_2,
                    'Song': song,
                    'Difference_in_Days': diff_in_days
                })

    differences_result_df = pd.DataFrame(differences_result_df, columns=['Country_1', 'Country_2', 'Song', 'Difference_in_Days'])
    differences_result_df.to_csv("differences_in_release_occurance2.csv")

def plot_diagram_for_countires_difference():
    excluded_countries = ['China', 'Chile', 'India', 'Japan', 'Taiwan']
    filepath = "differences_in_peak.csv"
    df = pd.read_csv(filepath)
    df = df[(~df['Country_1'].isin(excluded_countries)) & (~df['Country_2'].isin(excluded_countries))]
    countries_1 = df['Country_1'].unique()
    countries_2 = df['Country_2'].unique()
    countries = np.hstack((countries_1, countries_2))
    countries = list(set(countries))
    countries.sort()
    fig, axs = plt.subplots(nrows=22, ncols=22, sharex=True, sharey=True, figsize=(60,60))
    for (i, country_1) in enumerate(countries):
        for j in range(0, len(countries)):
            if i== j:
                continue
            country_2 = countries[j]
            countries_subset = df[(df['Country_1'] == country_1) & (df['Country_2'] == country_2)]['Difference_in_Days']
            if len(countries_subset) == 0:
                countries_subset = df[(df['Country_1'] == country_2) & (df['Country_2'] == country_1)]['Difference_in_Days']
                countries_subset = countries_subset * -1
            countries_subset = countries_subset[(countries_subset <=60) & (countries_subset >= -60)]
            color = "blue"
            if countries_subset.median() < 0 :
                color = "red"   
            if countries_subset.median() == 0 :
                color = "gray"          
            axs[j, i].hist(x=countries_subset.values, bins=range(-63, 70, 7), color=color, density=True) 
            axs[j, i].set_ylabel("Count")
            axs[j, i].set_title(f"{country_1} - {country_2}")
            
    fig.tight_layout()
    plt.savefig("./diagrams/time_differences_statistics/normalized_distribution_difference_in_peak.pdf", format="pdf")
    plt.show()


def calculate_posistion_from_occurance_to_peak(df):
    songs_first_occurance = calculate_first_occurance_of_song_in_country(df)
    songs_peak = calculate_peak_of_song_in_country(df)
    
    result_df = songs_first_occurance.merge(songs_peak, on=['Song', 'Country'], how='left', suffixes=('_Peak', '_Enter'))
    result_df['Days_Difference'] = (result_df['Date_Peak']-result_df['Date_Enter']).dt.days
    differences_result_df = []  
    unique_songs = df['Song'].unique()
    for song in unique_songs:
        song_df = result_df[result_df['Song'] == song]
        for i in range(len(song_df)):
            for j in range(i+1, len(song_df)):
                country_1 = song_df.iloc[i]['Country']
                country_2 = song_df.iloc[j]['Country']
                diff_in_days = song_df.iloc[i]['Days_Difference'] - song_df.iloc[j]['Days_Difference']
                
                # Add the calculated values to the result DataFrame
                differences_result_df.append({
                    'Country_1': country_1,
                    'Country_2': country_2,
                    'Song': song,
                    'Difference_in_Days': diff_in_days
                })

    differences_result_df = pd.DataFrame(differences_result_df, columns=['Country_1', 'Country_2', 'Song', 'Difference_in_Days'])
    differences_result_df.to_csv("differences_in_release_peak.csv")

def calculate_posistion_peak(df):
    df = df.sort_values(by=['Country','Song','Date'], ascending=[True, True, True])
    songs_peak = calculate_peak_of_song_in_country(df)
    differences_result_df = []  
    unique_songs = songs_peak['Song'].unique()
    for song in unique_songs:
        song_df = songs_peak[songs_peak['Song'] == song]
        for i in range(len(song_df)):
            for j in range(i+1, len(song_df)):
                country_1 = song_df.iloc[i]['Country']
                country_2 = song_df.iloc[j]['Country']
                diff_in_days = (song_df.iloc[i]['Date'] - song_df.iloc[j]['Date']).days
                
                # Add the calculated values to the result DataFrame
                differences_result_df.append({
                    'Country_1': country_1,
                    'Country_2': country_2,
                    'Song': song,
                    'Difference_in_Days': diff_in_days
                })

    differences_result_df = pd.DataFrame(differences_result_df, columns=['Country_1', 'Country_2', 'Song', 'Difference_in_Days'])
    differences_result_df.to_csv("differences_in_peak.csv")


def count_positive_negative_medians():
    excluded_countries = ['China', 'Chile', 'India', 'Japan', 'Taiwan']
    filepath = "differences_in_peak.csv"
    df = pd.read_csv(filepath)
    df = df[(~df['Country_1'].isin(excluded_countries)) & (~df['Country_2'].isin(excluded_countries))]
    countries_1 = df['Country_1'].unique()
    countries_2 = df['Country_2'].unique()
    countries = np.hstack((countries_1, countries_2))
    countries = list(set(countries))
    countries.sort()
    medians = {}
    for (i, country_1) in enumerate(countries):
        for j in range(0, len(countries)):
            if i== j:
                continue
            country_2 = countries[j]
            countries_subset = df[(df['Country_1'] == country_1) & (df['Country_2'] == country_2)]['Difference_in_Days']
            if len(countries_subset) == 0:
                countries_subset = df[(df['Country_1'] == country_2) & (df['Country_2'] == country_1)]['Difference_in_Days']
                countries_subset = countries_subset * -1
            countries_subset = countries_subset[(countries_subset <=60) & (countries_subset >= -60)]
            if country_1 not in medians:
                medians[country_1] = []          
            medians[country_1].append(countries_subset.median()) 

    data_list = [(key, value) for key, values in medians.items() for value in values]
    country_medians = pd.DataFrame(data_list, columns=["Country", "Medians"])
    result  = country_medians.groupby('Country')['Medians'].agg(positive_count=lambda x: (x > 0).sum(),
                                              negative_count=lambda x: (x < 0).sum()).reset_index()
    
    features = result[["positive_count", "negative_count"]]    
    linkage = hierarchy.linkage(features, method='ward')
    dendrogram = hierarchy.dendrogram(linkage, labels=result['Country'].tolist(), orientation='top')

    result["Cluster"] = fcluster(linkage, 20, criterion='distance')
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(x="positive_count", y="negative_count", data=result, hue="Cluster", palette="viridis")
    for i, row in result.iterrows():
        x, y = row["positive_count"], row["negative_count"]
        jitter_x = random.uniform(-0.5, 0.5)  # Add random jitter to x-coordinate
        jitter_y = random.uniform(-1, 1)  # Add random jitter to y-coordinate
        scatter.annotate(row["Country"], (x + jitter_x, y + jitter_y), fontsize=8, alpha=0.7)

        # scatter.annotate(row["positive_count"], row["negative_count"], row["Country"], fontsize=8, alpha=0.7)
    plt.title("Hierarchical Clustering of Countries")
    plt.savefig("./diagrams/time_differences_statistics/hierarchical_clustering_countries_diff_peak.pdf", format="pdf")
    plt.show()

    # clustered_data = result.loc[dendrogram['ivl'], dendrogram['ivl']]

    # plt.figure(figsize=(10, 8))
    # sns.heatmap(clustered_data, fmt='.0f', cmap='Blues', robust=True, xticklabels=True, yticklabels=True)
    # # plt.savefig(file_name, format="pdf", bbox_inches="tight")
    # plt.show()








# df = preprocessing(df)
# calculate_posistion_from_occurance_to_peak(df)
# calculate_posistion_from_release_to_occurance(df)
# calculate_posistion_peak(df)
# plot_diagram_for_countires_difference()
# count_positive_negative_medians()