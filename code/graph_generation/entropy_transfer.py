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
from code.graph_generation.ste import symbolic_TE
from code.graph_generation.summary_statistics.heatmaps import plot_heatmap

filepath = './data/Final data/all_data.csv'
df = pd.read_csv(filepath,encoding='latin-1')
df = preprocessing(df)

def extend_country_songs_with_missing_positions(df, desired_length, country):
    df.sort_values(['Song','Date'], inplace=True, ascending = [True, True])
    df['Weeks_in_chart'] = df.groupby('Song')['Date'].rank(ascending=True, method='first').fillna(41).astype(int)

    def extend_group(group):
        rows_to_add = desired_length - len(group)
        if rows_to_add > 0:
            new_rows = pd.DataFrame({'Country': country,
                                    'Position': [41] * rows_to_add,
                                    'Song': [group['Song'].iloc[0]] * rows_to_add,
                                    'Weeks_in_chart': range(group['Weeks_in_chart'].max() + 1, group['Weeks_in_chart'].max() + rows_to_add + 1)})
            return pd.concat([group, new_rows], ignore_index=True)
        elif rows_to_add < 0:
            rows_to_remove = len(group) - desired_length
            return group.iloc[:rows_to_remove]
        else:
            return group

    grouped = df.groupby('Song')
    extended_df = grouped.apply(extend_group).reset_index(drop=True)
    return extended_df
    


def calculate_and_plot_symboli_transfer_entropy_for_countries(df):
    countries = df['Country'].unique()
    # main_figure = make_subplots(rows=9, cols=3, shared_yaxes='all', shared_xaxes='all', subplot_titles = [c for c in countries])
    songs_extended_positions = pd.DataFrame()
    for (idx,c) in enumerate(countries):
        country_df = df[df['Country'] == c]
        desired_length = 100
        country_df = extend_country_songs_with_missing_positions(country_df, desired_length, c)
        songs_extended_positions = pd.concat([songs_extended_positions, country_df], ignore_index=True, axis=0)

    songs_extended_positions.to_csv("extended_position_countries.csv", encoding='utf-8', index=False)
    # uk_songs = songs_extended_positions[songs_extended_positions['Country'] == 'UK']
    # ireland_songs = songs_extended_positions[songs_extended_positions['Country'] == 'Ireland']
    # common_songs = pd.merge(uk_songs, ireland_songs, on=['Song'], how='inner').reset_index()
    # common_songs = common_songs['Song'].squeeze()
    # common_songs = common_songs.drop_duplicates()
    # entropies_uk = []
    # entropies_irl = []
    # for index, song in common_songs.items():
    #     uk_positions = uk_songs[uk_songs['Song'] == song]['Position']
    #     ireland_positions = ireland_songs[ireland_songs['Song'] == song]['Position']
    #     e = symbolic_TE(uk_positions, ireland_positions, 1, 3)
    #     entropies_uk.append(e)
    #     e = symbolic_TE(ireland_positions, uk_positions, 1, 3)
    #     entropies_irl.append(e)
        

    # print(statistics.mean(entropies_uk), statistics.median(entropies_uk)) 
    # print(statistics.mean(entropies_irl), statistics.median(entropies_irl)) 

def calculate_entropy_for_countries():
    filepath = "extended_position_countries.csv"
    df = pd.read_csv(filepath)
    countries = df['Country'].unique()
    countries = ['UK', 'Ireland', 'USA']
    entropies = {}
    try:
        for (i,c) in enumerate(countries[:5]):
            for (j, c2) in enumerate (countries):
                if i == j :
                    continue
                c1_songs = df[df['Country'] == c]
                c2_songs = df[df['Country'] == c2]
                common_songs = pd.merge(c1_songs, c2_songs, on=['Song'], how='inner').reset_index()
                common_songs = common_songs['Song'].squeeze()
                common_songs = common_songs.drop_duplicates()
                entropies_country = []
                for _, song in common_songs.items():
                    c1_positions = c1_songs[c1_songs['Song'] == song]['Position']
                    c2_positions = c2_songs[c2_songs['Song'] == song]['Position']
                    print(c, c2, song)
                    e = symbolic_TE(c1_positions, c2_positions, 1, 3)
                    entropies_country.append(e)
                    
                if c not in entropies:
                    entropies[c] = {}
                entropies[c][c2] = [statistics.mean(entropies_country), statistics.median(entropies_country)]
    except:
        pass

    data = pd.DataFrame.from_dict(entropies)
    data.to_csv("entropies_all_countries3.csv", encoding='utf-8')


def calculate_entropy_for_countries2():
    print('halooo')
    filepath = "extended_position_countries.csv"
    df = pd.read_csv(filepath)
    countries = ['UK', 'Ireland', 'USA']
    entropies = {}
    print('hallo')
    try:
        for (i,c) in enumerate(countries):
            for (j, c2) in enumerate (countries):
                if i == j :
                    continue
                c1_songs = df[df['Country'] == c]
                c2_songs = df[df['Country'] == c2]
                common_songs = pd.merge(c1_songs, c2_songs, on=['Song'], how='inner').reset_index()
                common_songs = common_songs['Song'].squeeze()
                common_songs = common_songs.drop_duplicates()
                entropies_country = []
                for _, song in common_songs.items():
                    c1_positions = c1_songs[c1_songs['Song'] == song]['Position']
                    c2_positions = c2_songs[c2_songs['Song'] == song]['Position']
                    print(c, c2, song)
                    e = symbolic_TE(c1_positions, c2_positions, 1, 3)
                    entropies_country.append(e)
                    
                if c not in entropies:
                    entropies[c] = {}
                entropies[c][c2] = statistics.mean(entropies_country)
    except:
        pass

    data3 = pd.DataFrame.from_dict(entropies)
    data3.to_csv("entropies.csv")
    plot_heatmap(data3,  "./diagrams/entropy/entropy_m3.pdf", 'Mean entropy')







print("halo1")
calculate_entropy_for_countries2()
        