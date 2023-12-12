from collections import Counter
import math
import statistics
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import colorcet as cc
import csv
import datetime
import itertools

from matplotlib import patches
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
from sympy.utilities.iterables import multiset_permutations

from code.graph_generation.initial_graphs import preprocessing
from code.graph_generation.ste import _get_symbol_sequence, symbolic_TE
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

def get_common_songs_for_countries(country1_df, country2_df):
    common_songs = pd.merge(country1_df, country2_df, on=['Song'], how='inner').reset_index()
    common_songs = common_songs['Song'].squeeze()
    common_songs = common_songs.drop_duplicates()
    return common_songs

def get_missing_positions_for_countries(country1_df, country2_df):
    merged_df = pd.merge(country1_df, country2_df, on='Song', how='inner', suffixes=('_Country1', '_Country2'))
    merged_df['Date_Country1'] = list(zip(merged_df['Date_Country1'], merged_df['Position_Country1']))
    merged_df['Date_Country2'] = list(zip(merged_df['Date_Country2'], merged_df['Position_Country2']))

    def combine_lists(list1, list2):
        combined = list(set([list1, list2]))
        combined.sort(key=lambda x: x[0])
        result = []
        for date, pos1 in combined:
            pos2 = 41
            if date == list2[0]:
                pos2 = list2[1]
            result.append((date, pos1, pos2))
        return result

    merged_df['Combined'] = merged_df.apply(lambda row: combine_lists(row['Date_Country1'], row['Date_Country2']), axis=1)
    result_df = pd.concat([merged_df['Combined'].apply(pd.Series), merged_df.drop('Combined', axis=1)], axis=1)
    result_df = result_df.drop(['Date_Country1', 'Position_Country1', 'Date_Country2', 'Position_Country2'], axis=1)
    result_df.reset_index(drop=True, inplace=True)

    return result_df

def get_date_position_for_song(country1_df, country2_df):
    merged_df = pd.merge(country1_df, country2_df, on='Week_Number_Year', how='outer', suffixes=('_Country1', '_Country2'))
    # print(merged_df.head(50))
    # print(merged_df.columns)
    merged_df['Position_Country1'] = merged_df['Position_Country1'].fillna(41)
    merged_df['Position_Country2'] = merged_df['Position_Country2'].fillna(41)
    merged_df = merged_df.loc[:, ['Week_Number_Year', 'Position_Country1', 'Position_Country2']]
    merged_df[['Week_Number', 'Year']] = merged_df['Week_Number_Year'].str.split('_', expand=True)

    # Convert the columns back to numeric if needed
    merged_df['Week_Number'] = pd.to_numeric(merged_df['Week_Number'])
    merged_df['Year'] = pd.to_numeric(merged_df['Year'])
    merged_df = merged_df.sort_values(by=['Year','Week_Number'], ascending=[True, True])
    # print(merged_df.head())
    return merged_df



def pp(df):
    countries = df['Country'].unique()
    # countries = ['USA', 'UK', 'Ireland']
    file_name = './correct_extended_countries_with_entropy_m4.csv' 

    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["Country1", "Country2", "Song", "Entropy"])

        for (i,c) in enumerate(countries):
            print(c)
            for(j, c2) in enumerate(countries):
                if i == j : 
                    continue
                country1_df = df[df['Country'] == c]
                country2_df = df[df['Country'] == c2]

                common_songs = get_common_songs_for_countries(country1_df, country2_df)
                for _, song in common_songs.items():
                    song_country1 = country1_df[country1_df['Song'] == song]
                    song_country2 = country2_df[country2_df['Song'] == song]
                    res = get_date_position_for_song(song_country1, song_country2)
                    try:                                 
                        e = symbolic_TE(res.loc[:, 'Position_Country1'], res.loc[:, 'Position_Country2'], 1, 4)
                        writer.writerow([c, c2, song, e])
                    except Exception as error:
                        print('error occured:', error)


def plot_diagram_for_entropy():
    filepath = "correct_extended_countries_with_entropy_all.csv"
    df = pd.read_csv(filepath,encoding='latin-1')

    countries_1 = df['Country1'].unique()
    countries_2 = df['Country2'].unique()
    countries = np.hstack((countries_1, countries_2))
    countries = list(set(countries))
    countries.sort()
    fig, axs = plt.subplots(nrows=len(countries), ncols=len(countries), sharex=True, sharey=True, figsize=(60,60))
    for (i, country_1) in enumerate(countries):
        for j in range(0, len(countries)):
            if i== j:
                continue
            country_2 = countries[j]
            countries_subset = df[(df['Country1'] == country_1) & (df['Country2'] == country_2)]['Entropy']
            countries_subset = countries_subset[countries_subset >= 0.1]
            color = "blue"       
            axs[j, i].hist(x=countries_subset.values, bins=10, color=color, density=True) 
            axs[j, i].set_ylabel("Count")
            axs[j, i].set_title(f"{country_1} - {country_2}")
            
    fig.tight_layout()
    plt.savefig("./diagrams/entropy/entropy_histogram_bins_bigger.pdf", format="pdf")
    # plt.show()

def get_the_biggest_smallest_entropy(df):
    df['Week_Number_Year']  = df['Week_Number'].astype(str) + "_" +df['Year'].astype(str)
    filepath = "correct_extended_countries_with_entropy_all.csv"
    entropy_df = pd.read_csv(filepath,encoding='latin-1')
    entropy_df['Entropy'] = pd.to_numeric(entropy_df['Entropy'], errors='coerce')
    entropy_df = entropy_df[entropy_df['Country1'] == 'USA'].reset_index()
    idx_max = entropy_df['Entropy'].idxmin()
    bigg = entropy_df.iloc[idx_max]
    positions = df[(df['Country'] == 'USA') & (df['Song'] == bigg['Song'])]
    positions_uk = df[(df['Country'] == bigg['Country2']) & (df['Song'] == bigg['Song'])]
    res = get_date_position_for_song(positions, positions_uk)
    res['Transform_Position_Country1'] = res['Position_Country1'].apply(transform_positions)
    res['Transform_Position_Country2'] = res['Position_Country2'].apply(transform_positions)
    x_sym = _get_symbol_sequence(res['Transform_Position_Country1'],1,3)
    y_sym = _get_symbol_sequence(res['Transform_Position_Country2'],1,3)
    
    e1 = symbolic_TE(res['Transform_Position_Country1'], res['Transform_Position_Country2'], 1, 3)
    e2 = symbolic_TE(res['Transform_Position_Country2'], res['Transform_Position_Country1'], 1, 3)
    de = e1-e2

    # calculate random    
    e_random = symbolic_TE(res['Transform_Position_Country1'], np.random.permutation(res['Transform_Position_Country1']), 1, 3)
    de_random = e1 - e_random

    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(20,20), gridspec_kw={'height_ratios': [2 , 2, 1]})
    axs[0,0].invert_yaxis()
    axs[1,0].invert_yaxis()
    axs[2,0].invert_yaxis()
    rect_high = patches.Rectangle((-1, 0), 35, 5, linewidth=1, edgecolor='blue', facecolor='blue', alpha=0.3)
    rect_medium = patches.Rectangle((-1, 5), 35, 5, linewidth=1, edgecolor='green', facecolor='green', alpha=0.3)
    rect_low = patches.Rectangle((-1, 10), 35, 35, linewidth=1, edgecolor='orange', facecolor='orange', alpha=0.3)
    axs[0, 0].add_patch(rect_high)
    axs[0, 0].add_patch(rect_medium)
    axs[0, 0].add_patch(rect_low)
    axs[0, 0].text(-5, 2.5, "high", size=10, alpha = 0.8, color='blue')
    axs[0, 0].text(-5, 7.5, "medium", size=10, alpha = 0.8, color='green')
    axs[0, 0].text(-5, 12.5, "low", size=10, alpha = 0.8, color='orange')
    axs[0,0].plot(res['Week_Number_Year'], res['Position_Country1'], marker='o', label='Position in USA')
    axs[0, 0].set_ylim([45, 0])
    axs[0, 0].set_xlim([-1, 30])
    axs[0, 0].legend()
    axs[0, 0].tick_params(axis='x', labelrotation=45)
    axs[1, 0].plot(res['Week_Number_Year'], res['Transform_Position_Country1'], marker='o', label='Transformed positions')
    axs[1, 0].set_ylim([4, 0])
    axs[1, 0].set_ylabel('Position')
    axs[1, 0].legend()
    axs[1, 0].tick_params(axis='x', labelrotation=45)
    for i in range(len(x_sym)):
        row_data = x_sym[i, :]
        x_values = np.arange(start=i*3,stop=(i+1)*len(row_data))
        axs[2, 0].plot(x_values, row_data, marker='o', color='green')
    axs[2, 0].set_title("symbolic encoding m=3")
    axs[2, 0].set_ylabel('Indices values')

    axs[0, 1].invert_yaxis()
    axs[2, 1].invert_yaxis()
    axs[1, 1].invert_yaxis()    
    rect_high = patches.Rectangle((-1, 0), 35, 5, linewidth=1, edgecolor='blue', facecolor='blue', alpha=0.3)
    rect_medium = patches.Rectangle((-1, 5), 35, 5, linewidth=1, edgecolor='green', facecolor='green', alpha=0.3)
    rect_low = patches.Rectangle((-1, 10), 35, 35, linewidth=1, edgecolor='orange', facecolor='orange', alpha=0.3)
    axs[0, 1].add_patch(rect_high)
    axs[0, 1].add_patch(rect_medium)
    axs[0, 1].add_patch(rect_low)
    axs[0, 1].text(-5, 2.5, "high", size=10, alpha = 0.8, color='blue')
    axs[0, 1].text(-5, 7.5, "medium", size=10, alpha = 0.8, color='green')
    axs[0, 1].text(-5, 12.5, "low", size=10, alpha = 0.8, color='orange')
    axs[0, 1].plot(res['Week_Number_Year'], res['Position_Country2'], marker='o', label='Position in UK')
    axs[0, 1].set_ylim([45, 0])
    axs[0, 1].set_xlim([-1, 30])
    axs[0, 1].tick_params(axis='x', labelrotation=45)
    axs[1, 1].plot(res['Week_Number_Year'], res['Transform_Position_Country2'], marker='o', label='Transformed positions')
    axs[1, 1].set_ylim([4, 0])
    axs[1, 1].set_ylabel('Position')
    axs[1, 1].tick_params(axis='x', labelrotation=45)
    axs[1, 1].legend()
    for i in range(len(y_sym)):
        row_data = y_sym[i, :]
        x_values = np.arange(start=i*3,stop=(i+1)*len(row_data))
        axs[2,1].plot(x_values, row_data, marker='o', color='green')
    axs[2,1].set_title("symbolic encoding m=3")
    axs[2,1].set_ylabel('Indices values')
    song = bigg['Song']
    plt.xticks(rotation=45)
    fig.suptitle(f'Entropy USA-UK: {e1: .3f}, UK-USA:{e2: .3f}, song: {song}, delta: {de: .3f}, delta random: {de_random: .3f}')
    plt.subplots_adjust( hspace=0.4, wspace=0.2)
    plt.savefig("./diagrams/entropy/entropy_min2.pdf", format="pdf")
    plt.show()
    # print(res)
    # 

def example():
    a = np.array([1,
    2,
    2,
    4,5,7,7,8,2])
    b = np.array([10,
    1,
    2,
    3,
    4,5,6,7,8])
    date = np.array([1,
    2,
    2,
    4,
    5,
    7,
    7,
    8,
    2])
    e1 = symbolic_TE(a, b, 1, 3)
    e2 = symbolic_TE(b, a, 1, 3)
    x_sym = _get_symbol_sequence(a,1,3).argsort(kind='quicksort')
    y_sym = _get_symbol_sequence(b,1,3).argsort(kind='quicksort')
    
    hashmult = np.power(3, np.arange(3))
    hashval_x = (np.multiply(x_sym, hashmult)).sum(1)
    hashval_y = (np.multiply(y_sym, hashmult)).sum(1)

    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(20,20), gridspec_kw={'height_ratios': [3, 1, 3,1]})
    axs[0].invert_yaxis()
    axs[1].invert_yaxis()
    axs[0].plot(date, a, marker='o', label='Position in USA')
    axs[0].plot(date[1:-1], hashval_x, marker='o', label='Values for symbolic encoding')
    axs[0].set_ylabel('Position')
    axs[0].legend()
    for i in range(len(x_sym)):
        row_data = x_sym[i, :]
        x_values = np.arange(start=i*3,stop=(i+1)*len(row_data))
        axs[1].plot(x_values, row_data, marker='o', color='green')
    axs[1].set_title("symbolic encoding m=3")
    axs[1].set_ylabel('Indices values')

    axs[2].invert_yaxis()
    axs[2].plot(date, b, marker='o', label='Position in UK')
    axs[2].plot(date[1:-1], hashval_y, marker='o', label='Values for symbolic encoding')
    axs[2].set_ylabel('Position')
    axs[2].legend()
    axs[3].invert_yaxis()
    for i in range(len(y_sym)):
        row_data = y_sym[i, :]
        x_values = np.arange(start=i*3,stop=(i+1)*len(row_data))
        axs[3].plot(x_values, row_data, marker='o', color='green')
    axs[3].set_title("symbolic encoding m=3")
    axs[3].set_ylabel('Indices values')
    fig.suptitle(f'Entropy a-b: {e1: .3f}, b-a:{e2: .3f}')
    fig.tight_layout() 
    plt.savefig("./diagrams/entropy/entropy_example4.pdf", format="pdf")
    plt.show()


def transform_positions(position):
    if position > 10 : 
        return 3
    if position > 5:
        return 2
    return 1

def unique_permutations(arr):
    unique_vals, counts = pd.unique(arr, return_counts=True)
    unique_perms = multiset_permutations(unique_vals)

    all_permutations = []
    for perm in unique_perms:
        current_perm = []
        for val in perm:
            current_perm.extend([val] * counts[unique_vals == val][0])
        all_permutations.append(current_perm)

    return all_permutations

def test_random(df):
    df['Week_Number_Year']  = df['Week_Number'].astype(str) + "_" +df['Year'].astype(str)
    filepath = "correct_extended_countries_with_entropy_all.csv"
    entropy_df = pd.read_csv(filepath,encoding='latin-1')
    entropy_df['Entropy'] = pd.to_numeric(entropy_df['Entropy'], errors='coerce')
    entropy_df = entropy_df[entropy_df['Country1'] == 'USA'].reset_index()
    idx_max = entropy_df['Entropy'].idxmin()
    bigg = entropy_df.iloc[idx_max]
    positions = df[(df['Country'] == 'USA') & (df['Song'] == bigg['Song'])]
    positions_uk = df[(df['Country'] == bigg['Country2']) & (df['Song'] == bigg['Song'])]
    res = get_date_position_for_song(positions, positions_uk)
    res['Transform_Position_Country1'] = res['Position_Country1'].apply(transform_positions)
    res['Transform_Position_Country2'] = res['Position_Country2'].apply(transform_positions)

    print('halo')
    e1 = symbolic_TE(res['Transform_Position_Country1'], res['Transform_Position_Country2'], 1, 3)
    all_possible_permutations =unique_permutations(res['Transform_Position_Country1'])
    print(len(res['Transform_Position_Country1']), len(all_possible_permutations))
    print(all_possible_permutations)
    e = []
    i=0
    for p in all_possible_permutations:
        print(p)
        e_random = symbolic_TE(res['Transform_Position_Country1'], p, 1, 3)
        de_random = e1 - e_random
        # print(de_random)
        print(i)
        i+=1
        e.append(de_random)

    print(e)

test_random(df)
# example()
        