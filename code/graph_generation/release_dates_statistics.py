import datetime
import random
from matplotlib import patches
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

from code.graph_generation.correlation import calculate_first_occurance_of_song, calculate_first_occurance_of_song_in_country, calculate_peak_of_song_in_country
from .initial_graphs import preprocessing

filepath = './data/Final data/all_data.csv'
df = pd.read_csv(filepath,encoding='latin-1')
df = preprocessing(df)

def calculate_days_after_release_for_country(df):
    songs_first_occurance = calculate_first_occurance_of_song(df)
    songs_first_occurance_in_country = calculate_first_occurance_of_song_in_country(df)

    result_df = songs_first_occurance_in_country.merge(songs_first_occurance, on=['Song'], how='left', suffixes=('_Country', '_World'))
    result_df['Days_Difference'] = (result_df['Date_Country']-result_df['Date_World']).dt.days

    return result_df

def plot_histogram_days_after_release(df):
    df = calculate_days_after_release_for_country(df)

    countries = df['Country'].unique()
    fig, axs = plt.subplots(nrows=9, ncols=3, sharey=True, figsize=(40,30))
    for (i, c) in enumerate(countries):
        result = df[df['Country'] == c]    
        j = i // 3 # row
        i = i % 3  # col

        year_subset = result['Days_Difference']
        count_changes = year_subset.value_counts().sort_index()

        axs[j, i].bar(count_changes.index, count_changes.values)
        
        perecentile_25 = year_subset.quantile(0.025)
        perecentile_975 = year_subset.quantile(0.975)
        perecentile_50 = year_subset.quantile(0.5)
        # add lines for percentiles
        axs[j, i].vlines(x = perecentile_25, ymin=0, ymax=12000, alpha=0.7, color="grey", linestyle='--')
        axs[j, i].vlines(x =perecentile_975, ymin=0, ymax=12000, alpha=0.7, color="grey", linestyle='--')
        axs[j, i].vlines(x= perecentile_50, ymin=0, ymax=12000, alpha=0.7, color="red", linestyle='--')
        # mark the percentiles 
        rect_left = patches.Rectangle((-40, 0), 40 + perecentile_25, 12000, linewidth=1, edgecolor='grey', facecolor='grey', alpha=0.3)
        axs[j, i].add_patch(rect_left)
        axs[j, i].text(perecentile_25 - 1, 7000, "2.5th", size=12, alpha = 0.8)
        rect_right = patches.Rectangle((perecentile_975, 0), 40 - perecentile_975, 12000, linewidth=1, edgecolor='grey', facecolor='grey', alpha=0.3)
        axs[j, i].add_patch(rect_right)
        axs[j, i].text(perecentile_975 + 1, 7000, "97.5th", size=12, alpha = 0.8)
        axs[j, i].text(perecentile_50 + 1, 10000, "Median", size=12, alpha = 0.8)

        axs[j, i].set_xlabel("Days from release to occurance in a country")
        axs[j, i].set_ylabel("Count")
        axs[j, i].set_xlim(-40, 40)
        axs[j, i].set_title(c)
        
    fig.tight_layout()
    plt.savefig("./diagrams/summary_statistics/histogram_curance_release_all_countries.pdf", format="pdf")

def plot_histogram_days_to_reach_peak(df):
    df = calculate_days_after_release_for_country(df)

    countries = df['Country'].unique()
    fig, axs = plt.subplots(nrows=9, ncols=3, sharey=True, figsize=(40,30))
    for (i, c) in enumerate(countries):
        result = df[df['Country'] == c]    
        j = i // 3 # row
        i = i % 3  # col

        year_subset = result['Days_Difference']
        count_changes = year_subset.value_counts().sort_index()

        axs[j, i].bar(count_changes.index, count_changes.values)
        
        perecentile_25 = year_subset.quantile(0.025)
        perecentile_975 = year_subset.quantile(0.975)
        perecentile_50 = year_subset.quantile(0.5)
        # add lines for percentiles
        axs[j, i].vlines(x = perecentile_25, ymin=0, ymax=12000, alpha=0.7, color="grey", linestyle='--')
        axs[j, i].vlines(x =perecentile_975, ymin=0, ymax=12000, alpha=0.7, color="grey", linestyle='--')
        axs[j, i].vlines(x= perecentile_50, ymin=0, ymax=12000, alpha=0.7, color="red", linestyle='--')
        # mark the percentiles 
        rect_left = patches.Rectangle((-40, 0), 40 + perecentile_25, 12000, linewidth=1, edgecolor='grey', facecolor='grey', alpha=0.3)
        axs[j, i].add_patch(rect_left)
        axs[j, i].text(perecentile_25 - 1, 7000, "2.5th", size=12, alpha = 0.8)
        rect_right = patches.Rectangle((perecentile_975, 0), 40 - perecentile_975, 12000, linewidth=1, edgecolor='grey', facecolor='grey', alpha=0.3)
        axs[j, i].add_patch(rect_right)
        axs[j, i].text(perecentile_975 + 1, 7000, "97.5th", size=12, alpha = 0.8)
        axs[j, i].text(perecentile_50 + 1, 10000, "Median", size=12, alpha = 0.8)

        axs[j, i].set_xlabel("Days reach peak")
        axs[j, i].set_ylabel("Count")
        axs[j, i].set_xlim(-40, 40)
        axs[j, i].set_title(c)
        
    fig.tight_layout()
    plt.savefig("./diagrams/summary_statistics/histogram_days_to_reach_peak_all_countries.pdf", format="pdf")

def plot_difference_in_days_to_peak():
    filepath = "differences_in_peak.csv" 
    df = pd.read_csv(filepath)

    excluded_countries = ['China', 'Chile', 'India', 'Japan', 'Taiwan']
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
            color = "red"
            if countries_subset.median() <= 0 :
                color = "blue"                
            axs[j, i].hist(x=countries_subset.values, bins=range(-63, 70, 7), color=color, density=True) 
            axs[j, i].set_ylabel("Count")
            axs[j, i].set_title(f"{country_1} - {country_2}")
            
    fig.tight_layout()
    plt.savefig("./diagrams/time_differences_statistics/normalized_distribution_difference_in_peak.pdf", format="pdf")
    # plt.show()

def plot_releases_uk_us(df):    
    df = df.sort_values(by=['Country', 'Song', 'Date'])
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce').fillna(0).astype(int)
    songs_uk = df[df['Country'] == 'UK']
    songs_us = df[df['Country'] == 'Portugal']
    merged_df = songs_us.merge(songs_uk, on=['Song'], how='inner')
    merged_df = merged_df.sort_values(by='Date_x', ascending=False).reset_index(drop=True)
    songs = merged_df['Song'].unique()[:40]
    fig, axs = plt.subplots(nrows=8, ncols=5, sharey=True, figsize=(60,60))
    for (i,song) in enumerate(songs): 
        j = i // 5 # row
        i = i % 5  # col
        s_uk = songs_uk[songs_uk['Song'] == song]
        s_us = songs_us[songs_us['Song'] == song]
        s_uk = s_uk.sort_values(by='Date').reset_index(drop=True)
        s_us = s_us.sort_values(by='Date').reset_index(drop=True)

        uk_peak_idx = s_uk['Position'].idxmin()
        us_peak_idx = s_us['Position'].idxmin()

        axs[j,i].plot(s_uk['Date'], s_uk['Position'], color='green', marker='.')
        axs[j, i].vlines(x =s_uk['Date'].iloc[0], ymin=0, ymax=40, alpha=0.7, color="grey", linestyle='--')
        axs[j, i].text(s_uk['Date'].iloc[0], 10, s_uk['Date'].iloc[0].strftime('%d-%m-%Y')+ " UK", size=8, alpha = 0.8)

        axs[j, i].vlines(x =s_uk['Date'].iloc[uk_peak_idx], ymin=0, ymax=40, alpha=0.7, color="grey", linestyle='--')
        axs[j, i].text(s_uk['Date'].iloc[uk_peak_idx], 25,"Peak UK", size=8, alpha = 0.8)

        axs[j,i].plot(s_us['Date'], s_us['Position'], color='blue', marker='.')
        axs[j, i].vlines(x =s_us['Date'].iloc[0], ymin=0, ymax=40, alpha=0.7, color="grey", linestyle='--')
        axs[j, i].text(s_us['Date'].iloc[0], 15, s_uk['Date'].iloc[0].strftime('%d-%m-%Y') + " Portugal", size=8, alpha = 0.8)
        
        axs[j, i].vlines(x =s_us['Date'].iloc[us_peak_idx], ymin=0, ymax=40, alpha=0.7, color="grey", linestyle='--')
        axs[j, i].text(s_us['Date'].iloc[us_peak_idx], 30,"Peak Portugal", size=8, alpha = 0.8)
        
    plt.ylim(40,0)
    fig.tight_layout()
    plt.savefig("./diagrams/time_differences_statistics/uk_portugal_songs.pdf", format="pdf")
    # plt.show()

def number_of_negative_positive_medians():

    excluded_countries = ['China', 'Chile', 'India', 'Japan', 'Taiwan']
    filepath = "differences_in_peak_weird_countries.csv"
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
            print(country_1, country_2, countries_subset.median())
            if country_1 not in medians:
                medians[country_1] = []          
            medians[country_1].append(countries_subset.median()) 

    data_list = [(key, value) for key, values in medians.items() for value in values]
    country_medians = pd.DataFrame(data_list, columns=["Country", "Medians"])
    result  = country_medians.groupby('Country')['Medians'].agg(positive_count=lambda x: (x > 0).sum(),
                                              negative_count=lambda x: (x <= 0).sum()).reset_index()
    
    features = result[["positive_count", "negative_count"]]    
    linkage = hierarchy.linkage(features, method='ward')
    dendrogram = hierarchy.dendrogram(linkage, labels=result['Country'].tolist(), orientation='top')

    result["Cluster"] = fcluster(linkage, 20, criterion='distance')
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(x="positive_count", y="negative_count", data=result, hue="Cluster", palette="viridis")
    for i, row in result.iterrows():
        x, y = row["positive_count"], row["negative_count"]
        jitter_x = random.uniform(-0.5, 0.5)  # Add random jitter to x-coordinate
        jitter_y = random.uniform(-2, 2)  # Add random jitter to y-coordinate
        scatter.annotate(row["Country"], (x + jitter_x, y + jitter_y), fontsize=8, alpha=0.7)

        # scatter.annotate(row["positive_count"], row["negative_count"], row["Country"], fontsize=8, alpha=0.7)
    plt.title("Hierarchical Clustering of Countries")
    plt.show()

def weird_portugal(df):
    df = df.sort_values(by=['Country', 'Song', 'Date'])
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce').fillna(0).astype(int)
    songs_uk = df[df['Country'] == 'UK']
    songs_us = df[df['Country'] == 'USA']
    merged_df = songs_us.merge(songs_uk, on=['Song'], how='inner')
    merged_df = merged_df.sort_values(by='Date_x', ascending=False).reset_index(drop=True)
    songs = merged_df['Song'].unique()

    diffs = []

    for (i,song) in enumerate(songs): 
        s_uk = songs_uk[songs_uk['Song'] == song]
        s_us = songs_us[songs_us['Song'] == song]
        s_uk = s_uk.sort_values(by='Date').reset_index(drop=True)
        s_us = s_us.sort_values(by='Date').reset_index(drop=True)
        uk_peak_idx = s_uk['Position'].idxmin()
        us_peak_idx = s_us['Position'].idxmin()

        uk_peak = s_uk['Date'].iloc[uk_peak_idx]
        us_peak = s_us['Date'].iloc[us_peak_idx]
        diffs.append((uk_peak - us_peak).days)

    return np.mean(diffs), np.median(diffs)

def weirdportugal(df):
    df = df.sort_values(by=['Country','Song','Date'], ascending=[True, True, True])
    songs_peak = calculate_peak_of_song_in_country(df)
    differences_result_df = []  
    unique_songs = songs_peak['Song'].unique()
    countries1 = ['USA', 'UK', 'Portugal']
    for song in unique_songs:
        song_df = songs_peak[songs_peak['Song'] == song]
        for i in range(len(countries1)):
            for j in range(0, len(song_df)):
                country_1 = countries1[i]
                if country_1 not in song_df['Country'].values:
                    continue
                country_2 = song_df.iloc[j]['Country']
                date_country_1 = song_df[song_df['Country'] == country_1]['Date']
                print(country_1, date_country_1)
                diff_in_days = (date_country_1.iloc[0]- song_df.iloc[j]['Date']).days
                print(diff_in_days, country_1, country_2)
                
                # Add the calculated values to the result DataFrame
                differences_result_df.append({
                    'Country_1': country_1,
                    'Country_2': country_2,
                    'Song': song,
                    'Difference_in_Days': diff_in_days
                })

    differences_result_df = pd.DataFrame(differences_result_df, columns=['Country_1', 'Country_2', 'Song', 'Difference_in_Days'])
    differences_result_df.to_csv("differences_in_peak_weird_countries.csv")





# plot_histogram_days_after_release(df)
# plot_difference_in_days_to_peak()
plot_releases_uk_us(df)
# number_of_negative_positive_medians()
# print(weird_portugal(df))
# weirdportugal(df)
