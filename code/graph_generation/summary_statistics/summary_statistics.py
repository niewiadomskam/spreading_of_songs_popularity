import math
import statistics
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.subplots as sp
import plotly.graph_objects as go

from scipy.cluster import hierarchy
from scipy.stats import median_abs_deviation
from code.graph_generation.correlation import calculate_first_occurance_of_song_in_country, calculate_last_occurance_of_song_in_country, calculate_peak_of_song_in_country
from code.graph_generation.initial_graphs import preprocessing

filepath = './data/Final data/all_data.csv'
df = pd.read_csv(filepath,encoding='latin-1')
df = preprocessing(df)


def std_error_of_median(data):
    mad = median_abs_deviation(data)
    std_median = mad / np.sqrt(len(data))
    return std_median

def plot_avg_lifetime_in_country(df):
    data = calculate_avg_lifetime_per_country(df)
    plt.scatter(data.keys(), data.values(), label='Data Points', color='b', marker='o')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.xticks(rotation='vertical')
    plt.savefig("./diagrams/summary_statistics/avg_song_lifetime.pdf", format="pdf", bbox_inches="tight")
    plt.show()

def plot_average_lifetime(df):
    df= calculate_weeks_on_chart(df)
    countries  = df['Country'].unique()
    fig, axs = plt.subplots(nrows=9, ncols=3, sharex=True, figsize=(40,15))
    for (i,c) in enumerate(countries):
        res = df[df['Country'] == c]
        j = i // 3 # row
        i = i % 3  # col

        median_weeks_on_chart = res[["First_day_of_week", "Weeks_on_top"]].groupby(
            by="First_day_of_week").median().rolling(window=30, min_periods=1).mean()
        median_std_error = res[["First_day_of_week", "Weeks_on_top"]].groupby(
            by="First_day_of_week")["Weeks_on_top"].apply(std_error_of_median).rolling(window=30, min_periods=1).mean().rename("Weeks_on_top")

        avg_weeks_on_chart = res[["First_day_of_week", "Weeks_on_top"]].groupby(
            by="First_day_of_week").mean().rolling(window=30, min_periods=1).mean()
        std_error = res[["First_day_of_week", "Weeks_on_top"]].groupby(
            by="First_day_of_week").sem().rolling(window=30, min_periods=1).mean()
        
        median_weeks_on_chart_plot = median_weeks_on_chart.plot(kind='line', legend=False, rot=0,ax=axs[j,i], color="orange")

        median_fill = median_weeks_on_chart_plot.fill_between(median_weeks_on_chart.index, median_weeks_on_chart["Weeks_on_top"].values - median_std_error.values,
                    median_weeks_on_chart["Weeks_on_top"].values + median_std_error.values, alpha=0.3, color="orange")

        avg_weeks_on_chart_plot = avg_weeks_on_chart.plot(ax=axs[j, i],kind='line', legend=False, rot=0)
        avg_fill = avg_weeks_on_chart_plot.fill_between(avg_weeks_on_chart.index, avg_weeks_on_chart["Weeks_on_top"] - std_error["Weeks_on_top"],
                            avg_weeks_on_chart["Weeks_on_top"] + std_error["Weeks_on_top"], alpha=0.3, color="C0")
        axs[j, i].set_xlabel("Week")
        axs[j, i].set_ylabel("Weeks on the chart")
        # axs[i,j].legend("Average", loc="upper left")
        axs[j, i].set_title(c)

    fig.tight_layout()
    plt.savefig("./diagrams/summary_statistics/average_weeks_on_chart2.pdf", format="pdf")
    plt.show()

def plot_average_lifetime_by_season(df):
    df= calculate_weeks_on_chart(df)
    countries  = df['Country'].unique()
    fig, axs = plt.subplots(nrows=9, ncols=3, sharex=True, sharey=True, figsize=(40,15))
    for (i,c) in enumerate(countries):
        res = df[df['Country'] == c]
        j = i // 3 # row
        i = i % 3  # col
        avg_weeks_on_chart = res[["Season", "Weeks_on_top"]].groupby(
            by="Season").mean()
        avg_weeks_on_chart_plot = avg_weeks_on_chart.plot(ax=axs[j, i], kind='bar', legend=False, rot=0, yticks=[0,2,4,6,8,10,12,14])
        axs[j, i].set_xlabel("Season")
        axs[j, i].set_ylabel("Weeks on the chart")
        axs[j, i].set_title(c)
        axs[j, i].grid(axis='y', linestyle='-', color='gray', alpha=0.5)

    fig.tight_layout()
    plt.savefig("./diagrams/summary_statistics/average_weeks_on_chart_season.pdf", format="pdf")
    plt.show()

def plot_average_lifetime_by_year(df):
    df= calculate_weeks_on_chart(df)
    countries  = df['Country'].unique()
    fig, axs = plt.subplots(nrows=9, ncols=3, sharex=True, sharey=True, figsize=(40,15))
    for (i,c) in enumerate(countries):
        res = df[df['Country'] == c]
        j = i // 3 # row
        i = i % 3  # col

        median_weeks_on_chart = res[["Year", "Weeks_on_top"]].groupby(
            by="Year").median().rolling(window=3, min_periods=1).mean()
        median_std_error = res[["Year", "Weeks_on_top"]].groupby(
            by="Year")["Weeks_on_top"].apply(std_error_of_median).rolling(window=3, min_periods=1).mean().rename("Weeks_on_top")

        avg_weeks_on_chart = res[["Year", "Weeks_on_top"]].groupby(
            by="Year").mean().rolling(window=3, min_periods=1).mean()
        std_error = res[["Year", "Weeks_on_top"]].groupby(
            by="Year").sem().rolling(window=3, min_periods=1).mean()
        
        median_weeks_on_chart_plot = median_weeks_on_chart.plot(kind='line', legend=False, rot=0,ax=axs[j,i], color="orange")

        median_fill = median_weeks_on_chart_plot.fill_between(median_weeks_on_chart.index, median_weeks_on_chart["Weeks_on_top"].values - median_std_error.values,
                    median_weeks_on_chart["Weeks_on_top"].values + median_std_error.values, alpha=0.3, color="orange")

        avg_weeks_on_chart_plot = avg_weeks_on_chart.plot(ax=axs[j, i],kind='line', legend=False, rot=0)
        avg_fill = avg_weeks_on_chart_plot.fill_between(avg_weeks_on_chart.index, avg_weeks_on_chart["Weeks_on_top"] - std_error["Weeks_on_top"],
                            avg_weeks_on_chart["Weeks_on_top"] + std_error["Weeks_on_top"], alpha=0.3, color="C0")
        axs[j, i].set_xlabel("Year")
        axs[j, i].set_ylabel("Weeks on the chart")
        # axs[i,j].legend("Average", loc="upper left")
        axs[j, i].set_title(c)

    fig.tight_layout()
    plt.savefig("./diagrams/summary_statistics/average_weeks_on_chart_sharedy_yearly.pdf", format="pdf")
    plt.show()

def calculate_avg_lifetime_per_country(df):
    avg_lifetime = {}
    countries = df['Country'].unique()
    for (idx, c) in enumerate(countries):
        result = df[df['Country'] == c]
        result["Days_in_chart"] = result.groupby(['Song'])['Date'].transform(lambda d: (max(d) - min(d)).days)
        avg_lifetime[c] = result["Days_in_chart"].mean()
    
    return avg_lifetime

def calculate_weeks_on_chart(df):
    df['First_day_of_week'] = df['Date'] - pd.to_timedelta((df['Date'].dt.dayofweek + 1) % 7, unit='D')
    df = df.sort_values(by=['Song','Date'])
    grouped = df.groupby(['Song','Country'])
    df['Weeks_on_top'] = grouped.cumcount() + 1

    return df

def weeks_on_chart_histogram(df):
    df= calculate_weeks_on_chart(df)
    countries  = df['Country'].unique()
    fig, axs = plt.subplots(nrows=9, ncols=3, sharex=True, sharey=True, figsize=(40,15))
    for (i,c) in enumerate(countries):
        res = df[(df['Country'] == c) & (df['Weeks_on_top'] <= 54)]["Weeks_on_top"]
        j = i // 3 # row
        i = i % 3  # col
        res.hist(ax=axs[j, i], bins=14, legend=False) # bin size =3 weeks
        res.hist(ax=axs[j, i], bins=27, legend=False, color="red") # bin size =3 weeks
        axs[j, i].set_ylabel("Count")
        axs[j, i].set_xlabel("Weeks on the chart")
        axs[j, i].set_title(c)

    fig.tight_layout()
    plt.savefig("./diagrams/summary_statistics/histogram_weeks_on_top_2hist.pdf", format="pdf")
    plt.show()

def plot_avg_days_on_chart_before_and_after_peak(df):
    df= calculate_weeks_on_chart(df)
    df["Max_weeks_on_chart"] = df.groupby(["Song"])["Weeks_on_top"].transform("max")
    countries  = df['Country'].unique()
    fig, axs = plt.subplots(nrows=9, ncols=3, sharex=True, sharey=True, figsize=(40,30))
    for (i,c) in enumerate(countries):
        res = df[(df['Country'] == c) & (df['Weeks_on_top'] <= 54)]
        res["Top_Position"] = res.groupby([ "Song"])["Position"].transform("min")
        j = i // 3 # row
        i = i % 3  # col
        x = np.arange(3)
        width = 0.2
        max_weeks_on_chart = get_average_from_max_weeks_on_chart(res)
        avg_weeks_on_chart_before_peak = get_average_weeks_on_chart_before_peak(res)
        avg_weeks_on_chart_after_peak = get_average_weeks_on_chart_after_peak(res)
        axs[j, i].bar(x - width, max_weeks_on_chart, width)
        axs[j, i].bar(x, avg_weeks_on_chart_before_peak, width)
        axs[j, i].bar(x + width, avg_weeks_on_chart_after_peak, width)

        axs[j, i].set_xticks(x, ['1-5', '6-10', '10+']) 
        axs[j, i].set_ylabel("Average weeks")
        axs[j, i].set_xlabel("Top position")
        axs[j, i].set_title(c)

    fig.tight_layout()
    plt.savefig("./diagrams/summary_statistics/plot_avg_days_on_chart_before_and_after_peak.pdf", format="pdf")
    plt.show()

def get_average_weeks_on_chart_before_peak(df):
    songs_first_occurance = calculate_first_occurance_of_song_in_country(df)
    songs_peak = calculate_peak_of_song_in_country(df)    
    result_df = songs_first_occurance.merge(songs_peak, on=['Song'], how='left', suffixes=('_Peak', '_Enter'))
    result_df['Weekss_Difference_from_initial_to_peak'] = (result_df['Date_Peak']-result_df['Date_Enter']).dt.days / 7

    top_1 = df[["Song", "Top_Position", "Max_weeks_on_chart"]][df["Top_Position"] <= 5]
    top_10 = df[["Song", "Top_Position", "Max_weeks_on_chart"]][df["Top_Position"] > 10]
    top_6_10 = df[["Song", "Top_Position", "Max_weeks_on_chart"]][(df["Top_Position"] <= 10) & (df["Top_Position"] > 5)]

    
    top_1  = top_1.merge(result_df, on=['Song'], how='left')
    top_1_avg = abs(top_1[["Weekss_Difference_from_initial_to_peak"]].mean().item())
    top_6_10  = top_6_10.merge(result_df, on=['Song'], how='left')
    top_6_10_avg = abs(top_6_10[["Weekss_Difference_from_initial_to_peak"]].mean().item())
    top_10  = top_10.merge(result_df, on=['Song'], how='left')
    top_10_avg = abs(top_10[["Weekss_Difference_from_initial_to_peak"]].mean().item())

    return [top_1_avg, top_6_10_avg, top_10_avg]

def get_average_weeks_on_chart_after_peak(df):
    songs_last_occurance = calculate_last_occurance_of_song_in_country(df)
    songs_peak = calculate_peak_of_song_in_country(df)    
    result_df = songs_last_occurance.merge(songs_peak, on=['Song'], how='left', suffixes=('_Peak', '_Last'))
    result_df['Weekss_Difference_from_last_to_peak'] = (result_df['Date_Last']-result_df['Date_Peak']).dt.days / 7

    top_1 = df[["Song", "Top_Position", "Max_weeks_on_chart"]][df["Top_Position"] <= 5]
    top_10 = df[["Song", "Top_Position", "Max_weeks_on_chart"]][df["Top_Position"] > 10]
    top_6_10 = df[["Song", "Top_Position", "Max_weeks_on_chart"]][(df["Top_Position"] <= 10) & (df["Top_Position"] > 5)]

    
    top_1  = top_1.merge(result_df, on=['Song'], how='left')
    top_1_avg = abs(top_1[["Weekss_Difference_from_last_to_peak"]].mean().item())
    top_6_10  = top_6_10.merge(result_df, on=['Song'], how='left')
    top_6_10_avg = abs(top_6_10[["Weekss_Difference_from_last_to_peak"]].mean().item())
    top_10  = top_10.merge(result_df, on=['Song'], how='left')
    top_10_avg = abs(top_10[["Weekss_Difference_from_last_to_peak"]].mean().item())

    return [top_1_avg, top_6_10_avg, top_10_avg]

def get_average_from_max_weeks_on_chart(df):    
    top_1 = df[["Top_Position", "Max_weeks_on_chart"]][df["Top_Position"] <= 5]
    top_10 = df[["Top_Position", "Max_weeks_on_chart"]][df["Top_Position"] > 10]
    top_6_10 = df[["Top_Position", "Max_weeks_on_chart"]][(df["Top_Position"] <= 10) & (df["Top_Position"] > 5)]

    top_1_avg = abs(top_1[["Max_weeks_on_chart"]].mean().item())
    top_6_10_avg = abs(top_6_10[["Max_weeks_on_chart"]].mean().item())
    top_10_avg = abs(top_10[["Max_weeks_on_chart"]].mean().item())
    # print(top_1_avg, type(top_1_avg))

    return [top_1_avg, top_6_10_avg, top_10_avg]

def weird_germany(df):
    df= calculate_weeks_on_chart(df)
    res = df[df['Country'] == 'Germany']
    median_weeks_on_chart = res[["First_day_of_week", "Weeks_on_top"]].groupby(
            by="First_day_of_week").median().rolling(window=30, min_periods=1).mean()
    
    median_std_error = res[["First_day_of_week", "Weeks_on_top"]].groupby(
            by="First_day_of_week")["Weeks_on_top"].apply(std_error_of_median).rolling(window=30).mean().rename("Weeks_on_top")

    avg_weeks_on_chart = res[["First_day_of_week", "Weeks_on_top"]].groupby(
            by="First_day_of_week").mean().rolling(window=30, min_periods=1).mean()
    std_error = res[["First_day_of_week", "Weeks_on_top"]].groupby(
            by="First_day_of_week").sem().rolling(window=30).mean()
    
# plot_avg_days_on_chart_before_and_after_peak(df)