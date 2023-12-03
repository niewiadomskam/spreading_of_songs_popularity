import math
import statistics
from matplotlib import patches
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.subplots as sp
import plotly.graph_objects as go

from scipy.cluster import hierarchy
from scipy.stats import median_abs_deviation
from code.graph_generation.initial_graphs import preprocessing
from code.graph_generation.summary_statistics.summary_statistics import calculate_weeks_on_chart

filepath = './data/Final data/all_data.csv'
countries_top_40 = [
    'Germany',
    'UK',
    'USA'
]

df = pd.read_csv(filepath,encoding='latin-1')
df = preprocessing(df)


def plot_average_lifetime_by_position(df):
    df = calculate_weeks_on_chart(df)
    countries  = df['Country'].unique()
    fig, axs = plt.subplots(nrows=9, ncols=3, sharex=True, figsize=(40,15))
    for (i,c) in enumerate(countries):
        res = df[df['Country'] == c]        
        j = i // 3 # row
        i = i % 3  # col
        top_1_on_chart_avg, top_1_err, top_6_10_on_chart_avg, top_6_10_err, top_10_on_chart_avg, top_10_err = get_split_tops_yearly(res)

        top_1_on_chart_avg_plot = top_1_on_chart_avg.plot(kind='line', legend=False, rot=0, ax=axs[j, i])
        top_6_10_on_chart_avg_plot = top_6_10_on_chart_avg.plot(kind='line', legend=False, rot=0, ax=axs[j, i])
        top_10_on_chart_avg_plot = top_10_on_chart_avg.plot(kind='line', legend=False, rot=0, ax=axs[j, i])

        top_1_on_chart_avg_plot.fill_between(top_1_on_chart_avg.index, top_1_on_chart_avg["Max_weeks_on_chart"] - top_1_err["Max_weeks_on_chart"],
                            top_1_on_chart_avg["Max_weeks_on_chart"] + top_1_err["Max_weeks_on_chart"], alpha=0.3, color="C0")
        top_6_10_on_chart_avg_plot.fill_between(top_6_10_on_chart_avg.index, top_6_10_on_chart_avg["Max_weeks_on_chart"] - top_6_10_err["Max_weeks_on_chart"],
                            top_6_10_on_chart_avg["Max_weeks_on_chart"] + top_6_10_err["Max_weeks_on_chart"], alpha=0.3, color="C1")
        top_10_on_chart_avg_plot.fill_between(top_10_on_chart_avg.index, top_10_on_chart_avg["Max_weeks_on_chart"] - top_10_err["Max_weeks_on_chart"],
                            top_10_on_chart_avg["Max_weeks_on_chart"] + top_10_err["Max_weeks_on_chart"], alpha=0.3, color="C2")

        axs[j, i].set_xlabel("Year")
        axs[j, i].set_ylabel("Average song's lifetime (weeks)")
        axs[j, i].legend(["Top 1-5", "Top 6-10", "Top 10+"], fontsize=13, loc="upper left")
        axs[j, i].set_title(c)

    fig.tight_layout()
    plt.savefig("./diagrams/summary_statistics/average_weeks_on_chart_by_position+notsharedy.pdf", format="pdf")
    plt.show()

def chart_position_change_distribution(df):
    countries = df['Country'].unique()
    fig, axs = plt.subplots(nrows=9, ncols=3, sharey=True, figsize=(40,30))
    for (i, c) in enumerate(countries):
        result = df[df['Country'] == c]    
        j = i // 3 # row
        i = i % 3  # col

        result.sort_values(by=['Song title', 'Song author', 'Date'], inplace=True, ascending=[False, False, False])
        result['Position_change'] = result.groupby(['Song title', 'Song author'])['Position'].diff().fillna(0).astype(int)
        year_subset = result['Position_change']
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

        axs[j, i].set_xlabel("Position differenece")
        axs[j, i].set_ylabel("Count")
        axs[j, i].set_xlim(-40, 40)
        axs[j, i].set_title(c)
        
    fig.tight_layout()
    plt.savefig("./diagrams/summary_statistics/position_change_distribution_all_countries.pdf", format="pdf")

def chart_start_and_end_positions(df):
    countries = df['Country'].unique()
    fig, axs = plt.subplots(nrows=9, ncols=3, sharey=True, figsize=(40,30))
    for (i, c) in enumerate(countries):
        result = df[df['Country'] == c] 
        if not c in countries_top_40:
            result = result[result['Position'] <= 20] 
        result = get_start_and_end_positions(result)  

        j = i // 3 # row
        i = i % 3  # col
        first_pos = result["First_Position"].value_counts().sort_index()
        last_pos =  result["Last_Position"].value_counts().sort_index()
        axs[j, i].bar(first_pos.index, first_pos.values, color="blue", alpha=0.6)
        axs[j, i].bar(last_pos.index, last_pos.values, color="orange", alpha=0.6)

        axs[j, i].set_xlabel("Position")
        axs[j, i].set_ylabel("Count")
        axs[j, i].set_xlim(1, 40)
        axs[j, i].set_title(c)
        
    fig.tight_layout()
    plt.savefig("./diagrams/summary_statistics/initial_and_final_position_all_countries2.pdf", format="pdf")

def transform_position(position):
    if position > 10:
        return "10+"
    if position > 5:
        return "6-10"
    return "1-5"

def get_split_tops_yearly(df):
    df["Top_Position"] = df.groupby([ "Song"])["Position"].transform("min")
    df["Max_weeks_on_chart"] = df.groupby(["Song"])["Weeks_on_top"].transform("max")

    top_1_yearly = df[["Year", "Top_Position", "Max_weeks_on_chart"]][df["Top_Position"] <= 5]

    top_10_yearly = df[["Year", "Top_Position", "Max_weeks_on_chart"]][df["Top_Position"] > 10]

    top_6_10_yearly = df[["Year", "Top_Position", "Max_weeks_on_chart"]][(df["Top_Position"] <= 10) & (df["Top_Position"] > 5)]
    
    top_1_on_chart_avg = top_1_yearly[["Year", "Max_weeks_on_chart"]].groupby(by="Year").mean()
    top_1_err = top_1_yearly[["Year", "Max_weeks_on_chart"]].groupby(by="Year").sem()

    top_10_on_chart_avg = top_10_yearly[["Year", "Max_weeks_on_chart"]].groupby(by="Year").mean()
    top_10_err = top_10_yearly[["Year", "Max_weeks_on_chart"]].groupby(by="Year").sem()

    top_6_10_on_chart_avg = top_6_10_yearly[["Year", "Max_weeks_on_chart"]].groupby(by="Year").mean()
    top_6_10_err = top_6_10_yearly[["Year", "Max_weeks_on_chart"]].groupby(by="Year").sem()

    return top_1_on_chart_avg, top_1_err, top_6_10_on_chart_avg, top_6_10_err, top_10_on_chart_avg, top_10_err

def get_start_and_end_positions(df):
    df = df.sort_values(by="Date")
    result = df.groupby(["Song"]).agg({
        "Position": ['first', 'last']
    }).reset_index()
    result.columns = ["Song", "First_Position", "Last_Position"]
    result = result.reset_index(drop=True)
    return result

chart_start_and_end_positions(df)