import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from code.graph_generation.initial_graphs import preprocessing

filepath = './data/Final data/all_data.csv'
df = pd.read_csv(filepath,encoding='latin-1')
df = preprocessing(df)


def chart_boxplot_lifetime_on_top(df):
    result = df.groupby(['Year', 'Song', 'Country'])['Position'].count().reset_index(name = 'counts')
    countries = result['Country'].unique()
    countries = df['Country'].unique()
    fig, axs = plt.subplots(nrows=9, ncols=3, sharey=True, figsize=(40,30))
    for (i, c) in enumerate(countries):
        result = df[df['Country'] == c]    
        j = i // 3 # row
        i = i % 3  # col

        axs[j, i].box(result['Year'], result['counts'])
        axs[j, i].set_xlabel("Year")
        axs[j, i].set_ylabel("Count")
        axs[j, i].set_title(c)

    fig.tight_layout()
    plt.savefig("./diagrams/final/lifetime_boxplot.pdf", format="pdf")

chart_boxplot_lifetime_on_top(df)
