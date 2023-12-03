import pandas as pd
import plotly.express as px

from plotly.subplots import make_subplots
from .initial_graphs import preprocessing
from ..data_scraping.get_artist_metadata import get_release_date

filepath = './data/Final data/all_data.csv'
df = pd.read_csv(filepath,encoding='latin-1')


def select_top_songs_per_year(df):
    df['Min_Position'] = df.groupby('Song')['Position'].transform('min')

    df = df[df['Min_Position'] == 1]

    most_popular_songs = df.groupby(['Year', 'Song']).agg(
        Occurance = ('Date', 'count')
    ).reset_index()
    most_popular_songs.sort_values(by=['Year', 'Occurance'], inplace=True, ascending=[False, False])
    return most_popular_songs

def select_top_1_song_for_each_year(df):
    most_popular_songs = select_top_songs_per_year(df)
    most_popular_songs = most_popular_songs.groupby(['Year']).agg(
        Max_Occurance = ('Occurance', 'max'),
        Song_name = ('Song', lambda x: x.iloc[0])
    ).reset_index()

    return most_popular_songs


def plot_when_song_first_appeared(df):
    top_songs = select_top_1_song_for_each_year(df)

    years = df['Year'].unique()
    main_figure = make_subplots(rows=9, cols=3, subplot_titles = [str(y) for y in years])

    for (idx, year) in enumerate(years):
        song = top_songs[top_songs['Year'] == year]['Song_name']
        result_in_countries = df[(df['Year'] == year) & (df['Song'].isin(song))]
        result_in_countries.sort_values(by=['Song', 'Week_Number'], inplace=True, ascending=[True, True])

        fig = px.line(result_in_countries, x='Week_Number', y='Position', color='Country', markers=True)
        fig.update_yaxes(autorange="reversed")
        # fig.show()
        for k in range(len(fig.data)):
            main_figure.add_trace(fig.data[k], row= (idx // 3)+1 ,col = (idx % 3)+1)
    main_figure.update_layout(height=6000)
    main_figure.update_yaxes(autorange="reversed")
    # main_figure.show()
    main_figure.write_html("top_song_in_each_year_in_all_countries.html")

def histogram_when_how_many_days_to_reach_top(df):
    top_songs = select_top_songs_per_year(df)
    result_in_countries = df[df['Song'].isin(top_songs['Song'])]

    result_in_countries = result_in_countries.sort_values(by=['Year', 'Country', 'Song', 'Position', 'Date'])
    min_date_df = result_in_countries.groupby(['Year', 'Country', 'Song']).agg(
        Min_Date=('Week_Number', 'min')
    ).reset_index()

    position_1_df = result_in_countries[result_in_countries['Position'] == 1][['Year', 'Country', 'Song', 'Week_Number']]
    position_1_df = position_1_df.groupby(['Year', 'Country', 'Song']).agg(
        Date_Position_1=('Week_Number', 'min')
    ).reset_index()

    result_df = min_date_df.merge(position_1_df, on=['Year', 'Country', 'Song'], how='left')

    result_df['Days_to_Peak'] = result_df['Date_Position_1']-result_df['Min_Date']
    
    fig = px.box(result_df, x='Country', y='Days_to_Peak', color='Country', labels={'Country':'Country', 'Days_to_Peak':'Days from appearence in chart to reachig top 1'})
    # fig.show()
    fig.write_html("histogram_weeks_to_peak.html")


def histogram_delays_days_per_country(df):
    top_songs = select_top_songs_per_year(df)
    result_in_countries = df[df['Song'].isin(top_songs['Song'])]
    min_date_countries_df = result_in_countries.groupby(['Year', 'Country', 'Song']).agg(
        Min_Date=('Week_Number', 'min')
    ).reset_index()

    min_date_df = result_in_countries.groupby(['Year', 'Song']).agg(
        Min_Date_Abs=('Week_Number', 'min')
    ).reset_index()

    result_df = min_date_countries_df.merge(min_date_df, on=['Year', 'Song'], how='left')

    result_df['Delay_Days'] = result_df['Min_Date']-result_df['Min_Date_Abs']
    
    fig = px.box(result_df, x='Country', y='Delay_Days', color='Country', labels={'Country':'Country', 'Delay_Days':'Days from first apperance in the world chart until apperance in country chart'})
    fig.update_traces(boxmean=True)
    # fig.show()
    fig.write_html("histogram_delay_weeks.html")

def histogram_delays_to_release_days_per_country(df):
    top_songs = select_top_songs_per_year(df)
    top_songs['Release_Date'] = top_songs.apply(lambda row: get_release_date(row[1].split('-')[0], row[1].split('-')[1]), axis=1)

    top_songs['Release_Date'] = pd.to_datetime(top_songs['Release_Date'])
    top_songs['Release_Week_Number'] = top_songs['Release_Date'].dt.strftime('%U').astype(int)

    result_in_countries = df[df['Song'].isin(top_songs['Song'])]
    min_date_countries_df = result_in_countries.groupby(['Year', 'Country', 'Song']).agg(
        Min_Date=('Week_Number', 'min')
    ).reset_index()

    result_df = min_date_countries_df.merge(top_songs, on=['Year', 'Song'], how='left')

    result_df['Delay_Days'] = result_df['Min_Date']-result_df['Release_Week_Number']
    
    fig = px.box(result_df, x='Country', y='Delay_Days', color='Country', labels={'Country':'Country', 'Delay_Days':'Days from first apperance in the world chart until apperance in country chart'})
    fig.update_traces(boxmean=True)
    fig.show()
    # fig.write_html("histogram_delay_weeks.html")


df = preprocessing(df)
histogram_delays_to_release_days_per_country(df)
