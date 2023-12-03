#!/usr/bin/env python3

import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from plotly.subplots import make_subplots
from scipy.stats import chisquare, chi2_contingency, chi2

from code.data_scraping.const import country_artist_map

filepath = './data/Final data/all_data.csv'
df = pd.read_csv(filepath,encoding='latin-1')
chosen_countries = [
    'Denmark',
    'Finland',
    'Belgium',
    'German',
    'Netherlands',
    'Norway'
]

def preprocessing(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True, ascending=False)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week_Number'] = df['Date'].dt.strftime('%U')
    df['Season'] = df['Date'].apply(get_season)
    # select only this countries which occures every year    
    # countries = select_countries_which_appear_each_year(df)
    # df = df[df['Country'].isin(countries)]
    df['Country'] =df['Country'].apply(lambda x: country_artist_map[x] if x in country_artist_map.keys() else x)
    df['Song'] = df['Song author'] +  " - " + df['Song title']
    df['Week_Number'] = df['Week_Number'].astype(int)
    df['Artist country'] =df['Artist country'].fillna('nan')
    df = df.drop(columns=['Unnamed: 0'])
    df = df.drop_duplicates()
    # remove wrong data for Germany 
    mask = (df['Country'] != 'Germany') | (df['Date'] >= '2007-09-15')
    df = df[mask]
    df = clear_out_genre(df)
    # df['Artist country'] =df['Artist country'].apply(lambda x: country_artist_map[x])
    return df

def get_season(date):
    if date.month in [3, 4, 5]:
        return 'Spring'
    elif date.month in [6, 7, 8]:
        return 'Summer'
    elif date.month in [9, 10, 11]:
        return 'Autumn'
    else:
        return 'Winter'
    
def clear_out_genre(df):
    weird_genre = ['AxÃ?Â?Ã?Â©', 'MÃ?Â?Ã?Âºsica tropical', 'ForrÃ\x83Â\x83Ã\x82Â³', 'MÃ\x83Â\x83Ã\x82Âºsica Mexicana', 
                   'MÃ\x83Â\x83Ã\x82Âºsica tropical', 'RaÃ\x83Â\x83Ã\x82Â\xadces', 'AxÃ\x83Â\x83Ã\x82Â©']
    df = df[~df['Genre'].isin(weird_genre)]
    return df

def select_countries_which_appear_each_year(df):
    countries_in_years = df.groupby('Year')['Country'].apply(set)
    countries = countries_in_years.agg(lambda x: set.intersection(*x))
    return countries

def histogram_songs_on_top_10(df):
    top_10_songs = df[df['Position'] > 11]
    time_on_chart = top_10_songs.groupby(['Year', 'Country', 'Song title', 'Song author']).agg(
        weeks_on_top=('Position', 'count')
    ).reset_index()
    print(time_on_chart.head())
    median_time_per_country = time_on_chart.groupby(['Country', 'Year']).agg(
        median=('weeks_on_top', 'median')
    ).reset_index()
    median_time_per_country = median_time_per_country[median_time_per_country['Country'].isin(chosen_countries)]
    fig = px.line(median_time_per_country, x='Year', y='median', color='Country')
    fig.write_image('./images/median_top10_songs_chosen_country.jpeg')
    fig = px.box(time_on_chart, x='Country', y='weeks_on_top', color='Country', labels={'Country':'Country', 'weeks_on_top':'Weeks on Top 10'})
    fig.write_image('./images/histogram_top10_songs_per_country.jpeg')

def chart_for_one_song_in_different_countries(df, song_title, song_artist):
    df = df[df['Song title'] == song_title]
    df = df[df['Song author'] == song_artist]
    df = df[df['Country'].isin(chosen_countries)]
    fig = px.line(df, x='Date', y='Position', color = 'Country', title = song_title+ ' by '+ song_artist)
    fig.update_yaxes(autorange="reversed")
    fig.write_image('./images/top1_song_chosen_countries.jpeg')

def chart_countrychart_artistcountry(df):
    fig = px.scatter(df, x='Country', y='Artist country')
    fig.write_image('./images/country_artist_country.jpeg')

def chart_line_average(df):
    # replace position number
    df['Position'] = df['Position'].apply(lambda row: replace_position(row))
    result = df.groupby(['Month', 'Year', 'Song title', 'Song author', 'Country', 'Position']).agg(
        counts=('Position', 'count')
    ).reset_index()

    result = result.groupby(['Country', 'Year', 'Position']).agg(
        median=('counts', 'median'),
        avg = ('counts', 'mean')
    ).reset_index()
    result['Date'] = pd.to_datetime(dict(year=result['Year'], month=1, day=1))
    result.sort_values(by='Date', inplace=True, ascending=False)


    countries = result['Country'].unique()
    main_figure = make_subplots(rows=9, cols=3, shared_yaxes='all', shared_xaxes='all', subplot_titles = [c for c in countries])

    for (idx,c) in enumerate(countries):
        res = result[result['Country'] == c]
        res.sort_values(['Date','Position'], inplace=True, ascending=[True, True])
        fig = px.line(res, x='Date', y='avg', color='Position', title=c, markers=True)
        fig.update_traces(showlegend=False)
        fig.update_traces(legendgroup=1, selector=dict(name='1'))
        fig.update_traces(legendgroup=2, selector=dict(name='10'))
        fig.update_traces(legendgroup=3, selector=dict(name='20'))
        for k in range(len(fig.data)):
            main_figure.add_trace(fig.data[k], row= (idx // 3)+1 ,col = (idx % 3)+1)
    main_figure.update_layout(height=3000)
    main_figure.write_html('mean_all_countries_yearly.html')

def chart_boxplot_lifetime_on_top(df):
    result = df.groupby(['Year', 'Song title', 'Song author', 'Country'])['Position'].count().reset_index(name = 'counts')
    countries = result['Country'].unique()
    main_figure = make_subplots(rows=9, cols=3, shared_yaxes='all', shared_xaxes='all', subplot_titles = [c for c in countries])
    for (idx,c) in enumerate(countries):
        res = result[result['Country'] == c]
        fig = px.box(res, x='Year', y='counts', title =c)
        # fig.write_image('./images/lifetime_box/box_lifetime_' + c + '.jpeg')
        for k in range(len(fig.data)):
            main_figure.add_trace(fig.data[k], row= (idx // 3)+1 ,col = (idx % 3)+1)
        # fig.write_image('./images/distribution_centiles/distribution_' + c + '.jpeg')
    main_figure.update_layout(height=3500)
    main_figure.write_html('lifetime_all_countries_yearly.html')
        
def trajectory_chart(df):
    df.drop(columns=['Week_Number', 'Artist country', 'Genre'], inplace=True)
    countries = df['Country'].unique()
    for c in countries[:1]:
        songsTrajectory = df[df['Country'] == "USA"]
        # fill the missing data with value 50 # end of chart
        songsTrajectory.sort_values('Date', inplace=True, ascending = True)

        all_weeks_df = songsTrajectory[['Date']].drop_duplicates()
        unique_songs = songsTrajectory[['Song title', 'Song author']].drop_duplicates()
        all_combinations_df = pd.DataFrame({'Date': np.repeat(all_weeks_df['Date'], len(unique_songs)),
                                        'Song title': np.tile(unique_songs['Song title'], len(all_weeks_df)),
                                        'Song author': np.tile(unique_songs['Song author'], len(all_weeks_df))})
        
        merged_df = pd.merge(all_combinations_df, songsTrajectory, on=['Date','Song title', 'Song author'], how='left')
        merged_df.sort_values(['Date'], inplace=True, ascending = [True])
        # Fill missing positions with 41
        merged_df['Position'].fillna(41, inplace=True)
        # fill out missing year and month columns
        merged_df['Year'] = merged_df['Date'].dt.year
        merged_df['Month'] = merged_df['Date'].dt.month
        merged_df['Week_Number'] = merged_df['Date'].dt.strftime('%U')

        min_position_df = merged_df.groupby(['Song title', 'Song author']).agg(
            Min_Position=('Position', 'min')
        ).reset_index()

        songsTrajectory = pd.merge(merged_df, min_position_df, on=['Song title', 'Song author'], how='left')
        songsTrajectory['Top_Group'] = songsTrajectory['Min_Position'].apply(lambda x: replace_position(x))

        songsTrajectory = songsTrajectory.groupby(['Year', 'Week_Number', 'Top_Group']).agg(
            trajectory=('Position', 'mean')
        ).reset_index()

        songsTrajectory['Date'] = songsTrajectory.apply(create_datetime, axis=1)
        songsTrajectory.sort_values(by='Date', inplace=True, ascending=False)
        fig = px.line(songsTrajectory, x='Date', y = 'trajectory', color='Top_Group', title=c)
        fig.update_yaxes(autorange="reversed")
        fig.show()

def trajectory_chart2(df, title = None):
    df.drop(columns=['Week_Number', 'Artist country', 'Genre', 'Unnamed: 0'], inplace=True)
    countries = df['Country'].unique()
    main_figure = make_subplots(rows=9, cols=3, shared_yaxes='all', shared_xaxes='all', subplot_titles = [c for c in countries])

    for (idx,c) in enumerate(countries):
        songsTrajectory = df[df['Country'] == c]
        songsTrajectory['Artist_song'] = songsTrajectory['Song author'] +  " - " + songsTrajectory['Song title']
        songsTrajectory.drop(columns=['Song title', 'Song author', 'Country'], inplace=True)
        # save all weeks that appear in chart
        # all_weeks_df = songsTrajectory[['Date']].drop_duplicates()
        desired_length = 100
        # fill the missing data with value 50 # end of chart
        songsTrajectory.sort_values(['Artist_song','Date'], inplace=True, ascending = [True, True])
        songsTrajectory['Weeks_in_chart'] = songsTrajectory.groupby('Artist_song')['Date'].rank(ascending=True, method='first').fillna(41).astype(int)
        songsTrajectory.drop(columns=['Date', 'Year', 'Month'], inplace=True)

        def extend_group(group):
            rows_to_add = desired_length - len(group)
            if rows_to_add > 0:
                new_rows = pd.DataFrame({'Position': [41] * rows_to_add,
                                        'Artist_song': [group['Artist_song'].iloc[0]] * rows_to_add,
                                        'Weeks_in_chart': range(group['Weeks_in_chart'].max() + 1, group['Weeks_in_chart'].max() + rows_to_add + 1)})
                return pd.concat([group, new_rows], ignore_index=True)
            elif rows_to_add < 0:
                rows_to_remove = len(group) - desired_length
                return group.iloc[:rows_to_remove]
            else:
                return group

        grouped = songsTrajectory.groupby('Artist_song')
        extended_df = grouped.apply(extend_group).reset_index(drop=True)
        extended_df['Min_Position'] = extended_df.groupby('Artist_song')['Position'].transform('min')
        extended_df['Top_Group'] = extended_df['Min_Position'].apply(replace_position).reset_index(drop=True)
        result = extended_df.groupby(['Weeks_in_chart','Top_Group']).agg(
            trajectory=('Position', 'mean')
        ).reset_index()
        fig = px.line(result, x='Weeks_in_chart', y = 'trajectory', color='Top_Group', title=c)
        idx_max_trajectory = result['trajectory'].argmin()
        fig.add_vline(x= result.iloc[idx_max_trajectory, 0] , line_width=2 ,line_color="red", line_dash='dash')
        fig.update_yaxes(autorange="reversed")
        # fig.show()
        # fig.write_html('./images/trajectory/trajectory_' + c + '2.html')
        for k in range(len(fig.data)):
            main_figure.add_trace(fig.data[k], row= (idx // 3)+1 ,col = (idx % 3)+1)
    main_figure.update_layout(height=3000)
    main_figure.update_yaxes(autorange="reversed")
    main_figure.update_xaxes(range=(0, 100))
    if title is not None:
        main_figure.update_layout(title=title)
        main_figure.write_html(title.replace(' ', '_')+'.html')
    # main_figure.show()
    main_figure.write_html("trajectories_till_100_week_all_new_distribution.html")

def trajectory_chart_chosen_countries(df):
    df.drop(columns=['Week_Number', 'Artist country', 'Genre', 'Unnamed: 0'], inplace=True)
    countries = ["Spain", "Portugal"]
    main_figure = make_subplots(rows=2, cols=2, shared_yaxes='all', shared_xaxes='all', subplot_titles = ['Spain 2000-2010', 'Spain 2010-2020', 'Portugal 2000-2010', 'Portugal 2010-2020' ])

    for (i, year) in enumerate([2000, 2010]):
        for (idx,c) in enumerate(countries):
            songsTrajectory = df[df['Country'] == c]
            songsTrajectory = songsTrajectory[(songsTrajectory['Year'] > year) & (songsTrajectory['Year'] <= year+10)]
            songsTrajectory['Artist_song'] = songsTrajectory['Song author'] +  " - " + songsTrajectory['Song title']
            songsTrajectory.drop(columns=['Song title', 'Song author', 'Country'], inplace=True)
            # save all weeks that appear in chart
            # all_weeks_df = songsTrajectory[['Date']].drop_duplicates()
            desired_length = 100
            # fill the missing data with value 50 # end of chart
            songsTrajectory.sort_values(['Artist_song','Date'], inplace=True, ascending = [True, True])
            songsTrajectory['Weeks_in_chart'] = songsTrajectory.groupby('Artist_song')['Date'].rank(ascending=True, method='first').fillna(41).astype(int)
            songsTrajectory.drop(columns=['Date', 'Year', 'Month'], inplace=True)

            def extend_group(group):
                rows_to_add = desired_length - len(group)
                if rows_to_add > 0:
                    new_rows = pd.DataFrame({'Position': [41] * rows_to_add,
                                            'Artist_song': [group['Artist_song'].iloc[0]] * rows_to_add,
                                            'Weeks_in_chart': range(group['Weeks_in_chart'].max() + 1, group['Weeks_in_chart'].max() + rows_to_add + 1)})
                    return pd.concat([group, new_rows], ignore_index=True)
                elif rows_to_add < 0:
                    rows_to_remove = len(group) - desired_length
                    return group.iloc[:rows_to_remove]
                else:
                    return group

            grouped = songsTrajectory.groupby('Artist_song')
            extended_df = grouped.apply(extend_group).reset_index(drop=True)
            extended_df['Min_Position'] = extended_df.groupby('Artist_song')['Position'].transform('min')
            extended_df['Top_Group'] = extended_df['Min_Position'].apply(replace_position).reset_index(drop=True)
            result = extended_df.groupby(['Weeks_in_chart','Top_Group']).agg(
                trajectory=('Position', 'mean')
            ).reset_index()
            fig = px.line(result, x='Weeks_in_chart', y = 'trajectory', color='Top_Group', title=c)
            idx_max_trajectory = result['trajectory'].argmin()
            fig.add_vline(x= result.iloc[idx_max_trajectory, 0] , line_width=2 ,line_color="red", line_dash='dash')
            fig.update_yaxes(autorange="reversed")
            # fig.show()
        # fig.write_html('./images/trajectory/trajectory_' + c + '2.html')
            for k in range(len(fig.data)):
                main_figure.add_trace(fig.data[k], col= (idx // 2)+1+i ,row = (idx % 2)+1)
    # main_figure.update_layout(height=600)
    main_figure.update_yaxes(autorange="reversed")
    main_figure.update_xaxes(range=(0, 100))
    # main_figure.show()
    main_figure.write_html("trajectories_till_100_week_Spain_Portugal.html")

def create_datetime(row):
    year = row['Year']
    week_number = row['Week_Number']
    date = datetime.datetime.strptime(f'{year}-W{week_number}-1', '%Y-W%U-%w')
    return date

def replace_position(position):
    if position >=15:
        return "15+"
    if position >= 10:
        return "10-14"
    if position >=5:
        return "5-9"
    return "1-4"

def chart_position_change_distribution(df):
    countries = df['Country'].unique()
    main_figure = make_subplots(rows=9, cols=3, shared_yaxes='all', shared_xaxes='all', subplot_titles = [c for c in countries])
    for (idx, c) in enumerate(countries):
        result = df[df['Country'] == c]
        result.sort_values(by=['Song title', 'Song author', 'Date'], inplace=True, ascending=[False, False, False])
        result['Position_change'] = result.groupby(['Song title', 'Song author'])['Position'].diff().fillna(0).astype(int)
        year_subset = result['Position_change']
        count_changes = year_subset.value_counts().sort_index()

        fig = px.bar(x =count_changes.index, y= count_changes.values, title = c)
        fig.update_yaxes(title='Count')
        fig.update_xaxes(title='Position change')
        perecentile_25 = year_subset.quantile(0.025)
        perecentile_975 = year_subset.quantile(0.975)
        perecentile_50 = year_subset.quantile(0.5)
        # add lines for percentiles
        fig.add_vline(x = perecentile_25, line_width=2 ,line_color="grey", line_dash='dash', annotation_text = "2.5th", annotation_font_size=16)
        fig.add_vline(x =perecentile_975, line_width=2 ,line_color="grey", line_dash='dash', annotation_text = "97.5th", annotation_font_size=16)
        fig.add_vline(x= perecentile_50, line_width=2 ,line_color="red", line_dash='dash', annotation_text = "Median", annotation_font_size=16)
        # mark the percentiles 
        fig.add_vrect(x0 =count_changes.index.min() , x1 =perecentile_25, line_width =0, fillcolor='grey', opacity=0.2)
        fig.add_vrect(x0 =perecentile_975 , x1 =count_changes.index.max(), line_width =0, fillcolor='grey', opacity=0.2)
        for k in range(len(fig.data)):
            main_figure.add_trace(fig.data[k], row= (idx // 3)+1 ,col = (idx % 3)+1)
        # fig.write_image('./images/distribution_centiles/distribution_' + c + '.jpeg')
    main_figure.update_layout(height=3000)
    main_figure.update_xaxes(range=(-50, 50))
    main_figure.update_yaxes(range=(0, 11000))
    main_figure.write_html('position_change_distribution_all_countries.html')
    
def plot_histogram_of_lifetime_on_top(df, number_of_bins, years):
    data = df[df['Year'].isin(years)]
    data = data.groupby(['Country','Year', 'Song title', 'Song author']).agg(
        weeks_on_top=('Position', 'count')
    ).reset_index()
    data = data.groupby(['Year', 'Song title', 'Song author']).agg(
        mean_weeks_on_top=('weeks_on_top', 'mean')
    ).reset_index()
    result = []
    size_of_the_bin =2
    for year in years:
        for i in range(0, 64, size_of_the_bin):
            d = data[(data['Year'] == year) & (data['mean_weeks_on_top'] >= i) & (data['mean_weeks_on_top'] < i+size_of_the_bin)]['mean_weeks_on_top']
            result.append([year, len(d)/len(data[data['Year'] == year])/size_of_the_bin, i+size_of_the_bin/2])
    hist_data = pd.DataFrame(result, columns=['Year', 'Porbability_density', 'Week_number'])
    
    # fig = px.histogram(data, x = 'mean_weeks_on_top', nbins=number_of_bins, histnorm='probability density')
    # fig.show()
    fig = px.line(hist_data, x = 'Week_number', y = 'Porbability_density', color='Year', markers=True)
    fig.write_html('song_lifetime_probability_distribution_for_selected_years.html')

def trajectory_top_song_among_countries(df):
    df.drop(columns=['Week_Number', 'Artist country', 'Genre', 'Unnamed: 0'], inplace=True)
    main_figure = make_subplots(rows=2, cols=2, shared_yaxes='all', shared_xaxes='all', subplot_titles = ['Spain 2000-2010', 'Spain 2010-2020', 'Portugal 2000-2010', 'Portugal 2010-2020' ])
    countries = ['Spain', 'Portugal']

    for (i, year) in enumerate([2000, 2010]):
        for (idx,c) in enumerate(countries):
            songsTrajectory = df[df['Country'] == c]
            songsTrajectory = songsTrajectory[(songsTrajectory['Year'] > year) & (songsTrajectory['Year'] <= year+10)]
            songsTrajectory['Artist_song'] = songsTrajectory['Song author'] +  " - " + songsTrajectory['Song title']
            songsTrajectory.drop(columns=['Song title', 'Song author', 'Country'], inplace=True)
            # save all weeks that appear in chart
            # all_weeks_df = songsTrajectory[['Date']].drop_duplicates()
            desired_length = 100
            # fill the missing data with value 50 # end of chart
            songsTrajectory.sort_values(['Artist_song','Date'], inplace=True, ascending = [True, True])
            songsTrajectory['Weeks_in_chart'] = songsTrajectory.groupby('Artist_song')['Date'].rank(ascending=True, method='first').fillna(41).astype(int)
            songsTrajectory.drop(columns=['Date', 'Year', 'Month'], inplace=True)

            def extend_group(group):
                rows_to_add = desired_length - len(group)
                if rows_to_add > 0:
                    new_rows = pd.DataFrame({'Position': [41] * rows_to_add,
                                            'Artist_song': [group['Artist_song'].iloc[0]] * rows_to_add,
                                            'Weeks_in_chart': range(group['Weeks_in_chart'].max() + 1, group['Weeks_in_chart'].max() + rows_to_add + 1)})
                    return pd.concat([group, new_rows], ignore_index=True)
                elif rows_to_add < 0:
                    rows_to_remove = len(group) - desired_length
                    return group.iloc[:rows_to_remove]
                else:
                    return group

            grouped = songsTrajectory.groupby('Artist_song')
            extended_df = grouped.apply(extend_group).reset_index(drop=True)
            extended_df['Min_Position'] = extended_df.groupby('Artist_song')['Position'].transform('min')
            extended_df['Top_Group'] = extended_df['Min_Position'].apply(replace_position).reset_index(drop=True)
            result = extended_df.groupby(['Weeks_in_chart','Top_Group']).agg(
                trajectory=('Position', 'mean')
            ).reset_index()
            fig = px.line(result, x='Weeks_in_chart', y = 'trajectory', color='Top_Group', title=c)
            idx_max_trajectory = result['trajectory'].argmin()
            fig.add_vline(x= result.iloc[idx_max_trajectory, 0] , line_width=2 ,line_color="red", line_dash='dash')
            fig.update_yaxes(autorange="reversed")
            # fig.show()
        # fig.write_html('./images/trajectory/trajectory_' + c + '2.html')
            for k in range(len(fig.data)):
                main_figure.add_trace(fig.data[k], col= (idx // 2)+1+i ,row = (idx % 2)+1)
    # main_figure.update_layout(height=600)
    main_figure.update_yaxes(autorange="reversed")
    main_figure.update_xaxes(range=(0, 100))
    # main_figure.show()
    main_figure.write_html("trajectories_till_100_week_Spain_Portugal.html")

def trajectory_of_the_most_popular_songs_in_decades(df):
    decades = [ 2000, 2010]

    for d in decades:
        result, title = dataframe_with_the_most_popular_song_in_decade(df, d)
        chart_title= f'{title} decades {d}-{d+10}'
        trajectory_chart2(result, chart_title)
        
def dataframe_with_the_most_popular_song_in_decade(df, decade_start):
    df = df[(df['Year'] >= decade_start) & (df['Year'] < decade_start +10)]
    songs = df[['Song title', 'Song author']].value_counts().reset_index()
    song_title = songs.iloc[0]['Song title'] 
    song_artist = songs.iloc[0]['Song author'] 
    result = df[(df['Song title'] == song_title) & (df['Song author'] == song_artist)]
    return result, song_title + ' by ' + song_artist


def difference_when_song_appeared_and_disappeared(df):
    countries = df['Country'].unique()
    main_figure = make_subplots(rows=9, cols=3, shared_yaxes='all', shared_xaxes='all', subplot_titles = [c for c in countries])
    for (idx, c) in enumerate(countries[:1]):
        result = df[df['Country'] == c]
        result.sort_values(by=['Song title', 'Song author', 'Date'], inplace=True, ascending=[False, False, False])
        result["Days_in_chart"] = result.groupby(['Song title', 'Song author'])['Date'].transform(lambda d: (max(d) - min(d)).days)
        result['Min_Position'] = result.groupby(['Song title', 'Song author'])['Position'].transform('min')
        result = result[result['Min_Position'] == 1]
        top_10_songs = get_top_10_songs(result)
        result = result[result['Days_in_chart'].isin(top_10_songs['Days_in_chart'])]
        result['Week_Number'] = result['Week_Number'].astype(int)
        result.sort_values(by=['Song title', 'Song author', 'Week_Number'], inplace=True, ascending=[True, False, True])
        # print(result.head(30))
        # r = result.groupby(['Song title', 'Song author']).agg(
        #     appeared=('Date', 'min'),
        #     disappeared=('Date', 'max'),
        #     peak=()
        # ).reset_index()

        # r['Number_of_days'] = r.apply(calculate_days_in_chart)

        # r.sort_values(by="Number_of_days", ascending=True, inplace=True)
        # r = r.iloc[:, :10]        

        # fig =px.scatter(result, x='appeared', y=result.index, color='Song title')
        fig = px.line(result, x='Week_Number', y='Position', color='Song title')
        fig.update_yaxes(autorange="reversed")
        fig.show()

def calculate_days_in_chart(group):
    min_date = min(group['appeared'])
    max_date = max(group['disappeared'])
    return (max_date - min_date).days

def get_top_10_songs(df):
    grouped_df = df.groupby(['Song title', 'Song author'])['Days_in_chart'].mean().reset_index()

    # Sort the grouped DataFrame by the sum of 'weeks_in_chart' in descending order
    sorted_df = grouped_df.sort_values(by='Days_in_chart', ascending=False)

    # Select the top 10 songs with the highest cumulative weeks
    return sorted_df.head(5)
    

# df = preprocessing(df)
# print(df['Country'].unique())
# chart_countrychart_artistcountry(df)
# songs = df[['Song title', 'Song author']].value_counts().reset_index()
# song_title = songs.iloc[0]['Song title'] #'Blinding Lights'
# son_artist = songs.iloc[0]['Song author'] #'Weeknd'
# chart_for_one_song_in_different_countries(df, song_title, son_artist)
# histogram_songs_on_top_10(df)
# print(chart_line_average(df))
# chart_boxplot_lifetime_on_top(df)
# trajectory_chart2(df)
# plot_histogram_of_lifetime_on_top(df, 27, [2010, 2015, 2020, 2022, 2023])
# chart_position_change_distribution(df)
# trajectory_chart_chosen_countries(df)
# trajectory_of_the_most_popular_songs_in_decades(df)
# calculate_histogram_difference(df)
# difference_when_song_appeared_and_disappeared(df)