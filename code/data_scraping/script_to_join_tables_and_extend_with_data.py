import asyncio
import pandas as pd
from code.data_scraping.shazam_api import get_genre
from code.data_scraping.get_artist_metadata import get_artist_country

filepath1 = 'C:\\Users\\niew\\OneDrive - Netcompany\\Documents\\private\\masters\\data scraping\\code\\data_scraping\\all_data.csv'
filepath2 = './top_charts_only_countries5.csv'

def get_genre_for_row(song_title, song_author):
    loop = asyncio.get_event_loop()
    genre = loop.run_until_complete(get_genre(song_title, song_author))
    return genre

def get_genre_from_df(df, song_title, song_author):
    genre = df.loc[(df['Song title'] == song_title) & (df['Song author'] == song_author), 'Genre']
    if genre.values.size == 0:
        return None
    return genre.values[0]

def get_country_from_df(df, song_title, song_author):
    country = df.loc[(df['Song title'] == song_title) & (df['Song author'] == song_author), 'Artist country']
    if country.values.size == 0:
        return None
    return country.values[0]

df1 = pd.read_csv(filepath1,encoding='latin-1')
# df2 = pd.read_csv(filepath2,encoding='latin-1')

# df = pd.concat([df1, df2], ignore_index=True).reset_index()
df = df1.drop_duplicates()



songs =df.drop_duplicates(['Song title','Song author'])[['Song title','Song author']]
# songs['Genre'] = songs.apply(lambda row: get_genre_for_row(row[0], row[1]), axis=1)
songs['Artist country'] = songs.apply(lambda row: get_artist_country(row[1]), axis=1)
songs.to_csv('songs_country2.csv')

# filepath1 = './songs_country.csv'
# filepath2 = './songs_genre.csv'

# df1 = pd.read_csv(filepath1,encoding='latin-1')
# df2 = pd.read_csv(filepath2,encoding='latin-1')

# df_ex = pd.merge(df1, df2, on=['Song title', 'Song author'], how='outer').reset_index()
# df_ex = df_ex.drop_duplicates()

# df['Artist country'] = df.apply(lambda row: get_country_from_df(df_ex, row[4], row[5]), axis=1) #df_ex.loc[(df_ex['Song title'] == row[4]) & (df_ex['Song author'] == row[5]), 'Genre'].values[0], axis=1)
# print(df_ex.info())
# df.to_csv('all_with_country.csv')
# print(df_ex["Genre"][(df_ex['Song title'] == 'Dynamite') & (df_ex['Song author'] == 'BTS')])

# filepath1 = 'all_with_country.csv'
# filepath2 = 'all_with_genre.csv'

# df1 = pd.read_csv(filepath1,encoding='latin-1', index_col='index')
# df2 = pd.read_csv(filepath2,encoding='latin-1', index_col='index')
# # df1.drop(['index'], axis=1, inplace=True)
# # df2.drop('index')

# df = pd.merge(df1, df2, left_on=['Country', 'Position', 'Date', 'Song title', 'Song author'], right_on=['Country', 'Position', 'Date', 'Song title', 'Song author'])
# df.drop('Unnamed: 0_x', axis=1, inplace=True)
# df.drop('Unnamed: 0_y', axis=1, inplace=True)
# print(df.head())
# print(df.columns)
# df.to_csv('./all_data.csv')