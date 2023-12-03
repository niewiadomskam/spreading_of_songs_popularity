import pandas as pd

df = pd.read_csv('./top_charts2.csv',encoding='latin-1')

songs = df[['Song title', 'Song author']].value_counts()
print(df.describe())
print(songs.head())
print(df['Song title'].nunique())
print(df['Song author'].value_counts().sort_values())
print(df['Country artist'].value_counts()[:6])
print(df['Genre'].nunique())
print(df['Genre'].value_counts())