import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import asyncio
import concurrent.futures


from code.graph_generation.initial_graphs import preprocessing
from code.graph_generation.entropy.ste_vectorized import _get_symbol_sequence, symbolic_TE
from code.graph_generation.entropy.generate_dict import load_dict_as_vectorized

filepath = './data/Final data/all_data.csv'
df = pd.read_csv(filepath,encoding='latin-1')
df = preprocessing(df)

def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped

def transform_positions(position):
    if position > 10 : 
        return 4
    if position > 7:
        return 3
    if position > 3 :
        return 2
    return 1

def get_date_position_for_song(country1_df, country2_df):
    merged_df = pd.merge(country1_df, country2_df, on='Week_Number_Year', how='outer', suffixes=('_Country1', '_Country2'))
    merged_df['Position_Country1'] = merged_df['Position_Country1'].fillna(41)
    merged_df['Position_Country2'] = merged_df['Position_Country2'].fillna(41)
    merged_df = merged_df.loc[:, ['Week_Number_Year', 'Position_Country1', 'Position_Country2']]
    merged_df[['Week_Number', 'Year']] = merged_df['Week_Number_Year'].str.split('_', expand=True)

    merged_df['Week_Number'] = pd.to_numeric(merged_df['Week_Number'])
    merged_df['Year'] = pd.to_numeric(merged_df['Year'])
    merged_df = merged_df.sort_values(by=['Year','Week_Number'], ascending=[True, True])
    return merged_df


def get_dataframe_with_transformed_positions_for_countries_and_song(df, c1, c2, song):
    positions = df[(df['Country'] == c1) & (df['Song'] == song)]
    positions_uk = df[(df['Country'] == c2) & (df['Song'] == song)]
    res = get_date_position_for_song(positions, positions_uk)
    res['Transform_Position_Country1'] = res['Position_Country1'].apply(transform_positions)
    res['Transform_Position_Country2'] = res['Position_Country2'].apply(transform_positions)

    return res

def load_initial_data(df):
    df['Week_Number_Year']  = df['Week_Number'].astype(str) + "_" +df['Year'].astype(str)
    filepath = "correct_extended_countries_with_entropy_all.csv"
    entropy_df = pd.read_csv(filepath,encoding='latin-1')
    entropy_df['Entropy'] = pd.to_numeric(entropy_df['Entropy'], errors='coerce')

    return df, entropy_df


def test_random(df):
    df['Week_Number_Year']  = df['Week_Number'].astype(str) + "_" +df['Year'].astype(str)
    filepath = "correct_extended_countries_with_entropy_all.csv"
    entropy_df = pd.read_csv(filepath,encoding='latin-1')
    entropy_df['Entropy'] = pd.to_numeric(entropy_df['Entropy'], errors='coerce')
    entropy_df = entropy_df[entropy_df['Country1'] == 'USA'].reset_index()
    idx_max = 300 # entropy_df['Entropy'].idxmin()
    bigg = entropy_df.iloc[idx_max]
    positions = df[(df['Country'] == 'USA') & (df['Song'] == bigg['Song'])]
    positions_uk = df[(df['Country'] == bigg['Country2']) & (df['Song'] == bigg['Song'])]
    res = get_date_position_for_song(positions, positions_uk)
    res['Transform_Position_Country1'] = res['Position_Country1'].apply(transform_positions)
    res['Transform_Position_Country2'] = res['Position_Country2'].apply(transform_positions)
    vectorized_lookup_dict = load_dict_as_vectorized()

    e1 = symbolic_TE(res['Transform_Position_Country1'], res['Transform_Position_Country2'], 1, 3, vectorized_lookup_dict)
    e2 = symbolic_TE(res['Transform_Position_Country2'], res['Transform_Position_Country1'], 1, 3, vectorized_lookup_dict)
    de = e1-e2

    # calculate deltas
    deltas= calculate_deltas_for_random_permutations(res, vectorized_lookup_dict)
    # plot deltas
    plot_histogram_for_perutations(deltas, de)


def test_time(df):
    start = time.time()
    file_name="entropies_delta_all_countries.csv"
    print('start', start)
    df, entropy_df = load_initial_data(df)
    vectorized_lookup_dict = load_dict_as_vectorized()
    countries = df['Country'].unique() #["UK", "USA", "Ireland", "Denmark", "Belgium"]

    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["Country1", "Country2", "Song", "Delta_entropy"])

        for i, c1 in enumerate(countries): 
            for j, c2 in enumerate(countries):
                if i <=j :
                    continue
                songs = entropy_df[(entropy_df['Country1'] ==c1) & (entropy_df['Country2'] == c2)]['Song'].unique()
                # loop = asyncio.get_event_loop()  
                # looper = asyncio.gather(*[my_async_loop(df, c1, c2, song, vectorized_lookup_dict) for song in songs])         # Run the loop                               
                # results = loop.run_until_complete(looper)
                for song in songs:
                    try:
                        results = my_async_loop(df, c1, c2, song, vectorized_lookup_dict)
                        if results:
                            writer.writerow(results) 
                    except:
                        print("error")

    end = time.time()
    print(end - start)

# @background
def my_async_loop(df, c1,c,song, vectorized_lookup_dict):
    res = get_dataframe_with_transformed_positions_for_countries_and_song(df, c1, c, song)
    if len(res) < 3 :
        return
    e1 = symbolic_TE(res['Transform_Position_Country1'], res['Transform_Position_Country2'], 1, 3, vectorized_lookup_dict)
    e2 = symbolic_TE(res['Transform_Position_Country2'], res['Transform_Position_Country1'], 1, 3, vectorized_lookup_dict)
    de = e1-e2

    # calculate deltas
    deltas= calculate_deltas_for_random_permutations(res, vectorized_lookup_dict)
    result = decide_if_actual_delta_is_significant(deltas, de)
    return [c1, c, song, result]



def calculate_deltas_for_random_permutations(res, vectorized_lookup_dict):
    res['Transform_Position_Country1'] = res['Position_Country1'].apply(transform_positions)
    res['Transform_Position_Country2'] = res['Position_Country2'].apply(transform_positions)
    e = []

    for i in range(0, 100):
        random_per_1 = np.random.permutation(res['Transform_Position_Country1'].values)
        random_per_2 = np.random.permutation(res['Transform_Position_Country2'].values)
        e_random_1 = symbolic_TE(random_per_1, random_per_2, 1, 3, vectorized_lookup_dict)        
        e_random_2 = symbolic_TE(random_per_2, random_per_1, 1, 3, vectorized_lookup_dict)
        de_random = e_random_1 - e_random_2
        e.append(de_random)

    return e

def decide_if_actual_delta_is_significant(deltas, de):
    deltas = pd.Series(deltas)
    perecentile_25 = deltas.quantile(0.025)
    perecentile_975 = deltas.quantile(0.975)
    if de < perecentile_25 :
        return -1
    if de > perecentile_975 :
        return 1
    return 0



def plot_histogram_for_perutations(deltas, de):
    deltas = pd.DataFrame(deltas)
    perecentile_25 = deltas.quantile(0.025)
    perecentile_975 = deltas.quantile(0.975)
    perecentile_50 = deltas.quantile(0.5)
       

    fig, ax = plt.subplots()

    # the histogram of the data
    n, bins, patches = ax.hist(deltas, 10)

    ax.axvline(de, color='red')
    ax.text(de + 0.01, 2, f"actual delta: {de: .3f} ", size=10, alpha = 0.8, color='red')

     # add lines for percentiles
    ax.vlines(x = perecentile_25, ymin=0, ymax=20, alpha=0.7, color="grey", linestyle='--')
    ax.vlines(x =perecentile_975, ymin=0, ymax=20, alpha=0.7, color="grey", linestyle='--')

    ax.set_xlabel("Entropy values")
    ax.set_ylabel("Count")
    # plt.savefig("./diagrams/entropy/entropy_test_4_groups.pdf", format="pdf")
    plt.show()

def concurrent_test_loop(df):
    start = time.time()
    file_name="entropies_delta_all_countries.csv"
    print('start', start)
    df, entropy_df = load_initial_data(df)
    vectorized_lookup_dict = load_dict_as_vectorized()
    countries = df['Country'].unique() #["UK", "USA", "Ireland", "Denmark", "Belgium"]

    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["Country1", "Country2", "Song", "Delta_entropy"])

        for i, c1 in enumerate(countries): 
            for j, c2 in enumerate(countries):
                if i <=j :
                    continue
                songs = entropy_df[(entropy_df['Country1'] ==c1) & (entropy_df['Country2'] == c2)]['Song'].unique()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Start the operations and mark each future with its parameter
                    future_to_param = {executor.submit(my_async_loop,(df, c1,c2,song, vectorized_lookup_dict)): (df, c1,c2,song, vectorized_lookup_dict) for song in songs}
                    for future in concurrent.futures.as_completed(future_to_param):
                        param = future_to_param[future]
                        try:
                            result = future.result()
                            if(result):
                                writer.writerows(result)
                        except Exception as exc:
                            print(f"{param} generated an exception: {exc}")


    end = time.time()
    print(end - start)

test_time(df)
