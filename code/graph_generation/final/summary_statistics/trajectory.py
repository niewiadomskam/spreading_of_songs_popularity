import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from code.graph_generation.initial_graphs import preprocessing, replace_position


filepath = './data/Final data/all_data.csv'
df = pd.read_csv(filepath,encoding='latin-1')
df = preprocessing(df)

def calculate_trajectory_data_for_country(songsTrajectory):
    desired_length = 100
    songsTrajectory = songsTrajectory.sort_values(['Song','Date'], inplace=False, ascending = [True, True])
    songsTrajectory['Weeks_in_chart'] = songsTrajectory.groupby('Song')['Date'].rank(ascending=True, method='first').fillna(41).astype(int)
    songsTrajectory = songsTrajectory.drop(columns=['Date', 'Year', 'Month'], inplace=False)

    def extend_group(group):
        rows_to_add = desired_length - len(group)
        if rows_to_add > 0:
            new_rows = pd.DataFrame({'Position': [41] * rows_to_add,
                                    'Song': [group['Song'].iloc[0]] * rows_to_add,
                                    'Weeks_in_chart': range(group['Weeks_in_chart'].max() + 1, group['Weeks_in_chart'].max() + rows_to_add + 1)})
            return pd.concat([group, new_rows], ignore_index=True)
        elif rows_to_add < 0:
            rows_to_remove = len(group) - desired_length
            return group.iloc[:rows_to_remove]
        else:
            return group

    grouped = songsTrajectory.groupby('Song')
    extended_df = grouped.apply(extend_group).reset_index(drop=True)
    extended_df['Min_Position'] = extended_df.groupby('Song')['Position'].transform('min')
    extended_df['Top_Group'] = extended_df['Min_Position'].apply(replace_position).reset_index(drop=True)
    result = extended_df.groupby(['Weeks_in_chart','Top_Group']).agg(
        trajectory=('Position', 'mean')
    ).reset_index()
    return result


def trajectory_chart(df, title = None):
    df = df.drop(columns=['Week_Number', 'Artist country', 'Genre'], inplace=False)
    countries = df['Country'].unique()
    fig, axs = plt.subplots(nrows=9, ncols=3, sharey=True, figsize=(10,20))

    for (i,c) in enumerate(countries):
        songsTrajectory = df[df['Country'] == c]
        songsTrajectory = songsTrajectory.drop(columns=['Song title', 'Song author', 'Country'], inplace=False)
        result = calculate_trajectory_data_for_country(songsTrajectory)

        j = i // 3 # row
        i = i % 3  # col
        groups = result.groupby('Top_Group')
        for key, values in groups:
            axs[j,i].plot(values['Weeks_in_chart'],values['trajectory'], label=key)

        axs[j, i].set_xlabel("Week")
        axs[j, i].set_ylabel("Position")
        axs[j, i].set_title(c)
        axs[j,i].set_xlim([0,100])
        axs[j,i].set_ylim([42,0])

        idx_max_trajectory = result['trajectory'].idxmin()
        axs[j, i].vlines(x =result.loc[idx_max_trajectory,'Weeks_in_chart'], ymin=0, ymax=42, alpha=0.7, color="red", linestyle='--')
        axs[j, i].text(result.loc[idx_max_trajectory,'Weeks_in_chart'] + 1, 5, "Maximum trajectory", size=10, alpha = 0.8)
    
    myorder = [0,3, 1,2]
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [handles[i] for i in myorder]
    labels = [labels[i] for i in myorder]
    fig.legend(handles, labels, loc='upper right')
    fig.tight_layout()
    plt.savefig("./diagrams/final/trajectory.pdf", format="pdf")
    plt.show()

trajectory_chart(df)