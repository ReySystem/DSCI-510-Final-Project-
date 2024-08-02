import os
import pandas as pd
import numpy as np
from functools import reduce
from sklearn.preprocessing import LabelEncoder
from src.visualize_results import get_bar, get_line_chart, \
    heatmap, get_pie, get_dotted_bar, get_scatter

# Create a directory to save the HTML file
plot_directory_path = 'results/plots/analysis_plots'
os.makedirs(plot_directory_path, exist_ok=True)


def get_team_standings(df):
    """
    function to analyze the team standings on the group stages
    :param df: points table csv
    :return: None
    """

    # get a sub dataframe for dataframe to analyze relevant columns
    sub_df_points = df[['Teams', 'Won', 'Qualification_Status']]

    # plot bar chart to analyse the points obtained by each team
    # and their qualification status
    get_bar(sub_df_points, 'Teams', 'Won', 'Matches won by Teams',
            {'Q': 'limegreen', 'E': 'greenyellow'}, 'Teams',
            'matches_won', 'teams_matches_plot', color='Qualification_Status')


def get_columnwise_records(df):
    """
    function to return all records of a column
    :param df: dataframe
    :return: list of records for a column
    """

    df = pd.DataFrame(df)

    # Convert each column to a separate list
    column_lists = {column: df[column].tolist() for column in df.columns}

    return column_lists


def get_overall_team_stats(df_match_summary, df_points):
    """
    function to analyze average runs per wicket and
    average wickets lost per match for all teams
    :param df_match_summary: match summary csv
    :param df_points: points table csv
    :return: None
    """

    # Include only the group stage matches
    df_match_summary = df_match_summary[:-3]

    # make a sub dataframe with relevant columns for analysis
    team_runs_per_wicket_df_team = df_match_summary[
        ['Team 1', 'Team 1 Runs Scored', 'Team 1 Wickets Lost',
         'Team 2', 'Team 2 Runs Scored', 'Team 2 Wickets Lost']]

    # get column records in the form of dict with key as Column name
    # and value as list of team names
    dict_column_records = get_columnwise_records(team_runs_per_wicket_df_team)

    # Combine stats of Team 1 and Team 2
    teams, runs, wickets = [[] for _ in range(3)]

    for k, v in dict_column_records.items():
        if k == 'Team 1':
            teams = v + dict_column_records['Team 2']
        elif k == 'Team 1 Runs Scored':
            runs = v + dict_column_records['Team 2 Runs Scored']
        elif k == 'Team 1 Wickets Lost':
            wickets = v + dict_column_records['Team 2 Wickets Lost']

    # make a df with new columns
    df_team_stats = pd.DataFrame({'Teams': teams, 'runs': runs,
                                  'wickets': wickets})

    # Calculate average runs per wicket for each team
    result_df_avg_runs = df_team_stats.groupby('Teams').apply(
        lambda x: (x['runs'].sum()) / (x['wickets'].sum())). \
        reset_index(name='Average Runs per Wicket')

    # Sort the result DataFrame in descending order
    result_df_avg_runs = \
        result_df_avg_runs.sort_values(by='Average Runs per Wicket',
                                       ascending=False)

    # merge qualification status on the existing dataframe
    # Merge dataframes on the 'Team' column
    final_avg_runs_per_wicket_df = \
        pd.merge(result_df_avg_runs, df_points, on='Teams')

    # plot bar chart to analyse the average runs scored
    # by each team and their qualification status
    get_bar(final_avg_runs_per_wicket_df, 'Teams', 'Average Runs per Wicket',
            'Average Runs per Wicket for all teams',
            {'Q': 'limegreen', 'E': 'greenyellow'}, 'Team Name',
            'Average Runs', 'teams_matches_average_runs_plot')

    # average_wickets_lost
    result_df_avg_wickets = df_team_stats.groupby('Teams').apply(
        lambda x: (x['wickets'].sum()) / len(x['wickets'])). \
        reset_index(name='Average Wickets lost')

    # Sort the result DataFrame in descending order
    result_df_avg_wickets = result_df_avg_wickets. \
        sort_values(by='Average Wickets lost', ascending=False).reset_index(
                    drop=True)

    # merge qualification status on the existing dataframe
    # Merge dataframes on the 'Team' column
    final_result_df_avg_wickets = \
        pd.merge(result_df_avg_wickets, df_points, on='Teams')

    # plot bar chart to analyse the average wickets lost by
    # each team and their qualification status
    get_bar(final_result_df_avg_wickets, 'Teams', 'Average Wickets lost',
            'Average Wickets lost per match all teams',
            {'Q': 'limegreen', 'E': 'greenyellow'}, 'Team Name',
            'Wickets Lost', 'teams_matches_average_wickets_plot')

    # find out the correlation of the overall team stats

    # Dropping the Total matches, Pts, NRR column
    # as it's irrelevant for this analysis
    # Dropping lost column as winning and lost are complement of each other
    final_avg_runs_per_wicket_df. \
        drop(['Mat', 'Lost', 'Pts', 'NRR'], axis=1, inplace=True)

    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Merge the avg runs per wicket df and avg wickets lost per match
    overall_team_stats_df = \
        pd.merge(final_avg_runs_per_wicket_df, result_df_avg_wickets,
                 on='Teams')

    # Fit and transform the Qualification_Status column
    overall_team_stats_df['Qualification_Status'] = \
        label_encoder.fit_transform(
            overall_team_stats_df['Qualification_Status'])

    # Generate a correlation heatmap
    heatmap(overall_team_stats_df,
            'overall_team_stats_heatmap', 'Correlation of Team statistics')


def get_teams_matchwise_trend(df_match_summary):
    """
    function to analyze the match by match performance of teams
    :param df_match_summary: match summary csv
    :return: None
    """

    # get a sub dataframe for dataframe to analyze relevant columns
    sub_df_match_summary = df_match_summary[['Team 1', 'Team 2', 'Winner']]

    # Create a new column for the team that lost
    sub_df_match_summary['Team Lost'] = sub_df_match_summary.apply(
        lambda record: record['Team 1']
        if record['Winner'] != record['Team 1'] else record['Team 2'], axis=1)
    df_team_won_lost = sub_df_match_summary[['Winner', 'Team Lost']]

    # Exclude the last three records (only considering group stages
    # for this analysis)
    df_team_won_lost = df_team_won_lost[:-3]

    # Create a dictionary with keys as team names and
    # values as lists containing 'Win' or 'Lose'
    team_results = {}

    for index, row in df_team_won_lost.iterrows():
        winner_team = row['Winner']
        team_lost = row['Team Lost']

        if winner_team not in team_results:
            team_results[winner_team] = ['Win']
        else:
            team_results[winner_team].append('Win')

        if team_lost not in team_results:
            team_results[team_lost] = ['Lose']
        else:
            team_results[team_lost].append('Lose')

    # Replace 'Win' with 2 and 'Lose' with 0
    # (0 and 2 are the points obtained on losing and winning respectively)
    team_results_points = {team: [2 if result == 'Win' else 0 for result
                                  in results] for team, results in
                           team_results.items()}

    # Generate cumulative list for each team
    team_results_cumulative = {team: np.cumsum(results) for
                               team, results in team_results_points.items()}

    # Convert the dictionary to a DataFrame
    df_team_matchwise_points = pd.DataFrame(
        [(team, match + 1, points) for team, results
         in team_results_cumulative.items() for match, points in
         enumerate(results)],
        columns=['Team', 'Match', 'Points'])

    get_line_chart(df_team_matchwise_points, 'Match', 'Points',
                   'Team matchwise winning trends', 'matchwise_points',
                   'Team')


def get_first_second_innings_stats(df_match_summary):
    """
    function to generate general statistics of first and second innings
    :param df_match_summary: match summary csv
    :return: first innings wins df, second innings wins df
    """
    sub_first_innings_df = df_match_summary[
                               ['Team 1', 'Team 2', 'Team 1 Runs Scored',
                                'Team 1 Overs', 'Team 1 Wickets Lost',
                                'Winner']][:-3]

    # Overall 1st and 2nd batting trend (Win and losing)
    tournament_1st_2nd = sub_first_innings_df[['Team 1', 'Team 2', 'Winner']]

    first_bat_win = tournament_1st_2nd[tournament_1st_2nd['Team 1']
                                       == tournament_1st_2nd['Winner']]
    second_bat_win = tournament_1st_2nd[tournament_1st_2nd['Team 2']
                                        == tournament_1st_2nd['Winner']]

    overall_data = {
        'Batting': ['1st Batting', '2nd Batting'],
        'Matches_Won': [first_bat_win.shape[0], second_bat_win.shape[0]]
    }

    overall_data = pd.DataFrame(overall_data)

    # plot the pie chart
    get_pie(overall_data, 'Batting',
            'Matches Won Batting 1st v/s 2nd', 'matches_won_batting_1_2')

    # get the times teams have batted first
    first_batting_count = sub_first_innings_df['Team 1'].value_counts()

    # get the total number of times teams have batted second
    second_batting_count = list(map(lambda x: 9 - x,
                                    list(sub_first_innings_df['Team 1'].
                                         value_counts())))

    batting_times_df = pd.DataFrame(list(first_batting_count.to_dict()
                                         .items()),
                                    columns=['Team', '1st Batting Count'])
    batting_times_df['2nd Batting Count'] = second_batting_count

    # Melt the DataFrame to reshape it for side-by-side bar chart
    df_melted = pd.melt(batting_times_df, id_vars='Team',
                        var_name='Batting Type', value_name='Count')

    # plot the bar chart to show total number
    # of time teams have batted first v/s second
    get_bar(df_melted, 'Team', 'Count',
            'Total count of teams batted 1st v/s 2nd',
            {'Q': 'limegreen', 'E': 'greenyellow'},
            'Team Name', 'Total Count',
            '1st_2nd_batting_counts', color='Batting Type', barmode='group')

    # Wins batting 1st and batting 2nd
    # Create a new column 'Winning_Team' based on the condition
    sub_first_innings_df['Wins Batting First'] = (
            sub_first_innings_df['Team 1'] ==
            sub_first_innings_df['Winner']).astype(int)

    # assigning a copy of the sub dataframe
    first_batting_df = sub_first_innings_df.copy()

    # Rename the 'Team' column to 'Team 1'
    first_batting_df = first_batting_df.rename(columns={'Team 1': 'Team'})

    # get value counts for each team
    total_counts_first = \
        first_batting_df.groupby('Team')['Wins Batting First']. \
        sum().reset_index()

    first_in_wins = pd.merge(batting_times_df, total_counts_first, on='Team')

    first_in_wins.drop('2nd Batting Count', axis=1, inplace=True)

    # Melt the DataFrame to reshape it for side-by-side bar chart
    df_melted_first_bat_wins = pd.melt(first_in_wins,
                                       id_vars='Team',
                                       var_name='Batting First',
                                       value_name='Count')

    # plot the bar chart to show total number of
    # time teams have batted first v/s second
    get_bar(df_melted_first_bat_wins, 'Team', 'Count',
            'Teams Batting 1st Count and Total wins batting 1st',
            {'Q': 'limegreen', 'E': 'greenyellow'}, 'Team Name',
            'Total Count', 'First_Batting_Wins',
            color='Batting First', barmode='group')

    # Create a new column 'Winning_Team' based on the condition

    sub_first_innings_df['Wins Batting Second'] = (
            sub_first_innings_df['Team 2'] ==
            sub_first_innings_df['Winner']).astype(int)

    # assigning a copy of the sub dataframe
    second_batting_df = sub_first_innings_df.copy()

    # Rename the 'Team' column to 'Team 1'
    second_batting_df = second_batting_df.rename(columns={'Team 2': 'Team'})

    # get value counts for each team
    total_counts_second = \
        second_batting_df.groupby('Team')['Wins Batting Second'] \
        .sum().reset_index()

    second_in_wins = pd.merge(batting_times_df, total_counts_second,
                              on='Team')
    second_in_wins.drop('1st Batting Count', axis=1, inplace=True)

    # Melt the DataFrame to reshape it for side-by-side bar chart
    df_melted = pd.melt(second_in_wins, id_vars='Team',
                        var_name='Batting Second Wins', value_name='Count')

    # plot the bar chart to show total number of time teams have
    # batted first v/s second
    get_bar(df_melted, 'Team', 'Count',
            'Teams Batting 2nd Count and Total wins batting 2nd',
            {'Q': 'limegreen', 'E': 'greenyellow'},
            'Team Name', 'Total Count',
            'Second_batting_Wins', color='Batting Second Wins',
            barmode='group')

    # frequency of teams getting bowled out before 50 overs

    # Filter the DataFrame based on the condition
    bowled_out_df = sub_first_innings_df[
        (sub_first_innings_df['Team 1 Overs'] < 50) &
        (sub_first_innings_df['Team 1 Wickets Lost'] == 10)]

    # Count occurrences and handle teams not bowled out
    team_counts = bowled_out_df['Team 1'].value_counts(). \
        reindex(df_match_summary['Team 1'].unique(), fill_value=0)

    # Sort counts in descending order
    team_counts = team_counts.sort_values(ascending=False)

    bowled_out_df = pd.DataFrame(list(
        team_counts.to_dict().items()), columns=['Team', 'Bowled Out Times'])

    # plot the bar chart to show total number of time teams
    # have gotten bowled out
    get_bar(bowled_out_df, 'Team', 'Bowled Out Times',
            'Frequency of Teams getting '
            'bowled out before playing 50 overs (1st innings)',
            None, 'Team Name',
            'Bowled out count', 'Bowled_out_frequency')

    # %age of teams winning and lossing getting bowled out
    # for less than 300 in lesser than 50 overs

    # getting a df copy
    bowled_match_winning_df = sub_first_innings_df.copy()

    # Filter teams with overs less than 50, runs less than 300
    # and lost 10 wickets
    # (to remove the ambiguity of playing
    # less overs due to DLS)
    bowled_match_winning_df = bowled_match_winning_df[
        (bowled_match_winning_df['Team 1 Overs'] < 50)
        & (bowled_match_winning_df['Team 1 Wickets Lost'] == 10) & (
                bowled_match_winning_df['Team 1 Runs Scored'] < 300)]

    # Create a new column 'Winning Team' with 1 if 'Team'
    # equals 'Winner', else 0
    bowled_match_winning_df['Winning Team'] = np.where(
        bowled_match_winning_df['Team 1'] ==
        bowled_match_winning_df['Winner'], 1, 0)

    # Map values in 'Winning Team' column to labels
    bowled_match_winning_df['Result'] = \
        bowled_match_winning_df['Winning Team']. \
        map({0: 'Losing', 1: 'Winning'})

    # plot the pie chart
    get_pie(bowled_match_winning_df, 'Result',
            'Winning and Losing %age of teams getting '
            'bowled out less than 300 before playing 50 overs',
            'less_than_300_less_than_50')

    bowled_teams_before_df = bowled_match_winning_df.copy()

    bowled_teams_before_df = bowled_teams_before_df[['Team 1', 'Winner']]

    # Get total value counts where 'Team 1' is equal to 'Winner'
    total_lost_counts = bowled_teams_before_df[
        bowled_teams_before_df['Team 1'] !=
        bowled_teams_before_df['Winner']].groupby('Team 1') \
        .size().reset_index(name='Total Losses')

    # Display the total value counts
    total_lost_counts = total_lost_counts. \
        sort_values(by='Total Losses', ascending=False)

    # plot the bar chart to show total number of time
    # teams have lost after getting bowled out
    # for less than 300 before 50 overs
    get_bar(total_lost_counts, 'Team 1', 'Total Losses',
            'Teams losing after getting bowled out before '
            'playing 50 overs and scoring less than 300 (1st innings)',
            None, 'Team Name',
            'Losses count', 'Teams_Bowled_out_losses')

    # %age of teams winning after scoring 300 in th first innings

    # getting a df copy
    batting_match_winning_df = sub_first_innings_df.copy()

    # Filter teams with over 3000 runs in first innings
    batting_match_winning_df = batting_match_winning_df[
        (batting_match_winning_df['Team 1 Runs Scored'] >= 300)]

    # Create a new column 'Winning Team' with 1
    # if 'Team' equals 'Winner', else 0
    batting_match_winning_df['Winning Team'] = np.where(
        batting_match_winning_df['Team 1'] ==
        batting_match_winning_df['Winner'], 1, 0)

    # Map values in 'Winning Team' column to labels
    batting_match_winning_df['Result'] = \
        batting_match_winning_df['Winning Team'].map(
            {0: 'Losing', 1: 'Winning'})

    # plot the pie chart
    get_pie(batting_match_winning_df, 'Result',
            'Winning and Losing %age of teams scoring over 300 (1st innings)',
            'over_300')

    batting_300_df = batting_match_winning_df.copy()

    batting_300_df = batting_300_df[['Team 1', 'Winner']]

    # Get total value counts where 'Team 1' is equal to 'Winner'
    total_lost_counts = batting_300_df[batting_300_df['Team 1'] ==
                                       batting_300_df['Winner']].groupby(
        'Team 1').size().reset_index(name='Total Wins')

    # Display the total value counts
    total_lost_counts = \
        total_lost_counts.sort_values(by='Total Wins', ascending=False)

    # plot the bar chart to show total number of time teams
    # have lost after getting bowled out for less than 300 before 50 overs
    get_bar(total_lost_counts, 'Team 1', 'Total Wins',
            'Teams winning after scoring over 300 (1st innings)',
            None, 'Team Name',
            'Win count', 'Teams_over_300_wins')

    return first_in_wins, second_in_wins


def first_innings_detailed_stats(df_match_summary, df_first_inning_wins,
                                 df_second_inning_wins,
                                 df_batting_scorecards,
                                 df_bowling_scorecards):
    """
    function to get detailed first innings stats
    :param df_match_summary: match summary df
    :param df_first_inning_wins: first innings wins df
    :param df_second_inning_wins: second innings wins df
    :param df_batting_scorecards: batting scorecard df
    :param df_bowling_scorecards: bowling scorecard df
    :return: None
    """

    # 1st innings batting stats

    # calculate average run rate for all teams batting 1st

    # exclude last 3 groups
    sub_first_innings_df = df_match_summary[
                               ['Team 1',
                                'Team 1 Runs Scored', 'Team 1 Overs',
                                'Team 1 Wickets Lost', 'Winner']][:-3]

    # Groupby team name
    first_team_totals = sub_first_innings_df.groupby(
        'Team 1', as_index=False).agg(
        {'Team 1 Runs Scored': 'sum',
         'Team 1 Wickets Lost': 'sum', 'Team 1 Overs': 'sum'})

    # Calculate run rate for each row
    first_team_totals['Run Rate'] = ((first_team_totals['Team 1 Runs Scored'] /
                                      first_team_totals['Team 1 Overs']).
                                     round(2))

    # Calculate the average run rate
    average_run_rate = first_team_totals['Run Rate'].mean().round(2)

    # plot bar plot for average 1st innings run rate for all teams
    get_dotted_bar(first_team_totals, 'Team 1', 'Run Rate',
                   'Average Run Rate for all teams (Batting 1st)',
                   'Team Name', 'Run Rate',
                   'first_innings_detailed_stats_1', average_run_rate)

    # average wickets lost while batting in first innings

    # Calculate total wickets lost and total matches played for each team
    team_first_wickets = sub_first_innings_df.groupby('Team 1').agg(
        {'Team 1 Wickets Lost': 'sum', 'Team 1': 'count'}
    ).rename(columns={'Team 1': 'Matches Played'})

    # Calculate average wickets lost
    team_first_wickets['Average Wickets Lost'] = \
        team_first_wickets['Team 1 Wickets Lost'] / team_first_wickets[
            'Matches Played']

    # Calculate the product of Matches Played and Average Wickets Lost
    team_first_wickets['Weighted Wickets Lost'] = \
        team_first_wickets['Matches Played'] * team_first_wickets[
            'Average Wickets Lost']

    # Calculate the total matches played and the weighted average wickets lost
    total_matches = team_first_wickets['Matches Played'].sum()
    weighted_average_wickets_lost_all_teams = \
        team_first_wickets['Weighted Wickets Lost'].sum() / total_matches

    # drop the index column
    team_first_wickets.reset_index(inplace=True)

    # plot bar plot for average 1st innings wickets lost by all teams
    get_dotted_bar(team_first_wickets, 'Team 1',
                   'Average Wickets Lost',
                   '1st innings all teams average wickets lost',
                   'Team Name', 'Wickets',
                   'first_innings_detailed_stats_2',
                   weighted_average_wickets_lost_all_teams)

    # Calculate Phasewise powerplay runs scored

    # powerplay phase-by-phase runs and wickets (only group stage matches)
    first_innings_pp = df_match_summary[
                           ['Team 1', 'Team 1 Overs',
                            'Team 1 PP-1 Runs Scored',
                            'Team 1 PP-1 Wickets Lost',
                            'Team 1 PP-2 Runs Scored',
                            'Team 1 PP-2 Wickets Lost',
                            'Team 1 PP-3 Runs Scored',
                            'Team 1 PP-3 Wickets Lost',
                            'Winner']][:-3]

    # Calculate run rate for each row pp-1
    first_innings_pp['PP-1 Run Rate'] = \
        (first_innings_pp['Team 1 PP-1 Runs Scored'] / 10).round(2)

    # pp-1 was different for the rain curtailed match
    # (Match 15 Netherlands v/s South Africa. Only 9 overs for the pp-1)
    first_innings_pp.iloc[
        14, first_innings_pp.columns.get_loc('PP-1 Run Rate')] = (
            (first_innings_pp.iloc[14]['PP-1 Run Rate'] * 10) / 9).round(2)

    # Group by 'Team' and calculate the mean of 'RunRate' in pp-1
    first_innings_pp1_runs = first_innings_pp.groupby(
        'Team 1')['PP-1 Run Rate'].mean().round(2).reset_index()

    # Calculate run rate for each row pp-2

    # calculate pp-2 overs played and pp-3 Overs played
    # (teams getting bowled out)
    first_innings_pp['Overs without pp-1'] = (
            first_innings_pp['Team 1 Overs'] - 10).round(1)
    pp2_pp3_overs_list = first_innings_pp['Overs without pp-1'].to_list()

    pp2_overs, pp3_overs = [[] for _ in range(2)]

    for overs in pp2_pp3_overs_list:
        if overs == 40.0:
            pp2_overs.append(30.0)
            pp3_overs.append(10.0)
        elif overs < 30.0:
            pp2_overs.append(overs)
            pp3_overs.append(0)
        elif overs < 40.0:
            pp2_overs.append(30.0)
            pp3_overs.append(round((overs - 30.0), 1))

    # adding pp-2 and pp-3 overs played
    first_innings_pp['PP-2 Overs'] = pp2_overs
    first_innings_pp['PP-3 Overs'] = pp3_overs

    # Calculate run rate for each row pp-2
    first_innings_pp['PP-2 Run Rate'] = (
            first_innings_pp['Team 1 PP-2 Runs Scored'] /
            first_innings_pp['PP-2 Overs']).round(2)

    # Calculate run rate for each row pp-3
    first_innings_pp['PP-3 Run Rate'] = (
            first_innings_pp['Team 1 PP-3 Runs Scored'] /
            first_innings_pp['PP-3 Overs']).round(2)

    # pp-2 was different for the rain curtailed match
    # (Match 15 Netherlands v/s South Africa. Only 26 overs for the pp-2)
    first_innings_pp.iloc[14, first_innings_pp.columns.get_loc(
        'PP-2 Run Rate')] = (
            (first_innings_pp.iloc[14]['PP-2 Run Rate'] * 30) / 26
    ).round(2)

    # Note: - pp-3 run rate has some Nan values
    # due to teams getting bowled out

    # Group by 'Team' and calculate the mean of 'RunRate' in pp-2
    first_innings_pp2_runs = first_innings_pp.groupby('Team 1')[
        'PP-2 Run Rate'].mean().round(2).reset_index()

    # Convert 'RunRate' to numeric to handle NaN values
    first_innings_pp['PP-3 Run Rate'] = pd.to_numeric(
        first_innings_pp['PP-3 Run Rate'], errors='coerce')

    # pp-3 was different for the rain curtailed match
    # (Match 15 Netherlands v/s South Africa. Only 8 overs for the pp-3)
    first_innings_pp.iloc[14, first_innings_pp.columns.get_loc(
        'PP-3 Run Rate')] = (
            (first_innings_pp.iloc[14]['PP-3 Run Rate'] * 3) / 8
    ).round(2)

    # Group by 'Team' and calculate the mean of 'RunRate' in pp-3
    first_innings_pp3_runs = first_innings_pp.groupby('Team 1')[
        'PP-3 Run Rate'].mean().round(2).reset_index()

    # merging the dataframes
    df_temp = pd.merge(first_innings_pp1_runs, first_innings_pp2_runs,
                       on='Team 1')
    powerplay_run_rates_innings_1 = pd.merge(df_temp, first_innings_pp3_runs,
                                             on='Team 1')

    # Melt the DataFrame to reshape it for side-by-side bar chart
    df_melted_runs = pd.melt(powerplay_run_rates_innings_1, id_vars='Team 1',
                             var_name='Powerplay',
                             value_name='RunRate')

    # plot bar chart to analyse the average runs rate
    # for all teams in different phases
    get_bar(df_melted_runs, 'Team 1', 'RunRate',
            '1st innings run rate for all teams in different phases',
            {'Q': 'limegreen', 'E': 'greenyellow'},
            'Team Name', 'Run rate',
            'first_innings_detailed_stats_3', color='Powerplay',
            barmode='group')

    # Calculate Phasewise powerplay wickets lost

    # pp-1 wickets lost
    first_innings_pp1_wickets = first_innings_pp.groupby('Team 1')[
        'Team 1 PP-1 Wickets Lost'].mean().reset_index()

    # pp-2 wickets lost
    first_innings_pp2_wickets = first_innings_pp.groupby('Team 1')[
        'Team 1 PP-2 Wickets Lost'].mean().reset_index()

    # pp-3 wickets lost
    first_innings_pp3_wickets = first_innings_pp.groupby('Team 1')[
        'Team 1 PP-3 Wickets Lost'].mean().reset_index()

    # plot bar chart to analyse the wickets lost by all
    # teams in different phases

    first_innings_pp1_wickets_nc = first_innings_pp1_wickets.copy()
    first_innings_pp2_wickets_nc = first_innings_pp2_wickets.copy()
    first_innings_pp3_wickets_nc = first_innings_pp3_wickets.copy()
    first_innings_pp1_wickets_nc.rename(columns={
        'Team 1 PP-1 Wickets Lost': 'PP-1 Wickets Lost'}, inplace=True)
    first_innings_pp2_wickets_nc.rename(columns={
        'Team 1 PP-2 Wickets Lost': 'PP-2 Wickets Lost'}, inplace=True)
    first_innings_pp3_wickets_nc.rename(columns={
        'Team 1 PP-3 Wickets Lost': 'PP-3 Wickets Lost'}, inplace=True)

    # merging the dataframes
    df_temp = pd.merge(first_innings_pp1_wickets_nc,
                       first_innings_pp2_wickets_nc,
                       on='Team 1')
    powerplay_wickets_innings_1_nc = pd.merge(df_temp,
                                              first_innings_pp3_wickets_nc,
                                              on='Team 1')

    # Melt the DataFrame to reshape it for side-by-side bar chart
    df_melted_wickets_nc = pd.melt(powerplay_wickets_innings_1_nc,
                                   id_vars='Team 1',
                                   var_name='Powerplay',
                                   value_name='Wickets')

    get_bar(df_melted_wickets_nc, 'Team 1', 'Wickets',
            '1st innings wicket lost for all teams in different phases',
            {'Q': 'limegreen', 'E': 'greenyellow'},
            'Team Name', 'Wickets',
            'first_innings_detailed_stats_4', color='Powerplay',
            barmode='group')

    # Get Win percentage while batting in 1st innings

    pp_runs_wickets_combined = pd.merge(
        powerplay_run_rates_innings_1, powerplay_wickets_innings_1_nc,
        on='Team 1')

    df_first_inning_wins['Win %age'] = (
            df_first_inning_wins['Wins Batting First'] * 100 /
            df_first_inning_wins['1st Batting Count'])

    # Rename the 'Team' column to 'Team 1'
    df_first_inning_wins = \
        df_first_inning_wins.rename(columns={'Team': 'Team 1'})

    pp_runs_wickets_combined_final = \
        pd.merge(pp_runs_wickets_combined, df_first_inning_wins, on='Team 1')
    pp_runs_wickets_combined_final.drop(
        ['1st Batting Count', 'Wins Batting First'], axis=1, inplace=True)

    # Renaming the column back to Team
    pp_runs_wickets_combined_final = \
        pp_runs_wickets_combined_final.rename(columns={'Team 1': 'Team'})

    # Get contribution of Top order (1-3 Batsman),
    # middle order (4-7) and lower order (8-11) batsmen

    # get group stage data
    df_batting_scorecards = df_batting_scorecards[:-6]

    # getting 1st innings batting scorecards
    first_batting_scorecards_df = \
        df_batting_scorecards[~df_batting_scorecards['Match'].duplicated()]

    # dropping columns having name, strike rate and match
    first_batting_scorecards_df.drop(
        columns=first_batting_scorecards_df.filter(
            like='name'
        ).columns.tolist() + first_batting_scorecards_df.filter(
            like='strike_rate'
        ).columns.tolist() + first_batting_scorecards_df.filter(
            like='Match').columns.tolist(),
        inplace=True)

    # grouping by team name and getting get sum of all the records
    grouped_first_batting_scorecards_df = \
        first_batting_scorecards_df.groupby('Team').sum()

    # get top order, middle order, lower order runs, balls
    temp_list = ['runs', 'balls', '4s', '6s']
    run_phasing_df, fours_sixes_df = pd.DataFrame(), pd.DataFrame()

    for item in temp_list:
        grouped_first_batting_scorecards_df[f'Top order {item}'] = \
            grouped_first_batting_scorecards_df[f'Batting_1_{item}'] + \
            grouped_first_batting_scorecards_df[f'Batting_2_{item}'] + \
            grouped_first_batting_scorecards_df[f'Batting_3_{item}']

        grouped_first_batting_scorecards_df[f'Middle order {item}'] = \
            grouped_first_batting_scorecards_df[f'Batting_4_{item}'] + \
            grouped_first_batting_scorecards_df[f'Batting_5_{item}'] + \
            grouped_first_batting_scorecards_df[f'Batting_6_{item}'] + \
            grouped_first_batting_scorecards_df[f'Batting_7_{item}']

        grouped_first_batting_scorecards_df[f'Lower order {item}'] = \
            grouped_first_batting_scorecards_df[f'Batting_8_{item}'] + \
            grouped_first_batting_scorecards_df[f'Batting_9_{item}'] + \
            grouped_first_batting_scorecards_df[f'Batting_10_{item}'] + \
            grouped_first_batting_scorecards_df[f'Batting_11_{item}']

        run_phasing_df[f'Top order {item}'] = \
            grouped_first_batting_scorecards_df[
                f'Top order {item}'].astype(int)
        run_phasing_df[f'Middle order {item}'] = \
            grouped_first_batting_scorecards_df[
                f'Middle order {item}'].astype(int)
        run_phasing_df[f'Lower order {item}'] = \
            grouped_first_batting_scorecards_df[
                f'Lower order {item}'].astype(int)

    # Calculate total runs for each row
    run_phasing_df['TotalRuns'] = \
        run_phasing_df['Top order runs'] + \
        run_phasing_df['Middle order runs'] + \
        run_phasing_df['Lower order runs']

    # Calculate the percentage of runs for each category
    run_phasing_df['Top Order Percentage'] = round(
        (run_phasing_df['Top order runs'] /
         run_phasing_df['TotalRuns']) * 100, 2)
    run_phasing_df['Middle Order Percentage'] = round(
        (run_phasing_df['Middle order runs'] /
         run_phasing_df['TotalRuns']) * 100, 2)
    run_phasing_df['Lower Order Percentage'] = round(
        (run_phasing_df['Lower order runs'] /
         run_phasing_df['TotalRuns']) * 100, 2)

    # remove team name from being as index
    run_phasing_df = run_phasing_df.reset_index()

    percentage_batting_df = run_phasing_df[
        ['Team', 'Top Order Percentage',
         'Middle Order Percentage', 'Lower Order Percentage']]

    melted_df = pd.melt(percentage_batting_df, id_vars=['Team'],
                        var_name='Order', value_name='Percentage')

    # plot bar chart to analyse the 'run %age contribution in different phases
    get_bar(melted_df, 'Team', 'Percentage',
            '%age contribution (batting phases)',
            {'Q': 'limegreen', 'E': 'greenyellow'},
            'Team Name', '%age',
            'first_innings_detailed_stats_5', color='Order',
            barmode='group')

    # get strike rate %age of top, middle and lower order
    run_phasing_df['Top order strike rate'] = round(
        (run_phasing_df['Top order runs'] /
         run_phasing_df['Top order balls']) * 100, 2)
    run_phasing_df['Middle order strike rate'] = round(
        (run_phasing_df['Middle order runs'] /
         run_phasing_df['Middle order balls']) * 100, 2)
    run_phasing_df['Lower order strike rate'] = round(
        (run_phasing_df['Lower order runs'] /
         run_phasing_df['Lower order balls']) * 100, 2)

    strike_rate_batting_df = run_phasing_df[
        ['Team', 'Top order strike rate',
         'Middle order strike rate', 'Lower order strike rate']]

    melted_df = pd.melt(strike_rate_batting_df, id_vars=['Team'],
                        var_name='Order', value_name='Strike rate')

    # plot bar chart to analyse the strike rate of
    # all teams in different phases
    get_bar(melted_df, 'Team', 'Strike rate',
            'Strike Rate (batting phases)',
            {'Q': 'limegreen', 'E': 'greenyellow'},
            'Team Name', 'Strike Rate',
            'first_innings_detailed_stats_6', color='Order',
            barmode='group')

    # get boundary percentage for teams

    # get total batting runs
    batting_runs_columns = grouped_first_batting_scorecards_df.columns[
        grouped_first_batting_scorecards_df.columns.str.match(
            r'Batting_\d+_runs')]
    batting_4s_columns = grouped_first_batting_scorecards_df.columns[
        grouped_first_batting_scorecards_df.columns.str.match(
            r'Batting_\d+_4s')]
    batting_6s_columns = grouped_first_batting_scorecards_df.columns[
        grouped_first_batting_scorecards_df.columns.str.match(
            r'Batting_\d+_6s')]

    # Calculate total boundary percentage

    # Sum the runs for each team and create a new column 'total_runs'
    fours_sixes_df['total_runs'] = \
        grouped_first_batting_scorecards_df[
            batting_runs_columns].sum(axis=1).astype(int)
    fours_sixes_df['total_runs_in_4s'] = \
        grouped_first_batting_scorecards_df[
            batting_4s_columns].sum(axis=1).astype(int)
    fours_sixes_df['total_runs_in_6s'] = \
        grouped_first_batting_scorecards_df[
            batting_6s_columns].sum(axis=1).astype(int)

    # Boundary percentage
    fours_sixes_df['Boundary %age'] = \
        round(((fours_sixes_df['total_runs_in_4s'] * 4 + fours_sixes_df[
            'total_runs_in_6s'] * 6) * 100 / fours_sixes_df['total_runs']), 2)

    average_boundary_percentage_all_teams = \
        fours_sixes_df['Boundary %age'].mean().round(2)

    fours_sixes_df.reset_index(inplace=True)

    # plot bar plot for average 1st innings boundary %age for all teams
    get_dotted_bar(fours_sixes_df, 'Team', 'Boundary %age',
                   '1st innings all teams average Boundary %age',
                   'Team Name', 'Percentage',
                   'first_innings_detailed_stats_7',
                   average_boundary_percentage_all_teams)

    # Batting correlation
    batting_dfs = [pp_runs_wickets_combined_final,
                   percentage_batting_df, strike_rate_batting_df]

    # Use reduce to merge DataFrames based on 'Team Name'
    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on='Team', how='outer'),
        batting_dfs)

    # Generate a correlation heatmap for statistics during 1st batting
    heatmap(merged_df, 'overall_first_batting_heatmap_1',
            'Correlation of Team statistics (1st Batting)')

    # Team Bowling First stats

    # Get Bowling df

    sub_df_bowling_scorecards = df_bowling_scorecards[
                                    ['Match', 'Team',
                                     'Maidens', 'Overs',
                                     'Runs Conceded',
                                     '0s', '4s', '6s', 'WD',
                                     'NB']][:-35]

    # getting team_order
    team_order = df_batting_scorecards[['Match', 'Team']]

    # getting 1st innings bowling order
    first_batting_scorecards_df = team_order[
        team_order['Match'].duplicated(keep='first') |
        ~team_order['Match'].duplicated(keep=False)]

    # Group by match number and team name, then sum the remaining columns
    df_bowling_stats = \
        sub_df_bowling_scorecards.groupby(['Match', 'Team']).agg(
            {'Maidens': 'sum', 'Overs': 'sum',
             'Runs Conceded': 'sum', '0s': 'sum',
             '4s': 'sum', '6s': 'sum', 'WD': 'sum',
             'NB': 'sum'}).reset_index()

    # Merge based on 'Match number' and 'team'
    first_team_bowling = pd.merge(
        first_batting_scorecards_df,
        df_bowling_stats, on=['Match', 'Team'])

    # get average 2nd innings run rate and wickets lost

    # removing match column and grouping by team name
    first_team_bowling.drop(['Match'], axis=1, inplace=True)
    first_team_bowling = first_team_bowling.groupby(['Team']).agg(
        {'Maidens': 'sum', 'Overs': 'sum', 'Runs Conceded': 'sum',
         '0s': 'sum', '4s': 'sum', '6s': 'sum', 'WD': 'sum',
         'NB': 'sum'}).reset_index()

    # Phasewise Economy calculation

    first_team_bowling['Economy'] = \
        round((first_team_bowling['Runs Conceded'] /
               first_team_bowling['Overs']), 2)

    average_economy_all_teams = first_team_bowling['Economy'].mean().round(2)

    # plot bar plot for average 1st innings boundary
    # %age runs conceded for all teams
    get_dotted_bar(first_team_bowling, 'Team', 'Economy',
                   '1st Bowling innings all teams average economy',
                   'Team Name', 'Economy Rate',
                   'first_innings_detailed_stats_8',
                   average_economy_all_teams)

    # economy by phase
    first_bowling_runs = df_match_summary[['Team 2', 'Team 1 Overs',
                                           'Team 1 PP-1 Runs Scored',
                                           'Team 1 Wickets Lost',
                                           'Team 1 PP-2 Runs Scored',
                                           'Team 1 PP-1 Wickets Lost',
                                           'Team 1 PP-2 Wickets Lost',
                                           'Team 1 PP-3 Wickets Lost',
                                           'Team 1 PP-3 Runs Scored',
                                           'Team 1 Wickets Lost',
                                           'Winner']][:-3]
    first_bowling_runs['PP-1 Overs'] = 10.0

    # getting pp-2 and pp-3 overs
    first_bowling_runs['PP-2 Overs'] = pp2_overs
    first_bowling_runs['PP-3 Overs'] = pp3_overs

    # pp-1 was different for the rain curtailed match
    # (Match 15 Netherlands v/s South Africa. Only 9 overs for the pp-2)
    first_bowling_runs.at[14, 'PP-1 Overs'] = 9.0

    # pp-2 was different for the rain curtailed match
    # (Match 15 Netherlands v/s South Africa. Only 26 overs for the pp-2)
    first_bowling_runs.at[14, 'PP-2 Overs'] = 26.0

    # pp-3 was different for the rain curtailed match
    # (Match 15 Netherlands v/s South Africa. Only 8 overs for the pp-2)
    first_bowling_runs.at[14, 'PP-3 Overs'] = 8.0

    first_bowling_runs['PP-1 Economy'] = round(
        first_bowling_runs['Team 1 PP-1 Runs Scored'] /
        first_bowling_runs['PP-1 Overs'], 2)
    first_bowling_runs['PP-2 Economy'] = round(
        first_bowling_runs['Team 1 PP-2 Runs Scored'] /
        first_bowling_runs['PP-2 Overs'], 2)
    first_bowling_runs['PP-3 Economy'] = round(
        first_bowling_runs['Team 1 PP-3 Runs Scored'] /
        first_bowling_runs['PP-3 Overs'], 2)

    # Convert 'RunRate' to numeric to handle NaN values
    first_bowling_runs['PP-3 Economy'] = \
        pd.to_numeric(first_bowling_runs['PP-3 Economy'], errors='coerce')

    # Group by 'Team' and calculate the mean of 'RunRate' in pp-3
    first_bowling_runs_pp1 = \
        first_bowling_runs.groupby(
            'Team 2')['PP-1 Economy'].mean().round(2).reset_index()
    first_bowling_runs_pp2 = \
        first_bowling_runs.groupby(
            'Team 2')['PP-2 Economy'].mean().round(2).reset_index()
    first_bowling_runs_pp3 = \
        first_bowling_runs.groupby(
            'Team 2')['PP-3 Economy'].mean().round(2).reset_index()

    # merging the dataframes
    df_econ = pd.merge(first_bowling_runs_pp1,
                       first_bowling_runs_pp2, on='Team 2')
    first_bowling_economy = pd.merge(df_econ,
                                     first_bowling_runs_pp3, on='Team 2')

    # Melt the DataFrame to reshape it for side-by-side bar chart
    df_melted_runs = pd.melt(first_bowling_economy, id_vars='Team 2',
                             var_name='Powerplay', value_name='Economy')

    # plot bar chart to analyse the average economy rate
    # conceded for all teams in different phases
    get_bar(df_melted_runs, 'Team 2', 'Economy',
            '1st innings economy rate conceded for '
            'all teams in different phases',
            {'Q': 'limegreen', 'E': 'greenyellow'},
            'Team Name', 'Economy rate',
            'first_innings_detailed_stats_9',
            color='Powerplay', barmode='group')

    # Phasewise wickets taken
    first_bowling_wicket_pp1 = \
        first_bowling_runs.groupby('Team 2')[
            'Team 1 PP-1 Wickets Lost'].mean().round(
            2).reset_index()
    first_bowling_wicket_pp2 = \
        first_bowling_runs.groupby('Team 2')[
            'Team 1 PP-2 Wickets Lost'].mean().round(
            2).reset_index()
    first_bowling_wicket_pp3 = \
        first_bowling_runs.groupby('Team 2')[
            'Team 1 PP-3 Wickets Lost'].mean().round(
            2).reset_index()

    # merging the dataframes
    df_wick = pd.merge(first_bowling_wicket_pp1,
                       first_bowling_wicket_pp2,
                       on='Team 2')
    first_bowling_wicket = pd.merge(df_wick,
                                    first_bowling_wicket_pp3,
                                    on='Team 2')

    first_bowling_wicket_pp1_nc = first_bowling_wicket_pp1.copy()
    first_bowling_wicket_pp2_nc = first_bowling_wicket_pp2.copy()
    first_bowling_wicket_pp3_nc = first_bowling_wicket_pp3.copy()
    first_bowling_wicket_pp1_nc.rename(
        columns={'Team 1 PP-1 Wickets Lost': 'PP-1 Wickets Taken'},
        inplace=True)
    first_bowling_wicket_pp2_nc.rename(
        columns={'Team 1 PP-2 Wickets Lost': 'PP-2 Wickets Taken'},
        inplace=True)
    first_bowling_wicket_pp3_nc.rename(
        columns={'Team 1 PP-3 Wickets Lost': 'PP-3 Wickets Taken'},
        inplace=True)

    df_nc_10 = pd.merge(first_bowling_wicket_pp1_nc,
                        first_bowling_wicket_pp2_nc,
                        on='Team 2')
    first_bowling_wicket_nc = pd.merge(df_nc_10,
                                       first_bowling_wicket_pp3_nc,
                                       on='Team 2')
    # Melt the DataFrame to reshape it for side-by-side bar chart
    df_melted_wickets = pd.melt(first_bowling_wicket_nc, id_vars='Team 2',
                                var_name='Powerplay', value_name='Wickets')

    # plot bar chart to analyse the average economy rate
    # conceded for all teams in different phases
    get_bar(df_melted_wickets, 'Team 2', 'Wickets',
            '1st innings wickets taken by all teams in different phases',
            {'Q': 'limegreen', 'E': 'greenyellow'},
            'Team Name', 'Wickets',
            'first_innings_detailed_stats_10',
            color='Powerplay', barmode='group')

    # get average wickets per match in 1st bowling innings

    first_bowling_all_wicket = pd.DataFrame()
    first_bowling_all_wicket['Team 2'] = first_bowling_wicket['Team 2'].copy()
    first_bowling_all_wicket['Total Wickets per match'] = \
        first_bowling_wicket['Team 1 PP-1 Wickets Lost'] + \
        first_bowling_wicket['Team 1 PP-2 Wickets Lost'] + \
        first_bowling_wicket['Team 1 PP-3 Wickets Lost']
    average_wicket_per_match_all_teams = \
        first_bowling_all_wicket['Total Wickets per match'].mean().round(2)

    # plot bar plot for average 1st innings
    # boundary %age runs conceded for all teams
    get_dotted_bar(first_bowling_all_wicket, 'Team 2',
                   'Total Wickets per match',
                   '1st Bowling innings all teams average Wicket taken',
                   'Team Name', 'Wickets',
                   'first_innings_detailed_stats_11',
                   average_wicket_per_match_all_teams)

    # Calculate Bowling Boundary percentage

    first_team_bowling['Boundary %age'] = round(
        ((first_team_bowling['4s'] * 4 +
          first_team_bowling['6s'] * 6) * 100 /
         first_team_bowling['Runs Conceded']), 2)
    average_boundary_percentage_all_teams = \
        first_team_bowling['Boundary %age'].mean().round(2)

    # plot bar plot for average 1st innings boundary
    # %age runs conceded for all teams
    get_dotted_bar(first_team_bowling, 'Team', 'Boundary %age',
                   '1st Bowling innings all teams '
                   'average runs conceded Boundary %age',
                   'Team Name', 'Percentage',
                   'first_innings_detailed_stats_12',
                   average_boundary_percentage_all_teams)

    # Calculate maiden over count

    # plot bar plot for maiden overs count for all teams
    average_first_bowling_maidens = \
        first_team_bowling['Maidens'].mean().round(2)
    get_dotted_bar(first_team_bowling, 'Team', 'Maidens',
                   '1st Bowling innings Maidens count', 'Team Name',
                   'Maidens', 'first_innings_detailed_stats_13',
                   average_first_bowling_maidens)

    # Calculate total extras conceded

    # plot bar plot for total extras for all teams
    first_team_bowling['total extras conceded'] = \
        first_team_bowling['WD'] + first_team_bowling['NB']
    average_extras_conceded = \
        first_team_bowling['total extras conceded'].mean().round(2)

    get_dotted_bar(first_team_bowling, 'Team', 'total extras conceded',
                   '1st Bowling innings total extras count',
                   'Team Name', 'Extras',
                   'first_innings_detailed_stats_14', average_extras_conceded)

    # Calculate Win %age
    df_second_inning_wins['Win %age'] = (
            (df_second_inning_wins['Wins Batting Second'] * 100) /
            df_second_inning_wins['2nd Batting Count'])

    # Drop 2nd batting count
    df_second_inning_wins.drop(['2nd Batting Count', 'Wins Batting Second'],
                               axis=1, inplace=True)

    # Rename the 'Team' column to 'Team 1'
    df_second_inning_wins.rename(columns={'Team': 'Team 2'}, inplace=True)

    # bowling stats correlation

    bowling_dfs = [first_bowling_economy,
                   first_bowling_wicket_nc, df_second_inning_wins]

    # Use reduce to merge DataFrames based on 'Team Name'
    merged_df = reduce(lambda left, right:
                       pd.merge(left, right, on='Team 2', how='outer'),
                       bowling_dfs)

    # Generate a correlation heatmap
    heatmap(merged_df, 'overall_first_bowling_heatmap_1',
            'Correlation of Team statistics (1st Bowling)')


def second_innings_detailed_stats(df_match_summary, df_first_inning_wins,
                                  df_second_inning_wins,
                                  df_batting_scorecards,
                                  df_bowling_scorecards):
    """
    function to get detailed second innings stats
    :param df_match_summary: match summary df
    :param df_first_inning_wins: first innings wins df
    :param df_second_inning_wins: second innings wins df
    :param df_batting_scorecards: batting scorecard df
    :param df_bowling_scorecards: bowling scorecard df
    :return: None
    """

    # 2nd innings batting stats

    # get average wickets lost in chase

    # exclude last 3 groups
    sub_second_innings_batting_wickets_df = \
        df_match_summary[['Team 2', 'Team 2 Wickets Lost', 'Winner']][:-3]

    # Groupby team name
    second_innings_wickets = \
        sub_second_innings_batting_wickets_df.groupby(
            'Team 2', as_index=False).agg(
            {'Team 2 Wickets Lost': 'mean'})

    weighted_average_wickets_lost_all_teams = \
        second_innings_wickets['Team 2 Wickets Lost'].mean()

    # plot bar plot for average 1st innings wickets lost by all teams
    get_dotted_bar(second_innings_wickets, 'Team 2', 'Team 2 Wickets Lost',
                   '2nd innings all teams average wickets lost',
                   'Team Name', 'Wickets',
                   'second_innings_detailed_stats_1',
                   weighted_average_wickets_lost_all_teams)

    # Calculate powerplay phasewise %age of target scored
    second_innings_pp = df_match_summary[
                            ['Team 1 Runs Scored', 'Team 2',
                             'Team 2 Overs', 'Team 2 PP-1 Runs Scored',
                             'Team 2 PP-1 Wickets Lost',
                             'Team 2 PP-2 Runs Scored',
                             'Team 2 PP-2 Wickets Lost',
                             'Team 2 PP-3 Runs Scored',
                             'Team 2 PP-3 Wickets Lost', 'Winner']][:-3]

    second_innings_pp['Team 2 PP-1 %age target scored'] = round(
        second_innings_pp['Team 2 PP-1 Runs Scored'] * 100 /
        second_innings_pp['Team 1 Runs Scored'], 2)

    second_innings_pp['Team 2 PP-2 %age target scored'] = round(
        second_innings_pp['Team 2 PP-2 Runs Scored'] * 100 /
        second_innings_pp['Team 1 Runs Scored'], 2)

    second_innings_pp['Team 2 PP-3 %age target scored'] = round(
        second_innings_pp['Team 2 PP-3 Runs Scored'] * 100 /
        second_innings_pp['Team 1 Runs Scored'], 2)

    get_ppwise_run_percent = second_innings_pp[
        ['Team 2', 'Team 2 PP-1 %age target scored',
         'Team 2 PP-2 %age target scored',
         'Team 2 PP-3 %age target scored']]

    get_ppwise_run_percent.rename(columns={
        'Team 2 PP-1 %age target scored': '%age target scored in PP-1',
        'Team 2 PP-2 %age target scored': '%age target scored in PP-2',
        'Team 2 PP-3 %age target scored': '%age target scored in PP-3'},
                                  inplace=True)

    get_ppwise_run_percent = \
        get_ppwise_run_percent.groupby([
            'Team 2'], as_index=False).mean().round(2)

    # Melt the DataFrame to long format
    melted_df_run_percent = pd.melt(get_ppwise_run_percent,
                                    id_vars=['Team 2'], var_name='Percentage',
                                    value_name='Percentage target')

    # plot bar chart to analyse the phasewise
    # average % runs scored by all teams while chasing
    get_bar(melted_df_run_percent, 'Team 2', 'Percentage target',
            '2nd innings %age runs scored by all teams in different phases',
            {'Q': 'limegreen', 'E': 'greenyellow'},
            'Team Name', 'Percentage',
            'second_innings_detailed_stats_2', color='Percentage target')

    # Get phasewise wickets lost

    # pp-1 wickets lost
    second_innings_pp1_wickets = second_innings_pp.groupby('Team 2')[
        'Team 2 PP-1 Wickets Lost'].mean().reset_index()

    # pp-2 wickets lost
    second_innings_pp2_wickets = second_innings_pp.groupby('Team 2')[
        'Team 2 PP-2 Wickets Lost'].mean().reset_index()

    # pp-3 wickets lost
    second_innings_pp3_wickets = \
        second_innings_pp.groupby('Team 2')[
            'Team 2 PP-3 Wickets Lost'].mean().reset_index()

    second_innings_pp1_wickets_nc = second_innings_pp1_wickets.copy()
    second_innings_pp2_wickets_nc = second_innings_pp2_wickets.copy()
    second_innings_pp3_wickets_nc = second_innings_pp3_wickets.copy()

    second_innings_pp1_wickets_nc.rename(columns={
        'Team 2 PP-1 Wickets Lost': 'PP-1 Wickets Lost'}, inplace=True)
    second_innings_pp2_wickets_nc.rename(columns={
        'Team 2 PP-2 Wickets Lost': 'PP-2 Wickets Lost'}, inplace=True)
    second_innings_pp3_wickets_nc.rename(columns={
        'Team 2 PP-3 Wickets Lost': 'PP-3 Wickets Lost'}, inplace=True)

    # merging the dataframes
    df_temp_ppwise_wickets_nc = \
        pd.merge(second_innings_pp1_wickets_nc,
                 second_innings_pp2_wickets_nc, on='Team 2')
    powerplay_wickets_innings_2_nc = \
        pd.merge(df_temp_ppwise_wickets_nc,
                 second_innings_pp3_wickets_nc, on='Team 2')

    # Melt the DataFrame to reshape it for side-by-side bar chart
    df_melted_wickets_innings_2_nc = pd.melt(powerplay_wickets_innings_2_nc,
                                             id_vars='Team 2',
                                             var_name='Powerplay',
                                             value_name='Wickets')

    # plot bar chart to analyse the wickets
    # lost by all teams in different phases
    get_bar(df_melted_wickets_innings_2_nc, 'Team 2', 'Wickets',
            '2nd innings wicket lost for all teams in different phases',
            {'Q': 'limegreen', 'E': 'greenyellow'},
            'Team Name', 'Wickets',
            'second_innings_detailed_stats_3',
            color='Powerplay', barmode='group')

    # Get contribution of Top order (1-3 Batsman),
    # middle order (4-7) and lower order (8-11) batsmen

    # get group stage data
    df_batting_scorecards = df_batting_scorecards[:-6]

    # getting 1st innings batting scorecards
    second_batting_scorecards_df = df_batting_scorecards[
        df_batting_scorecards['Match'].duplicated(keep='first') |
        ~df_batting_scorecards['Match'].duplicated(
            keep=False)]

    # dropping columns having name, strike rate and match
    second_batting_scorecards_df.drop(
        columns=second_batting_scorecards_df.filter(
            like='name'
        ).columns.tolist() + second_batting_scorecards_df.filter(
            like='strike_rate'
        ).columns.tolist() + second_batting_scorecards_df.filter(
            like='Match').columns.tolist(),
        inplace=True)

    # grouping by team name and getting sum of all the records
    grouped_second_batting_scorecards_df = \
        second_batting_scorecards_df.groupby('Team').sum()

    # get top order, middle order, lower order runs, balls
    temp_list = ['runs', 'balls', '4s', '6s']
    run_phasing_df = pd.DataFrame()
    for item in temp_list:
        grouped_second_batting_scorecards_df[f'Top order {item}'] = \
            grouped_second_batting_scorecards_df[f'Batting_1_{item}'] + \
            grouped_second_batting_scorecards_df[f'Batting_2_{item}'] + \
            grouped_second_batting_scorecards_df[f'Batting_3_{item}']

        grouped_second_batting_scorecards_df[f'Middle order {item}'] = \
            grouped_second_batting_scorecards_df[f'Batting_4_{item}'] + \
            grouped_second_batting_scorecards_df[f'Batting_5_{item}'] + \
            grouped_second_batting_scorecards_df[f'Batting_6_{item}'] + \
            grouped_second_batting_scorecards_df[f'Batting_7_{item}']

        grouped_second_batting_scorecards_df[f'Lower order {item}'] = \
            grouped_second_batting_scorecards_df[f'Batting_8_{item}'] + \
            grouped_second_batting_scorecards_df[f'Batting_9_{item}'] + \
            grouped_second_batting_scorecards_df[f'Batting_10_{item}'] + \
            grouped_second_batting_scorecards_df[f'Batting_11_{item}']

        run_phasing_df[f'Top order {item}'] = \
            grouped_second_batting_scorecards_df[
                f'Top order {item}'].astype(int)
        run_phasing_df[f'Middle order {item}'] = \
            grouped_second_batting_scorecards_df[
                f'Middle order {item}'].astype(int)
        run_phasing_df[f'Lower order {item}'] = \
            grouped_second_batting_scorecards_df[
                f'Lower order {item}'].astype(int)

    # Calculate total runs for each row
    run_phasing_df['TotalRuns'] = \
        run_phasing_df['Top order runs'] + \
        run_phasing_df['Middle order runs'] + \
        run_phasing_df['Lower order runs']

    # Calculate the percentage of runs for each category
    run_phasing_df['Top Order Percentage'] = round(
        (run_phasing_df['Top order runs'] /
         run_phasing_df['TotalRuns']) * 100, 2)
    run_phasing_df['Middle Order Percentage'] = round(
        (run_phasing_df['Middle order runs'] /
         run_phasing_df['TotalRuns']) * 100, 2)
    run_phasing_df['Lower Order Percentage'] = round(
        (run_phasing_df['Lower order runs'] /
         run_phasing_df['TotalRuns']) * 100, 2)

    # remove team name from being as index
    run_phasing_df = run_phasing_df.reset_index()

    percentage_batting_df_second = run_phasing_df[
        ['Team', 'Top Order Percentage',
         'Middle Order Percentage', 'Lower Order Percentage']]

    melted_df = pd.melt(percentage_batting_df_second, id_vars=['Team'],
                        var_name='Order', value_name='Percentage')

    # plot bar chart to analyse the 'run %age contribution in different phases
    get_bar(melted_df, 'Team', 'Percentage',
            '%age contribution (batting phases)',
            {'Q': 'limegreen', 'E': 'greenyellow'},
            'Team Name', '%age',
            'second_innings_detailed_stats_4', color='Order',
            barmode='group')

    # batting stats correlation

    # Rename the 'Team' column to 'Team 1'
    df_second_inning_wins = \
        df_second_inning_wins.rename(columns={'Team': 'Team 2'})

    percentage_batting_df_second = \
        percentage_batting_df_second.rename(columns={'Team': 'Team 2'})

    # combine the necessary dfs
    batting_dfs = [percentage_batting_df_second, get_ppwise_run_percent,
                   powerplay_wickets_innings_2_nc, df_second_inning_wins]

    # Use reduce to merge DataFrames based on 'Team Name'
    merged_df = reduce(lambda left, right: pd.merge(left, right,
                                                    on='Team 2',
                                                    how='outer'), batting_dfs)

    # Generate a correlation heatmap
    heatmap(merged_df, 'overall_second_batting_heatmap_1',
            'Correlation of Team statistics (2nd Batting)')

    # 2nd innings bowling stats

    # Get Bowling df

    sub_df_bowling_scorecards = df_bowling_scorecards[
                                    ['Match', 'Team', 'Maidens',
                                     'Overs', 'Runs Conceded',
                                     '0s', '4s', '6s', 'WD',
                                     'NB']][:-35]

    # getting team_order
    team_order = df_batting_scorecards[['Match', 'Team']]

    # getting 2nd innings bowling order
    second_batting_scorecards_df = team_order[
        team_order['Match'].duplicated(keep='first') |
        ~team_order['Match'].duplicated(keep=False)]

    # Group by match number and team name, then sum the remaining columns
    df_bowling_stats = \
        sub_df_bowling_scorecards.groupby(['Match', 'Team']).agg(
            {'Maidens': 'sum', 'Overs': 'sum', 'Runs Conceded': 'sum',
             '0s': 'sum', '4s': 'sum', '6s': 'sum', 'WD': 'sum',
             'NB': 'sum'}).reset_index()

    # Merge based on 'Match number' and 'team'
    second_team_bowling = pd.merge(
        second_batting_scorecards_df,
        df_bowling_stats, on=['Match', 'Team'])

    # get average 2nd innings run rate and wickets lost]

    # removing match column and grouping by team name
    second_team_bowling.drop(['Match'], axis=1, inplace=True)

    # Economy by phase

    second_bowling_runs = df_match_summary[['Team 1', 'Team 2 Overs',
                                            'Team 2 PP-1 Runs Scored',
                                            'Team 2 Wickets Lost',
                                            'Team 2 PP-2 Runs Scored',
                                            'Team 2 PP-1 Wickets Lost',
                                            'Team 2 PP-2 Wickets Lost',
                                            'Team 2 PP-3 Wickets Lost',
                                            'Team 2 PP-3 Runs Scored',
                                            'Team 2 Wickets Lost',
                                            'Winner']][:-3]
    second_bowling_runs['PP-1 Overs'] = 10.0

    # calculate pp-2 overs played and pp-3 Overs played
    # (teams getting bowled out)
    second_bowling_runs['Overs without pp-1'] = \
        (second_bowling_runs['Team 2 Overs'] - 10).round(1)
    pp2_pp3_overs_list = second_bowling_runs['Overs without pp-1'].to_list()

    # getting pp-2 and pp-3 overs
    pp2_overs, pp3_overs = [[] for _ in range(2)]

    for overs in pp2_pp3_overs_list:
        if overs == 40.0:
            pp2_overs.append(30.0)
            pp3_overs.append(10.0)
        elif overs < 30.0:
            pp2_overs.append(overs)
            pp3_overs.append(0)
        elif overs < 40.0:
            pp2_overs.append(30.0)
            pp3_overs.append(round((overs - 30.0), 1))

    # adding pp-2 and pp-3 overs played
    second_bowling_runs['PP-2 Overs'] = pp2_overs
    second_bowling_runs['PP-3 Overs'] = pp3_overs

    # pp-1 was different for the rain curtailed match
    # (Match 15 Netherlands v/s South Africa.
    # Only 9 overs for the pp-1)
    second_bowling_runs.at[14, 'PP-1 Overs'] = 9.0

    # pp-2 was different for the rain curtailed match
    # (Match 15 Netherlands v/s South Africa.
    # Only 26 overs for the pp-2)
    second_bowling_runs.at[14, 'PP-2 Overs'] = 26.0

    # pp-3 was different for the rain curtailed match
    # (Match 15 Netherlands v/s South Africa.
    # Only 7.5 overs for the pp-3 (sa bowled out in 42.5))
    second_bowling_runs.at[14, 'PP-3 Overs'] = 7.5

    second_bowling_runs = \
        second_bowling_runs.groupby('Team 1').sum().reset_index()

    second_bowling_runs['PP-1 Economy'] = round(
        second_bowling_runs['Team 2 PP-1 Runs Scored'] /
        second_bowling_runs['PP-1 Overs'], 2)
    second_bowling_runs['PP-2 Economy'] = round(
        second_bowling_runs['Team 2 PP-2 Runs Scored'] /
        second_bowling_runs['PP-2 Overs'], 2)
    second_bowling_runs['PP-3 Economy'] = round(
        second_bowling_runs['Team 2 PP-3 Runs Scored'] /
        second_bowling_runs['PP-3 Overs'], 2)

    # Convert 'RunRate' to numeric to handle NaN values
    second_bowling_runs['PP-3 Economy'] = \
        pd.to_numeric(second_bowling_runs['PP-3 Economy'],
                      errors='coerce')

    second_bowling_runs_pp1 = \
        second_bowling_runs[['Team 1', 'PP-1 Economy']]
    second_bowling_runs_pp2 = \
        second_bowling_runs[['Team 1', 'PP-2 Economy']]
    second_bowling_runs_pp3 = \
        second_bowling_runs[['Team 1', 'PP-3 Economy']]

    # merging the dataframes
    df_econ = \
        pd.merge(second_bowling_runs_pp1, second_bowling_runs_pp2,
                 on='Team 1')
    second_bowling_economy = \
        pd.merge(df_econ, second_bowling_runs_pp3,
                 on='Team 1')

    # Melt the DataFrame to reshape it for side-by-side bar chart
    df_melted_runs = pd.melt(second_bowling_economy, id_vars='Team 1',
                             var_name='Powerplay', value_name='Economy')

    # plot bar chart to analyse the average economy rate conceded
    # for all teams in different phases
    get_bar(df_melted_runs, 'Team 1', 'Economy',
            '2nd innings economy rate conceded for all '
            'teams in different phases',
            {'Q': 'limegreen', 'E': 'greenyellow'}, 'Team Name',
            'Economy rate', 'second_innings_detailed_stats_5',
            color='Powerplay', barmode='group')

    # Calculate Phasewise powerplay wickets lost

    # pp-1, 2, 3 wickets lost
    second_innings_pp1_wickets = \
        second_bowling_runs[['Team 1', 'Team 2 PP-1 Wickets Lost']]
    second_innings_pp2_wickets = \
        second_bowling_runs[['Team 1', 'Team 2 PP-2 Wickets Lost']]
    second_innings_pp3_wickets = \
        second_bowling_runs[['Team 1', 'Team 2 PP-3 Wickets Lost']]

    # Rename the 'Team' column to 'Team 1'
    df_first_inning_wins = df_first_inning_wins. \
        rename(columns={'Team': 'Team 1'})

    second_innings_pp1_wickets_merged = \
        pd.merge(second_innings_pp1_wickets, df_first_inning_wins,
                 on='Team 1')
    second_innings_pp2_wickets_merged = \
        pd.merge(second_innings_pp2_wickets, df_first_inning_wins,
                 on='Team 1')
    second_innings_pp3_wickets_merged = \
        pd.merge(second_innings_pp3_wickets, df_first_inning_wins,
                 on='Team 1')

    second_innings_pp1_wickets_merged['Team 2 PP-1 Wickets Lost'] = \
        round((second_innings_pp1_wickets_merged['Team 2 PP-1 Wickets Lost'] /
               second_innings_pp1_wickets_merged[
                   '1st Batting Count']), 2)
    second_innings_pp2_wickets_merged['Team 2 PP-2 Wickets Lost'] = \
        round((second_innings_pp2_wickets_merged['Team 2 PP-2 Wickets Lost'] /
               second_innings_pp2_wickets_merged['1st Batting Count']), 2)
    second_innings_pp3_wickets_merged['Team 2 PP-3 Wickets Lost'] = \
        round((second_innings_pp3_wickets_merged['Team 2 PP-3 Wickets Lost'] /
               second_innings_pp3_wickets_merged['1st Batting Count']), 2)

    # dropping irrelevant columns

    second_innings_pp1_wickets_merged.drop(['1st Batting Count',
                                            'Wins Batting First',
                                            'Win %age'], axis=1,
                                           inplace=True)
    second_innings_pp2_wickets_merged.drop(['1st Batting Count',
                                            'Wins Batting First',
                                            'Win %age'], axis=1,
                                           inplace=True)
    second_innings_pp3_wickets_merged.drop(['1st Batting Count',
                                            'Wins Batting First',
                                            'Win %age'], axis=1,
                                           inplace=True)

    # plot bar chart to analyse the wickets lost
    # by all teams in different phases

    second_innings_pp1_wickets_merged_nc =\
        second_innings_pp1_wickets_merged.copy()
    second_innings_pp2_wickets_merged_nc =\
        second_innings_pp2_wickets_merged.copy()
    second_innings_pp3_wickets_merged_nc =\
        second_innings_pp3_wickets_merged.copy()

    second_innings_pp1_wickets_merged_nc.rename(columns={
        'Team 2 PP-1 Wickets Lost': 'PP-1 Wickets Taken'},
                                                inplace=True)
    second_innings_pp2_wickets_merged_nc.rename(columns={
        'Team 2 PP-2 Wickets Lost': 'PP-2 Wickets Taken'},
                                                inplace=True)
    second_innings_pp3_wickets_merged_nc.rename(columns={
        'Team 2 PP-3 Wickets Lost': 'PP-3 Wickets Taken'},
                                                inplace=True)

    # merging the dataframes
    df_temp_nc = \
        pd.merge(second_innings_pp1_wickets_merged_nc,
                 second_innings_pp2_wickets_merged_nc, on='Team 1')
    powerplay_wickets_bowling_innings2_nc = \
        pd.merge(df_temp_nc, second_innings_pp3_wickets_merged_nc, on='Team 1')

    # Melt the DataFrame to reshape it for side-by-side bar chart
    df_melted_wickets_nc = pd.melt(powerplay_wickets_bowling_innings2_nc,
                                   id_vars='Team 1', var_name='Powerplay',
                                   value_name='Wickets')

    get_bar(df_melted_wickets_nc, 'Team 1', 'Wickets',
            '2nd innings wicket taken by all teams in different phases',
            {'Q': 'limegreen', 'E': 'greenyellow'}, 'Team Name',
            'Wickets', 'second_innings_detailed_stats_6', color='Powerplay',
            barmode='group')

    # bowling stats correlation

    # Drop 2nd batting count
    df_first_inning_wins.drop(['1st Batting Count',
                               'Wins Batting First'],
                              axis=1, inplace=True)

    # combine the necessary dfs
    bowling_dfs = [second_bowling_economy,
                   powerplay_wickets_bowling_innings2_nc,
                   df_first_inning_wins]

    # Use reduce to merge DataFrames based on 'Team Name'
    merged_df = reduce(lambda left, right: pd.merge(left, right,
                                                    on='Team 1', how='outer'),
                       bowling_dfs)

    # Generate a correlation heatmap
    heatmap(merged_df, 'overall_second_bowling_heatmap_1',
            'Correlation of Team statistics (2nd Bowling)')


def toss_analysis(df_match_summary, df_points):
    """
    function to generate toss-related stats
    :param df_match_summary: match summary df
    :param df_points: points table df
    :return: None
    """

    toss_stats = \
        df_match_summary[['Team 1', 'Team 2',
                          'Won_Toss', 'Decision_Toss',
                          'Winner']][:-3]

    # Get the decision pie after winning toss

    # plot the pie chart
    get_pie(toss_stats, 'Decision_Toss',
            'Decision after winning toss', 'toss_analysis_1')

    # Get frequency of each team winning the toss
    team_toss = toss_stats['Won_Toss'].value_counts().reset_index()

    # Rename the columns
    team_toss.columns = ['Team', 'Count']

    # plot bar chart to analyse the total tosses won by each team
    get_bar(team_toss, 'Team', 'Count', 'Toss won by Teams',
            {'Q': 'limegreen', 'E': 'greenyellow'}, 'Teams',
            'Toss', 'toss_analysis_2')

    # Calculate toss winn percentage
    team_toss['Toss Win %age'] = team_toss['Count'] * 100 / 9

    # Toss and win percent correlation

    df_points_temp = df_points.copy()

    # Renaming the column in points table
    df_points_temp.rename(columns={'Teams': 'Team'}, inplace=True)

    # calculate match win percentage
    df_points_temp['Match Win %age'] = \
        df_points_temp['Won'] * 100 / df_points_temp['Mat']

    team_toss.drop(['Count'], axis=1, inplace=True)
    df_points_temp. \
        drop(['Mat', 'Won', 'Lost', 'Pts', 'Qualification_Status', 'NRR'],
             axis=1, inplace=True)

    toss_corr_dfs = [team_toss, df_points_temp]

    # Use reduce to merge DataFrames based on 'Team Name'
    merged_df = reduce(lambda left, right:
                       pd.merge(left, right, on='Team', how='outer'),
                       toss_corr_dfs)

    # Generate a correlation heatmap for toss statistics
    heatmap(merged_df, 'toss_win_heatmap_1',
            'Correlation of Team winning the toss with winning the match')


def venue_trends(df_match_summary):
    """
    function to generate general trends for each venue
    :param df_match_summary: match summary df
    :return: None
    """

    team_venues_dict = {}

    sub_df_match_summary = df_match_summary[:-3]

    # Iterate over the rows of the DataFrame
    for _, row in sub_df_match_summary.iterrows():
        teams = [row['Team 1'], row['Team 2']]
        venue = row['Stadium']

        # Update the dictionary for each team
        for team in teams:
            if team in team_venues_dict:
                team_venues_dict[team].append(venue)
            else:
                team_venues_dict[team] = [venue]

    # Create a list to store data
    data = []

    # Iterate through the dictionary
    for team, venues in team_venues_dict.items():
        # Count the matches played by the team at each venue
        venue_counts = {venue: venues.count(venue) for venue in set(venues)}

        # Add a row for each team-venue combination
        for venue, matches_played in venue_counts.items():
            data.append({'Team': team, 'Venue': venue,
                         'MatchesPlayed': matches_played})
    df_venues_teams = pd.DataFrame(data)

    # plot bar chart to present the matches played by teams at different venues
    get_bar(df_venues_teams, 'Team', 'MatchesPlayed',
            'Matches Played by Teams at different venues',
            {'Q': 'limegreen', 'E': 'greenyellow'}, 'Teams',
            'Count', 'venue_trends_1', color='Venue', barmode='group')

    # Separate matches where Team 1 won and Team 2 won
    team1_wins = \
        sub_df_match_summary[sub_df_match_summary['Winner'] ==
                             sub_df_match_summary['Team 1']]
    team2_wins = \
        sub_df_match_summary[sub_df_match_summary['Winner'] ==
                             sub_df_match_summary['Team 2']]

    # Count occurrences at each venue for Team 1 wins
    team1_wins_count = \
        team1_wins.groupby('Stadium')['Team 1'].count().reset_index()
    team1_wins_count = \
        team1_wins_count.rename(columns={'Team 1': '1st Batting Wins'})

    # Count occurrences at each venue for Team 2 wins
    team2_wins_count = \
        team2_wins.groupby('Stadium')['Team 2'].count().reset_index()
    team2_wins_count = \
        team2_wins_count.rename(columns={'Team 2': '2nd Batting Wins'})

    # Merge the two DataFrames on 'Stadium'
    result_df = pd.merge(team1_wins_count, team2_wins_count,
                         on='Stadium', how='outer').fillna(0)

    # Melt the DataFrame to reshape it for side-by-side bar chart
    df_melted = pd.melt(result_df, id_vars='Stadium',
                        var_name='Batting/Chasing Team', value_name='Count')

    # plot bar chart to present the matches won
    # batting first v/s chasing at a venue
    get_bar(df_melted, 'Stadium', 'Count',
            'Matches won batting first v/s chasing at different venues',
            {'Q': 'limegreen', 'E': 'greenyellow'}, 'Stadium',
            'Count', 'venue_trends_2', color='Batting/Chasing Team',
            barmode='group')

    # Get average runs per wicket and avg run rate
    venue_wise_stats = \
        sub_df_match_summary.groupby('Stadium', as_index=False).agg(
            {'Team 1 Runs Scored': 'sum', 'Team 1 Wickets Lost': 'sum',
             'Team 1 Overs': 'sum', 'Team 2 Runs Scored': 'sum',
             'Team 2 Wickets Lost': 'sum', 'Team 2 Overs': 'sum'})

    venue_wise_stats['Batting first avg runs per wicket'] = round(
        (venue_wise_stats['Team 1 Runs Scored'] /
         venue_wise_stats['Team 1 Wickets Lost']), 2)
    venue_wise_stats['Batting second avg runs per wicket'] = round(
        (venue_wise_stats['Team 2 Runs Scored'] /
         venue_wise_stats['Team 2 Wickets Lost']), 2)

    venue_wise_stats['Batting first avg run rate'] = round(
        (venue_wise_stats['Team 1 Runs Scored'] /
         venue_wise_stats['Team 1 Overs']), 2)
    venue_wise_stats['Batting second avg run rate'] = round(
        (venue_wise_stats['Team 2 Runs Scored'] /
         venue_wise_stats['Team 2 Overs']), 2)

    avg_rpo = venue_wise_stats[['Stadium',
                                'Batting first avg runs per wicket',
                                'Batting second avg runs per wicket']]

    # Melt the DataFrame to reshape it for side-by-side bar chart
    df_melted = pd.melt(avg_rpo, id_vars='Stadium',
                        var_name='Batting/Chasing Team', value_name='rpo')

    # plot bar chart to present the matches won
    # batting first v/s chasing at a venue
    get_bar(df_melted, 'Stadium', 'rpo',
            'Average Runs per wicket batting first v/s '
            'chasing at different venues',
            {'Q': 'limegreen', 'E': 'greenyellow'}, 'Stadium',
            'Count', 'venue_trends_3',
            color='Batting/Chasing Team', barmode='group')

    avg_run_rate = \
        venue_wise_stats[['Stadium', 'Batting first avg run rate',
                          'Batting second avg run rate']]

    # Melt the DataFrame to reshape it for side-by-side bar chart
    df_melted = pd.melt(avg_run_rate, id_vars='Stadium',
                        var_name='Batting/Chasing Team',
                        value_name='avg run rate')

    # plot bar chart to present the Average run rate batting 1st v/s chasing
    get_bar(df_melted, 'Stadium', 'avg run rate',
            'Average Run Rate batting first v/s chasing at different venues',
            {'Q': 'limegreen', 'E': 'greenyellow'}, 'Stadium',
            'Count', 'venue_trends_4',
            color='Batting/Chasing Team', barmode='group')


def individual_batting_stats(df):
    """
    function to generate individual tournament batting stats
    :param df: dataframe
    :return: None
    """

    df = df[df['Runs'] > 100]
    get_scatter(df, 'Runs', 'SR', 'Team', 'Player',
                'Runs v/s Strike Rate scatter chart',
                'Runs', 'Strike Rate',
                'scatter_1', 'Runs')

    get_scatter(df, 'Runs', 'Ave', 'Team', 'Player',
                'Runs v/s Average scatter chart',
                'Runs', 'Average', 'scatter_2',
                'Runs')


def individual_bowling_stats(df):
    """
    function to generate individual bowling stats
    :param df: dataframe
    :return: none
    """

    df = df[df['Wkts'] > 5]
    get_scatter(df, 'Wkts', 'SR', 'Team', 'Player',
                'Wickets v/s Strike Rate scatter chart',
                'Wickets', 'Strike Rate',
                'scatter_3', 'Wkts')

    get_scatter(df, 'Wkts', 'Econ', 'Team', 'Player',
                'Wickets v/s Economy scatter chart',
                'Wickets', 'Economy',
                'scatter_4', 'Wkts')
    get_scatter(df, 'Wkts', 'Ave', 'Team', 'Player',
                'Wickets v/s Bowling Average scatter chart',
                'Wickets', 'Average',
                'scatter_5', 'Wkts')


def run_analysis():
    """
    function to perform analysis
    :return: None
    """

    # get team standings on the points table (group stages)
    df_points = pd.read_csv('data/processed/points_table.csv')
    get_team_standings(df_points)

    # analyze matchwise trend of each team during group stages
    df_match_summary = pd.read_csv('data/processed/match_summary.csv')
    get_teams_matchwise_trend(df_match_summary)

    # get average runs per wicket and
    # average wickets lost per match for all teams (group stages)
    # get correlation among these factors
    get_overall_team_stats(df_match_summary, df_points)

    # get winning and losing trend of teams batting 1st and second
    df_first_inning_wins, df_second_inning_wins = \
        get_first_second_innings_stats(df_match_summary)

    # make a copy of 2nd innings win %age df
    df_second_inning_wins_batting_2nd = df_second_inning_wins.copy()

    # get stats of teams batting and bowling first
    df_batting_scorecards = \
        pd.read_csv('data/processed/Batting_Scorecards.csv')
    df_bowling_scorecards = \
        pd.read_csv('data/processed/Bowling_Scorecards.csv')
    first_innings_detailed_stats(df_match_summary, df_first_inning_wins,
                                 df_second_inning_wins, df_batting_scorecards,
                                 df_bowling_scorecards)

    # second_innings_bowling
    second_innings_detailed_stats(df_match_summary, df_first_inning_wins,
                                  df_second_inning_wins_batting_2nd,
                                  df_batting_scorecards, df_bowling_scorecards)

    # toss analysis
    toss_analysis(df_match_summary, df_points)

    # venue trends analysis
    venue_trends(df_match_summary)

    # Individual Batting stat analysis
    df_bat_stat = pd.read_csv('data/processed/'
                              'batting-most-runs-career.csv')
    individual_batting_stats(df_bat_stat)

    # Individual Bowling stat analysis
    df_bowl_stat = pd.read_csv('data/processed/'
                               'bowling-most-wickets-career.csv')
    individual_bowling_stats(df_bowl_stat)
