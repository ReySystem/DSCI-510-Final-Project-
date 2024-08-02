import os
import shutil
import pandas as pd
from src.visualize_results import get_msno_matrix
from src.utils.utils import get_file_list
import re

files_with_null_values = dict()


def move_files_to_new_path(file_list, new_path='data/processed/'):
    """
    function to move files from raw directory to processed directory
    :param file_list: raw directory path
    :param new_path: processed directory path
    :return: None
    """

    # Move each file to the new path
    for file_path in file_list:
        # Extract the filename from the path
        file_name = os.path.basename(file_path)

        # Build the destination path in the new directory
        destination_path = os.path.join(new_path, file_name)

        # Copy the file
        shutil.copy(file_path, destination_path)


def remove_files(file_list):
    """
    function to remove files from directory
    :param file_list: list of files to be removed
    :return: None
    """

    for file_path in file_list:

        try:

            # Use os.remove to delete the file
            os.remove(file_path)

        except OSError:

            raise Exception('No file found')


def remove_non_ascii_characters(df):
    """
    function to filter out non-ascii characters from records
    :param df: dataframe
    :return: cleaned dataframe with only ascii characters
    """

    # Use applymap to apply the lambda function to each element
    # in the DataFrame
    df_cleaned = df.applymap(lambda x: ''.join(char for char in str(x)
                                               if 32 <= ord(char) <= 126))

    return df_cleaned


def remove_columns_from_csv(df, file, columns_to_remove):
    """
    function to remove dataframe columns
    :param df: dataframe
    :param file: csv file
    :param columns_to_remove: list of columns
    :return: None
    """

    try:

        # Identify columns that exist in both the DataFrame
        # and the list to remove
        common_columns = set(df.columns). \
            intersection(columns_to_remove)

        # Remove specified columns if they exist
        df = df.drop(columns=common_columns, errors='ignore')

        # Save the modified DataFrame back to the CSV file
        df.to_csv(file, index=False)

    except Exception:

        raise Exception('Some error')


def extract_and_remove_text_within_parentheses(df, file):
    """
    function to map country names with codes and separating them from names
    :param df: dataframe
    :param file: csv filename
    :return: None
    """

    # mapping of country codes and country names
    country_codes = {'IND': 'India', 'SA': 'South Africa',
                     'NZ': 'New Zealand', 'AUS': 'Australia',
                     'PAK': "Pakistan", 'AFG': 'Afghanistan',
                     'ENG': "England", 'BAN': 'Bangladesh',
                     'SL': 'Sri Lanka', 'NED': 'Netherlands'}

    # Use regular expression to extract text within parentheses
    df['Team'] = df['Player'].str.extract(r'\((.*?)\)',
                                          expand=False).map(country_codes)

    # Remove the extracted text from the original column
    df['Player'] = df['Player'].str.replace(r'\(.*?\)', '').str.strip()

    df.to_csv(file, index=False)


def get_columns_with_null_values(df, file):
    """
    get list of columns having null records
    :param df: dataframe
    :param file: filename
    :return: None
    """

    # Get a list of columns with null values
    columns_with_null = df.columns[df.isnull().any()].tolist()

    if columns_with_null:
        files_with_null_values[file] = columns_with_null


def fill_null_values_by_column(df, fill_values):
    """
    function to fill null values by given values in the dict
    :param df: dataframe
    :param fill_values: dict containing values to be filled in
    :return: None
    """

    # Fill specified columns with the corresponding fill values
    df_filled = df.fillna(fill_values)
    return df_filled


def impute_missing(file):
    """
    function to fill missing records
    :param file: filepath
    :return: None
    """

    df = pd.read_csv(file)

    # Replace all occurrences of '-' with 0
    df.replace('-', 0, inplace=True)

    # Save the csv file
    df.to_csv(file, index=False)


def get_team_runs_wickets(df_facts, runs_column, wickets_column, main_column):
    """
    function to get teamwise runs and wickets
    :param df_facts: input dataframe
    :param runs_column: Column name for runs
    :param wickets_column: Column name for wickets
    :param main_column: column from where runs and wickets data is retrieved
    :return: modified dataframe
    """

    # Create two new columns by splitting the original column
    df_facts[[runs_column, wickets_column]] = df_facts[main_column].str. \
        split('/', n=1, expand=True)

    # Convert the 'Runs' column to numeric
    df_facts[runs_column] = pd.to_numeric(df_facts[runs_column],
                                          errors='coerce').astype('Int64')

    # Fill missing values in 'Wickets' with 10
    df_facts[wickets_column].fillna('10', inplace=True)

    # Convert the 'Wickets' column to numeric
    df_facts[wickets_column] = pd.to_numeric(df_facts[wickets_column],
                                             errors='coerce').astype('Int64')

    # Drop the original column
    df_facts.drop(main_column, axis=1, inplace=True)

    return df_facts


def clean_match_facts(df_facts, df_schedule_results, df_venues):
    """
    function to clean match_facts.csv
    :param df_facts: match_facts dataframe
    :param df_schedule_results: match_schedule_results dataframe
    :param df_venues: venues dataframe
    :return: None
    """

    # Replace values in the specified column
    df_facts['Time'] = df_facts['Time'].replace({'Hours of play '
                                                 '(local time)14.00 '
                                                 'start, First Session '
                                                 '14.00-17.30, '
                                                 'Interval 17.30-18.00, '
                                                 'Second Session '
                                                 '18.00-21.30': 'DN',
                                                 'Hours of play '
                                                 '(local time)10.30 start, '
                                                 'First Session 10.30-14.00, '
                                                 'Interval 14.00-14.30, '
                                                 'Second Session '
                                                 '14.30-18.00': 'D'})

    # Extract team name and decision from the "toss" column
    df_facts['Won_Toss'] = df_facts['Toss'].str.extract(r'Toss([\w\s]+),')
    df_facts['Decision_Toss'] = df_facts['Toss']. \
        str.extract(r'elected to (\w+)')

    # Replace missing values in the "Decision" column with 'Bat'
    df_facts['Decision_Toss'].fillna('Bat', inplace=True)

    # Drop the original "toss" column
    df_facts.drop('Toss', axis=1, inplace=True)

    # Concatenate the two DataFrames along the columns (axis=1)
    df_facts = pd.concat([df_facts, df_schedule_results], axis=1)

    # Remove 'overs' and convert to numeric
    df_facts['Team 1 Overs'] = pd.to_numeric(df_facts['Team 1 Overs'].
                                             str.replace(' overs', ''),
                                             errors='coerce')

    # Issue with sa vs ned match thus removing
    # the '(data)' part from the 'Column_Name' column
    df_facts['Team 1 Score'] = \
        df_facts['Team 1 Score'].str.replace(r'\(.*?\)', '', regex=True)

    df_facts = get_team_runs_wickets(df_facts,
                                     'Team 1 Runs Scored',
                                     'Team 1 Wickets Lost',
                                     'Team 1 Score')

    old_df = df_facts.copy()

    # regex to get team 2 overs for match 34
    pattern = r'\b\d+\.\d+'

    # getting team 2 overs for match 34
    dls_match_score = old_df.loc[34, 'Team 2 Score']

    matches = re.findall(pattern, dls_match_score)
    dls_match_overs = float(matches[0])

    # regex to get team 2 overs
    pattern = r'\((\d+(?:\.\d+)?)(?:\/\d+)? ov, T:\d+\)'

    # Apply the pattern using str.extract
    extracted_info = df_facts['Team 2 Score'].str.extract(pattern,
                                                          expand=True)

    # Rename the column
    extracted_info.columns = ['Team 2 Overs']

    # Convert the extracted column to numeric
    extracted_info['Team 2 Overs'] = \
        pd.to_numeric(extracted_info['Team 2 Overs'], errors='coerce')

    # Fill missing values (NaN) with the value 50
    extracted_info['Team 2 Overs'].fillna(50, inplace=True)

    # Join with the original DataFrame
    df_facts = df_facts.join(extracted_info)

    # handle the case of rain curtailed match (dls)
    df_facts.loc[34, 'Team 2 Overs'] = dls_match_overs

    # Remove the data within parentheses part from the 'Team 2 Score' column
    df_facts['Team 2 Score'] = \
        df_facts['Team 2 Score'].str.replace(r'\(.*?\)', '', regex=True)

    df_facts = \
        get_team_runs_wickets(df_facts, 'Team 2 Runs Scored',
                              'Team 2 Wickets Lost',
                              'Team 2 Score')

    # Remove outer part in powerplays
    powerplay_list = ['Team 1 PP-1 Score', 'Team 1 PP-2 Score',
                      'Team 1 PP-3 Score', 'Team 2 PP-1 Score',
                      'Team 2 PP-2 Score', 'Team 2 PP-3 Score']

    # Get team and powerplay-wise runs and wickets
    for powerplay in powerplay_list:
        df_facts[powerplay] = df_facts[powerplay].str.extract(r'\((.*?)\)')
        runs_wickets_info = \
            df_facts[powerplay].str.extract(r'(\d+)\s*run.*?(\d+)\s*wicket')

        # create column_names
        runs_column = powerplay[:-6] + ' Runs Scored'
        wickets_column = powerplay[:-6] + ' Wickets Lost'

        # Create new columns for runs and wickets
        df_facts[runs_column] = \
            pd.to_numeric(runs_wickets_info[0],
                          errors='coerce').astype('Int64')
        df_facts[wickets_column] = \
            pd.to_numeric(runs_wickets_info[1],
                          errors='coerce').astype('Int64')

        # Drop the existing powerplay column
        df_facts.drop(powerplay, axis=1, inplace=True)

    # Fill empty values with 0 (PP3 Runs and wickets)
    df_facts.fillna(0, inplace=True)

    # Incorrect Ground name replacement
    df_facts['Ground'].replace('Wankhede', 'Mumbai', inplace=True)
    df_facts['Ground'].replace('Eden Gardens', 'Kolkata', inplace=True)

    # Renaming Column Ground as City
    df_facts.rename(columns={'Ground': 'City'}, inplace=True)

    # Merge DataFrames and keep the 'city' column from df_venues
    df_facts = \
        pd.merge(df_facts, df_venues[['Stadium', 'City']],
                 on='City', how='left')

    # Reorder the columns
    desired_order = ['Time', 'Stadium', 'City', 'Team 1', 'Team 2',
                     'Won_Toss', 'Decision_Toss',
                     'Team 1 Runs Scored', 'Team 1 Wickets Lost',
                     'Team 1 Overs', 'Team 2 Runs Scored',
                     'Team 2 Wickets Lost', 'Team 2 Overs',
                     'Team 1 PP-1 Runs Scored',
                     'Team 1 PP-1 Wickets Lost',
                     'Team 1 PP-2 Runs Scored',
                     'Team 1 PP-2 Wickets Lost',
                     'Team 1 PP-3 Runs Scored',
                     'Team 1 PP-3 Wickets Lost',
                     'Team 2 PP-1 Runs Scored',
                     'Team 2 PP-1 Wickets Lost',
                     'Team 2 PP-2 Runs Scored',
                     'Team 2 PP-2 Wickets Lost',
                     'Team 2 PP-3 Runs Scored',
                     'Team 2 PP-3 Wickets Lost',
                     'MOM', 'Winner']

    df_summary = df_facts[desired_order]

    # Save the modified DataFrame to a new file
    df_summary.to_csv('data/processed/match_summary.csv', index=False)


def change_dtype_of_columns(df, column_names, new_dtype):
    """
    function to change data type of columns
    :param df: input dataframe
    :param column_names: list of column names
    :param new_dtype: new dtype to be assigned
    :return: modified df
    """

    # Change the data type of specified columns in-place
    df[column_names] = df[column_names].astype(new_dtype)
    return df


def preprocess_pts_table(df, file):
    """
    function to separate qualification column from team
    :param df: points table csv
    :param file: file path
    :return: None
    """

    # Use regular expression to extract text within parentheses
    df['Qualification_Status'] = \
        df['Teams'].str.extract(r'\((.*?)\)', expand=False)

    # Remove the extracted text from the original column
    df['Teams'] = df['Teams'].str.replace(r'\(.*?\)', '').str.strip()

    df.to_csv(file, index=False)


def correct_team_positions_records(df_match_summary, df_batting_scorecards):
    # Group by 'Match No.' and create a list of teams for each match
    grouped_df = df_batting_scorecards.groupby('Match')['Team']. \
        apply(list).reset_index()

    # Convert the grouped DataFrame to a list of lists
    result_list = grouped_df['Team'].values.tolist()

    # Replace the old values in the records with the new ones
    df_match_summary[['Team 1', 'Team 2']] = pd.DataFrame(result_list)

    # missing data for sl vs ned match 19
    df_match_summary.at[18, 'Team 2 PP-3 Runs Scored'] = 40
    df_match_summary.at[18, 'Team 2 PP-3 Wickets Lost'] = 1

    # missing data for ind vs aus final match
    df_match_summary.at[47, 'Team 2 PP-3 Runs Scored'] = 18
    df_match_summary.at[47, 'Team 2 PP-3 Wickets Lost'] = 1

    # save the modified df
    df_match_summary.to_csv('data/processed/match_summary.csv')


def clean_data():
    """
    function to clean data
    :return: None
    """

    directory_path = 'data/raw'
    file_pattern = '*.csv'
    new_directory_path = 'data/processed'

    # get a list of CSV files
    csv_files = get_file_list(directory_path, file_pattern)

    # move to processed directory
    move_files_to_new_path(csv_files)

    # remove unnecessary file
    rm_files = [new_directory_path + '/bowling-best-figures-innings.csv',
                new_directory_path + '/bowling-most-5wi-career.csv',
                new_directory_path + '/batting-most-hundreds-career.csv',
                new_directory_path + '/batting-most-ducks-career.csv',
                new_directory_path +
                '/batting-highest-career-batting-average.csv',
                new_directory_path +
                '/batting-highest-career-strike-rate.csv',
                new_directory_path + '/batting-most-fifties-career.csv',
                new_directory_path + '/batting-most-sixes-career.csv',
                new_directory_path +
                '/bowling-best-career-bowling-average.csv',
                new_directory_path + '/bowling-best-career-economy-rate.csv',
                new_directory_path + '/bowling-best-career-strike-rate.csv',
                new_directory_path + '/bowling-most-4wi-career.csv']

    print(rm_files)
    remove_files(rm_files)

    # get updated file list
    csv_files = get_file_list(new_directory_path, file_pattern)
    csv_files = [x for x in csv_files if x not in rm_files]

    # remove non-ascii characters and unimportant columns
    for file in csv_files:
        df = pd.read_csv(file)
        df_ascii = remove_non_ascii_characters(df)
        columns_to_remove = ['Span', '10', 'BBI', 'Scorecard', 'Tied',
                             'NR', 'HS', 'Balls', 'Margin',
                             'Match Date', 'Inns']
        remove_columns_from_csv(df_ascii, file, columns_to_remove)

    # replacing country codes with country names and
    # placing it in a new column
    for file in csv_files:

        if file.startswith(new_directory_path + '\\batting') or \
                file.startswith(new_directory_path + '\\bowling'):
            df = pd.read_csv(file)
            extract_and_remove_text_within_parentheses(df, file)

    # get dict where keys are file names and
    # values are columns having null values
    for file in csv_files:
        df = pd.read_csv(file)
        get_columns_with_null_values(df, file)

    # iterating over columns with null value
    for key, value in files_with_null_values.items():

        df = pd.read_csv(key)

        # string splitting the path
        filename = key.split('\\')[1]

        # Subdirectories
        with_null_value_directory = 'plots/with_null_value'
        without_null_value_directory = 'plots/without_null_value'

        # get null value matrix without cleaning
        get_msno_matrix(df, with_null_value_directory, filename)

        # filling null values
        fill_values = {}

        for column in value:

            if column.endswith('name') or column.endswith('Score'):

                fill_values[column] = 'Not Applicable'

            elif column.endswith('strike_rate'):

                fill_values[column] = 0.0

            else:

                fill_values[column] = 0

        df_filled = fill_null_values_by_column(df, fill_values)

        # saving the updated df
        df_filled.to_csv(key, index=False)

        # get null value matrix after cleaning
        get_msno_matrix(df_filled, without_null_value_directory, filename)

    # filling records where value is missing (i.e. '-' is present)
    for file in csv_files:
        impute_missing(file)

    # creating match summary
    df_facts = pd.read_csv(new_directory_path + '/match_facts.csv')
    df_schedule_results = \
        pd.read_csv(new_directory_path + '/match_schedule_results.csv')
    df_venues = \
        pd.read_csv(new_directory_path + '/venues.csv')
    clean_match_facts(df_facts, df_schedule_results, df_venues)

    # remove venues.csv, match_facts.csv and match_schedule_results.csv
    remove_files([new_directory_path + '/match_facts.csv',
                  new_directory_path + '/match_schedule_results.csv',
                  new_directory_path + '/venues.csv'])

    # cleaning Bowling_Scorecards.csv
    df_bowl_score = pd.read_csv(new_directory_path + '/Bowling_Scorecards.csv')
    df_bowl_score = change_dtype_of_columns(df_bowl_score,
                                            ['Maidens', 'Runs Conceded',
                                             '0s', '4s', '6s', 'WD', 'NB'],
                                            'Int64')

    # Drop the column with an empty name
    df_bowl_score = df_bowl_score.drop(columns=[''], errors='ignore')

    # remove the unnamed first column
    df_bowl_score.to_csv(new_directory_path +
                         '/Bowling_Scorecards.csv', index=False)

    # separating q column from points table
    df_pts = pd.read_csv(new_directory_path + '/points_table.csv')
    preprocess_pts_table(df_pts, new_directory_path + '/points_table.csv')

    # Correcting the order of Team 1 and Team 2
    # (i.e. team batting 1st - Team 1 and team batting 2nd - Team 2)
    df_match_summary = pd.read_csv(new_directory_path + '/match_summary.csv')
    df_batting_scorecards = pd.read_csv(new_directory_path +
                                        '/Batting_Scorecards.csv')
    correct_team_positions_records(df_match_summary, df_batting_scorecards)
