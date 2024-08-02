from src.utils.utils import save_df, get_soup, \
    segregate_odd_even_indices, get_links, get_first_innings_overs
import pandas as pd

# main url that has data
root_link = 'https://www.espncricinfo.com/'


def get_points_table(url):
    """
    function to get the points table from group stages csv
    :param url: url containing points table
    :return: None
    """

    columns_points_table, final_records = [[] for _ in range(2)]

    # Web scraping contents using beautiful soup
    soup = get_soup(url)

    # getting table columns
    for tag in soup.find_all('tr', class_='cb-srs-gray-strip'):

        for inner_tag in tag:

            if inner_tag.text.strip() != '':

                columns_points_table.append(inner_tag.text.strip())

    # getting table contents
    odd_tr_count = 0

    for tag in soup.find('tbody'):

        if odd_tr_count % 2 == 0:

            country_record = []

            for inner_tag in tag:

                if inner_tag.text.strip() != '':

                    country_record.append(inner_tag.text.strip())
                    continue

                final_records.append(country_record)

        odd_tr_count += 1

    # save dataframes
    save_df(columns_points_table, final_records, 'raw', 'points_table')


def get_match_schedule_results(url):
    """
    function to return the match schedule and results csv
    :param url: url containing match schedule and results
    :return: None
    """

    column_matches, final_records = [[] for _ in range(2)]

    # Web scraping contents using beautiful soup
    soup = get_soup(url)

    # getting table columns
    for tag in soup.find('thead',
                         class_='ds-bg-fill-content-alternate ds-text-left'):

        for inner_tag in tag:

            column_matches.append(inner_tag.text)

    # getting table rows
    for tr_content in soup.find('tbody'):

        td_elements = tr_content.find_all('td')

        match_record = []

        for td_content in td_elements[:-1]:

            match_record.append(td_content.text)

        # getting the link to match scorecard
        match_record.append(td_elements[-1].find('a').get('href'))
        final_records.append(match_record)

    # save dataframes
    # final_records = final_records[::-1]

    save_df(column_matches, final_records, 'raw', 'match_schedule_results')


def get_match_facts(file_path):
    """
    function to return general match facts csv
    :param file_path: csv file containing match schedule and results
    :return: None
    """

    # Read the dataframe containing match schedule and results
    df = pd.read_csv(file_path)

    team_scores, man_of_match, pp_1, pp_2, pp_3, \
        team_1_pp_1, team_1_pp_2, team_1_pp_3, \
        team_2_pp_1, team_2_pp_2, team_2_pp_3, \
        toss, play_time, final_records, team_1_overs = \
        [[] for _ in range(15)]

    match_num = 1  # match counter

    # loop over the match links
    for url in df['Scorecard']:

        # Web scraping contents using beautiful soup
        soup = get_soup(root_link + url)

        # Team scores
        for inner_div in soup.find('div',
                                   class_='ds-flex ds-flex-col ds-mt-3 '
                                          'md:ds-mt-0 ds-mt-0 ds-mb-1'):
            count = 0

            for inner_tag in inner_div:
                count += 1

                if count == 1:
                    continue

                team_scores.append(inner_tag.text)

        # Man of Match
        tag = soup.find('span',
                        class_='ds-text-tight-m ds-font-medium ds-text-typo '
                               'ds-underline ds-decoration-ui-stroke '
                               'hover:ds-text-typo-primary '
                               'hover:ds-decoration-ui-stroke-primary '
                               'ds-block ds-cursor-pointer')

        man_of_match.append(tag.text)

        # Powerplay_scores
        for tag in soup.find_all('ul', class_='ds-text-tight-s '
                                              'ds-font-regular '
                                              'ds-list-disc ds-pt-2 '
                                              'ds-px-4 ds-mb-4'):
            flag = False

            for inner_div in tag:

                if 'Powerplay 1:' in inner_div.text:
                    pp_1.append(inner_div.text)

                elif 'Powerplay 2:' in inner_div.text:
                    pp_2.append(inner_div.text)

                elif 'Powerplay 3:' in inner_div.text:
                    pp_3.append(inner_div.text)
                    flag = True

                elif 'Innings Break' in inner_div.text:
                    team_1_overs.append(get_first_innings_overs
                                        (inner_div.text))

            if not flag and match_num != 36:  # Match 36 has redundant li
                pp_3.append('NA')

        match_num += 1

        # Toss results and playing time
        for tbody in soup.find('table', class_='ds-w-full ds-table '
                                               'ds-table-sm ds-table-auto'):

            for tr_element in tbody:

                if 'Toss' in tr_element.text:
                    toss.append(tr_element.text)

                elif 'Hours of play (local time)' in tr_element.text:
                    play_time.append(tr_element.text)

    # Segregating odd and even indices of lists for both teams
    team_1_scores, team_2_scores = segregate_odd_even_indices(team_scores)
    team_1_pp_1, team_2_pp_1 = segregate_odd_even_indices(pp_1)
    team_1_pp_2, team_2_pp_2 = segregate_odd_even_indices(pp_2)
    team_1_pp_3, team_2_pp_3 = segregate_odd_even_indices(pp_3)

    # setting table rows
    columns = ['Time', 'Toss', 'Team 1 Score', 'Team 1 Overs', 'Team 2 Score',
               'Team 1 PP-1 Score', 'Team 1 PP-2 Score',
               'Team 1 PP-3 Score', 'Team 2 PP-1 Score',
               'Team 2 PP-2 Score', 'Team 2 PP-3 Score', 'MOM']

    # getting table columns
    for i in range(len(toss)):
        final_records.append(
            [play_time[i], toss[i], team_1_scores[i],
             team_1_overs[i], team_2_scores[i],
             team_1_pp_1[i], team_1_pp_2[i], team_1_pp_3[i],
             team_2_pp_1[i], team_2_pp_2[i], team_2_pp_3[i], man_of_match[i]])

    # save dataframes
    save_df(columns, final_records, 'raw', 'match_facts')


def get_records(links):
    """
    function to generate batting and bowling stats of world cup csv
    :param links: links to individual stats page
    :return: None
    """

    for url in links:

        # Web scraping contents using beautiful soup
        soup = get_soup(root_link + url)
        thead = soup.find('thead')

        # getting table columns
        columns, final_records = [[] for _ in range(2)]

        for inner_tr in thead:

            for td_element in inner_tr:

                columns.append(td_element.text)

        # getting table rows
        tbody = soup.find('tbody')

        for inner_tr in tbody:

            record = []

            for inner_td in inner_tr:

                record.append(inner_td.text)

            final_records.append(record)

        save_df(columns, final_records, 'raw', url[20:].split('/')[0])


def get_match_bowling_stats(file_path):
    """
    function to retrieve match by match bowling figures csv
    :param file_path: csv file containing match schedule and results
    :return: None
    """

    # Read the dataframe containing match schedule and results
    df = pd.read_csv(file_path)

    match_count = 1
    records = []

    # loop over the match links
    for link in df['Scorecard']:

        # Web scraping contents using beautiful soup
        soup = get_soup(root_link + link)

        batting_team_name = []

        for tag in soup.find_all('span', class_='ds-text-title-xs '
                                                'ds-font-bold ds-capitalize'):

            batting_team_name.append(tag.text)

        bowling_team_name = batting_team_name[::-1]  # 1st batting = 2nd bowl

        i = 0  # team name index

        # bowling stats per match
        for tag in soup.find_all('thead', class_='ds-bg-fill-content-'
                                                 'alternate '
                                                 'ds-text-left')[1:4:2]:

            tbody = tag.find_next_sibling()

            for tr in tbody.find_all('tr',
                                     class_=lambda cl: cl != 'ds-hidden'):

                bowler_stat_list, bowler = [], ''
                td_tag_bowler = tr.find('td', class_='ds-flex '
                                                     'ds-items-center')

                if td_tag_bowler:

                    bowler = td_tag_bowler.text

                for td_tag in tr.find_all('td', class_='ds-w-0 '
                                                       'ds-whitespace'
                                                       '-nowrap '
                                                       'ds-min-w-max '
                                                       'ds-text-right'):

                    bowler_stat_list.append(float(td_tag.text))

                bowler_stat_list.insert(0, bowler)
                bowler_stat_list.insert(0, bowling_team_name[i])
                bowler_stat_list.insert(0, match_count)
                records.append(bowler_stat_list)

            i += 1

        match_count += 1

    # getting table columns
    columns = ['Match', 'Team', 'Bowler Name', 'Overs', 'Maidens',
               'Runs Conceded', 'Economy', '0s', '4s', '6s', 'WD',
               'NB']

    # save dataframes
    save_df(columns, records, 'raw', 'Bowling_Scorecards')


def get_match_batting_stats(file_path):
    """
    function to retrieve match by match batting figures csv
    :param file_path: csv file containing match schedule and results
    :return: None
    """

    # Read the dataframe containing match schedule and results
    df = pd.read_csv(file_path)

    match_count = 1  # counter for match number
    final_records = []

    # loop over the match links
    for link in df['Scorecard']:

        batting_team_name = []

        # Web scraping contents using beautiful soup
        soup = get_soup(root_link + link)

        # get batting team names
        for tag in soup.find_all('span', class_='ds-text-title-xs '
                                                'ds-font-bold ds-capitalize'):

            batting_team_name.append(tag.text)

        final_batting_scorecard = []

        for table in soup.find_all('table',
                                   class_='ds-w-full ds-table '
                                          'ds-table-md '
                                          'ds-table-auto '
                                          'ci-scorecard-table'):

            team_scorecard = []

            # lambda function to avoid tr tags with specific class names
            for tr in table.find_all('tr',
                                     class_=lambda v: v != 'ds-hidden'
                                     and v != 'ds-text-tight-s')[1:-2]:

                ignore_td = 0

                for td_tag in tr:

                    if td_tag.text == 'TOTAL':

                        break

                    if ignore_td != 1 and ignore_td != 4:

                        team_scorecard.append(td_tag.text)

                    ignore_td += 1

            final_batting_scorecard.append(team_scorecard)

        i = 0  # team name index

        for scorecard in final_batting_scorecard:

            scorecard.insert(0, batting_team_name[i])
            scorecard.insert(0, match_count)
            final_records.append(scorecard)
            i += 1

        match_count += 1

    # setting table columns
    columns = ['Match', 'Team']

    for i in range(1, 12):

        columns.append(f'Batting_{i}_name')
        columns.append(f'Batting_{i}_runs')
        columns.append(f'Batting_{i}_balls')
        columns.append(f'Batting_{i}_4s')
        columns.append(f'Batting_{i}_6s')
        columns.append(f'Batting_{i}_strike_rate')

    save_df(columns, final_records, 'raw', 'Batting_Scorecards')


def get_venue_list(link):
    """
    function to get venue info csv
    :param link: url containing tournament venues
    :return: None
    """

    # Web scraping contents using beautiful soup
    soup = get_soup(link)

    # getting table records
    records = []

    for div in soup.find_all('div', class_='cb-col cb-col-100 '
                                           'cb-lst-itm cb-pos-rel '
                                           'cb-lst-itm-lg'):

        venue_name = div.find('h2', class_='cb-nws-hdln cb-font-18 line-ht24')
        city_name = div.find('div', class_='cb-nws-intr')
        records.append([venue_name.text, city_name.text])

    # setting table columns
    columns = ['Stadium', 'City']

    # save dataframes
    save_df(columns, records, 'raw', 'venues')


def get_data():
    """
    Function to call the methods in get_data
    :return: None
    """

    # retrieve points table raw data
    points_table_url = ('https://www.cricbuzz.com/cricket-series/6732/'
                        'icc-cricket-world-cup-2023/points-table')
    get_points_table(points_table_url)

    # retrieve matches table raw data
    matches_url = ('https://www.espncricinfo.com/records/tournament/'
                   'team-match-results/icc-cricket-world-cup-2023-24-15338')
    get_match_schedule_results(matches_url)

    # reverse the records in match schedule results
    df_reverse = pd.read_csv('data/raw/match_schedule_results.csv')
    df_reversed = df_reverse[::-1]
    df_reversed.to_csv('data/raw/match_schedule_results.csv', index=False)

    # retrieve general match facts
    get_match_facts('data/raw/match_schedule_results.csv')

    # retrieve bowling record links and remove links for redundant records
    link = get_links(root_link + 'records/tournament/icc-cricket-world-cup'
                                 '-2023-24-15338', 'Bowling records')
    del link[-3:]

    # retrieve bowling stats
    get_records(link)

    # retrieve batting record links and remove links for redundant records
    link = get_links('https://www.espncricinfo.com/records/tournament/'
                     'icc-cricket-world-cup-2023-24-15338', 'Batting records')
    del link[-3:]
    del link[1]

    # retrieve batting stats
    get_records(link)

    # retrieve bowling and batting scorecard for every match
    get_match_bowling_stats('data/raw/match_schedule_results.csv')
    get_match_batting_stats('data/raw/match_schedule_results.csv')

    # retrieve tournament venues
    get_venue_list('https://www.cricbuzz.com/cricket-series/6732/'
                   'icc-cricket-world-cup-2023/venues')
