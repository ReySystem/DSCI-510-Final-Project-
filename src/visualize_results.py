import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import geopandas as gpd
import plotly.express as px
import pandas as pd
import folium
from src.utils.utils import get_longitude_and_latitude
import os

# Create a directory to save the HTML files
plot_directory_path = 'results/plots/analysis_plots'
os.makedirs(plot_directory_path, exist_ok=True)


def get_msno_matrix(df, directory_path, filename):
    """
    function to generate missing value matrix
    :param df: dataframe
    :param directory_path: path of the file
    :param filename: file_name
    :return: None
    """

    # Create the missing values matrix
    msno.matrix(df)

    # base folder
    base_directory = 'results'

    # Create full paths
    full_directory_path = os.path.join(base_directory, directory_path)

    # Check and create directories
    if not os.path.exists(full_directory_path):
        os.makedirs(full_directory_path)

    # Save the visualization as an image
    output_path = os.path.join(full_directory_path, filename.split('.')[0])

    plt.savefig(output_path)


def get_participating_nations(country_list):
    """
    function to generate map of nations participating
    :param country_list: list of countries
    :return: None
    """

    # Host country
    countries_to_highlight_red = [country_list[0]]

    # Remaining countries
    countries_to_highlight_blue = country_list[1:]

    # Create a DataFrame with all countries
    all_countries = country_list
    df = pd.DataFrame({'Country': all_countries})

    # Load GeoDataFrame of world countries
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # Merge the GeoDataFrame with the DataFrame based on country names
    world = world.merge(df, how='left', left_on='name', right_on='Country')

    # Create a new column to indicate the color category
    world.loc[world['Country'].isin(countries_to_highlight_red),
              'color_category'] = 'Host'
    world.loc[world['Country'].isin(countries_to_highlight_blue),
              'color_category'] = 'Participating Nations'

    # Plotly Choropleth map with highlighted India in blue,
    # specified countries in red, and others in white
    fig = px.choropleth(
        world,
        geojson=world.geometry,
        locations=world.index,
        color='color_category',  # Use the 'color_category' column for color
        color_discrete_map={'Red': 'red', 'Blue': 'blue',
                            'Other': 'white'},  # Map colors
        title='Countries part of Men\'s ICC Cricket World Cup 2023',
        projection='natural earth',
        hover_data=[],  # Remove all hover effects
        labels={'color_category': 'Countries'}  # Rename the legend label
    )

    # Customize the layout to center the title horizontally
    fig.update_layout(title_x=0.5)

    # Create a folder to save the image
    folder_path = 'results/plots/maps'
    os.makedirs(folder_path, exist_ok=True)

    # Save the image locally
    image_path = os.path.join(folder_path, 'world_map.png')
    fig.write_image(image_path)


def plot_point_on_map(co_ordinates_list):
    """
    function to generate map of indian venues for world cup
    :param co_ordinates_list: list of tuples with latitudes and longitudes
    :return: None
    """

    # Create a folium map centered around India
    india_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

    for latitude, longitude, address in co_ordinates_list:
        # Add a marker for the specified location with a label on hover
        marker = folium.Marker(location=[latitude, longitude],
                               icon=folium.Icon())
        marker.add_child(folium.Tooltip(address))
        marker.add_to(india_map)

    # Save the map to an HTML file or display it
    india_map.save('results/plots/maps/india_map.html')


def get_line_chart(df, xlabel, ylabel, title, filename, color=None):
    """
    function to generate plotly line chart
    :param df: dataframe
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param title: figure title
    :param filename: filename to be saved with
    :param color: df feature on which color is based on
    :return: None
    """

    # Create a line chart
    fig = px.line(df, x=xlabel, y=ylabel, color=color, markers=True,
                  title=title)

    # Customize layout
    fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel, title_x=0.5)

    # Save the plot as an HTML file in the specified directory
    file_path = os.path.join(plot_directory_path, f'{filename}.html')
    fig.write_html(file_path)


def get_bar(df, x, y, title, color_discrete_map,
            xaxis_title, yaxis_title, filename,
            color=None, barmode='relative'):
    """
    function to generate plotly barplot
    :param df: dataframe
    :param x: x feature
    :param y: y feature
    :param title: figure title
    :param color_discrete_map: dictionary with color mapping
    :param xaxis_title: xlabel
    :param yaxis_title: ylabel
    :param filename: filename to be saved with
    :param color: df feature on which color is based on
    :barmode: type of bar plot (relative/grouped)
    :return: None
    """

    # Create a Plotly Express bar plot
    fig = px.bar(df, x=x, y=y, color=color,
                 title=title,
                 color_discrete_map=color_discrete_map, barmode=barmode)

    # Customize layout
    fig.update_layout(xaxis_title=xaxis_title,
                      yaxis_title=yaxis_title, title_x=0.5)

    # Save the plot as an HTML file in the specified directory
    file_path = os.path.join(plot_directory_path, f'{filename}.html')
    fig.write_html(file_path)


def get_dotted_bar(df, x, y, title, xaxis_title,
                   yaxis_title, filename, line_data):
    """
    function to plot the bar chart with the average value as a dotted line
    :param df: dataframe
    :param x: x feature
    :param y: y feature
    :param title: figure title
    :param xaxis_title: xlabel
    :param yaxis_title: ylabel
    :param filename: filename to be saved with
    :param color: df feature on which color is based on
    :param line_data: average value as dotted line
    :return: None
    """

    # Plot bar chart
    fig = px.bar(df, x=x, y=y, title=title)

    # Customize layout
    fig.update_layout(title_x=0.5,
                      xaxis_title=xaxis_title,
                      yaxis_title=yaxis_title)

    # Add average run rate as dotted lines
    fig.add_shape(
        type='line',
        x0=-0.5,
        x1=len(df[x]) - 0.5,
        y0=line_data,
        y1=line_data,
        line=dict(color='red', width=2, dash='dash')
    )

    # Save the plot as an HTML file in the specified directory
    file_path = os.path.join(plot_directory_path, f'{filename}.html')
    fig.write_html(file_path)


def heatmap(df, filename, title):
    """
    function to generate correlation heatmap
    :param df: dataframe
    :param filename: filename to be saved with
    :param title: Heatmap title
    :return: None
    """

    # setting the figure size
    plt.figure(figsize=(10, 8))

    # Create a mask to hide the upper triangle
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))

    # plotting the heatmap
    sns.heatmap(df.corr(), annot=True, cmap='inferno',
                fmt=".2f", linewidths=.5, mask=mask)

    plt.title(title)

    # Save the figure as an image file (e.g., PNG)
    plt.tight_layout()
    plt.savefig(plot_directory_path + '/' + filename + '.png')


def get_pie(df, names, title, filename):
    """
    function to generate pie chart
    :param df: dataframe
    :param names: column name
    :param title: Title of the pie chart
    :param filename: filename to be saved with
    :return: None
    """

    # Create a pie chart using Plotly
    fig = px.pie(df, names=names, title=title)

    # Set the title to center
    fig.update_layout(title=title, title_x=0.5)

    # Save the plot as an HTML file in the specified directory
    file_path = os.path.join(plot_directory_path, f'{filename}.html')
    fig.write_html(file_path)


def get_scatter(df, x, y, color, hover, title,
                x_axis_title, y_axis_title, filename, size):
    """
    function to generate a scatter chart
    :param df: dataframe
    :param x: x feature
    :param y: y feature
    :param color: column based on which the points are color coded
    :param hover: data to show on hover
    :param title: figure title
    :param x_axis_title: xlabel
    :param y_axis_title: ylabel
    :param filename: filename to be saved with
    :param size column which determines the size of the points
    :return: None
    """

    # Create a scatter plot with color-coded countries
    fig = px.scatter(df, x=x, y=y, color=color, hover_data=[hover], size=size)

    # Customize the layout
    fig.update_layout(
        title=title,
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        showlegend=True,
        title_x=0.5
    )

    # Save the plot as an HTML file in the specified directory
    file_path = os.path.join(plot_directory_path, f'{filename}.html')
    fig.write_html(file_path)


def visualize_results():
    """
    function to run the visualization
    :return: None
    """

    # get participating nations map
    df_points = pd.read_csv('data/processed/points_table.csv')

    get_participating_nations(list(df_points['Teams']))

    # get indian venues map
    df_match_summary = pd.read_csv('data/processed/match_summary.csv')

    venue_city_list = df_match_summary[['Stadium', 'City']]. \
        to_records(index=False).tolist()
    co_ordinates_list = []

    for venue, city in list(set(venue_city_list)):
        address = venue + ', ' + city
        latitude, longitude = get_longitude_and_latitude(address)
        co_ordinates_list.append([latitude, longitude, address])

    plot_point_on_map(co_ordinates_list)
