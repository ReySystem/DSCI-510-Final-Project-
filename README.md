[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/h_LXMCrc)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=12837995&assignment_repo_type=AssignmentRepo)
# DSCI 510 Final Project

## Name of the Project: Unveiling Insights through Data-driven Exploration of Cricket World Cup 2023

## Team Members (Name and Student IDs): Anshuman Mohanty (4257570790) and Kaustubh Sharma (1765749035)
## Project Structure:
 ```sh
├── data/
│ ├── processed/
│ └── raw/
├── results/
│ └── Plots/
│   └──analysis_plots/ # contains plots obtained from analysis
│   └──maps/ # contains the visualization maps
│   └──with_null_value/ # contains visualization for dataframes with missing values
│   └──without_null_value/ # contains dataframes after filling out missing values
│ └── final_report.pdf
   └── Inferences for Visualization.ipynb # ipynb containing detailed explanation for visualizations
   └── Inferences for Visualization.html # html file version having detailed explanation for visualizations
├── src/
│ ├──utils/
│   ├──utils.py # contains helper functions
│ ├──clean_data.py
│ ├──get_data.py
│ ├──run_analysis.py
│ ├──visualize_results.py
├──.gitignore
├──main.py # main file which calls execution of other project files
├──project_proposal
├──README.md
├──requirements.txt
  ```
## Instructions to clone the repository:
Clone the repository to your local directory by entering the following command:
  ```sh
  git clone 
  ```

## Instructions to create a conda environment: 

After navigating inside the project directory, we need to type in the command: 
  ```sh
  conda create -n venv
  ```
Here, venv is the name of virtual environment.

Upon creation of the virtual environment, the following command needs to be entered to activate the virtual environment.
  ```sh
  conda activate venv
  ```

## Instructions on how to install the required libraries: 
The following command installs the required libraries: 
  ```sh
  pip install -r requirements.txt
  ```

## Instructions on how to download the data
The following command saves the scraped data in 'data/raw/' directory: 
  ```sh
  python main.py -get
  ```

## Instructions on how to clean the data
The following command cleans the scraped data and saves the files in 'data/processed/' directory: 
  ```sh
  python main.py -clean
  ```
## Instrucions on how to run analysis code
The following command runs the analysis
  ```sh
  python main.py -analyze
  ```

## Instructions on how to create visualizations
The following command runs the visualizations
  ```sh
  python main.py -visualize
  ```

Note:- A detailed explaination for all the visualizations is provided in 'Inferences for Visualization.ipynb', which is present in the 'results/' directory. Execute the entire notebook once as the plots (drawn with plotly) might not be visible. 
