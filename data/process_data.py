import sys
import sqlite3
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads and merges datasets
    Args:
    messages_filepath: String. Filepath for the csv file containing the messages.
    categories_filepath: String. Filepath for the csv file containing the categories.
    Returns:
    df: pandas dataframe. Dataframe containing messages and respective categories.
    """
    categories = pd.read_csv(categories_filepath)
    messages = pd.read_csv(messages_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """
    Clean dataframes from uneeded columns, duplicates and text artifacts
    Args:
    df: pandas dataframe. Dataframe containing messages and categories.
    Returns:
    df: pandas dataframe. Dataframe containing cleaned version of messages and categories.
    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply (lambda x: x.split('-')[0]).tolist()
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column], errors = 'coerce')
    
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories], axis = 1,)
    df.drop_duplicates(inplace = True)
    df = df[df['related'] != 2]
    return df
    
def save_data(df, database_filename):
    """
    Save the cleaned data.

    Input:
    df: pandas dataframe. Dataframe containing cleaned version of messages and respective categories.
    database_filename: String. Filename for the output database.

    Output:
    None.
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('MessageCategories2', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()