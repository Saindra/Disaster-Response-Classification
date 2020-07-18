import sys
import pandas as pd
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
    
    '''
    INPUT:
    messages_filepath - filepath to disaster_messages csv data
    categories_filepath - filepath to categories_messages csv data
    
    OUTPUT:
    df - a merged dataframe of disaster messages and categories data
    
    '''
    
    #read disaster_messages csv data
    df_messages=pd.read_csv(messages_filepath)
    
    #read categories_messages csv data
    df_categories=pd.read_csv(categories_filepath)
    
    #merge both the data
    df=pd.merge(df_messages,df_categories,on='id')
    
    return df


def clean_data(df):
    
    '''
    INPUT:
    df - dataframe having disaster messages and its category data
    
    OUTPUT:
    df - a cleaned dataframe after pre processing
    
    '''
    #create a category dataframe based on the categories column in df
    df_categories=df['categories'].str.split(';',expand=True)
    
    #finding new columns names for df_categories and replace the names
    col_names=[x.split('-')[0] for x in df_categories.loc[0,:]]  
    df_categories.columns=col_names
    
    #clean the data in df_categories dataframe
    for column in df_categories:
        
        #set each value to be the last character of the string
        df_categories[column] = [val.split('-')[1] for val in df_categories[column]]
        
        # convert column from string to numeric
        df_categories[column] = [int(val) for val in df_categories[column]]
        
    #drop the categories column in input data frame
    df.drop(columns='categories',inplace=True)
    
    #append the cleaned categoies data back to input dataframe
    df=df.join(df_categories)
    
    #fix
    df.related=df.related.replace(2,1)
    
    #drop duplicates
    df.drop_duplicates(inplace=True)
        
    return df




def save_data(df, database_filename):
    
    '''
    INPUT:
    df - dataframe to be saved to database
    database_filename - file path with name of the database

    '''
    
     # Create sqlite engine and save the input dataframe to the database
    sql_engine = create_engine('sqlite:///{}'.format(database_filename))
    print(df.shape[0])
    df.to_sql('disaster_messages', sql_engine, index=False, if_exists='replace')
    return  


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