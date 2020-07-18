import sys
import pandas as pd 
import numpy as np
from sqlalchemy import create_engine
import re
import nltk
import pickle
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def load_data(database_filepath):
    
    '''
    INPUT:
    database_filepath - db filepath for the input
    
    OUTPUT:
    X - input features
    Y - Output
    category_names - Output columns names
    
    '''
    
    sql_engine=create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('SELECT * from disaster_messages', sql_engine)
    X = df['message'].values
    Y = df.iloc[:, 4:]
    category_names = list(Y.columns)
    return X, Y.values, category_names


def tokenize(text):
    
    '''
    INPUT:
    text - input text for pre processing
    
    OUTPUT:
    clear_tokens - normalized words ready for modeling
    
    '''
    
    #Regular expression for url
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls=re.findall(url_regex,text)
    
    for url in detected_urls:
        text.replace(url,'urlplaceholder')
        
    tokens=word_tokenize(text)
    lemmatizer=WordNetLemmatizer()
    
    clear_tokens=[]
    for tok in tokens:
        clear_tokens.append(lemmatizer.lemmatize(tok).lower().strip())
        
    return clear_tokens


def build_model():
    
    '''
    INPUT:
    
    OUTPUT:
    cv - Machine Learning model for multi-output classification

    '''
    
    pipeline = Pipeline([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'text_pipeline__vect__max_features': (None, 2500, 5000),
        'text_pipeline__tfidf__use_idf': (True, False)
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv
    


def evaluate_model(model, X_test, Y_test, category_names):
    
    '''
    INPUT:
    model - ML model
    X_test - input test data
    Y_test - output test data
    category_names - output columns names
    
    
    OUTPUT:
    
    '''
    
    y_pred = model.predict(X_test)
    report = classification_report(Y_test, y_pred, target_names=category_names)
    print(report)



def save_model(model, model_filepath):
    
    '''
    INPUT:
    model - ML model
    model_filepath - output path to save ML model to a pickel object
    
    OUTPUT:
    
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()