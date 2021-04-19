import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import warnings
import re
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from sklearn.metrics import classification_report



def load_data(database_filepath):
    """
    Load the data

    Inputs:
    database_filepath: String. Filepath for the db file containing the cleaned data.

    Output:
    X: dataframe. Contains the feature data.
    y: dataframe. Contains the labels (categories) data.
    category_names: List of strings. Contains the labels names.
    """

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('MessageCategories', engine)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """
    Normalize, tokenize and stems texts.

    Input:
    text: string. Sentence containing a message.

    Output:
    stemmed_tokens: list of strings. A list of strings containing normalized and stemmed tokens.
    """
    text = re.sub(r'[^a-zA-Z0-9]',' ',text.lower())
    
    #token messages
    words = word_tokenize(text)
    tokens = [w for w in words if w not in stopwords.words("english")]
    
    #sterm and lemmatizer
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Builds a ML pipeline and performs gridsearch.
    Args:
    None
    Returns:
    cv: gridsearchcv object.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters = {
        'clf__estimator__n_estimators': [50, 100]
        } 
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose=3)    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Returns test accuracy, number of 1s and 0s, recall, precision and F1 Score.

    Inputs:
    model: model object. Instanciated model.
    X_test: pandas dataframe containing test features.
    y_test: pandas dataframe containing test labels.
    category_names: list of strings containing category names.

    Returns:
    None

    """

    y_pred = model.predict(X_test)
    print(classification_report(y_pred, Y_test.values, target_names=category_names))
    # print raw accuracy score 
    print('Accuracy Score: {}'.format(np.mean(Y_test.values == y_pred)))


def save_model(model, model_filepath):
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