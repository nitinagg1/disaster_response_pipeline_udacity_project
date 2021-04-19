# Disaster Response Pipeline Project

### Summary
In this project I will be using a data set containing messages provided by Figure8, that were sent during disaster events and build a classifier to identify messages or events in need of attention or relief. 
I create a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model. Finally, I export the model to a pickle file.

The best performing machine learning model will be deployed as a webapp using bootstrap and Flask where the user can test their own tentative messages to see how they would be classified with the models I selected and trained. Through the web app the user can also consult visualizations of the clean and transformed data.

### Files
* \app
  * run.py -> this file runs the flask application and uses the model created during data processing to analyse messages entered by users. It also has some data visualization.
  
* data
    *  disaster_categories.csv
    *  disaster_messages.csv
    *  process_data.py -> python which uses the csv files provided to perform an ETL job. Finally the cleaned data is exported into a sqllite table.
    
* models
   * train_classifier.py -> this file loads table, builds a model, evaluates the model and finally saves the model to be used in webapp.
    

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

