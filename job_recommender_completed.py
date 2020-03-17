#!/usr/bin/env python

import numpy as np
import pandas as pd
import sklearn
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

'''
The Combined_Jobs_Final.csv file: has the main jobs data(title, description, company, etc.)
The Job_Views.csv file: the file with the jobs seeing for the user.
The Experience.csv: the file containing the experience from the user.
The Positions_Of_Interest.csv: contains the interest the user previously has manifested.
'''
def tokenizer(row):
    '''
    Will tokenize the given text
    input: dataframe row
    returns: a tokenized row
    '''
    return nltk.word_tokenize(row)

def drop_stopwords(row, language):
    '''
    removes all the stop words in the given row
    input: row | language => english for now
    returns: a list of words without stop words
    '''
    stop_words = set(stopwords.words(language))
    return [word for word in row if word not in stop_words]

def stemming(row):
    '''
    returns the root of each word in the given row
    input: dataframe row
    returns: returns a list of stemmed words
    '''
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in row]

def rejoin_words(row):
    '''
    joins all the words in the list
    input: dataframe row
    returns: a single string for each row (list)
    '''
    return (' '.join(row))

def simple_data_cleaning(dataframe):
    dataframe = dataframe.str.lower()
    dataframe = dataframe.str.replace('[^a-z \n\.]', ' ')
    return dataframe

def advance_data_cleaning(dataframe):
    dataframe = simple_data_cleaning(dataframe)
    dataframe = dataframe.apply(lambda row: tokenizer(row))
    dataframe = dataframe.apply(lambda row: drop_stopwords(row, 'english'))
    dataframe = dataframe.apply(lambda row: stemming(row))
    dataframe = dataframe.apply(lambda row: rejoin_words(row))
    return dataframe

jobs = pd.read_csv('Combined_Jobs_Final.csv')

# Dropping all the unnecessary columns and renaming them to easier and more readable names
header = ['Job.ID', 'Slug', 'Title', 'Position', 'Company', 'City', 'Employment.Type', 'Job.Description']
jobs = pd.DataFrame(jobs, columns=header)
jobs.columns = ['JobID', 'Slug', 'Title', 'Position', 'Company','City', 'EmploymentType','JobDescription']

# Find and fill in company addresses where the value was nan.(Googled the headquarter)
jobs.loc[jobs.Company == 'St. Francis Hospital', 'City'] = 'New York'
jobs.loc[jobs.Company == 'CHI Payment Systems', 'City'] = 'Edmond'
jobs.loc[jobs.Company == 'Genesis Health Systems', 'City'] = 'Davenport'
jobs.loc[jobs.Company == 'Genesis Health System', 'City'] = 'Davenport'
jobs.loc[jobs.Company == 'Driveline Retail', 'City'] = 'Coppell'
jobs.loc[jobs.Company == 'Volvo Group', 'City'] = 'Washington'
jobs.loc[jobs.Company == 'Home Instead Senior Care', 'City'] = 'Nebraska'
jobs.loc[jobs.Company == 'Academic Year In America', 'City'] = 'Stamford'
jobs.loc[jobs.Company == 'Educational Testing Services', 'City'] = 'Princeton'
jobs.loc[jobs.Company == 'CBS Healthcare Services and Staffing', 'City'] = 'Urbandale'
# Correcting a mistake in the company name
jobs['Company'] = jobs['Company'].replace(['Genesis Health Systems'], 'Genesis Health System')
# Fill the nan values for the employment type
jobs.EmploymentType = jobs['EmploymentType'].fillna('Full-Time/Part-Time')

# we would use these columns in the recommender.
header = ['JobID', 'Title']
jobs_title = pd.DataFrame(jobs, columns=header)

'''
 Corpus
 I create a corpus from the columns below and drop all unnecessary columns.
 ['Position',  'Company',  'City',  'EmploymentType',  'JobDescription']
'''

jobs['Corpus'] = jobs['Position'].map(str)+' '+jobs.Slug+' '+jobs.Company+' '+jobs.City+' '+jobs.EmploymentType+' '+jobs.JobDescription
jobs = jobs.drop(['Title', 'Slug', 'Position', 'Company', 'City', 'EmploymentType', 'JobDescription',], axis=1).fillna(' ')

# finally cleaning and preparing the data
jobs.Corpus = advance_data_cleaning(jobs['Corpus'])

########### Job_Views

jViews = pd.read_csv('Job_Views.csv')

jViews = jViews.drop(['Job.ID', 'Title', 'State.Name', 'State.Code', 'Industry', 'View.Start', 'View.End', 'View.Duration', 'Created.At', 'Updated.At'], axis=1)
jViews['jViewCorpus'] = jViews['Position'].map(str) +' '+jViews["Company"] +"  "+ jViews["City"]
jViews = jViews.drop(['Position', 'Company', 'City'], axis=1)
jViews.columns = ['ApplicantID', 'jViewCorpus']
jViews.jViewCorpus = simple_data_cleaning(jViews['jViewCorpus'])

########### Experience
experience = pd.read_csv('experience.csv')

experience = experience.drop(['Employer.Name', 'City', 'State.Name', 'State.Code', 'Start.Date', 'End.Date', 'Job.Description', 'Salary', 'Can.Contact.Employer', 'Created.At', 'Updated.At'], axis=1)
experience.columns = ['ApplicantID', 'Position']
experience.Position = experience.Position.fillna(' ')
# As we see, there are sometimes more than one application for the applicant. Those will be marged to one.
experience = experience.groupby('ApplicantID', sort=False)['Position'].apply(' '.join).reset_index()

########### Positions_Of_Interest
poi = pd.read_csv('Positions_Of_Interest.csv')

poi = poi.sort_values(by='Applicant.ID')
poi = poi.drop(['Created.At', 'Updated.At'], axis=1)
poi.columns = ['ApplicantID', 'POI']
poi.POI = poi.POI.fillna(' ')
poi = poi.groupby('ApplicantID', sort=True)['POI'].apply(' '.join).reset_index()

########### Marge DataFrames
user = jViews.merge(experience, how='outer', left_on='ApplicantID', right_on='ApplicantID').fillna(' ')

user = user.merge(poi, how='outer', left_on='ApplicantID', right_on='ApplicantID').fillna('')
user = user.sort_values(by='ApplicantID')
user['Corpus'] = user.jViewCorpus.map(str)+' '+user.Position+' '+user.POI
user = user.drop(['jViewCorpus', 'Position', 'POI'], axis=1)
user.Corpus = user.Corpus.fillna(' ')
user.columns = ['ApplicantID', 'Corpus']
# drop all the rows with a empty corpus and cleaning the data.
user = user.drop(user[user.Corpus == ' '].index, axis=0)
user.Corpus = simple_data_cleaning(user['Corpus'])
user = user.sort_values(by='ApplicantID')

########### MODEL
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
tf_idf_job = vectorizer.fit_transform((jobs['Corpus']))
tf_idf_user = vectorizer.transform(user['Corpus'])

_cosine = map(lambda x: cosine_similarity(tf_idf_user, x), tf_idf_job)
cosine_output = list(_cosine)

users_set = set(user.ApplicantID.values)

def disply_applicant(user_id, dataframe):
    '''
    Selects the given users first application
    input: user id
    returns: selected applicants row
    '''
    if user_id in users_set:
        index = np.where(dataframe.ApplicantID == user_id)[0][0]
        return dataframe.iloc[[index]]
    return 'User id is not in the DB.'

def _recommended_jobs(top, user):
    '''
    Returns all the recommended jobs for the given user.
    input: top => nr of top recommended jobs | user => user id
    return: a dataframe with two columns (ApplicantID, JobID)
    '''
    result = pd.DataFrame()

    list_range = range(len(cosine_output))
    _top = sorted(list_range, key=lambda i: cosine_output[i].any(), reverse=True)[:top]
    
    for index, item in enumerate(_top):
        result.at[index, 'ApplicantID'] = user
        result.at[index, 'JobID'] = jobs['JobID'][item]
    
    return result

def _select_job(recommended_jobs):
    '''
    Returns the description (Corpus) for the given list.
    input: recommended jobs dataframe
    return: a dataframe with two columns (JobID, Corpus)
    '''
    result = pd.DataFrame()
    
    for i, recomend in enumerate(recommended_jobs['JobID']):
        index = np.where(jobs.JobID == recomend)[0][0]
        result.at[i, 'JobID'] = recomend
        result.at[i, 'Corpus'] = jobs['Corpus'][index]
    
    return result

_recommended = _recommended_jobs(10, 326)
print(f'\n recommender_output:\n{_recommended_jobs(10, 326)}')
print(f'\n _select_job:\n{_select_job(_recommended)}')

# we could make it more user friendly by adding some more info about what user applied for and what we recommend...
def recommender_output(top, user):
    '''
    Returns the final output for the recommender system
    input: top => nr of top recommended jobs | user => user id
    return: a dataframe with three columns (ApplicantID, JobID, Corpus)
    '''
    __recommended = _recommended_jobs(top, user)
    __selected = _select_job(__recommended)
    
    return __recommended.merge(__selected, on='JobID')

print(f'\n recommender_output:\n{recommender_output(5, 326)}')

######### KNN
from sklearn.neighbors import NearestNeighbors

number_neighbers = 10
model = NearestNeighbors(number_neighbers, p=2)
model.fit(tf_idf_job)
result = model.kneighbors(tf_idf_user, return_distance=True)

def get_recommendation(user, recommended_jobs, scores):
    _result = pd.DataFrame()

    for index, recomend in enumerate(recommended_jobs):
        _result.at[index, 'ApplicantID'] = user
        _result.at[index, 'JobID'] = jobs_title['JobID'][recomend]
        _result.at[index, 'title'] = jobs_title['Title'][recomend]
        _result.at[index, 'score'] =  scores[index]
    
    return _result

print(f'\n get_recommendation\n{get_recommendation(326, result[1][0][1:], result[0][0][1:])}')
print('\nEnd of the script!\n')
