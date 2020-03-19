#!/usr/bin/env python

import numpy as np
import pandas as pd
import sklearn
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

'''
The Combined_Jobs_Final.csv file: has the main jobs data(title, description, company, etc.)
The Job_Views.csv file: the file with the jobs seeing for the user.
The Experience.csv: the file containing the experience from the user.
The Positions_Of_Interest.csv: contains the interest the user previously has manifested.
'''

def drop_stopwords(row, language):
    '''
    removes all the stop words in the given row
    input: row | language => english for now
    returns: a list of words without stop words
    '''
    stop_words = set(stopwords.words(language))
    return [word for word in row if word not in stop_words and word not in list(string.punctuation)]

def lemmitize(row):
    '''
    returns the root of each word in the given row
    input: dataframe row
    returns: returns a list of lemmatized words
    '''
    _lemmitize = WordNetLemmatizer()
    return [_lemmitize.lemmatize(word, pos='v') for word in row]

def _data_cleaning(row):
    '''
    returns a clean/lemmitize string
    input: dataframe row
    returns: returns a string
    '''
    row = row.lower().replace('[^a-z \n\.]', ' ')
    row = nltk.word_tokenize(row)
    row = drop_stopwords(row, 'english')
    row = lemmitize(row)
    return ' '.join(row)

jobs = pd.read_csv('Combined_Jobs_Final.csv')

# Dropping all the unnecessary columns and renaming them to easier and more readable names
header = ['Job.ID', 'Title', 'Position', 'Company', 'City', 'Employment.Type', 'Job.Description']
jobs = pd.DataFrame(jobs, columns=header)
jobs.columns = ['JobID', 'Title', 'Position', 'Company','City', 'EmploymentType','JobDescription']

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

jobs['Corpus'] = jobs['Position'].map(str)+' '+jobs.Company+' '+jobs.City+' '+jobs.EmploymentType+' '+jobs.JobDescription+' '+jobs.Title
jobs = jobs.drop(['Title', 'Position', 'Company', 'City', 'EmploymentType', 'JobDescription',], axis=1).fillna(' ')

# finally cleaning and preparing the data
jobs.Corpus = jobs['Corpus'].map(str).apply(_data_cleaning)

########### Job_Views

jViews = pd.read_csv('Job_Views.csv')

jViews = jViews.drop(['Job.ID', 'Title', 'State.Name', 'State.Code', 'Industry', 'View.Start', 'View.End', 'View.Duration', 'Created.At', 'Updated.At'], axis=1)
jViews['jViewCorpus'] = jViews['Position'].map(str) +' '+jViews["Company"] +"  "+ jViews["City"]
jViews = jViews.drop(['Position', 'Company', 'City'], axis=1).fillna(' ')
jViews.jViewCorpus = jViews['jViewCorpus'].map(str).apply(_data_cleaning)
jViews.columns = ['ApplicantID', 'jViewCorpus']

########### Experience
experience = pd.read_csv('experience.csv')

experience = experience.drop(['Employer.Name', 'City', 'State.Name', 'State.Code', 'Start.Date', 'End.Date', 'Job.Description', 'Salary', 'Can.Contact.Employer', 'Created.At', 'Updated.At'], axis=1)
experience.columns = ['ApplicantID', 'Position']
experience.Position = experience.Position.fillna(' ')
# As we see, there are sometimes more than one application for the applicant. Those will be marged to one.
experience = experience.groupby('ApplicantID', sort=False)['Position'].apply(' '.join).reset_index()
experience.Position = experience['Position'].map(str).apply(_data_cleaning)
experience = experience.sort_values(by='ApplicantID')

########### Positions_Of_Interest
poi = pd.read_csv('Positions_Of_Interest.csv')

poi = poi.sort_values(by='Applicant.ID')
poi = poi.drop(['Created.At', 'Updated.At'], axis=1).fillna(' ')
poi.columns = ['ApplicantID', 'POI']
poi.POI = poi['POI'].map(str).apply(_data_cleaning)
poi = poi.groupby('ApplicantID', sort=False)['POI'].apply(' '.join).reset_index()

########### Marge DataFrames
user = jViews.merge(experience, how='outer', left_on='ApplicantID', right_on='ApplicantID').fillna(' ')
user = user.sort_values(by='ApplicantID')
user = user.merge(poi, how='outer', left_on='ApplicantID', right_on='ApplicantID').fillna('')
user = user.sort_values(by='ApplicantID')

user['Corpus'] = user.jViewCorpus.map(str)+' '+user.Position+' '+user.POI
user = user.drop(['jViewCorpus', 'Position', 'POI'], axis=1).fillna(' ')
user.columns = ['ApplicantID', 'Corpus']

# drop all the rows with a empty corpus and cleaning the data.
user = user.drop(user[user.Corpus == ' '].index, axis=0)
user.Corpus = user['Corpus'].apply(_data_cleaning)

########### MODEL
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def disply_applicant(user_id, dataframe=user):
    '''
    Selects the given users first application
    input: user id
    returns: selected applicants row
    '''
    users_set = set(user.ApplicantID.values)

    if user_id in users_set:
        index = np.where(dataframe.ApplicantID == user_id)[0][0]
        return dataframe.iloc[[index]]
    return 'User id is not in the DB.'

selected_user = disply_applicant(326)

vectorizer = TfidfVectorizer()
tf_idf_job = vectorizer.fit_transform((jobs['Corpus']))
tf_idf_user = vectorizer.transform(selected_user['Corpus'])

_cosine = map(lambda x: cosine_similarity(tf_idf_user, x), tf_idf_job)
cosine_output = list(_cosine)

list_range = range(len(cosine_output))
_top = sorted(list_range, key=lambda i: cosine_output[i], reverse=True)[:10]
list_scores = [cosine_output[i][0][0] for i in _top]

def get_recommendation(user, top_recommended, scores, jobs_title=jobs_title):
    _result = pd.DataFrame()

    for index, recomend in enumerate(top_recommended):
        _result.at[index, 'ApplicantID'] = user
        _result.at[index, 'JobID'] = jobs_title['JobID'][recomend]
        _result.at[index, 'title'] = jobs_title['Title'][recomend]
        _result.at[index, 'score'] =  scores[index]
    
    return _result

print(f'\n get_recommendation\n{get_recommendation(326, _top, list_scores)}')

########### KNN
from sklearn.neighbors import NearestNeighbors

model = NearestNeighbors(n_neighbors=10, p=2)
model.fit(tf_idf_job)
model_output = model.kneighbors(tf_idf_user, return_distance=True)

_top = model_output[1][0][1:]
list_scores = model_output[0][0][1:]

print(f'KNN model recommendations:\n{get_recommendation(326, _top, list_scores)}')

print('\nEnd of the script!\n')
