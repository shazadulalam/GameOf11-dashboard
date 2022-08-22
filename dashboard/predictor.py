#!/usr/bin/env python
# coding: utf-8

# In[241]:


#import libs
import pandas as pd
import matplotlib.pyplot as plt
import os

#all models
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

#accuracy measures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score
import numpy as np



#list of ODI teams with their ICC rank value
teams_dict = {'australia':15, 'india':18, 'bangladesh':20, 'srilanka':13, 'southafrica':16, 'pakistan':14, 
            'zimbabwe':9,'newzealand':17, 'westindies':12, 'england':19, 'afghanistan':11, 'ireland':10, 
            'netherlands':8,'oman':7, 'scotland':6, 'nepal':5, 'namibia':4, 'uae':3, 'usa':2, 'papuanewguinea':1
             }

#list of other t20 teams but this does not match the teams that we have in the dataset
teams_t20={'Pakistan': '1', 'Australia': '2', 'England': '3', 'India': '4', 'South Africa': '5', 'New Zealand': '6', 
           'Afghanistan': '7', 'Sri Lanka': '8', 'Bangladesh': '9', 'West Indies': '10', 'Zimbabwe': '11', 
           'Ireland': '12', 'Scotland': '13', 'UAE': '14', 'Nepal': '15', 'Netherlands': '16', 'PNG': '17', 
           'Oman': '18', 'Namibia': '19', 'Singapore': '20', 'Qatar': '21', 'Canada': '22', 'Hong Kong': '23', 
           'Jersey': '24', 'Saudi Arabia': '25', 'Italy': '26', 'Kuwait': '27', 'Kenya': '28', 'Denmark': '29', 
           'Bermuda': '30', 'Malaysia': '31', 'Germany': '32', 'USA': '33', 'Uganda': '34', 'Botswana': '35', 
           'Ghana': '36', 'Norway': '37', 'Austria': '38', 'Guernsey': '39', 'Romania': '40', 'Nigeria': '41', 
           'Sweden': '42', 'Spain': '43', 'Tanzania': '44', 'Cayman Islands': '45', 'Philippines': '46', 
           'Bahrain': '47', 'Argentina': '48', 'France': '49', 'Vanuatu': '50', 'Belize': '51', 'Luxembourg': '52', 
           'Malawi': '53', 'Peru': '54', 'Fiji': '55', 'Panama': '56', 'Belgium': '57', 'Samoa': '58', 'Japan': '59',
           'Costa Rica': '60', 'Mexico': '61', 'Hungary': '62', 'Bulgaria': '63', 'Czech Republic': '64', 
           'Israel': '65', 'Thailand': '66', 'Portugal': '67', 'Finland': '68', 'South Korea': '69', 
           'Isle of Man': '70', 'Chile': '71', 'Bhutan': '72', 'Mozambique': '73', 'Sierra Leone': '74', 
           'Brazil': '75', 'Maldives': '76', 'St Helena': '77', 'Malta': '78', 'Myanmar': '79', 
           'Indonesia': '80', 'China': '81', 'Gambia': '82', 'Gibraltar': '83', 'Swaziland': '84', 
           'Rwanda': '85', 'Lesotho': '86'}


#load all the rank files csv
path_for_rank_files = os.path.join(os.getcwd(), 'fantasy_game/dataset')

bbl = pd.read_csv(path_for_rank_files + '/bbl.csv')
bpl = pd.read_csv(path_for_rank_files + '/bpl.csv')
icc_women_odi = pd.read_csv(path_for_rank_files + '/icc-women-odi.csv')
icc_women_t20 = pd.read_csv(path_for_rank_files + '/icc-women-t20.csv')
ipl = pd.read_csv(path_for_rank_files + '/ipl.csv')
mzansi = pd.read_csv(path_for_rank_files + '/mzansi.csv')
psl = pd.read_csv(path_for_rank_files + '/psl.csv')
t10 = pd.read_csv(path_for_rank_files + '/t10-league.csv')

path_for_rank_files = '/Users/mustakimsunny/Desktop/ML/go11ai/data'

cpl = pd.read_csv(path_for_rank_files + '/carribean-premier-league.csv')
global_canada = pd.read_csv(path_for_rank_files + '/global-canada.csv') 
karnataka = pd.read_csv(path_for_rank_files + '/karnataka.csv')
oman_pent_t20 = pd.read_csv(path_for_rank_files + '/oman-pentagular-t20-series.csv')
syed_muali_trophy = pd.read_csv(path_for_rank_files + '/syed-mushtaq-ali-trophy.csv')
vbnd = pd.read_csv(path_for_rank_files + '/vitality-blast-nothern-division.csv')
vbsd = pd.read_csv(path_for_rank_files + '/vitality-blast-southern-division.csv')
womens_big_bash = pd.read_csv(path_for_rank_files + '/womens-big-bash.csv')
#covered_league = pd.read_csv(path_for_rank_files + '/covered_league.csv') 


#reverse the ranking to make meaningful score
def reverse_score(dataframe):
    score = [dataframe['Score'][ind] for ind in dataframe.index]
    score.reverse()
    dataframe['score_processed'] = score
    dataframe = dataframe.drop(['Score'], axis=1)
    return dataframe

#convert dataframe into dictionary
def convert_to_dict(dataframe):
    dic = {}
    for ind in dataframe.index:
        team, score = dataframe['Teams'][ind],dataframe['score_processed'][ind] 
        team = team.lower().replace(" ", '')
        dic[team] = score
    
    return dic




#convert to score in reverse order and then transform into dictionary
bbl = reverse_score(bbl)
bbl_dict = convert_to_dict(bbl)

bpl = reverse_score(bpl)
bpl_dict = convert_to_dict(bpl) 

icc_women_odi = reverse_score(icc_women_odi)
icc_women_odi_dict = convert_to_dict(icc_women_odi) 

icc_women_t20 = reverse_score(icc_women_t20)
icc_womend_t20_dict = convert_to_dict(icc_women_t20) 

ipl = reverse_score(ipl)
ipl_dict = convert_to_dict(ipl) 

mzansi = reverse_score(mzansi)
mzansi_dict = convert_to_dict(mzansi) 

psl = reverse_score(psl)
psl_dict = convert_to_dict(psl) 

t10 = reverse_score(t10)
t10_dict = convert_to_dict(t10) 




cpl = reverse_score(cpl)
cpl_dict = convert_to_dict(cpl) 


global_canada = reverse_score(global_canada)
global_canada_dict = convert_to_dict(global_canada)


karnataka = reverse_score(karnataka)
karnataka_dict = convert_to_dict(karnataka)

oman_pent_t20 = reverse_score(oman_pent_t20)
oman_pent_t20_dict = convert_to_dict(oman_pent_t20)


syed_muali_trophy = reverse_score(syed_muali_trophy)
syed_muali_trophy_dict = convert_to_dict(syed_muali_trophy)

vbnd = reverse_score(vbnd)
vbnd_dict = convert_to_dict(vbnd)


vbsd = reverse_score(vbsd)
vbsd_dict = convert_to_dict(vbsd)


womens_big_bash = reverse_score(womens_big_bash)
womens_big_bash_dict = convert_to_dict(womens_big_bash)




#set paths for dataset loading
#path = os.path.join(os.getcwd(), 'final_df.xlsx')
path = os.path.join(os.getcwd(), 'final_df_upgraded.xlsx')
#file_path = os.path.join(path, 'final_df.xlsx')
print(path)



#read files and see data types
final_data = pd.read_excel(path)



#preprocessing
#drop unnecessary columns, first argument = data, second argument = all the cols needed to be dropped

def drop_cols(dataframe,*arg):
    columns = list(arg)
    dataframe = dataframe.drop(columns, axis=1)
    return dataframe

#convert matchTime to datetime
def convert_to_datetime(dataframe, date_col):
    dataframe[date_col] = pd.to_datetime(dataframe[date_col])
    return dataframe


def add_gap_between_current_and_last_match(df_initial_model):
    #print(df_initial_model)
    df_initial_model = df_initial_model.sort_values(['matchTime'], ascending=[False])
    df_initial_model['prev'] = df_initial_model.matchTime.shift(-1)
    df_initial_model['diff'] = df_initial_model['matchTime'] - df_initial_model['prev']
    df_initial_model['diff'] = df_initial_model['diff'].dt.days
    df_initial_model = drop_cols(df_initial_model, 'prev')
    
    return df_initial_model

#seperating the hour from the match time and adding it into the dataframe
def extract_and_add_hour(dataframe,date_col):
    hour = list(dataframe[date_col].dt.hour)
    dataframe['hour'] = hour
    return dataframe

#add count of instances took place on the same date
def extract_and_add_date_count(dataframe, date_col):
    dataframe['date_only'] = dataframe[date_col].dt.to_period('D')
    dataframe['date_count'] = dataframe.date_only.map(dataframe.groupby('date_only').size())
    dataframe = drop_cols(dataframe, 'date_only')
    
    return dataframe
    

#match weight calculation
def match_weight(team1, team2, dictionary):
    if team1 in dictionary and team2 in dictionary:
        normalized_sum = dictionary[team1]/len(dictionary) + dictionary[team2]/len(dictionary)
        weight = normalized_sum/2
        return weight
    else:
        return None
    
#train test split
def split_train_test(dataframe, target):
    train_set, test_set = train_test_split(dataframe, test_size=0.2, random_state=42)
    X_train, X_test = train_set.drop(target, axis=1), test_set.drop(target, axis=1)
    y_train, y_test = train_set[target], test_set[target]
    
    return X_train, X_test, y_train, y_test


#creating a pipeline for scaling and normalization, this will be helpful for future scaling, we can just create these 
#pipelines which will consist of all different types of cleaning functions. In goes noisy data, out comes the cleanes processed data



#num_pipeline = Pipeline([
#        ('dc', drop_cols(dataframe,*arg)),
#    ])


#select model, fit and predict
#then measure different accuracy score for each of them

def fit_predict(X_train, X_test, y_train, y_test):
    predicted_vals = []
    algorithms = [LinearRegression(), SVC(), LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier(),
                  RandomForestRegressor(bootstrap=False, max_depth=40, max_features='sqrt',
                                        min_samples_leaf=1,min_samples_split=2,n_estimators=80),
                  RandomForestClassifier()]
    for algorithm in algorithms:
        model = algorithm
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        x_pred = model.predict(X_train)
        predicted_vals.append(y_pred)
        print('Model: ', model)
        print('\n')
        print('mean squared error: ', mean_squared_error(y_test, y_pred))
        print('explained variance score: ', explained_variance_score(y_test, y_pred))
        print('mean absolute error: ', mean_absolute_error(y_test, y_pred))
        print('median squared error: ', median_absolute_error(y_test, y_pred))
        print('r2 score on test: ', r2_score(y_test, y_pred))
        print('r2 score on train: ', r2_score(y_train, x_pred))

        print('---------------------------------------------')
    
    return predicted_vals
        


#copy main daaset to another variable for pre-processing
df_initial_model = final_data

#peek through the dataset
df_initial_model


#drop columns that are not intended to be used for now
#df_initial_model = drop_cols(df_initial_model,'match_id','TotalContestPerMatch', 'Unnamed: 10')
df_initial_model = drop_cols(df_initial_model,'match_id','TotalContestPerMatch', 'TotalWinner', 'TotalSeat')

#convert match time column to datetime for further processing
df_initial_model = convert_to_datetime(df_initial_model,'matchTime')

#add difference between two matches in terms of days
df_initial_model = add_gap_between_current_and_last_match(df_initial_model)

#add date_count column
df_initial_model = extract_and_add_date_count(df_initial_model, 'matchTime')

#extract the hour values and as an additional column
df_initial_model = extract_and_add_hour(df_initial_model,'matchTime')

#now that the hour is in the dataframe, there is no need of the matchtime column, so drop it
df_initial_model = drop_cols(df_initial_model,'name', 'matchTime')

# fill missing values with mean column values
#df_initial_model.fillna(df_initial_model.mean(), inplace=True)
df_initial_model = df_initial_model.dropna()

#average total contest VS number of matches on the same date plot
df_count = df_initial_model.groupby('date_count')
count_vs_contest = df_count['total_contest_given'].mean().reset_index()

plt.figure(figsize=(20,10))
plt.plot(count_vs_contest['date_count'], count_vs_contest['total_contest_given'])
plt.xlabel("Number of mathces occured on the same date")
plt.ylabel("Number of total contests given on average")



#split into basic train, test 
X_train, X_test, y_train, y_test = split_train_test(df_initial_model, 'total_contest_given')


#convert target col to int
y_test = y_test.astype('int64')
y_train = y_train.astype('int64')

#check correlation 
corr_matrix = df_initial_model.corr()
print(corr_matrix['total_contest_given'].sort_values(ascending=False))

predicted_vals = fit_predict(X_train, X_test, y_train, y_test)



#looks like RandomForestRegressor giving the best answer, let's calculate with cross validation
from sklearn.model_selection import cross_val_score

rf_reg = RandomForestRegressor(bootstrap=False, max_depth=40, max_features='sqrt',
                                min_samples_leaf=1,min_samples_split=2,n_estimators=80)
cv_score = cross_val_score(rf_reg, X_train, y_train, scoring='r2', cv=10)

print(cv_score) 
print("Mean: ", cv_score.mean())


# In[161]:


#stratified sampling

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=10, random_state=42)
rf_reg = RandomForestRegressor(bootstrap=False, max_depth=40, max_features='sqrt',
                                        min_samples_leaf=1,min_samples_split=2,n_estimators=80)
r2 = []

for train_index, test_index in skfolds.split(X_train, y_train):
    #clone_clf = clone(sgd_clf)
    X_train_folds = X_train.iloc[train_index]
    y_train_folds = y_train.iloc[train_index]
    X_test_fold = X_train.iloc[test_index]
    y_test_fold = y_train.iloc[test_index]
    
    
    
    rf_reg.fit(X_train_folds, y_train_folds)
    y_pred = rf_reg.predict(X_test_fold)
    
    r2.append(r2_score(y_test_fold, y_pred))
    
print(sum(r2)/len(r2))


#build another dataset with hype values
hypes = []
dict_of_dicts = {'bbl_dict': bbl_dict,
     'bpl_dict': bpl_dict, 
     'icc_women_odi_dict': icc_women_odi_dict, 
     'icc_womend_t20_dict': icc_womend_t20_dict,
     'ipl_dict': ipl_dict,
     'mzansi_dict': mzansi_dict,
     'psl_dict': psl_dict,
     't10_dict': t10_dict,
     'teams_dict' : teams_dict,
     'cpl_dict':cpl_dict, 
     'global_canada_dict': global_canada_dict, 
     'karnataka_dict':karnataka_dict, 
     'oman_pent_t20_dict':oman_pent_t20_dict,
     'syed_muali_trophy_dict': syed_muali_trophy_dict, 
     'vbnd_dict':vbnd_dict,
     'vbsd_dict':vbsd_dict,
     'womens_big_bash_dict':womens_big_bash_dict}

i = 0
x = {}
teams_with_hype = pd.DataFrame()
for teams in final_data['name']:
    i += 1
    teams= teams.replace('vs', ',')
    teams = teams.replace(" ", '')
    team1,team2 = teams.lower().split(',')
    
    for key,val in dict_of_dicts.items():
        if team1 in dict_of_dicts[key] and team2 in dict_of_dicts[key]:
            #print(team1 + ' vs ' + team2 + ' found in ' + key)
            #print(match_weight(team1, team2,val))
            hype = match_weight(team1, team2,val)
            
            for key, val in final_data.iloc[i-1].items():
                x[key] = val
            x['hype'] = hype
            teams_with_hype = teams_with_hype.append(x, ignore_index=True)
            break

#datset to predict total contest
teams_with_hype_1 = teams_with_hype

#dataset to predict total seat
teams_with_hype_2 = teams_with_hype

#pre-processing for total contest prediction

#drop columns that are not intended to be used for now
#teams_with_hype = drop_cols(teams_with_hype,'match_id','TotalContestPerMatch', 'Unnamed: 10')
#teams_with_hype = drop_cols(teams_with_hype,'match_id','TotalContestPerMatch', 'TotalWinner')
teams_with_hype_1 = drop_cols(teams_with_hype_1,'match_id','TotalContestPerMatch', 'TotalWinner', 'TotalSeat')


#convert match time column to datetime for further processing
teams_with_hype_1 = convert_to_datetime(teams_with_hype_1,'matchTime')

#add difference between two matches in terms of days
teams_with_hype_1 = add_gap_between_current_and_last_match(teams_with_hype_1)

#add number of matches occured at the same date
teams_with_hype_1 = extract_and_add_date_count(teams_with_hype_1, 'matchTime')

#extract the hour values and as an additional column
teams_with_hype_1 = extract_and_add_hour(teams_with_hype_1,'matchTime')

#now that the hour is in the dataframe, there is no need of the matchtime column, so drop it
teams_with_hype_1 = drop_cols(teams_with_hype_1,'name', 'matchTime')

# fill missing values with mean column values
teams_with_hype_1.fillna(teams_with_hype_1.mean(), inplace=True)
#teams_with_hype = teams_with_hype.dropna()


#split into basic train, test 
X_train, X_test, y_train, y_test = split_train_test(teams_with_hype_1, 'total_contest_given')
#X_train, X_test, y_train, y_test = split_train_test(teams_with_hype, 'TotalSeat')

#convert target col to int
y_test = y_test.astype('int64')
y_train = y_train.astype('int64')



predicted_vals = fit_predict(X_train, X_test, y_train, y_test)


#looks like RandomForestRegressor giving the best answer, let's calculate with cross validation
from sklearn.model_selection import cross_val_score

rf_reg = RandomForestRegressor(bootstrap=False, max_depth=40, max_features='sqrt',
                                        min_samples_leaf=1,min_samples_split=2,n_estimators=80)
cv_score = cross_val_score(rf_reg, X_train, y_train, scoring='r2', cv=10)

print(cv_score) 
print("Mean: ", cv_score.mean())


#stratified sampling

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=10, random_state=42)
rf_reg = RandomForestRegressor(bootstrap=False, max_depth=40, max_features='sqrt',
                                        min_samples_leaf=1,min_samples_split=2,n_estimators=80)
r2 = []

for train_index, test_index in skfolds.split(X_train, y_train):
    #clone_clf = clone(sgd_clf)
    X_train_folds = X_train.iloc[train_index]
    y_train_folds = y_train.iloc[train_index]
    X_test_fold = X_train.iloc[test_index]
    y_test_fold = y_train.iloc[test_index]
    
    
    
    rf_reg.fit(X_train_folds, y_train_folds)
    y_pred = rf_reg.predict(X_test_fold)
    
    r2.append(r2_score(y_test_fold, y_pred))
    
print(sum(r2)/len(r2))

#plot actual VS predicted value for any model from the list in the cell above
x = []
for i in range(111):
    x.append(i)
y1 = y_test
y2 = predicted_vals[5] #change the index of the predicted_vals according to what model needs to be visualized

plt.figure(figsize=(30,10))
plt.plot(x, y1, label = "truth", color='red')
plt.plot(x, y2, label = "predicted", color='green')

plt.xlabel('Observations')
plt.ylabel('Total Contest')
plt.title('Actual VS Predicted plot')
plt.legend()

plt.show()


#another copy of the dataset used to predict number of seats

#drop columns that are not intended to be used for now
#teams_with_hype = drop_cols(teams_with_hype,'match_id','TotalContestPerMatch', 'Unnamed: 10')
#teams_with_hype = drop_cols(teams_with_hype,'match_id','TotalContestPerMatch', 'TotalWinner')
teams_with_hype_2 = drop_cols(teams_with_hype_2,'match_id','TotalContestPerMatch', 'TotalWinner')


#convert match time column to datetime for further processing
teams_with_hype_2 = convert_to_datetime(teams_with_hype_2,'matchTime')

#add difference between two matches in terms of days
teams_with_hype_2 = add_gap_between_current_and_last_match(teams_with_hype_2)

#add number of matches occured at the same date
teams_with_hype_2 = extract_and_add_date_count(teams_with_hype_2, 'matchTime')

#extract the hour values and as an additional column
teams_with_hype_2 = extract_and_add_hour(teams_with_hype_2,'matchTime')

#now that the hour is in the dataframe, there is no need of the matchtime column, so drop it
teams_with_hype_2 = drop_cols(teams_with_hype_2,'name', 'matchTime')

# fill missing values with mean column values
teams_with_hype_2.fillna(teams_with_hype_2.mean(), inplace=True)
#teams_with_hype = teams_with_hype.dropna()



#split into basic train, test 
X_train, X_test, y_train, y_test = split_train_test(teams_with_hype_2, 'TotalSeat')
#X_train, X_test, y_train, y_test = split_train_test(teams_with_hype, 'TotalSeat')

#convert target col to int
y_test = y_test.astype('int64')
y_train = y_train.astype('int64')


predicted_vals = fit_predict(X_train, X_test, y_train, y_test)


#looks like RandomForestRegressor giving the best answer, let's calculate with cross validation

rf_reg = RandomForestRegressor(bootstrap=False, max_depth=40, max_features='sqrt',
                                        min_samples_leaf=1,min_samples_split=2,n_estimators=80)
cv_score = cross_val_score(rf_reg, X_train, y_train, scoring='r2', cv=10)

print(cv_score) 
print("Mean: ", cv_score.mean())


#plot actual VS predicted value for any model from the list in the cell above
x = []
for i in range(111):
    x.append(i)
y1 = y_test
y2 = predicted_vals[5] #change the index of the predicted_vals according to what model needs to be visualized

plt.figure(figsize=(30,10))
plt.plot(x, y1, label = "truth", color='red')
plt.plot(x, y2, label = "predicted", color='green')

plt.xlabel('Observations')
plt.ylabel('Seats')
plt.title('Actual VS Predicted plot')
plt.legend()

plt.show()

#Grid search to find the best parameter, be careful about this block of code
#running in local machine may take forever




#next steps
#1. pickle model to save
#2. create a pipeline to predict values in real time
#3. integretion with the app admin section

