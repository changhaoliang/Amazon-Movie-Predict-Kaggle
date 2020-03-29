import re

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.externals import joblib
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer,
                                             TfidfVectorizer)
                                             
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy import sparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn import datasets , linear_model
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

def get_small_dataset(number):
    '''
        get train.csv head(number) data to test functions
    '''
    data = pd.read_csv('./data/train.csv', index_col = 0)
    res = data.head(number)
    res.to_csv('./data/small_data.csv')
    return res

def filter_data(read_data):
    data = read_data.drop(['Summary', 'Text'], axis = 1)
    return data

def get_test_index(data):
    target_data= data[data['Score'].isnull() == True] 
    return target_data


# 找到电影最初的timestamp
# 用户评价时的timestamp-电影最初的timestamp
# 用户最初的timestamp

def get_user_mean(data):
    # calculate each user's mean rating
    user_group = data.groupby('UserId')

    user_id = list(user_group.groups.keys())
    user_time_min = user_group['Time'].agg([np.min])
    user_time_max = user_group['Time'].agg([np.max])


    user_time_min.to_csv('./data/features/USER_TMIN.csv')
    user_time_max.to_csv('./data/features/USER_TMAX.csv')

    # user_mean_rating = user_group['Score'].agg([np.min])
    # user_mean_rating = user_mean_rating.rename(columns={'mean': 'user_mean'})
    # user_mean_rating.to_csv('./data/features/USER_MIN.csv')

def get_movie_mean(data):
    # calculate each movie's mean rating
    movie_group = data.groupby('ProductId')
    movie_id = list(movie_group.groups.keys())

    movie_time_min = movie_group['Time'].agg([np.min])
    movie_time_max = movie_group['Time'].agg([np.max])

    movie_time_min.to_csv('./data/features/MOVIE_TMIN.csv')
    movie_time_max.to_csv('./data/features/MOVIE_TMAX.csv')

    # movie_mean_rating = movie_group['Score'].agg([np.min])
    # movie_mean_rating = movie_mean_rating.rename(columns={'mean': 'movie_max'})
    # movie_mean_rating.to_csv('./data/features/SCR_MIN.csv')
def merge_time_feature():
    movie_min = pd.read_csv('./data/features/time/MOVIE_TMIN.csv', index_col = 0)
    movie_min = movie_min.rename(columns={'amin':'MOVIE_MIN'})

    movie_max = pd.read_csv('./data/features/time/MOVIE_TMAX.csv', index_col = 0)
    movie_max= movie_max.rename(columns={'amax':'MOVIE_MAX'})

    user_min = pd.read_csv('./data/features/time/USER_TMIN.csv', index_col = 0)
    user_min = user_min.rename(columns={'amin':'USER_MIN'})

    user_max = pd.read_csv('./data/features/time/USER_TMAX.csv', index_col = 0)
    user_max = user_max.rename(columns={'amax':'USER_MAX'})

    movie = pd.read_csv('./data/features/MOVIE.csv', index_col = 0)
    movie = pd.merge(movie, movie_min, on = 'ProductId')
    movie = pd.merge(movie, movie_max, on = 'ProductId')
    movie.to_csv('./data/features/MOVIE_2.csv', index_label = 'Id')


    user = pd.read_csv('./data/features/USER.csv', index_col = 0)
    user = pd.merge(user, user_min, on = 'UserId')
    user = pd.merge(user, user_max, on = 'UserId')


    user.to_csv('./data/features/User_2.csv', index_label = 'Id')

def get_new_data():
    data = pd.read_csv('./data/train.csv')
    movies = pd.read_csv('./data/movie.csv')
    users = pd.read_csv('./data/user_rating.csv')
    data = pd.merge(data, movies, on = 'ProductId')
    data = pd.merge(data, users, on = 'UserId')

    data.to_csv('new_data.csv')

def get_user_rating_preference():
    users = pd.read_csv('./data/user_rating.csv')
    data = pd.read_csv('train.csv')

    user_group = data.groupby('UserId')

    preference= []
    size = []
    for name, group in user_group:
        rating_var = -1

        if (group.shape[0] != 1):
            rating_diff = group['Score'] - group['movie_mean']
            rating_var = np.var(rating_diff)
        preference.append(rating_var)
        size.append(group.shape[0])
    
    users['user_preference'] = pd.DataFrame(preference)
    users['rating_numbers'] = pd.DataFrame(size)
    print(users.shape)
    
    users.to_csv('user.csv', index_label='Id')



def process_summary():
    data = pd.read_csv('./data/train.csv', index_col = 0)

    data['Summary'] = data['Summary'].str.replace("[^a-zA-Z#]", " ")
    data['Summary'] = data['Summary'].fillna(' ')
    data['Summary'] = data['Summary'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
    data['Summary'] = data['Summary'].apply(lambda x: x.lower())

    stop_words = stopwords.words('english')

    tokens = data['Summary'].apply(lambda x: x.split())
    wnl = WordNetLemmatizer() 
    tokens = tokens.apply(lambda x: [wnl.lemmatize(item) for item in x if item not in stop_words])

    detokens = []
    for i in range(len(tokens)):
        t = ' '.join(tokens[i])
        detokens.append(t)

    data['Summary Word'] = detokens
    

    score = data['Score']
    new_data = pd.merge(data['Summary Word'], score, on = 'Id')
    new_data = pd.merge(data['HelpfulnessNumerator'], new_data, on = 'Id')
    new_data = pd.merge(data['HelpfulnessDenominator'], new_data, on = 'Id')

    print(new_data.shape)
    new_data.to_csv('./data/Summary_train2.csv')

def add_flag():
    train = pd.read_csv('./data/train.csv', index_col = 0)
    train = train[['ProductId','UserId' ,'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Score']]
    train['Rate Flag'] = np.where(train['Score'] >= 3.0, 1, 0)
    train['Summary Flag'] = np.where(train['HelpfulnessNumerator'] / train['HelpfulnessDenominator'] >= 0.5, 1, 0)

    movie = pd.read_csv('./data/movie.csv', index_col = 0)
    user = pd.read_csv('user.csv', index_col = 0)
    print(user.shape)


    train = pd.merge(train, movie, on = 'ProductId')
    train = pd.merge(train, user, on = 'UserId')
    train = train.drop(['UserId', 'ProductId'], axis = 1)
    train['movie_mean'] = train['movie_mean'].round(3)
    train['user_mean'] = train['user_mean'].round(3)
    train['user_preference'] = train['user_preference'].round(3)

    print(train.shape)
    train.to_csv('train_flag1.csv', index_label = 'Id')

def get_final_train():
    train = pd.read_csv('./data/newnew_train.csv', index_col = 0)
    train = train.dropna(axis = 0, how = 'any')

    train = train.drop(['Summary Word'], axis = 1)

    idf_matrix = pd.read_csv('./data/idf_train.csv', index_col = 0)
    new_data = pd.merge(train, idf_matrix, on = 'Id')
    new_data.to_csv('./data/train_22.csv')

def train_flag_knn():
    train = pd.read_csv('train_flag1.csv')
    train = train.dropna(subset = ["Score"], axis = 0, how = 'any')

    labels2 = train['Rate Flag']

    # only scale features 
    features = train[['HelpfulnessNumerator' ,'HelpfulnessDenominator', 'Summary Flag' , 'movie_mean', 'user_mean', 'user_preference', 'rating_numbers']]
    features = features.fillna(value = 0)
    features = pd.DataFrame(preprocessing.scale(features), columns = ['HelpfulnessNumerator' ,'HelpfulnessDenominator', 'Summary Flag' , 'movie_mean', 'user_mean', 'user_preference', 'rating_numbers'])
    
    x_train, x_test, y_train, y_test = train_test_split(features, labels2, test_size=0.2)

    model = rfc()
    model.fit(features, labels2)
    predict = model.predict(x_test)
    res = mean_squared_error(y_test, predict)

    print(res)

def train_rate_knn():
    train = pd.read_csv('train_flag1.csv')
    train = train.dropna(subset = ["Score"], axis = 0, how = 'any')
    labels = train['Score'].astype('int')

    # only scale features 
    features = train[[ 'Summary Flag' , 'movie_mean', 'user_mean', 'user_preference']]
    features = features.fillna(value = 0)
    features = pd.DataFrame(preprocessing.scale(features), columns = ['Summary Flag' , 'movie_mean', 'user_mean', 'user_preference'])
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    model = rfc()
    model.fit(features, labels)
    model.score(x_test, y_test)
    joblib.dump(model, 'lr_noflag.model')
    predict = model.predict(x_test)
    res = mean_squared_error(y_test, predict)
    print(res)


def get_test_data():
    train = pd.read_csv('./data/new_train.csv', index_col = 0)
    test = pd.read_csv('./data/test.csv')
    target_id = np.array(test['Id'])
    test_data = train.loc[target_id, :]

    test_data = test_data.drop(['Score'], axis = 1)

    vectorizer = joblib.load('vector.model')
    svd_model = joblib.load('svd_model.model')

    test_data['Summary Word'] = test_data['Summary Word'].fillna(value = " ")

    idf_test = vectorizer.transform(test_data['Summary Word'])
    idf_test = svd_model.transform(idf_test)

    idf_test = pd.DataFrame(idf_test)
    idf_test.index.name = 'Id'

    train = test_data.drop(['Summary Word'], axis = 1)
    idf_matrix = pd.read_csv('./data/idf_test.csv', index_col = 0)

    new_data = pd.merge(train, idf_matrix, on = 'Id')
    new_data.to_csv('./data/test_final.csv')

def get_new_test():
    train = pd.read_csv('train_flag1.csv')
    test = pd.read_csv('./data/test.csv')

    target_id = np.array(test['Id'])
    test_data = train.loc[target_id, :]

    test_data = test_data.drop(['Score'], axis = 1)
    test_data = test_data.drop(['Id'], axis = 1)

    test_data.to_csv('./data/test_data.csv')

def data_prune(data):
    '''
        data must contains HelpfulnessNumerator, HelpfulnessDenominator two columns and summary
    '''
    data = data[data['user_preference'] < 1.0]
    data = data[data['HelpfulnessNumerator'] > 0]
    data = data.dropna(subset = ["Score"], axis = 0, how = 'any')

    print(data.shape)

    return data

from sklearn.naive_bayes import GaussianNB
def train_idf_svd():
    train = pd.read_csv('mydata666.csv', index_col = 0)
    train = train.dropna(axis = 0, how = 'any')

    label = train['Score'].astype('int')
    summary = train['Summary Word'].fillna(value = "")
    vector = TfidfVectorizer(max_df=0.9, stop_words='english')  
    summary_vector = vector.fit_transform(summary)
    joblib.dump(vector, 'vector.model')

    # feats_names = ["desc_" + x for x in vector.get_feature_names()]
    # print(feats_names)
    # exit(0)
    #vectorizer = TfidfVectorizer(stop_words='english', max_df= 0.9)
    # joblib.dump(vectorizer, 'vector.model')

    #other_features = train[['HelpfulnessDenominator','HelpfulnessNumerator','Time','SUMMARY_LEN','TEXT_LEN','SURP','CAPS','Rate Flag','Summary Flag','USER_MEAN','RATE_NUM','MOVIE_MEAN','MOVIE_SCRMAX','MOVIE_SCRMIN','MOVIE_CREATED','MOVIE_TMAX','USER_SCRMAX','USER_SCRMIN','USER_DEV','USER_CREATED','USER_TMAX','USER_TIME','MOVIE_TIME']]
    #other_features = train[['HelpfulnessDenominator','HelpfulnessNumerator', 'MOVIE_MEAN','USER_MEAN',  'USER_DEV', 'Time', 'Rate Flag','Summary Flag', 'RATE_NUM']]
    other_features = train[['HelpfulnessDenominator','HelpfulnessNumerator']]
    stda = StandardScaler()  
    other_features = stda.fit_transform(np.array(other_features))  
    
    new_feature = sparse.hstack((summary_vector, other_features)).tocsr()

    x_train, x_test, y_train, y_test = train_test_split(new_feature, label, test_size = 0.2)
    # lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')

    model = linear_model.LinearRegression()
    model.fit(x_train, y_train)
    # joblib.dump(model, 'lr_123.model')

    test_accuracy = model.score(x_test, y_test)
    print('test', test_accuracy)

    predict = model.predict(x_test)
    res = mean_squared_error(y_test, predict)
    print('msa', res)



def prediction():
    train = pd.read_csv('mydata666.csv', index_col = 0)
    print(train.shape)

    test = pd.read_csv('./data/test.csv')
    test_data = train.loc[test['Id']]

    summary = test_data['Summary Word'].fillna(value="")
    vector = joblib.load('vector.model')
    summary_vector = vector.transform(summary)
    other_features = test_data[['HelpfulnessDenominator','HelpfulnessNumerator', 'MOVIE_MEAN','USER_MEAN',  'USER_DEV', 'Time', 'Rate Flag','Summary Flag', 'RATE_NUM']].fillna(value=0)
    stda = StandardScaler()  
    other_features = stda.fit_transform(np.array(other_features))  
    
    features = sparse.hstack((summary_vector, other_features)).tocsr()
    #features = test_data[['SUMMARY_LEN', 'TEXT_LEN', 'SURP', 'CAPS','HelpfulnessDenominator','HelpfulnessNumerator','Time', 'USER_TIME','MOVIE_TIME', 'USER_MEAN', 'MOVIE_MEAN', 'USER_DEV', 'MOVIE_SCRMAX','MOVIE_SCRMIN', 'USER_SCRMIN', 'USER_SCRMAX']].fillna(value=0)

    model = joblib.load("lr_123.model")
    score = model.predict(features)

    test['Score'] = score

    test['Score'][test['Score'] <= 0] = 1.0
    test['Score'][test['Score'] > 5] = 5.0
    

    #test_data['Predict'] = test_data.apply(lambda x : 5 - x['Predict'] if (x['HelpfulnessDenominator'] != 0 and x['HelpfulnessNumerator'] / x['HelpfulnessDenominator'] < 0.5 and x['USER_DEV'] < 1) else x['Predict'])

    # test_data['Predict'] = test_data.apply(lambda x : x['HelpfulnessDenominator'])
    # exit(0)
    # test['Score'] = test_data['Predict']

    test.to_csv('submit1234.csv', index = 0)



def check_update(data):
    '''
        data:score, dev, user_mean, movie_mean, HelpfulnessDenominator, HelpfulnessNumerator)
    '''
    #data['Predict'] = data['Predict'].apply(lambda x: [wnl.lemmatize(item) for item in x if item not in stop_words])
    for index, row in data.iterrows():
        predict_score = row['Predict']
        movie_mean = row['MOVIE_MEAN']
        user_mean = row['USER_MEAN']
        user_dev = row['USER_DEV']
        agree = row['HelpfulnessNumerator']
        total = row['HelpfulnessDenominator']

        # 如果大部分人不认同某个用户的评分, 说明他的评分与大众审美相反
        if total > 1 and agree / total < 0.5 and user_dev < 1:
            if movie_mean >= 4 :
                if predict_score == 1 or predict_score == 2:
                    row['Predict'] = 5 - row['Predict']
            elif movie_mean == 3:
                if predict_score == 5 or predict_score == 1:
                    row['Predict'] = abs(predict_score - movie_mean)
            else:
                if predict_score == 3 or predict_score == 4:
                    row['Predict'] = 5 - row['Predict']
                elif predict_score == 5:
                    row['Predict'] = 1
    
    print(data['Predict'].head())
    return data


# question = text.apply(lambda x: len(re.findall(r'[?]',x)))
# surprise = text.apply(lambda x: len(re.findall(r'[!]',x)))
# print(question.head())
# print(surprise.head())

train = pd.read_csv('mydata666.csv', index_col = 0)
train = train.dropna(axis = 0, how = 'any')

features = train[['SUMMARY_LEN', 'TEXT_LEN', 'SURP', 'CAPS','HelpfulnessDenominator','HelpfulnessNumerator','Time', 'USER_TIME','MOVIE_TIME', 'USER_MEAN', 'MOVIE_MEAN', 'USER_DEV', 'MOVIE_SCRMAX','MOVIE_SCRMIN', 'USER_SCRMIN', 'USER_SCRMAX']]

label = train[['Score']].astype('int')
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size = 0.2)

classifier = RandomForestRegressor(n_estimators=100,verbose=2,n_jobs=20,min_samples_split=5,random_state=1034324)
classifier.fit(x_train, y_train)

# joblib.dump(classifier, 'randomForest2.model')
importances = classifier.feature_importances_
indices = np.argsort(importances)[::-1]

features_label = train.columns[1:]
for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, features_label[indices[f]], importances[indices[f]]))

test_accuracy = classifier.score(x_test, y_test)
print('test', test_accuracy)

predict = classifier.predict(x_test)
res = mean_squared_error(y_test, predict, squared=False)
print('msa', res)


