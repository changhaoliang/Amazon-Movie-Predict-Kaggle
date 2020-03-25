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

def get_user_mean(data):
    # calculate each user's mean rating
    user_group = data.groupby('UserId')
    user_id = list(user_group.groups.keys())
    user_mean_rating = user_group['Score'].agg([np.mean])
    user_mean_rating = user_mean_rating.rename(columns={'mean': 'user_mean'})
    user_mean_rating.to_csv('./data/user_rating.csv')
    return user_mean_rating

def get_movie_mean(data):
    # calculate each movie's mean rating
    movie_group = data.groupby('ProductId')
    movie_id = list(movie_group.groups.keys())
    movie_mean_rating = movie_group['Score'].agg([np.mean])
    movie_mean_rating = movie_mean_rating.rename(columns={'mean': 'movie_mean'})
    movie_mean_rating.to_csv('./data/movie.csv')
    return movie_mean_rating

def get_new_data():
    data = pd.read_csv('./data/train.csv')
    movies = pd.read_csv('./data/movie.csv')
    users = pd.read_csv('./data/user_rating.csv')
    data = pd.merge(data, movies, on = 'ProductId')
    data = pd.merge(data, users, on = 'UserId')

    data.to_csv('new_data.csv')

def get_user_rating_preference():
    users = pd.read_csv('./data/user_rating.csv')
    data = pd.read_csv('train_flag.csv')

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

def train_idf_svd():
    train = pd.read_csv('./data/Summary_train2.csv', index_col = 0)
    # train = data_prune(train)
    train = train.dropna(subset = ["Score"], axis = 0, how = 'any')
    

    label = train['Score'].astype('int')
    summary = train['Summary Word'].fillna(value = "")
    vectorizer = TfidfVectorizer(stop_words='english', max_df= 0.5)
    features = vectorizer.fit_transform(summary)

    other_features = train[['HelpfulnessDenominator','HelpfulnessNumerator']]
    new_feature = sparse.hstack((features, other_features)).tocsr()
    print(new_feature.shape)

    x_train, x_test, y_train, y_test = train_test_split(features, label, test_size = 0.2)

    print(x_test.shape)
    joblib.dump(vectorizer, 'vector2.model')

    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    lr.fit(features, label)
    joblib.dump(lr, 'lr_123.model')

    # lr = joblib.load('lr_123.model')

    # print test acc and rmse
    test_accuracy = lr.score(x_test, y_test)
    print('test', test_accuracy)

    predict = lr.predict(x_test)
    res = mean_squared_error(y_test, predict)
    print('msa', res)

# train_idf_svd()
# exit(0)



def prediction():
    # dev = pd.read_csv('./data/features/SCR_DEV.csv')

    train = pd.read_csv('mydata.csv', index_col = 0)
    # mydata = pd.merge(train, dev, on = 'Id')

    # mydata.to_csv('mydata.csv')
    # exit(0)

    summary = train[['HelpfulnessNumerator', 'HelpfulnessDenominator', 'user_preference','user_mean', 'movie_mean', 'Time', 'Summary len', 'Text len', 'CAPS', 'SURP']].fillna(value=0)
    
    test = pd.read_csv('./data/test.csv')
    summary = summary.loc[test['Id']]
    # vectorizer = joblib.load('vector2.model')
    # features = vectorizer.transform(summary) 
 
    features = summary

    model = joblib.load("randomForest2.model")
    score = model.predict(features)

    test['Score'] = score.astype('float')

    test.to_csv('res_last_try_today.csv', index = 0)
    return test['Score']
# prediction()


# question = text.apply(lambda x: len(re.findall(r'[?]',x)))
# surprise = text.apply(lambda x: len(re.findall(r'[!]',x)))
# print(question.head())
# print(surprise.head())


train = pd.read_csv('mydata777.csv', index_col = 0)
train = train.dropna(axis = 0, how = 'any')

features = train[['HelpfulnessNumerator', 'HelpfulnessDenominator', 'user_preference', 'user_mean', 'movie_mean', 'Time', 'Summary len', 'Text len','CAPS', 'SURP']]

label = train[['Score']].astype('int')

x_train, x_test, y_train, y_test = train_test_split(features, label, test_size = 0.2)
classifier = DecisionTreeClassifier(max_depth=100)


# classifier = RandomForestRegressor(n_estimators=100,verbose=2,n_jobs=20,min_samples_split=5,random_state=1034324)
classifier.fit(x_train, y_train)

# joblib.dump(classifier, 'randomForest2.model')

test_accuracy = classifier.score(x_test, y_test)
print('test', test_accuracy)

predict = classifier.predict(x_test)
res = mean_squared_error(y_test, predict)
print('msa', res)


