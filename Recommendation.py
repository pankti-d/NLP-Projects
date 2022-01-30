import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer

df = pd.read_csv('yelp_training_set_review.csv')
df = df[['review_id', 'user_id', 'business_id', 'text', 'stars', 'date']]
df_business = pd.read_csv('yelp_training_set_business.csv')
#Check Null values in Dataframe
# print(df.isnull().sum())
#Select only stars and text
yelp_data = df[['business_id', 'user_id', 'stars', 'text']]

yelp_data = yelp_data.dropna()
import string
from nltk.corpus import stopwords
stop = []
for word in stopwords.words('english'):
    s = [char for char in word if char not in string.punctuation]
    stop.append(''.join(s))


def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    # print(type(mess)) is string
    try:
        nopunc = [char for char in mess if char not in string.punctuation]
        # Join the characters again to form the string.
        nopunc = ''.join(nopunc)
        # Now just remove any stopwords
        return " ".join([word for word in nopunc.split() if word.lower() not in stop])
    except:
        print(mess)



yelp_data['text'] = yelp_data['text'].apply(text_process)

userid_df = yelp_data[['user_id','text']]
business_df = yelp_data[['business_id', 'text']]

userid_df = userid_df.groupby('user_id').agg({'text': ' '.join})
business_df = business_df.groupby('business_id').agg({'text': ' '.join})

from sklearn.feature_extraction.text import TfidfVectorizer
#userid vectorizer
userid_vectorizer = TfidfVectorizer(tokenizer = WordPunctTokenizer().tokenize, max_features=5000)
userid_vectors = userid_vectorizer.fit_transform(userid_df['text'])
#Business id vectorizer
businessid_vectorizer = TfidfVectorizer(tokenizer = WordPunctTokenizer().tokenize, max_features=5000)
businessid_vectors = businessid_vectorizer.fit_transform(business_df['text'])
userid_rating_matrix = pd.pivot_table(yelp_data, values='stars', index=['user_id'], columns=['business_id'])

P = pd.DataFrame(userid_vectors.toarray(), index=userid_df.index, columns=userid_vectorizer.get_feature_names())
Q = pd.DataFrame(businessid_vectors.toarray(), index=business_df.index, columns=businessid_vectorizer.get_feature_names())


def matrix_factorization(R, P, Q, steps=25, gamma=0.001, lamda=0.02):
    for step in range(steps):
        for i in R.index:
            for j in R.columns:
                if R.loc[i, j] > 0:
                    eij = R.loc[i, j] - np.dot(P.loc[i], Q.loc[j])
                    P.loc[i] = P.loc[i] + gamma * (eij * Q.loc[j] - lamda * P.loc[i])
                    Q.loc[j] = Q.loc[j] + gamma * (eij * P.loc[i] - lamda * Q.loc[j])
        e = 0
        for i in R.index:
            for j in R.columns:
                if R.loc[i, j] > 0:
                    e = e + pow(R.loc[i, j] - np.dot(P.loc[i], Q.loc[j]), 2) + lamda * (
                                pow(np.linalg.norm(P.loc[i]), 2) + pow(np.linalg.norm(Q.loc[j]), 2))
        if e < 0.001:
            break
    return P, Q

P,Q = matrix_factorization(userid_rating_matrix, P, Q, steps=25, gamma=0.001, lamda=0.02)
words = "i want to have dinner with beautiful views"
test_df= pd.DataFrame([words], columns=['text'])
test_df['text'] = test_df['text'].apply(text_process)
test_vectors = userid_vectorizer.transform(test_df['text'])
test_v_df = pd.DataFrame(test_vectors.toarray(), index=test_df.index, columns=userid_vectorizer.get_feature_names())
predictItemRating=pd.DataFrame(np.dot(test_v_df.loc[0],Q.T),index=Q.index,columns=['Rating'])
topRecommendations=pd.DataFrame.sort_values(predictItemRating,['Rating'],ascending=[0])[:7]

for i in topRecommendations.index:
    print(df_business[df_business['business_id']==i]['name'].iloc[0])
    print(df_business[df_business['business_id']==i]['categories'].iloc[0])
    print(str(df_business[df_business['business_id']==i]['stars'].iloc[0])+ ' '+str(df_business[df_business['business_id']==i]['review_count'].iloc[0]))
    print('')