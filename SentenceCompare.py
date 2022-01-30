import pandas as pd
import json
import string
import contractions
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from pymongo import MongoClient
from bson.objectid import ObjectId
from difflib import SequenceMatcher

with open('./data.json') as f:
    result = json.load(f)

data = {}
conflictdf_list = list()
conflictdf = pd.DataFrame(columns=["intent_x", "intent_y", "text1","text2", "ratio", "lang"])

for intent in result:
    for lang, phrases in intent['trainingPhrases'].items():
        for text in phrases:
            if lang not in data:
                data[lang] = []
            data[lang].append({'intent': intent['_id'], 'text': ' '.join([t['text'] for t in text]), 'lang': lang})
print(data)

for key in data:
    df = pd.DataFrame(data[key])
    df.drop(df.columns.difference(['text', 'intent']), 1, inplace=True)
    intent = [pd.DataFrame(y) for x, y in df.groupby('intent', as_index=False)]
    intentList = df['intent'].unique().astype(str).tolist()
    for i in range(len(intent)):
        for j in range(len(intent)):
            if i == j:
                break
            else:
                for k in range(len(intent[i]["text"])):
                    compare = intent[i]["text"].iloc[k]
                    for l in range(len(intent[j]["text"])):
                        ratio = SequenceMatcher(None, compare, intent[j]["text"].iloc[l]).ratio()
                        if ratio >= 0.8:
                            # print(ratio)
                            conflictdf = pd.concat([pd.DataFrame([{
                                "intent_x": intent[i]["intent"].iloc[0],
                                "intent_y": intent[j]["intent"].iloc[0],
                                "text1" : compare,
                                "text2" : intent[j]["text"].iloc[l],
                                "ratio" : ratio,
                                "lang" : key
                                }]), conflictdf])

                            if len(conflictdf) != 0:
                                conflictdf_list.append(conflictdf)

print(conflictdf_list)



