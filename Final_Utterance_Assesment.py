import json
import string
from difflib import SequenceMatcher
import contractions
import nltk
import pandas as pd
from bson.objectid import ObjectId
from iso639 import languages
from nltk.stem.snowball import SnowballStemmer
from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tracemalloc

nltk.download('stopwords')
from nltk.corpus import stopwords


tracemalloc.start()


# client = MongoClient()
# collection = client['production-saas-botplatform']['dialogflowintents']
#
# result = list(collection.find({'isDefault': False, 'botId': ObjectId("5dfc6bcdcb44a2688c15af35")},
#                               {'botId': True, 'intentId': True, 'trainingPhrases': True}))[1:6]

with open('./data.json') as f:
    result = json.load(f)

data = {}

for intent in result:
    for lang in intent['trainingPhrases'].keys():
        if lang not in data.keys():
            data[lang] = []
        for text in intent['trainingPhrases'][lang]:
            data[lang].append({'intent': str(intent['_id']), 'text': ' '.join([t['text'] for t in text])})

result = {'conflicting_intents': [], 'intent_scores': {}}

for lang in data.keys():
    df = pd.DataFrame(data[lang])
    df.drop(df.columns.difference(['text', 'intent']), 1, inplace=True)

    intent = [pd.DataFrame(y) for x, y in df.groupby('intent', as_index=False)]
    intentList = df['intent'].unique().astype(str).tolist()

    training_data = {}
    for i in range(len(intent)):
        name = intent[i]['intent'].iloc[0]
        training_data[name] = []
        for j in range(len(intent[i]['text'])):
            training_data[name].append(intent[i]['text'].iloc[j])

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
                            result['conflicting_intents'].append({
                                "intent_x": intent[i]["intent"].iloc[0],
                                "intent_y": intent[j]["intent"].iloc[0],
                                "text_x": compare,
                                "text_y": intent[j]["text"].iloc[l],
                                "ratio": round(ratio, 2),
                                "lang": lang
                            })


    def clean_string(text):
        text = contractions.fix(text)
        text = ''.join([word.strip() for word in text if word not in string.punctuation])
        text = text.lower()
        text = ' '.join([stemmer.stem(word) for word in text.split()])
        text = ' '.join([word for word in text.split() if word not in sws])
        return text


    def cosine_sim_vectors(vec1, vec2):
        return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]


    stemmer = SnowballStemmer(languages.get(alpha2=lang).name.lower())
    sws = stopwords.words(languages.get(alpha2=lang).name.lower())

    for key in training_data:
        td = training_data[key]
        cleaned = list(map(clean_string, td))
        vectorizer = CountVectorizer()
        vectors = vectorizer.fit_transform(cleaned).toarray()
        scores = []
        for i in range(0, len(vectors)):
            score = 0
            for j in range(0, len(vectors)):
                score += cosine_sim_vectors(vectors[i], vectors[j])
            score /= len(vectors)
            scores.append(score)
        scores = [round((score / max(scores)) * 0.9, 2) for score in scores]
        intent_score = round(sum(scores) / len(vectors), 2)
        if key not in result['intent_scores'].keys():
            result['intent_scores'][key] = {
                'intent_score': intent_score,
                'training_phrases': {}
            }
        if lang not in result['intent_scores'][key]['training_phrases'].keys():
            result['intent_scores'][key]['training_phrases'][lang] = []
        for i in range(len(scores)):
            result['intent_scores'][key]['training_phrases'][lang].append({
                "text": td[i],
                "score": scores[i]
            })

with open("result.json", "w", encoding='utf-8') as outfile:
    json.dump(result, outfile, ensure_ascii=False, indent=4)

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
tracemalloc.stop()