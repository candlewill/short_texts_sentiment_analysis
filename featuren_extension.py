__author__ = 'NLP-PC'
from load_data import load_pickle
import nltk
from load_data import load_extend_anew

words, _, _=load_extend_anew()

feature_names = load_pickle('./data/features/feature_names.p')
print(feature_names)
english_stemmer=nltk.stem.SnowballStemmer('english')
stemmed_dict = [english_stemmer.stem(w) for w in words]
print(len(stemmed_dict))
overlapping_words= (set(feature_names) & set(stemmed_dict))
print(len(overlapping_words))
print(english_stemmer.stem(''))
features = load_pickle('./data/transformed_data/transformed_train.p')
print(features[1,249])
print(type(features))

d='We are very nice goes I am nicely'
sent = list(d.split())
print(sent)
stemmed_sent = [english_stemmer.stem(w) for w in sent]
print(stemmed_sent)