__author__ = 'NLP-PC'
from load_data import load_pickle
import nltk
from load_data import load_extend_anew

words, _, _=load_extend_anew()

feature_names = load_pickle('./data/features/feature_names.p')
print(feature_names)
english_stemmer=nltk.stem.SnowballStemmer('english')
stemmed_dict = [english_stemmer.stem(w) for w in words]
print(stemmed_dict)
print(set(feature_names) & set(stemmed_dict))