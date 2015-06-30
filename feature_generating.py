__author__ = 'NLP-PC'
from preprocess import preprocessor
from parameters import parameters
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk.stem
from load_data import load_train_data
from load_data import load_test_data
import pickle
from save_data import dump_picle

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        #利用NLTK进行词干化处理
        english_stemmer=nltk.stem.SnowballStemmer('english')
        analyzer=super(TfidfVectorizer,self).build_analyzer()
        return lambda doc:(english_stemmer.stem(w) for w in analyzer(doc))

vectorizer_param={'preprocessor': preprocessor, 'ngram_range': parameters['ngram_range'], 'analyzer':'word',
                                    'min_df':parameters['min_df'], 'max_df': parameters['max_df'],
                                    'binary': parameters['TF_binary'], 'norm': parameters['norm'],'sublinear_tf': parameters['sublinear_tf'], 'max_features': parameters['max_features']}

vectorizer = StemmedTfidfVectorizer(**vectorizer_param)
texts, _=load_train_data('Sentiment140')
transformed_train=vectorizer.fit_transform(texts)
testdata, _ = load_test_data()
transformed_test=vectorizer.transform(testdata)
dump_picle(transformed_train, "./data/transformed_data/transformed_train.p")
dump_picle(transformed_test, "./data/transformed_data/transformed_test.p")