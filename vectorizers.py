__author__ = 'NLP-PC'
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from load_data import load_anew, load_extend_anew
import re, nltk
from preprocess import preprocessor
from parameters import parameters
from logger_manager import log_state


def punctuation_estimator():
    # the method to design is referenced the site: http://goo.gl/BDe8Cb
    def get_punctuations(text):
        # only question mark and exclamation are considered
        punctuation = re.findall(r'[?!]', text)
        for mark in punctuation:
            yield mark

    analyzer = get_punctuations
    # v = CountVectorizer(analyzer=analyzer, binary=True)
    v = TfidfVectorizer(analyzer=analyzer, binary=True, norm='l1', use_idf=False, sublinear_tf=True, max_df=1)
    return v


def anew_estimator(words=None):
    english_stemmer = nltk.stem.SnowballStemmer('english')
    words, _, _ = load_extend_anew()
    words = [english_stemmer.stem(w) for w in words]
    words = set(words)
    # Note: the max_features parameter is ignored if vocabulary is not None
    vectorizer = TfidfVectorizer(vocabulary=words, binary=True, norm='l1', use_idf=False,
                                 sublinear_tf=True, max_df=0.5)
    return vectorizer


def TFIDF_estimator():
    log_state('Start generating features')

    class StemmedTfidfVectorizer(TfidfVectorizer):
        def build_analyzer(self):
            # 利用NLTK进行词干化处理
            english_stemmer = nltk.stem.SnowballStemmer('english')
            analyzer = super(TfidfVectorizer, self).build_analyzer()
            return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

    vectorizer_param = {'preprocessor': preprocessor, 'ngram_range': parameters['ngram_range'], 'analyzer': 'word',
                        'min_df': parameters['min_df'], 'max_df': parameters['max_df'],
                        'binary': parameters['TF_binary'], 'norm': parameters['norm'],
                        'sublinear_tf': parameters['sublinear_tf'], 'max_features': parameters['max_features']}
    log_state((sorted(list(vectorizer_param.items()))))
    log_state('Training data size: ' + str(parameters['test_data_size']))
    return StemmedTfidfVectorizer(**vectorizer_param)


if __name__ == "__main__":
    d = ['thank you! why are you googd???, and!, are you good$!@#$%^&*?(), ~$%^^*%$#???']
    pe = punctuation_estimator()
    a = pe.fit_transform(d)
    print(a, pe.get_feature_names())
