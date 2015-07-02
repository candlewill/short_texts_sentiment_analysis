__author__ = 'NLP-PC'
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re


def punctuation_estimator():
    # the method to design is referenced the site: http://goo.gl/BDe8Cb
    def get_punctuations(text):
        # only question mark and exclamation are considered
        punctuation = re.findall(r'[?!]', text)
        for mark in punctuation:
            yield mark

    analyzer = get_punctuations
    # v = CountVectorizer(analyzer=analyzer, binary=True)
    v = TfidfVectorizer(analyzer=analyzer, binary=True, norm='l1', use_idf=False, sublinear_tf=True)
    return v


if __name__ == "__main__":
    d = ['thank you! why are you googd???, and!, are you good$!@#$%^&*?(), ~$%^^*%$#???']
    pe = punctuation_estimator()
    a = pe.fit_transform(d)
    print(a, pe.get_feature_names())
