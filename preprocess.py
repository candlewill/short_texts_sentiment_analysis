__author__ = 'NLP-PC'
import re, const_values as const

# Ô¤´¦Àí
def preprocessor(tweet):
    emo_repl_order = const.emo_repl_order
    emo_repl = const.emo_repl
    re_repl = const.re_repl

    tweet = tweet.lower()
    for k in emo_repl_order:
        tweet = tweet.replace(k, emo_repl[k])
    tweet = tweet.replace("-", " ").replace("_", " ").replace('"', '').replace(".", '').replace(',', '').replace(';',
                                                                                                                 '').strip()
    for r, repl in re_repl.items():
        tweet = re.sub(r, repl, tweet)

    # stem²Ù×ö
    # english_stemmer=nltk.stem.SnowballStemmer('english')
    # tweet_list=nltk.word_tokenize(tweet)
    # tweet=' '.join([english_stemmer.stem(t) for t in tweet_list])

    return tweet