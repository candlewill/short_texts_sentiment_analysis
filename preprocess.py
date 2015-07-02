__author__ = 'NLP-PC'
# encoding: utf-8
import re, const_values as const
import string

def preprocessor(tweet):
    emo_repl_order = const.emo_repl_order
    emo_repl = const.emo_repl
    re_repl = const.re_repl

    tweet = tweet.lower()
    for k in emo_repl_order:
        tweet = tweet.replace(k, emo_repl[k])

    for r, repl in re_repl.items():
        tweet = re.sub(r, repl, tweet)

    # remove all punctuation
    for c in string.punctuation:
        tweet= tweet.replace(c, " ")
    tweet = re.sub(r" +", " ", tweet).strip()

    return tweet

if __name__ == '__main__':
    from load_data import load_train_data
    texts, _ = load_train_data()
    for line in texts:
        print(preprocessor(line))
