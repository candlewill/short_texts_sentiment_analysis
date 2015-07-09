__author__ = 'hs'
from load_data import load_word_embedding
import time

st = time.time()
# examplenshu
model = load_word_embedding()

print(model['computer'])
elapsed_time = time.time() - st
print("Elapsed time: %.3fmin" % (elapsed_time / 60))
# cosine similarity
print(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1))
elapsed_time = time.time() - st
print("Elapsed time: %.3fmin" % (elapsed_time / 60))
print(model.doesnt_match("breakfast cereal dinner lunch".split()))
elapsed_time = time.time() - st
print("Elapsed time: %.3fmin" % (elapsed_time / 60))
print(model.similarity('woman', 'man'))
elapsed_time = time.time() - st
print("Elapsed time: %.3fmin" % (elapsed_time / 60))
# Find the top-N most similar words, using the multiplicative combination objective proposed by Omer Levy and Yoav Goldberg in [4]. Positive words still contribute positively towards the similarity, negative words negatively, but with less susceptibility to one large distance dominating the calculation.
# In the common analogy-solving case, of two positive and one negative examples, this method is equivalent to the “3CosMul” objective (equation (4)) of Levy and Goldberg.
# Additional positive or negative examples contribute to the numerator or denominator, respectively – a potentially sensible but untested extension of the method. (With a single positive example, rankings will be the same as in the default most_similar.)
print(model.most_similar_cosmul(positive=['baghdad', 'england'], negative=['london'], topn=10))
elapsed_time = time.time() - st
print("Elapsed time: %.3fmin" % (elapsed_time / 60))
# Compute cosine similarity between two sets of words.
print(model.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant']))
elapsed_time = time.time() - st
print("Elapsed time: %.3fmin" % (elapsed_time / 60))
