__author__ = 'NLP-PC'
from load_data import load_anew
from visualization import draw_scatter_with_labels
from load_data import load_extend_anew
words, arousal, valence = load_extend_anew()

draw_scatter_with_labels(arousal,valence, words, 'arousal', 'valence')

word, a, v = load_anew()
draw_scatter_with_labels(a, v, word, 'arousal', 'valence')