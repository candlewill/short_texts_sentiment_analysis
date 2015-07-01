__author__ = 'NLP-PC'
from load_data import load_anew
from visualization import draw_scatter_with_labels

words, arousal, valence = load_anew()

draw_scatter_with_labels(arousal,valence, words, 'valence', 'arousal')