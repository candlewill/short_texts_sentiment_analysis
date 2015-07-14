from load_data import load_processed_data
from qrcode_generator import to_qrcode
import numpy as np

texts, labels = load_processed_data(data_type='train', stem=False)
feature_vec = []
i = 0
for text, label in zip(texts, labels):
    text_qrcode = to_qrcode(text)
    text_qrcode = list(text_qrcode.getdata())
    feature_vec.append(np.append(label, text_qrcode))

from save_data import csv_save

csv_save(feature_vec, './data/traindata/qrcode.csv')
