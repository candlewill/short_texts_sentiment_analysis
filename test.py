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
    i += 1
    print(i)

print(feature_vec)