__author__ = 'NLP-PC'

parameters = {

    # vectorizer参数选择
    'min_df': 0,  # 仅考虑频率出现在min_df之上的ngrams
    'ngram_range': (1, 1),  # ngram范围
    'test_data_size': 20000,  # 选择不同训练数据大小 # Please Note: even when value is 160 000, the true len is 1 600 000
    'max_df': 0.5,  # 除去太频繁出现的ngrams
    'TF_binary': True,  # 是否使用TF-IDF加权
    'norm': 'l1',  # 是否规格化
    'sublinear_tf': True,  # 是否对TF使用log(1+x)
    'max_features': 5000,
}