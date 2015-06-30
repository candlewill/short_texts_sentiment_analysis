__author__ = 'NLP-PC'

parameters = {

    # vectorizer参数选择
    'min_df': 0,  # 仅考虑频率出现在min_df之上的ngrams
    'ngram_range': (1, 3),  # ngram范围
    'test_data_size': 20000,  # 选择不同训练数据大小
    'max_df': 0.8,  # 除去太频繁出现的ngrams
    'TF_binary': True,  # 是否使用TF-IDF加权
    'norm': 'l1',  # 是否规格化
    'sublinear_tf': True,  # 是否对TF使用log(1+x)
    'max_features': 5000,

    # feature type
    'combine_feature': False,  # 是否使用更多的特征

    # 分类器
    'classifier': 'nb',  # 贝叶斯或者svm分类器，目前svm还有问题

    # 是否对training_data分群
    'clustering_training_data': True,  # 具体参数设置在后面的if中

    # 是否对test_data分群
    'clustering_test_data': False,

}

