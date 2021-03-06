__author__ = 'NLP-PC'
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from load_data import load_pickle, load_test_data
from matplotlib import pyplot as plt
from logger_manager import log_performance

def analysis_result(predict, true, figure=False):
    f1 = f1_score(true, predict, average='binary')
    precision_binary, recall_binary, fbeta_score_binary, _ = precision_recall_fscore_support(true, predict,
                                                                                             average='binary')
    accuracy = accuracy_score(true, predict)
    print('正确率(Accuracy)：%.3f\nF值(Macro-F score)：%.3f' % (accuracy, f1))
    print('精确度(Precision):%.3f\n召回率：%.3f\nF值: %.3f' % (precision_binary, recall_binary, fbeta_score_binary))
    log_performance(accuracy, f1,precision_binary, recall_binary, len(true))
    if figure==False:
        return
    # 画图
    n_groups = 5
    values = (accuracy, f1, precision_binary, recall_binary, fbeta_score_binary)
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    rects1 = plt.bar(index + bar_width / 2, values, bar_width, alpha=0.6, color='b')
    plt.xlabel('Result')
    plt.ylabel('Scores')
    plt.title('Experiment analysis')
    plt.xticks(index + bar_width, ('Accuracy', 'F', 'Precision', 'Recall', 'F'))
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    predict = load_pickle('./data/predict_labels/predict_labels.p')
    _, true_labels = load_test_data()
    analysis_result(predict, true_labels)
