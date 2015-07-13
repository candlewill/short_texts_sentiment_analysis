__author__ = 'hs'
import numpy as np
from save_data import csv_save


def count_matching(sentences, words, threshold=1):
    count = 0
    for sent in sentences:
        sent_list = sent.split(' ')
        word_count = [sent_list.count(w) for w in words]
        word_count = np.array(word_count)
        occur_times = (word_count >= 1).sum()
        if occur_times >= threshold:
            # print(sent, np.array(words)[word_count>=1])
            count = count + 1

    return count


def select_matching(texts, labels, words, threshold=1):
    selected_content = []
    for text, label in zip(texts, labels):
        sent_list = text.split(' ')
        word_count = [sent_list.count(w) for w in words]
        word_count = np.array(word_count)
        occur_times = (word_count >= 1).sum()
        if occur_times >= threshold:
            print(label, text, np.array(words)[word_count >= 1])
            selected_content.append((label, text))
    csv_save(selected_content, './data/traindata/Sentiment140/pre-processed/anew_part_of_nostem_160000.csv')
    return


def avg_valence(texts, words, valence):
    texts_scores = []
    for text in texts:
        sent_list = text.split(' ')
        word_count = [sent_list.count(w) for w in words]
        word_count = np.array(word_count)
        occur_times = (word_count >= 1).sum()
        if occur_times > 0:
            avg = np.average(np.array(valence)[word_count >= 1])
        else:
            avg = -1
        texts_scores.append(avg)
    return texts_scores


if __name__ == '__main__':
    from load_data import load_selected_data, load_anew, load_extend_anew

    # # print(count_matching(texts, words))


    texts, labels = load_selected_data(data_type='train', stem=False)
    words, valence, _ = load_anew()

    # select_matching(texts,labels, words)
    # exit()
    words, valence = np.array(words), np.array(valence)
    words_pos, valence_pos = words[valence > np.average(valence)], valence[
        valence > np.average(valence)]  # avg = 5.1511713456
    words_neg, valence_neg = words[valence < np.average(valence)], valence[
        valence < np.average(valence)]  # avg = 5.1511713456

    pos = avg_valence(texts, words_pos, valence_pos)
    neg = avg_valence(texts, words_neg, valence_neg)

    from visualization import draw_scatter_with_color

    draw_scatter_with_color(pos, neg, labels, 'pos', 'neg')
    # classify
    polarity = []
    for pos_score, neg_score in zip(pos, neg):
        if pos_score == -1:
            polarity.append(0)
        elif neg_score == -1:
            polarity.append(1)
        else:
            if pos_score > neg_score:
                polarity.append(1)
            else:
                polarity.append(0)

    from analysis import analysis_result

    print(labels)
    print(polarity)
    analysis_result(labels, polarity)
