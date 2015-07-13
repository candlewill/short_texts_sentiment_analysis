__author__ = 'NLP-PC'
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean


def draw_line_chart(x, y, x_labels, y_labels):
    plt.plot(x, y, 'o-')
    plt.xlabel(x_labels)
    plt.ylabel(y_labels)
    plt.show()


def draw_scatter_with_labels(x, y, labels, x_labels, y_labels):
    def on_pick(event):
        ind = event.ind
        print('label: %s, %s: %s, %s: %s' % (
        str(np.take(labels, ind)), x_labels, str(np.take(x, ind)), y_labels, str(np.take(y, ind))))

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(x, y, marker='o', picker=True)
    plt.axhline(mean(y), color='black')
    plt.axvline(mean(x), color='black')
    plt.xlabel(x_labels)
    plt.ylabel(y_labels)
    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()


def draw_scatter_with_color(x, y, labels, x_labels, y_labels):
    def on_pick(event):
        ind = event.ind
        print('label: %s, %s: %s, %s: %s' % (
            str(np.take(labels, ind)), x_labels, str(np.take(x, ind)), y_labels, str(np.take(y, ind))))

    colors = ['red' if labels[i] == 1 else 'green' for i in range(0, len(labels))]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(x, y, marker='o', picker=True, c=colors)
    plt.axhline(mean(y), color='black')
    plt.axvline(mean(x), color='black')
    plt.xlabel(x_labels)
    plt.ylabel(y_labels)
    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()


if __name__ == "__main__":
    v = [0.688022284123, 0.740947075209, 0.74930362117, 0.757660167131, 0.757660167131, 0.771587743733, 0.782729805014,
         0.779944289694, 0.782729805014, 0.791086350975, 0.793871866295, 0.782729805014, 0.788300835655]
    x = np.linspace(100, 1300, len(v))
    draw_line_chart(x, v, 'Feature number', 'Accuracy')
