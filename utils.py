"""
    File name: utils.py
    Author: Kshitij Shah
    Date created: 5/30/2018
    Python Version: 3.6
"""

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


def evaluate(y_true, y_score):
    conf_mat = confusion_matrix(y_true=np.argmax(y_true, axis=1), y_pred=np.argmax(y_score, axis=1))
    accuracy = accuracy_score(y_true=np.argmax(y_true, axis=1), y_pred=np.argmax(y_score, axis=1))
    roc = roc_auc_score(y_true=y_true, y_score=y_score)
    return accuracy, conf_mat, roc


def plot_me_nice(confusion_matrix, labels, style='gist_heat', title='unknown'):
    confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()
    hm = ax.pcolor(confusion_matrix, cmap=style)
    ax.set_xticks(np.arange(len(labels)) + 0.5, minor=False)
    ax.set_yticks(np.arange(len(labels)) + 0.5, minor=False)
    ax.set_xticklabels(labels, minor=False)
    ax.set_yticklabels(labels, minor=False)
    plt.xticks(rotation=90)
    plt.colorbar(hm)
    plt.title(title)
    return plt.show()