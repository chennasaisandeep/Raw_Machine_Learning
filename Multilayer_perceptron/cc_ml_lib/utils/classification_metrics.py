
import numpy as np


def get_precision(y_actual, y_pred):
    true_positives = np.sum(np.logical_and(y_pred == 1, y_actual == 1))
    false_positives = np.sum(np.logical_and(y_pred == 1, y_actual == 0))
    precision = true_positives / (true_positives + false_positives)
    return precision

def get_recall(y_actual, y_pred):
    true_positives = np.sum(np.logical_and(y_pred == 1, y_actual == 1))
    false_negatives = np.sum(np.logical_and(y_pred == 0, y_actual == 1))
    recall = true_positives / (true_positives + false_negatives)
    return recall

def get_f1_score(y_actual, y_pred):
    precision = get_precision(y_actual, y_pred)
    recall = get_recall(y_actual, y_pred)
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return f1_score
