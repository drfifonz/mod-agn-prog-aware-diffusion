"""Global metrics from all predictions of one test run; no batch averaging."""
from statistics import mean, stdev
import math


def classification_metrics(true: list[int], predicted: list[int], classes: list[str]) -> dict:
    if not classes or len(classes) != len(set(classes)):
        raise ValueError('Class names must be nonempty and unique')
    if not true or len(true) != len(predicted):
        raise ValueError('Prediction arrays must be nonempty and equally sized')
    n = len(classes)
    matrix = [[0 for _ in classes] for _ in classes]
    for actual, pred in zip(true, predicted, strict=True):
        if not (isinstance(actual, int) and isinstance(pred, int) and 0 <= actual < n and 0 <= pred < n):
            raise ValueError('Prediction index outside configured classes')
        matrix[actual][pred] += 1
    per_class = {}
    for i, name in enumerate(classes):
        tp = matrix[i][i]
        support = sum(matrix[i])
        positive = sum(row[i] for row in matrix)
        per_class[name] = {'precision': tp/positive if positive else 0., 'recall': tp/support if support else 0.,
                           'f1': 2*tp/(support+positive) if support+positive else 0., 'support': support}
    return {'macro_f1': mean(v['f1'] for v in per_class.values()),
            'accuracy': sum(matrix[i][i] for i in range(n))/len(true),
            'per_class': per_class, 'confusion_matrix': matrix, 'class_names': classes, 'n_test': len(true)}


def aggregate_runs(values: list[float]) -> dict:
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError('Expected nonempty finite run metrics')
    return {'n': len(values), 'mean': mean(values), 'sample_sd': stdev(values) if len(values) > 1 else None}
