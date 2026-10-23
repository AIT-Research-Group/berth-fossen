"""Read-only analysis of archived load forecasts; no threshold fitting on test."""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np

RUN = Path(__file__).resolve().parents[1] / 'outputs/force_training_v1'
MODELS = ('persistence', 'lstm_point', 'energy_lstm', 'gan_lstm')


def confusion(truth, predicted):
    truth, predicted = np.asarray(truth, bool), np.asarray(predicted, bool)
    tn = int(np.sum(~truth & ~predicted)); fp = int(np.sum(~truth & predicted))
    fn = int(np.sum(truth & ~predicted)); tp = int(np.sum(truth & predicted))
    ratio = lambda a, b: a / b if b else None
    return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'total': len(truth),
            'precision': ratio(tp, tp+fp), 'recall': ratio(tp, tp+fn),
            'specificity': ratio(tn, tn+fp), 'f1': ratio(2*tp, 2*tp+fp+fn),
            'accuracy': ratio(tp+tn, len(truth)), 'positives': tp+fn}


@lru_cache(maxsize=4)
def archive(name):
    with np.load(RUN / 'evaluation' / f'{name}_test.npz', allow_pickle=False) as saved:
        return {k: saved[k] for k in saved.files}


def events(samples, target, thresholds, horizon, offset=0):
    """A positive event is ANY exceedance within the next horizon minutes."""
    exceeded = samples[:, :, :horizon, offset:offset+3] / 1000 > thresholds
    actual = (target[:, :horizon, offset:offset+3] / 1000 > thresholds).any(axis=1)
    probability = exceeded.any(axis=2).mean(axis=1)
    any_probability = exceeded.any(axis=(2, 3)).mean(axis=1)
    return np.c_[actual, actual.any(axis=1)], np.c_[probability, any_probability]


def analysis(query):
    get = lambda k, default: query.get(k, [str(default)])[0]
    thresholds = np.array([float(get('x', 100)), float(get('y', 200)), float(get('n', 5000))])
    cutoff = float(get('p', .5)); horizon = int(get('h', 30))
    stat = get('stat', 'rms'); model = get('model', 'gan_lstm'); case = int(get('case', 0))
    if not np.isfinite(thresholds).all() or np.any(thresholds <= 0):
        raise ValueError('Thresholds must be finite and positive')
    if not 0 < cutoff <= 1 or horizon not in (5, 10, 30) or stat not in ('rms', 'absmax') or model not in MODELS:
        raise ValueError('Invalid analysis parameters')
    chosen = archive(model)
    if not 0 <= case < len(chosen['target']):
        raise ValueError('Unknown forecast case')
    offset = 0 if stat == 'rms' else 3
    compare, matrices, selected_p, selected_y = [], [], None, None
    for name in MODELS:
        saved = archive(name)
        if not np.array_equal(saved['scenario'], chosen['scenario']) or not np.array_equal(saved['origin_s'], chosen['origin_s']) or not np.array_equal(saved['target'], chosen['target']):
            raise ValueError('Model archives do not share identical evaluation cases')
        actual, probability = events(saved['ensemble'], saved['target'], thresholds, horizon, offset)
        all_matrices = [confusion(actual[:, j], probability[:, j] >= cutoff) for j in range(4)]
        compare.append({'model': name, **all_matrices[3]})
        if name == model:
            matrices, selected_p, selected_y = all_matrices, probability, actual
    with np.load(RUN / 'dataset/test.npz', allow_pickle=False) as dataset:
        match = np.flatnonzero((dataset['scenario'] == chosen['scenario'][case]) & (dataset['origin_s'] == chosen['origin_s'][case]))
        if len(match) != 1:
            raise ValueError('History cannot be linked unambiguously')
        history = dataset['x'][match[0]] / 1000
    samples = chosen['ensemble'][case] / 1000
    if model in ('lstm_point', 'persistence'):
        # The old evaluator may store repeated copies of a deterministic output.
        samples = samples[:1]
    config = json.loads((RUN / 'experiment.json').read_text(encoding='utf-8'))['config']
    cases = [{'index': i, 'scenario': int(s), 'origin_s': float(t), 'probability': selected_p[i].tolist(), 'actual': selected_y[i].tolist(),
              'cells': [('tp' if y else 'fp') if p >= cutoff else ('fn' if y else 'tn') for y, p in zip(selected_y[i], selected_p[i])]} for i, (s, t) in enumerate(zip(chosen['scenario'], chosen['origin_s']))]
    return {'run': 'outputs/force_training_v1', 'config': config, 'model': model, 'matrices': matrices, 'comparison': compare,
            'scenarios': len(np.unique(chosen['scenario'])), 'cutoff': cutoff, 'thresholds': thresholds.tolist(), 'horizon': horizon,
            'cases': cases, 'case': case, 'stat': stat, 'sample_count': samples.shape[0],
            'example': {'history': history.tolist(), 'target': (chosen['target'][case]/1000).tolist(),
                        'samples': samples[:12].tolist(), 'mean': samples.mean(axis=0).tolist(),
                        'lower': np.quantile(samples, .05, axis=0).tolist(), 'upper': np.quantile(samples, .95, axis=0).tolist()}}
