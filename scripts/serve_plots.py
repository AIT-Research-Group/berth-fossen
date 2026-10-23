"""Local Three.js viewer for saved load CSVs. No simulation/training is run."""
import argparse
import csv
import json
from functools import partial, lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))


@lru_cache(maxsize=8)
def forecasts(path, scenario):
    with np.load(path, allow_pickle=False) as archive:
        ids = np.flatnonzero(archive['scenario'] == scenario)
        if not len(ids):
            return None
        samples = archive['ensemble'][ids] / 1000
        if Path(path).name == 'lstm_point_test.npz':
            samples = samples[:, :1]
        return {'origins': archive['origin_s'][ids].tolist(),
                'mean': samples.mean(1).tolist(),
                'lower': np.quantile(samples, .05, axis=1).tolist(),
                'upper': np.quantile(samples, .95, axis=1).tolist(),
                'target': (archive['target'][ids] / 1000).tolist(),
                'peak_samples': samples.max(axis=2).tolist(),
                'samples': int(samples.shape[1]), 'step_s': 60}


def context(path, model):
    """Link scenes and archived predictions to the same dataset only."""
    result = {'forecast': None, 'scene': None}
    if path.parent.name == 'dataset':
        run = path.parent.parent
        manifest = json.loads((path.parent / 'manifest.json').read_text(encoding='utf-8'))
        sid = int(path.stem.split('_')[-1])
        entry = next(s for s in manifest['scenarios'] if s['id'] == sid)
        config = entry['config']
        result['split'] = entry['split']
        result['load_mode'] = manifest.get('load_mode', '')
        prediction = run / 'evaluation' / f'{model}_test.npz'
        if prediction.exists():
            result['forecast'] = forecasts(str(prediction), sid)
        result['forecast_note'] = 'Dự đoán đã lưu trên tập test, không chạy inference mới.'
        result['scene_mode'] = 'fixed'
    else:
        run = path.parent
        config = json.loads((run / 'metadata.json').read_text(encoding='utf-8'))['config']
        result['forecast_note'] = 'Không có dự đoán cùng kịch bản này. Chọn scenario test 048–055 để xem dự đoán đã lưu.'
        result['scene_mode'] = 'fixed'
        state_file = run / 'vessel.npz'
        if state_file.exists():
            with np.load(state_file, allow_pickle=False) as saved:
                # Keep every sample: interpolation must not alias wave-frequency motion.
                result['motion'] = {'time': saved['time_s'].tolist(), 'state': saved['state'].tolist()}
                for source in ('wind', 'wave'):
                    key = f'{source}_force_body'
                    if key in saved:
                        result['motion'][source] = saved[key].tolist()
                result['scene_mode'] = 'replay'
    from berth_sim.waves import WaveField
    rng = np.random.default_rng(np.random.SeedSequence(config['seed']).spawn(3)[1])
    field = WaveField.create(config['waves'], rng)
    # The viewer draws the probe line normal to the wall, so it needs k*cos(theta).
    result['scene'] = {'omega': field.omega.tolist(), 'k': field.kx.tolist(),
                       'direction': field.direction.tolist(),
                       'amplitude': field.amplitude.tolist(), 'phase': field.phase.tolist(),
                       'reflection': field.reflection.tolist(), 'reflection_phase': field.reflection_phase.tolist(),
                       'storm': config['storm'], 'initial_state': config['vessel']['initial_state'],
                       'length_m': config['vessel']['wind_load']['length_m'],
                       'hs_m': config['waves']['hs_m'], 'tp_s': config['waves']['tp_s']}
    result['model'] = model
    return result


def catalog():
    paths = sorted((ROOT / 'outputs').glob('*/dataset/scenario_*.csv'))
    paths += sorted((ROOT / 'outputs').glob('*/load_features.csv'))
    selected = {}
    for path in paths:
        with path.open(encoding='utf-8-sig', newline='') as handle:
            header = next(csv.reader(handle), [])
        if 'environment_X_rms_N' in header:
            selected[str(path.relative_to(ROOT)).replace('\\', '/')] = path
    return selected


def read_loads(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if 'time_s' not in columns or 'environment_X_rms_N' not in columns:
            raise ValueError('File does not contain load statistics')
        return {'columns': columns, 'rows': [[float(r[c]) for c in columns] for r in reader]}


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path.startswith('/api/'):
            try:
                files = catalog()
                if url.path == '/api/catalog':
                    result = list(files)
                elif url.path == '/api/analysis':
                    from scripts.plot_analysis import analysis
                    result = analysis(parse_qs(url.query))
                elif url.path == '/api/data':
                    key = parse_qs(url.query).get('file', [''])[0]
                    if key not in files:
                        self.send_error(404, 'Unknown dataset')
                        return
                    result = read_loads(files[key])
                    result['name'] = key
                elif url.path == '/api/context':
                    query = parse_qs(url.query)
                    key = query.get('file', [''])[0]
                    model = query.get('model', ['gan_lstm'])[0]
                    if key not in files or model not in ('gan_lstm', 'energy_lstm', 'lstm_point'):
                        self.send_error(404)
                        return
                    result = context(files[key], model)
                else:
                    self.send_error(404)
                    return
                payload = json.dumps(result, allow_nan=False).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (ValueError, OSError, KeyError, StopIteration) as exc:
                self.send_error(422, str(exc))
            return
        super().do_GET()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, directory=str(ROOT / 'web')))
    print(f'Load viewer: http://127.0.0.1:{args.port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
