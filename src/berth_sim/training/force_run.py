"""Train load forecasters on fixed-reference synthetic vessel loads."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from ..simulation import simulate, wave_field
from ..loads import load_features
from ..vessel import first_order_force
from ..wind import wind_load
from .data import SPLITS, scenario_plan, map_scenarios, data_workers
from .force_model import TARGETS, fit_force_scaler
from .train import train_model, load_model, predict, configure_device
from .evaluation import crps, cluster_interval, write_csv


_HYDRO = {}


def _hydro(reference):
    # Converted MSS tables (z-up BODY, exp(-i omega t)); scipy is optional.
    if reference not in _HYDRO:
        from ..hydrodynamics import TankerHydrodynamics
        _HYDRO[reference] = TankerHydrodynamics(Path(reference).parent)
    return _HYDRO[reference]


def scenario_loads(job):
    """One-minute load statistics for a fixed ship; runs in a worker process."""
    scenario, reference = job
    raw, truth, _, _ = simulate(scenario, features=False)
    field = wave_field(scenario)
    vessel = scenario['vessel']
    shape = (len(field.omega), 3)
    if reference:
        incident, reflected = _hydro(reference).berth_force_raos(field, float(vessel['initial_state'][2]))
    else:
        incident = np.broadcast_to(vessel['demo_incident_force_rao'], shape)
        reflected = np.broadcast_to(vessel['demo_reflected_force_rao'], shape)
    wave = first_order_force(raw['time_s'], field, incident, reflected,
        vessel['initial_state'][0], truth['wave_envelope'])
    wind = np.array([wind_load(w, vessel['initial_state'][2], np.zeros(3), vessel['wind_load']) for w in raw['wind_global_mps']])
    window = round(60/scenario['dt_s'])
    zero_axes = {axis for axis, column in zip('XYN', wave.T) if not np.any(column)}
    return load_features(raw['time_s'], wind, wave, window, window), zero_axes


def generate(config, output):
    dataset = output/'dataset'
    dataset.mkdir(parents=True, exist_ok=False)
    base = json.loads(Path(config['simulation_config']).read_text(encoding='utf-8-sig'))
    reference = config.get('hydro_reference')
    hydro = _hydro(reference) if reference else None
    plan = scenario_plan(base, config)
    simulated = map_scenarios(scenario_loads, [(scenario, reference) for _, scenario in plan], data_workers(config))
    manifest = {'task': 'loads', 'targets': TARGETS, 'synthetic_only': True,
        'load_mode': 'fixed vessel reference, zero velocity, no dynamics integration',
        'historical_input': 'simulated load RMS and maxima; requires load measurements or estimates in deployment',
        'wave_force_model': ('MSS tanker open-water force RAO, incident and specularly reflected'
                             if hydro is not None else 'constant demo force RAO from simulation config'),
        'wave_zero_axes': None, 'scenarios': [], 'splits': {},
        'hydro_reference': hydro.provenance() if hydro is not None else None}
    zero_axes = None
    splits = {}
    scenario_id = 0
    for split in SPLITS:
        xs, ys, ids, origins, unique = [], [], [], [], []
        for _ in range(config['scenario_counts'][split]):
            scenario = plan[scenario_id][1]
            rows, axes = next(simulated)
            vessel = scenario['vessel']
            zero_axes = axes if zero_axes is None else zero_axes & axes
            values = np.array([[row[key] for key in TARGETS] for row in rows], dtype=np.float32)
            write_csv(dataset/f'scenario_{scenario_id:03d}.csv', rows)
            unique.append(values)
            history, horizon = config['history_steps'], config['horizon_steps']
            stride = config['train_stride'] if split=='train' else config['eval_stride']
            count = 0
            for i in range(history-1, len(values)-horizon, stride):
                xs.append(values[i-history+1:i+1]); ys.append(values[i+1:i+horizon+1])
                ids.append(scenario_id); origins.append(rows[i]['time_s']); count += 1
            manifest['scenarios'].append({'id': scenario_id, 'split': split, 'config': scenario, 'windows': count,
                'effective_vessel_state': [*vessel['initial_state'][:3], 0., 0., 0.]})
            print(f'loads {split} scenario={scenario_id} windows={count}', flush=True)
            scenario_id += 1
        data = {'x': np.asarray(xs), 'y': np.asarray(ys), 'scenario': np.asarray(ids),
                'origin_s': np.asarray(origins), 'unique_features': np.concatenate(unique)}
        np.savez_compressed(dataset/f'{split}.npz', **data)
        splits[split] = data
        manifest['splits'][split] = {'scenarios': config['scenario_counts'][split], 'windows': len(xs)}
    manifest['wave_zero_axes'] = sorted(zero_axes)
    (dataset/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return splits, manifest


def evaluate_loads(config, splits, output):
    test = splits['test']
    destination = output/'evaluation'
    destination.mkdir()
    metrics = []
    examples = {'history': test['x'][:1], 'target': test['y'][:1]}
    for name in ('persistence', 'lstm_point', 'energy_lstm', 'gan_lstm'):
        if name=='persistence':
            ensemble = np.repeat(test['x'][:, None, -1:], config['horizon_steps'], axis=2)
        else:
            model, _ = load_model(output/name/'best.pt', config['device'])
            ensemble = predict(model, test['x'], config['ensemble_test'], config['batch_size'], config['seed']+2000)
        assert np.isfinite(ensemble).all()
        assert np.all(ensemble[..., 3:] >= ensemble[..., :3])
        examples[name] = ensemble[:1]
        np.savez_compressed(destination/f'{name}_test.npz', ensemble=ensemble, target=test['y'], scenario=test['scenario'], origin_s=test['origin_s'])
        mean = ensemble.mean(1)
        probabilistic = ensemble.shape[1]>1
        if probabilistic:
            score = crps(ensemble, test['y'])
            lower, upper = np.quantile(ensemble, [.05,.95], axis=1)
        for horizon in (5,10,30):
            for j, label in enumerate(TARGETS):
                error = mean[:, horizon-1,j]-test['y'][:,horizon-1,j]
                interval = cluster_interval(error**2, test['scenario'], config['bootstrap_replicates'], square_root=True)
                row = {'model':name,'horizon_min':horizon,'variable':label,
                    'unit':'Nm' if label.endswith('_Nm') else 'N',
                    'rmse':float(np.sqrt(np.mean(error**2))), 'mae':float(np.mean(abs(error))),
                    'rmse_ci_low':float(interval[0]), 'rmse_ci_high':float(interval[1]), 'crps':'','coverage_90':''}
                if probabilistic:
                    row.update(crps=float(score[:,horizon-1,j].mean()), coverage_90=float(((test['y'][:,horizon-1,j]>=lower[:,horizon-1,j])&(test['y'][:,horizon-1,j]<=upper[:,horizon-1,j])).mean()))
                metrics.append(row)
        print(f'evaluated loads {name}', flush=True)
    write_csv(destination/'load_metrics.csv', metrics)
    np.savez_compressed(destination/'examples.npz', **examples)
    return metrics


def wave_model_note(manifest):
    """Describe the wave-load source from the generated manifest, not a fixed claim."""
    if manifest.get('hydro_reference'):
        note = (f"Lực sóng bậc nhất dùng RAO lực của tàu {manifest['hydro_reference']['name']} (MSS/WAMIT, vận tốc 0, "
                "nước hở) cho cả sóng tới và sóng phản xạ gương; chưa xét ảnh hưởng của tường lên RAO hay cộng hưởng khe.")
    else:
        note = 'Lực sóng dùng RAO hằng số minh họa trong cấu hình.'
    zero = manifest.get('wave_zero_axes') or []
    if zero:
        note += f" Tải sóng theo trục {', '.join(zero)} bằng 0, nên các thành phần đó chỉ phản ánh tải gió."
    return note


def make_report(output, config, manifest, metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(output/'evaluation/examples.npz') as data:
        figure, axes = plt.subplots(3,1,figsize=(11,9),constrained_layout=True)
        for j, ax in enumerate(axes):
            ax.plot(np.arange(-59,1),data['history'][0,:,j]/1000,color='black',label='History')
            ax.plot(np.arange(1,31),data['target'][0,:,j]/1000,'k--',label='Future target')
            for name in ('energy_lstm','gan_lstm'):
                samples=data[name][0,:,:,j]/1000
                low,high=np.quantile(samples,[.05,.95],axis=0)
                ax.plot(np.arange(1,31),samples.mean(0),label=name)
                ax.fill_between(np.arange(1,31),low,high,alpha=.15)
            ax.set(title=TARGETS[j],xlabel='Minutes from forecast',ylabel='kNm' if j==2 else 'kN')
            ax.legend();ax.grid(alpha=.2)
        figure.suptitle('Synthetic fixed-reference vessel loads: RMS forecasts')
        figure.savefig(output/'evaluation/load_forecast.png',dpi=160);plt.close(figure)
    figure,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for j,name in enumerate(('lstm_point','energy_lstm','gan_lstm')):
        history=list(csv.DictReader((output/name/'history.csv').open()))
        for ax,key in ((axes[0,j],'train_loss'),(axes[1,j],'validation_score')):
            ax.plot([int(r['epoch']) for r in history],[float(r[key]) for r in history])
            ax.set(title=name,xlabel='Epoch',ylabel=key);ax.grid(alpha=.2)
    figure.savefig(output/'evaluation/load_loss.png',dpi=160);plt.close(figure)
    lines=['# Huấn luyện dự báo lực và mômen','',
        'Dự báo RMS và cực đại trị tuyệt đối X, Y, N của tổng tải gió + sóng trong mỗi phút. Lịch sử 60 phút, chân trời 30 phút. Đầu vào là lịch sử tải mô phỏng, không phải chỉ gió và Hm0.', '',
        'Tàu được giữ cố định tại vị trí/hướng tham chiếu, vận tốc bằng 0. Không tích phân động lực học trong bộ dữ liệu này. ' + wave_model_note(manifest) + ' Hệ số gió là minh họa. Không có nhãn đo thực.', '',
        f"Mỗi mô hình chạy {config['epochs']} epoch; checkpoint được chọn theo validation. Tập calibration được giữ riêng, chưa dùng do chưa có ngưỡng tải được xác định. Không xuất xác suất an toàn.", '',
        '| Tập | Kịch bản | Cửa sổ |','|---|---:|---:|']
    for split, info in manifest['splits'].items():lines.append(f"| {split} | {info['scenarios']} | {info['windows']} |")
    lines += ['', '| Mô hình | Đại lượng | RMSE +30 phút (kN hoặc kNm) | Coverage 90% |','|---|---|---:|---:|']
    for row in metrics:
        if row['horizon_min']==30:
            coverage=f"{100*row['coverage_90']:.1f}%" if row['coverage_90']!='' else '—'
            lines.append(f"| {row['model']} | {row['variable']} | {row['rmse']/1000:.4f} | {coverage} |")
    lines += ['', 'RMS và cực đại không chứa dấu lực; chưa dự báo lực tức thời hoặc hướng tác động. Cực đại được ràng buộc >= RMS >= 0. Chuẩn hóa fit trên train; chia theo kịch bản, bootstrap theo kịch bản. Một seed, cùng họ mô phỏng; không suy diễn kết quả sang vận hành cảng.', '',
        'Để sử dụng thực tế phải có lịch sử tải đo hoặc ước lượng cùng định nghĩa và phải hiệu chỉnh mô hình tải. Không suy ra lực tức thời duy nhất từ Hm0.', '',
        '![Loss](evaluation/load_loss.png)', '', '![Dự báo tải](evaluation/load_forecast.png)']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=Path('configs/force_training.json'))
    parser.add_argument('--output',type=Path,default=Path('outputs/force_training_v1'))
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8-sig'))
    if config.get('task')!='loads' or config['history_steps']!=60 or config['horizon_steps']!=30:
        raise ValueError('This runner requires task=loads, history=60, horizon=30')
    args.output.mkdir(parents=True,exist_ok=False)
    configure_device(config)
    source=Path(__file__).resolve().parents[1]
    shutil.copytree(source,args.output/'source'/'berth_sim',ignore=shutil.ignore_patterns('__pycache__'))
    (args.output/'experiment.json').write_text(json.dumps({'config':config,'torch':str(torch.__version__),'numpy':np.__version__,'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(0) if config['device'].startswith('cuda') else None,
        'source_sha256':{str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*.py')}},indent=2),encoding='utf-8')
    splits,manifest=generate(config,args.output)
    scaler=fit_force_scaler(splits['train']['unique_features'])
    (args.output/'scaler.json').write_text(json.dumps(scaler,indent=2),encoding='utf-8')
    for name in ('lstm_point','energy_lstm','gan_lstm'):
        train_model(name,config,scaler,splits['train'],splits['validation'],args.output)
    metrics=evaluate_loads(config,splits,args.output)
    make_report(args.output,config,manifest,metrics)
    print(json.dumps({'report':str(args.output/'report.md')}),flush=True)


if __name__=='__main__':main()
