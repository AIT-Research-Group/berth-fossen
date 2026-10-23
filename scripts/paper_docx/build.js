// Results document (bilingual): English figures/tables/captions, Vietnamese commentary.
// v3: 184-scenario dataset, base vs upgraded training over three seeds; detailed figures from
// the median-seed upgraded run. Numbers in the commentary are read from the result files.
const fs = require('fs');
const path = require('path');
const {Document, Packer, Paragraph, AlignmentType, Header, Footer, PageNumber, TextRun} = require('docx');
const {FONT, run, para, heading, note, caption, tableCaption, table, figure, fmt} = require('./lib');

const [results, outFile] = process.argv.slice(2);
const T = JSON.parse(fs.readFileSync(path.join(results, 'tables.json'), 'utf8'));
const T3 = JSON.parse(fs.readFileSync(path.join(results, 'figures_3d.json'), 'utf8'));
const A = T.ablation;
const img = name => path.join(results, name);
const NAMES = {persistence: 'Persistence', lstm_point: 'LSTM (point)', energy_lstm: 'Energy-LSTM', gan_lstm: 'GAN-LSTM'};
const MODELS = Object.keys(NAMES);
const PROB = ['energy_lstm', 'gan_lstm'];
const vi = (value, digits) => (+value).toFixed(digits).replace('.', ',');
// Relative reduction, used only in the Vietnamese commentary (decimal comma).
const pct = (a, b) => `${vi(100 * (1 - a / b), 1)}%`;
let tableNo = 0, figureNo = 0;
const nextTable = () => `Table ${++tableNo}.`;
const nextFigure = () => `Figure ${++figureNo}.`;
const B = T.base_config, E = T.experiment, W = B.waves;
const id3 = n => String(n).padStart(3, '0');

// ---------- lookups ----------
const metric = (model, h, variable) => T.forecast_metrics.find(r => r.model === model && +r.horizon_min === h && r.variable === variable);
const load = (model, variable) => T.load_metrics.find(r => r.model === model && +r.horizon_min === 30 && r.variable === variable);
const warn = (model, event) => T.warning_metrics.find(r => r.model === model && +r.horizon_min === 30 && r.event === event);
const cf = (model, event) => T.confusion.find(r => r.model === model && r.event === event);
const ab = (variant, key) => A[variant].metrics[key];
const ms = (variant, key, digits) => `${fmt(ab(variant, key).mean, digits)} ± ${fmt(ab(variant, key).std, digits)}`;
const tests = T.scenarios.filter(s => s.split === 'test');

// ---------- tables ----------
const setupRows = [
  ['Samples per scenario', `${E.scenario_duration_s / 3600} h at Δt = ${E.simulation_dt_s} s`, 'Independent scenarios, split before windowing'],
  ['Incident spectrum', `JONSWAP (DNV-RP-C205), ${E.wave_components} components`, 'Frequencies jittered within bins; no renormalisation to band'],
  ['Frequency band', `${W.f_min_hz}–${W.f_max_hz} Hz`, 'Retains > 97% of variance for 3.5 s ≤ T~p~ ≤ 7.5 s'],
  ['Water depth', `${W.depth_m} m`, 'Finite-depth linear dispersion'],
  ['Wall and reflection', 'Vertical wall at x = 0, constant K~r~, phase 0', 'Linear, specular; no gap resonance'],
  ['Wave probes', W.probe_x_m.map(x => `${x}`).join(', ') + ' m', 'Least-squares separation, cond. number ≤ ' + W.max_condition],
  ['Spectral estimate', `${B.feature_window_s} s window, Welch ${W.welch_segment_s} s, 50% overlap`, 'Causal (window ends at issue time)'],
  ['Feature step', `${B.feature_step_s} s`, 'Gust = max 3-s mean speed in the last minute'],
  ['Wind turbulence', 'Kaimal, FFT synthesis on record grid, L = 80 m', 'Gaussian, non-repeating (TurbSim-style)'],
  ['Weather episode', 'Prescribed Gaussian pulse on wind and wave amplitude', 'Scenario design, not coupled wind–wave physics'],
  ['Vessel (load task)', `${T.load_hydro ? 'MSS tanker force RAO (WAMIT)' : 'Demo RAO'}, fixed at x = −250 m, heading 90°`, 'Incident + reflected, open-water RAO, zero speed'],
  ['Forecast set-up', `${E.history_steps} min history → ${E.horizon_steps} min ahead`, `Train stride ${E.train_stride} min, evaluation stride ${E.eval_stride} min`],
];

const rangeRows = [
  ['Design H~s~', 'U(0.25, 0.95) m'], ['Peak period T~p~', 'U(3.5, 7.5) s'], ['JONSWAP γ', 'U(1, 5)'],
  ['Reflection coefficient K~r~', 'U(0.3, 0.9)'], ['Probe noise σ', 'U(2, 15) mm'], ['Mean wind speed U', 'U(4, 11) m/s'],
  ['Turbulence σ~u~', 'U(0.4, 1.5) m/s'], ['Wind direction (toward)', 'U(−180°, 180°)'],
  ['Storm centre', 'U(0.3, 0.7) × duration'], ['Storm width', 'U(15, 40) min'],
  ['Storm gain (wind, wave)', 'U(0.1, 1.1) each'], ['Wave lag behind wind', 'U(0, 20) min'],
];
const splitRows = ['train', 'validation', 'calibration', 'test'].map(s => [
  s, T.splits[s].scenarios, T.splits[s].windows, T.load_splits[s].scenarios, T.load_splits[s].windows]);

const testRows = tests.map(s => [id3(s.id), fmt(s.hs, 2), fmt(s.tp, 2), fmt(s.gamma, 2), fmt(s.kr, 2), fmt(s.u, 1),
  fmt(s.sigma_u, 2), fmt(s.direction_deg, 0), fmt(s.wind_gain, 2), fmt(s.wave_gain, 2), fmt(s.wave_lag_min, 1)]);
const sepAll = T.separation;
const sepRows = sepAll.map(s => [id3(s.id), `${s.valid}/${s.total}`, fmt(s.bias_pct, 2), fmt(s.std_pct, 2), fmt(s.rmse_m * 100, 2)]);
const worst = sepAll.reduce((a, b) => (Math.abs(b.bias_pct) > Math.abs(a.bias_pct) ? b : a));
const worstCfg = tests.find(s => s.id === worst.id);
const others = sepAll.filter(s => s.id !== worst.id);
const median = values => { const v = [...values].sort((a, b) => a - b); return v[Math.floor(v.length / 2)]; };

// Upgrade settings, read from the two configurations actually used.
const upgradeRows = [
  ['Energy score estimator', 'Plain (M^2^ pairs incl. self-pairs)', 'Fair, M(M−1) pairs (Ferro 2014)'],
  ['Critic loss', 'Non-saturating BCE', 'Hinge'],
  ['Critic input', '1 ensemble member, flattened future', '4 members; 1-min and 5-min convolutions'],
  ['Critic regularisation', 'Gradient clipping', 'Spectral normalisation + gradient clipping'],
  ['Critic learning rate', '= generator', '2 × generator (TTUR)'],
  ['Generator noise', 'One global draw per member', '+ 4 fresh noise channels per lead time'],
  ['Dropout / weight decay', '0 / 0', '0.2 / 10^−4^ (AdamW)'],
  ['Stopping', '100 epochs', 'Early stopping, patience 15 (max 150)'],
  ['Interval calibration', 'CQR conformal (both)', 'CQR conformal (both)'],
];

const abRows = [
  ['RMSE H~m0,R~, 30 min (m)', 'hs/30/rmse', 4, 'persistence/hs/30/rmse'],
  ['RMSE gust, 30 min (m/s)', 'gust/30/rmse', 3, 'persistence/gust/30/rmse'],
  ['CRPS H~m0,R~, 30 min (m)', 'hs/30/crps', 4, null],
  ['CRPS gust, 30 min (m/s)', 'gust/30/crps', 3, null],
  ['Trajectory energy score', 'joint_es', 4, null],
  ['Brier, H~m0,R~ event', 'brier_wave', 4, null],
  ['F1, H~m0,R~ event', 'f1_wave', 3, 'persistence/f1_wave'],
  ['POD, H~m0,R~ event', 'pod_wave', 3, 'persistence/pod_wave'],
  ['False-alarm ratio, H~m0,R~ event', 'far_wave', 3, 'persistence/far_wave'],
  ['Coverage 90%, raw (%)', 'hs/30/coverage_90', 1, null, 100],
  ['Coverage 90%, conformal (%)', 'hs/30/coverage_90_conformal', 1, null, 100],
  ['Epochs run', 'epochs', 0, null],
  ['Training time (s)', 'train_s', 0, null],
].map(([label, key, digits, reference, scale = 1]) => {
  const cell = (variant, model) => {
    const m = ab(variant, `${model}/${key}`);
    return `${fmt(m.mean * scale, digits)} ± ${fmt(m.std * scale, digits)}`;
  };
  return [label, cell('base', 'energy_lstm'), cell('upgraded', 'energy_lstm'), cell('base', 'gan_lstm'),
          cell('upgraded', 'gan_lstm'), reference ? fmt(ab('base', reference).mean, digits) : '—'];
});

const trainRows = [];
for (const [task, label] of [['training_env', 'Environment (v3, upgraded)'], ['training_loads', 'Loads (v2)']]) {
  for (const model of MODELS.slice(1)) {
    const s = T[task][model];
    trainRows.push([label, NAMES[model], s.parameter_count.toLocaleString('en-US'), s.epochs_run, s.best_epoch,
                    fmt(s.best_validation_score, 4), fmt(s.duration_s, 0)]);
  }
}

function metricRows(variable, digits) {
  const rows = [];
  for (const h of [5, 10, 30]) {
    for (const model of MODELS) {
      const r = metric(model, h, variable);
      const p = x => (r[x] === '' || r[x] === undefined ? '—' : `${(100 * r[x]).toFixed(1)}`);
      rows.push([h, NAMES[model], `${fmt(r.rmse, digits)} [${fmt(r.rmse_ci_low, digits)}, ${fmt(r.rmse_ci_high, digits)}]`,
                 fmt(r.crps, digits), p('coverage_90'), p('coverage_90_conformal'), fmt(r.width_90, digits),
                 fmt(r.width_90_conformal, digits)]);
    }
  }
  return rows;
}
const joint = Object.fromEntries(T.joint_metrics.map(r => [r.model, +r.normalized_trajectory_energy_score]));

const events = {wind: `Gust > ${E.demo_thresholds.gust_mps} m/s`, wave: `H~m0,R~ > ${E.demo_thresholds.reflected_hs_m} m`,
                simultaneous: 'Both in the same minute', any: 'Either'};
const warnRows = Object.keys(events).flatMap(event => PROB.map(m => {
  const r = warn(m, event);
  return [events[event], NAMES[m], fmt(r.brier_raw, 4), fmt(r.brier_calibrated, 4),
          `[${fmt(r.brier_ci_low, 4)}, ${fmt(r.brier_ci_high, 4)}]`, `${r.positive_origins}/${r.total_origins}`, r.positive_scenarios];
}));
const eventNames = {wind: 'Gust', wave: 'Reflected H~m0~', any: 'Either', simultaneous: 'Both (same minute)'};
const confusionRows = Object.keys(eventNames).flatMap(event => MODELS.map(m => {
  const r = cf(m, event);
  return [eventNames[event], NAMES[m], r.tp, r.fp, r.fn, r.tn, fmt(r.recall, 2), fmt(r.precision, 2), fmt(r.f1, 2),
          fmt(r.specificity, 2)];
}));

const loadTargets = [['environment_X_rms_N', 'RMS F~X~ (kN)'], ['environment_Y_rms_N', 'RMS F~Y~ (kN)'],
                     ['environment_N_rms_Nm', 'RMS M~N~ (kNm)'], ['environment_X_absmax_N', 'Max |F~X~| (kN)'],
                     ['environment_Y_absmax_N', 'Max |F~Y~| (kN)'], ['environment_N_absmax_Nm', 'Max |M~N~| (kNm)']];
const loadRows = loadTargets.map(([v, label]) => [label, ...MODELS.map(m => fmt(+load(m, v).rmse / 1000, 1)),
  ...PROB.map(m => `${(100 * load(m, v).coverage_90).toFixed(1)}`)]);
const loadGain = v => pct(+load('gan_lstm', v).rmse, +load('persistence', v).rmse);

// ---------- key numbers used in commentary ----------
const h30 = m => +metric(m, 30, 'reflected_hs_m').rmse, g30 = m => +metric(m, 30, 'gust_mps').rmse;
const cov = (m, v, key = 'coverage_90') => 100 * +metric(m, 30, v)[key];
const envTrain = T.training_env, loadTrain = T.training_loads;
const abMean = (variant, key) => ab(variant, key).mean;
const change = (key, model) => pct(abMean('upgraded', `${model}/${key}`), abMean('base', `${model}/${key}`));

// ---------- document ----------
const children = [
  new Paragraph({alignment: AlignmentType.CENTER, spacing: {after: 80},
                 children: [run('Results: Probabilistic Forecasting of Wind and Reflected Waves at a Berth', {bold: true, size: 32})]}),
  new Paragraph({alignment: AlignmentType.CENTER, spacing: {after: 80},
                 children: [run('Figures, tables and recorded parameters for the results section', {italics: true, size: 24})]}),
  new Paragraph({alignment: AlignmentType.CENTER, spacing: {after: 240},
                 children: [run('Thống kê kết quả huấn luyện, kịch bản thử nghiệm và bảng thông số — bản song ngữ (hình, bảng tiếng Anh; nhận xét tiếng Việt)', {size: 21, color: '1F4E79'})]}),
  para(`Source runs: environmental forecasting v3 — shared dataset outputs/v3_data (${T.splits.train.scenarios}/${T.splits.validation.scenarios}/${T.splits.calibration.scenarios}/${T.splits.test.scenarios} scenarios), base and upgraded training with seeds 1–3 (outputs/v3_base_s*, outputs/v3_upgraded_s*); detailed figures from the median-seed upgraded run ${T.env_run}. Load forecasting: outputs/${T.load_run}. Hardware: ${T.gpu}, CUDA ${T.cuda}, PyTorch ${T.env_versions.torch}. All data are synthetic.`, {size: 20}),
  note('Lưu ý quan trọng',
       `Bản này thay thế hoàn toàn số liệu v2. So với v2: (1) bộ dữ liệu lớn gấp ~3,3 lần (${T.splits.test.scenarios} kịch bản test thay vì 8); (2) mô hình được nâng cấp (Bảng 5) và so sánh base ↔ nâng cấp trên cùng dữ liệu với 3 seed huấn luyện; (3) khoảng dự báo được hiệu chỉnh conformal. Các hình chi tiết lấy từ lần chạy ${T.env_run} — seed có energy score GAN-LSTM ở giữa trong 3 seed (không chọn lần tốt nhất). Phần dự báo tải vẫn là v2 (kiến trúc cũ, 1 seed) vì chưa huấn luyện lại. Dữ liệu là mô phỏng; ngưỡng 13 m/s và 0,7 m là ngưỡng thử nghiệm.`),

  heading('1. Experimental set-up', 1),
  tableCaption(nextTable(), 'Simulation, measurement and forecasting configuration.'),
  table(['Item', 'Setting', 'Remark'], setupRows, [22, 38, 40],
        {align: [AlignmentType.LEFT, AlignmentType.LEFT, AlignmentType.LEFT]}),
  note('Nhận xét', 'Bảng 1 dùng làm mục "Thiết lập thí nghiệm". Các lựa chọn đều có căn cứ trích dẫn trong README: phổ JONSWAP theo DNV-RP-C205, tách sóng bình phương tối thiểu nhiều đầu đo (Mansard & Funke; Zelt & Skjelbreia), ước lượng phổ Welch, rối gió Kaimal theo cách tổng hợp của TurbSim. Đặc trưng được tính nhân quả: cửa sổ phổ kết thúc tại thời điểm phát hành dự báo.'),
  figure(img('fig00_berth_sea_ship_3d.png')),
  caption(nextFigure(), `Berth layout and simulated sea surface η(x, y, t) at the reflected-wave peak of test scenario ${id3(T3.sea_scenario.split('_').pop())} (t = ${T3.sea_time_min.toFixed(0)} min): incident plus specularly reflected irregular waves in front of a vertical quay wall, wave probes, and the moored tanker drawn at true scale as a Wigley hull with the proportions of the MSS/WAMIT tanker (B/L = 0.187, T/L = 0.041). η is exaggerated ×5 and the scene ×3 vertically; waves shorter than ${T3.sea_min_wavelength_m.toFixed(0)} m are omitted by the plotting grid (${(100 * T3.sea_kept_variance).toFixed(0)}% of variance shown).`),
  note('Nhận xét', 'Hình dùng cho phần mô tả bài toán: tường cầu tại x = 0, ba đầu đo trước tường, tàu dầu neo song song cầu. Sóng tới vuông góc tường nên đỉnh sóng song song thân tàu; bước sóng đỉnh (~20–90 m tùy T~p~) nhỏ hơn nhiều so với chiều dài tàu 350 m. Trường sóng là nước hở: chưa tính nhiễu xạ/bức xạ quanh thân tàu, tàu đứng yên.'),

  tableCaption(nextTable(), 'Scenario sampling distributions (all splits) and dataset sizes.'),
  table(['Parameter', 'Distribution'], rangeRows, [45, 55], {align: [AlignmentType.LEFT, AlignmentType.LEFT]}),
  new Paragraph({spacing: {after: 80}}),
  table(['Split', 'Scenarios (env.)', 'Windows (env.)', 'Scenarios (loads, v2)', 'Windows (loads, v2)'], splitRows, [18, 20, 20, 21, 21]),
  figure(img(T.figures.scenarios)),
  caption(nextFigure(), `Design parameters of the ${T.scenarios.length} synthetic scenarios by split: (a) H~s~–T~p~, (b) K~r~–γ, (c) mean wind–turbulence, (d) storm gains. Test scenarios (diamonds) lie inside the training envelope.`),
  note('Nhận xét', `${T.scenarios.length} kịch bản độc lập được chia theo kịch bản trước khi cắt cửa sổ (${T.splits.train.scenarios}/${T.splits.validation.scenarios}/${T.splits.calibration.scenarios}/${T.splits.test.scenarios}), không có cửa sổ chồng lấn giữa các tập. Kịch bản test nằm trong vùng tham số của tập huấn luyện: kết quả đo khả năng nội suy trong cùng họ mô phỏng, chưa phải ngoại suy. Độ sâu, vị trí đầu đo và hướng sóng (vuông góc tường) giữ cố định.`),

  heading('2. Test scenarios', 1),
  tableCaption(nextTable(), `Parameters of the ${tests.length} held-out test scenarios.`),
  table(['ID', 'H~s~ (m)', 'T~p~ (s)', 'γ', 'K~r~', 'U (m/s)', 'σ~u~ (m/s)', 'Dir. (°)', 'Wind gain', 'Wave gain', 'Lag (min)'],
        testRows, [6, 9, 9, 7, 7, 9, 10, 9, 11, 11, 10]),
  figure(img('fig02_test_spectra_3d.png')),
  caption(nextFigure(), `Reflected-wave spectral density S~R~(f, t) of eight representative test scenarios (${T.selected_tests.map(id3).join(', ')}; chosen evenly over the range of peak reflected H~m0~), estimated causally from three noisy probes (10-min windows, Welch 150-s segments, one estimate per minute over 4 h). Common logarithmic colour scale; MATLAB-style view (−37.5°, 30°).`),
  note('Nhận xét', 'Hình phổ 3D thể hiện trực tiếp bộ dữ liệu đầu vào: mỗi kịch bản là một bề mặt năng lượng sóng phản xạ theo tần số và thời gian. Đỉnh phổ nằm ở f ≈ 1/T~p~, năng lượng tăng rồi giảm theo xung bão, và mức năng lượng khác nhau rõ giữa các kịch bản (do H~s~, K~r~ và biên độ bão). Vùng tần số cao là nền nhiễu đầu đo và rò phổ. 8 kịch bản được chọn đều theo H~m0,R~ đỉnh để đại diện từ trạng thái sóng nhỏ tới lớn; bảng 3 và 4 liệt kê đủ cả 24.'),
  figure(img(T.figures.test_series)),
  caption(nextFigure(), 'Estimated reflected H~m0~ (blue, left axis) and one-minute gust (orange, right axis) in the same eight test scenarios; dotted lines mark the experimental thresholds (0.7 m, 13 m/s).'),
  note('Nhận xét', `Trong ${tests.length} kịch bản test có ${warn('energy_lstm', 'wind').positive_scenarios} kịch bản vượt ngưỡng gió, ${warn('energy_lstm', 'wave').positive_scenarios} kịch bản vượt ngưỡng sóng và ${warn('energy_lstm', 'simultaneous').positive_scenarios} kịch bản vượt đồng thời trong cùng phút — nhiều hơn hẳn v2 (2/4/1), nên các chỉ số cảnh báo ở mục 6 đáng tin cậy hơn. Gió thường đạt đỉnh trước sóng (độ trễ bão 0–20 phút), là tín hiệu báo trước duy nhất mà mô hình có thể khai thác.`),

  tableCaption(nextTable(), 'Accuracy of the three-probe reflected-wave estimate against the clean simulated reflected field (all test scenarios).'),
  table(['Scenario', 'Valid windows', 'Bias (%)', 'Std. of rel. error (%)', 'RMSE (cm)'], sepRows, [18, 18, 18, 26, 20]),
  figure(img(T.figures.separation)),
  caption(nextFigure(), `(a) Reflected H~m0~ estimated from three noisy probes versus the clean simulated value; (b) relative error per test scenario. Welch averaging with ${W.welch_segment_s} s segments.`),
  note('Nhận xét', `Tách sóng tới/phản xạ rất sát giá trị thật trên cả ${sepAll.length} kịch bản test: mọi cửa sổ đạt chất lượng, độ chệch trung vị ${vi(median(sepAll.map(s => s.bias_pct)), 2)}%, độ lệch chuẩn ${vi(Math.min(...sepAll.map(s => s.std_pct)), 1)}–${vi(Math.max(...sepAll.map(s => s.std_pct)), 1)}%. Độ chệch lớn nhất ở kịch bản ${id3(worst.id)} (+${vi(worst.bias_pct, 1)}%, H~s~ = ${vi(worstCfg.hs, 2)} m, T~p~ = ${vi(worstCfg.tp, 1)} s): sóng nhỏ nên nhiễu đầu đo chiếm tỷ trọng năng lượng lớn hơn. Ước lượng H~m0,R~ đủ tin cậy làm đầu vào dự báo, với lưu ý ở trạng thái sóng nhỏ.`),

  heading('3. Model upgrade and training', 1),
  tableCaption(nextTable(), 'Differences between the base and upgraded probabilistic models (identical data, architecture size and horizon).'),
  table(['Component', 'Base', 'Upgraded'], upgradeRows, [26, 34, 40],
        {align: [AlignmentType.LEFT, AlignmentType.LEFT, AlignmentType.LEFT]}),
  note('Nhận xét', 'Nâng cấp nhắm vào ba điểm yếu đã quan sát ở v2: (i) quá khớp sớm; (ii) khoảng dự báo hẹp — ngoài ra ước lượng energy score cũ tính cả cặp tự thân nên đạt cực tiểu khi ensemble hẹp hơn thực tế ~20% (đã kiểm chứng bằng mô phỏng, có kiểm thử); (iii) critic của GAN chỉ nhìn một mẫu. Không thêm đặc trưng vật lý (T~p~, K~r~) vì trong bộ mô phỏng chúng không đổi trong mỗi kịch bản, không mang thông tin về thời điểm bão.'),
  tableCaption(nextTable(), `Training summary (hidden size ${E.hidden_size}, batch ${E.batch_size}, lr ${E.learning_rate}; checkpoint selected on validation only). Environmental models: run ${T.env_run}; load models: run ${T.load_run}.`),
  table(['Task', 'Model', 'Parameters', 'Epochs run', 'Best epoch', 'Best val. score', 'Time (s)'],
        trainRows, [22, 16, 12, 11, 11, 14, 14]),
  figure(img(T.figures.training_env)),
  caption(nextFigure(), `Training loss and validation score per epoch, upgraded environmental models (${T.env_run}); training stops 15 epochs after the best validation score.`),
  note('Nhận xét', `Early stopping dừng LSTM, Energy-LSTM và GAN-LSTM sau ${envTrain.lstm_point.epochs_run}, ${envTrain.energy_lstm.epochs_run} và ${envTrain.gan_lstm.epochs_run} epoch (tốt nhất ở epoch ${envTrain.lstm_point.best_epoch}, ${envTrain.energy_lstm.best_epoch}, ${envTrain.gan_lstm.best_epoch}). Validation vẫn đạt cực tiểu sớm rồi tăng: mô hình còn xu hướng quá khớp dù đã có dropout, weight decay và dữ liệu gấp 4 lần — early stopping là cơ chế bảo vệ chính. Loss hinge của critic ổn định quanh ~0,97, không có dấu hiệu critic áp đảo generator.`),
  figure(img(T.figures.training_loads)),
  caption(nextFigure(), `Training loss and validation score per epoch, load forecasting (${T.load_run}, base architecture, 100 epochs).`),

  heading('4. Ablation: base versus upgraded (three seeds)', 1),
  tableCaption(nextTable(), `Test-set results of base and upgraded models, mean ± standard deviation over training seeds 1–3 on the same data (${T.splits.test.scenarios} test scenarios, ${T.splits.test.windows} issue times). Warnings: 30-min horizon, calibrated probability ≥ 0.5.`),
  table(['Metric', 'Energy base', 'Energy upgr.', 'GAN base', 'GAN upgr.', 'Persist.'], abRows, [26, 15, 15, 15, 15, 14]),
  figure(img(T.figures.ablation)),
  caption(nextFigure(), 'Ablation of the model upgrade: bars show the mean over three training seeds, error bars one standard deviation, dots the individual seeds; dashed lines show persistence.'),
  note('Nhận xét', `Nâng cấp cải thiện các chỉ số xác suất và cảnh báo nhưng mức độ khiêm tốn: với GAN-LSTM, Brier sự kiện sóng giảm ${change('brier_wave', 'gan_lstm')}, F1 tăng từ ${vi(abMean('base', 'gan_lstm/f1_wave'), 3)} lên ${vi(abMean('upgraded', 'gan_lstm/f1_wave'), 3)}, CRPS H~m0,R~ giảm ${change('hs/30/crps', 'gan_lstm')}; với Energy-LSTM, CRPS H~m0,R~ giảm ${change('hs/30/crps', 'energy_lstm')} và energy score quỹ đạo giảm ${change('joint_es', 'energy_lstm')}. Độ chính xác điểm hầu như không đổi (RMSE H~m0,R~ GAN ${vi(abMean('base', 'gan_lstm/hs/30/rmse'), 4)} → ${vi(abMean('upgraded', 'gan_lstm/hs/30/rmse'), 4)} m). Lợi ích chắc chắn nhất là độ ổn định giữa các seed (độ lệch chuẩn RMSE gió giật của GAN giảm từ ${vi(ab('base', 'gan_lstm/gust/30/rmse').std, 2)} xuống ${vi(ab('upgraded', 'gan_lstm/gust/30/rmse').std, 2)} m/s) và thời gian huấn luyện giảm ~${change('train_s', 'gan_lstm')}. Phần lớn khác biệt nằm trong khoảng 1 độ lệch chuẩn: với 3 seed chưa đủ để khẳng định ý nghĩa thống kê.`),
  note('Nhận xét (tiếp)', `Coverage thô của khoảng 90% vẫn thấp (${vi(100 * abMean('upgraded', 'gan_lstm/hs/30/coverage_90'), 1)}% với GAN-LSTM nâng cấp): sửa energy score không đủ khắc phục, nguyên nhân chính nhiều khả năng là quá khớp. Hiệu chỉnh conformal (CQR trên tập calibration) đưa coverage về ${vi(100 * abMean('upgraded', 'gan_lstm/hs/30/coverage_90_conformal'), 1)}% — đây là cách xử lý nên báo cáo trong bài. GAN-LSTM và Energy-LSTM nâng cấp tương đương nhau; thành phần đối kháng không cho lợi thế rõ rệt so với huấn luyện chỉ bằng energy score. Cả hai đều vượt persistence ổn định trên mọi seed (RMSE H~m0,R~ thấp hơn ~${pct(abMean('upgraded', 'energy_lstm/hs/30/rmse'), abMean('base', 'persistence/hs/30/rmse'))}).`),

  heading('5. Environmental forecasting on the test set', 1),
  tableCaption(nextTable(), `Gust forecast skill on the test set (m/s), run ${T.env_run}. RMSE with 95% scenario-bootstrap interval; coverage and width of the central 90% interval before and after conformal calibration.`),
  table(['Lead (min)', 'Model', 'RMSE [95% CI]', 'CRPS', 'Cov. raw (%)', 'Cov. conf. (%)', 'Width raw', 'Width conf.'],
        metricRows('gust_mps', 3), [9, 15, 24, 9, 11, 12, 10, 10], {shade: r => Math.floor(r / 4) % 2 === 1}),
  tableCaption(nextTable(), `Reflected H~m0~ forecast skill on the test set (m), run ${T.env_run}.`),
  table(['Lead (min)', 'Model', 'RMSE [95% CI]', 'CRPS', 'Cov. raw (%)', 'Cov. conf. (%)', 'Width raw', 'Width conf.'],
        metricRows('reflected_hs_m', 4), [9, 15, 24, 9, 11, 12, 10, 10], {shade: r => Math.floor(r / 4) % 2 === 1}),
  figure(img(T.figures.env_skill)),
  caption(nextFigure(), 'Skill versus lead time on the test set: RMSE with 95% scenario-bootstrap bands for (a) gust and (b) reflected H~m0~; CRPS (solid) and raw coverage of the 90% interval (dashed; dotted line = nominal 90%) for (c) gust and (d) reflected H~m0~.'),
  note('Nhận xét', `Ở 30 phút, Energy-LSTM và GAN-LSTM cho RMSE H~m0,R~ ${vi(h30('energy_lstm'), 3)} và ${vi(h30('gan_lstm'), 3)} m so với ${vi(h30('persistence'), 3)} m của persistence (giảm ${pct(h30('energy_lstm'), h30('persistence'))} và ${pct(h30('gan_lstm'), h30('persistence'))}); gió giật ${vi(g30('energy_lstm'), 2)}–${vi(g30('gan_lstm'), 2)} so với ${vi(g30('persistence'), 2)} m/s. Ở 5 phút, persistence gần tương đương vì H~m0,R~ biến đổi chậm. Sau hiệu chỉnh conformal, coverage đạt ${vi(cov('gan_lstm', 'reflected_hs_m', 'coverage_90_conformal'), 1)}% (H~m0,R~) và ${vi(cov('gan_lstm', 'gust_mps', 'coverage_90_conformal'), 1)}% (gió giật) cho GAN-LSTM, đổi lại khoảng rộng hơn. Energy score quỹ đạo: Energy-LSTM ${vi(joint.energy_lstm, 3)}, GAN-LSTM ${vi(joint.gan_lstm, 3)}.`),
  figure(img(T.figures.examples)),
  caption(nextFigure(), `Example forecasts at the most demanding issue time in the test set (scenario ${id3(T.example.scenario)}, ${T.example.origin_min.toFixed(0)} min): 60-min observed history, observed future, ensemble mean and central 90% interval (raw).`),
  note('Nhận xét', 'Trường hợp khó nhất: H~m0,R~ tăng từ ~0,75 lên ~1,2 m trong 30 phút sau khi gió đã tăng mạnh. Khác với v2 (trung bình ensemble đi ngang), cả hai mô hình v3 đã bắt được xu hướng tăng nhờ tín hiệu gió đi trước; GAN-LSTM bám đà tăng sát hơn, quan trắc nằm trong khoảng 90% khoảng 15 phút đầu rồi vượt ra ngoài. Với gió giật, mô hình dự báo đỉnh rồi giảm đúng xu hướng. Đỉnh sóng muộn vẫn bị đánh giá thấp — giới hạn của lịch sử 60 phút khi bão là xung quy định trước.'),

  heading('6. Event warnings (30-min horizon)', 1),
  tableCaption(nextTable(), 'Brier score of 30-min exceedance probabilities before and after logistic calibration (fitted on the calibration split). Positives are counted as issue times and as distinct scenarios.'),
  table(['Event', 'Model', 'Brier raw', 'Brier calibrated', '95% CI (calibrated)', 'Positive issue times', 'Positive scenarios'],
        warnRows, [18, 15, 12, 14, 17, 13, 11], {shade: r => Math.floor(r / 2) % 2 === 1}),
  figure(img(T.figures.reliability)),
  caption(nextFigure(), 'Reliability diagrams of calibrated 30-min exceedance probabilities on the test set (marker area ∝ number of issue times).'),
  note('Nhận xét', `Với ${T.splits.test.scenarios} kịch bản test, mỗi loại sự kiện có hàng trăm mốc dương tính từ 7–20 kịch bản, nên biểu đồ độ tin cậy ổn định hơn v2. Hiệu chỉnh logistic không phải lúc nào cũng có lợi: nó cải thiện sự kiện đồng thời (GAN-LSTM ${vi(warn('gan_lstm', 'simultaneous').brier_raw, 4)} → ${vi(warn('gan_lstm', 'simultaneous').brier_calibrated, 4)}) nhưng làm xấu sự kiện bất kỳ (${vi(warn('gan_lstm', 'any').brier_raw, 4)} → ${vi(warn('gan_lstm', 'any').brier_calibrated, 4)}) — nên chọn dùng hiệu chỉnh theo từng sự kiện dựa trên tập calibration, không áp dụng máy móc.`),
  tableCaption(nextTable(), `Contingency table of 30-min warnings on the test set (${T.splits.test.windows} issue times). Probabilistic models warn when the calibrated probability ≥ 0.5 (fixed a priori); persistence and the point LSTM give yes/no forecasts. POD = hits/(hits + misses); precision = hits/warnings.`),
  table(['Event', 'Model', 'TP', 'FP', 'FN', 'TN', 'POD', 'Precision', 'F1', 'Specificity'], confusionRows,
        [19, 15, 7, 7, 7, 8, 9, 10, 8, 10], {shade: r => Math.floor(r / 4) % 2 === 1}),
  figure(img(T.figures.confusion)),
  caption(nextFigure(), 'Confusion matrices of 30-min warnings for gust, reflected H~m0~ and either event (counts and row percentages). Rows: observed event within the next 30 min; columns: warning issued.'),
  note('Nhận xét', `Mô hình học phát hiện sự kiện tốt hơn hẳn persistence: sự kiện gió POD ${vi(cf('gan_lstm', 'wind').recall, 2)} so với ${vi(cf('persistence', 'wind').recall, 2)}; sự kiện sóng ${vi(cf('gan_lstm', 'wave').recall, 2)} so với ${vi(cf('persistence', 'wave').recall, 2)}; sự kiện đồng thời ${vi(cf('gan_lstm', 'simultaneous').recall, 2)} so với ${vi(cf('persistence', 'simultaneous').recall, 2)}. F1 tăng tương ứng (sóng ${vi(cf('gan_lstm', 'wave').f1, 2)} so với ${vi(cf('persistence', 'wave').f1, 2)}). Persistence có precision gần 1 nhưng chỉ cảnh báo khi ngưỡng đã bị vượt nên bỏ lỡ giai đoạn khởi phát — chính là giá trị cảnh báo sớm của mô hình. Đổi lại số báo động giả tăng (sự kiện bất kỳ: ${cf('gan_lstm', 'any').fp} so với ${cf('persistence', 'any').fp} mốc). GAN-LSTM và Energy-LSTM gần như tương đương. Ngưỡng quyết định 0,5 cố định trước; các mốc phát hành chồng lấn không độc lập, nên các tỷ lệ mang tính mô tả.`),

  heading('7. Load forecasting (tanker at berth, v2)', 1),
  tableCaption(nextTable(), `Load forecast RMSE at 30 min (kN, kNm) and coverage of the 90% interval (%), run ${T.load_run} (base architecture, one seed). Environmental load = wind + first-order wave (incident + reflected) on a fixed tanker.`),
  table(['Target', 'Persistence', 'LSTM', 'Energy-LSTM', 'GAN-LSTM', 'Cov. Energy', 'Cov. GAN'], loadRows, [20, 13, 11, 14, 13, 14.5, 14.5]),
  figure(img(T.figures.loads)),
  caption(nextFigure(), 'Load forecast RMSE versus lead time with 95% scenario-bootstrap bands: RMS and maximum of sway force F~Y~ and yaw moment M~N~.'),
  note('Nhận xét', `Phần tải chưa được huấn luyện lại với nâng cấp v3 (kiến trúc cũ, 1 seed, 8 kịch bản test). GAN-LSTM có RMSE thấp nhất tại 30 phút; so với persistence giảm ${loadGain('environment_Y_rms_N')} (RMS F~Y~), ${loadGain('environment_N_rms_Nm')} (RMS M~N~) và ${loadGain('environment_Y_absmax_N')} (cực đại |F~Y~|). Lợi thế xuất hiện ngay từ phút đầu vì chỉ số RMS/cực đại trong cửa sổ 1 phút chứa ~12 chu kỳ sóng nên dao động ngẫu nhiên mạnh; đường RMSE răng cưa là nhiễu lấy mẫu (đã kiểm tra tự tương quan). Coverage 63–83% cần hiệu chỉnh conformal như phần môi trường. Nên huấn luyện lại phần tải với cấu hình v3 trước khi đưa vào bài.`),

  heading('8. Limitations to state in the paper', 1),
  note('Giới hạn', `Dữ liệu hoàn toàn mô phỏng, cùng một họ mô hình cho huấn luyện và kiểm thử; 3 seed huấn luyện; ${T.splits.test.scenarios} kịch bản test trong cùng vùng tham số. Bão là xung Gauss quy định trước, không phải vật lý gió sinh sóng — tín hiệu báo trước duy nhất là gió đi trước sóng. Sóng tới vuông góc tường, hệ số phản xạ hằng số, không có cộng hưởng khe tàu–cầu, nhiễu xạ hay bức xạ quanh thân tàu. Tàu cố định, RAO nước hở. Ngưỡng cảnh báo là thử nghiệm. Để kết luận mô hình "vượt trội" cần thêm seed, tập ngoài miền (độ sâu, hướng sóng khác) và đối chiếu dữ liệu đo thực.`),
];

const doc = new Document({
  creator: 'AIT Research Group', title: 'Results — wind and reflected-wave forecasting at a berth',
  styles: {default: {document: {run: {font: FONT, size: 22}}},
           paragraphStyles: [
             {id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
              run: {font: FONT, size: 28, bold: true}, paragraph: {spacing: {before: 360, after: 120}, outlineLevel: 0}},
             {id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
              run: {font: FONT, size: 24, bold: true}, paragraph: {spacing: {before: 240, after: 120}, outlineLevel: 1}}]},
  sections: [{
    properties: {page: {size: {width: 11906, height: 16838}, margin: {top: 1440, bottom: 1440, left: 1440, right: 1440}}},
    headers: {default: new Header({children: [new Paragraph({alignment: AlignmentType.RIGHT,
      children: [run('Wind & reflected-wave forecasting — results (synthetic data, v3)', {size: 16, color: '7F7F7F'})]})]})},
    footers: {default: new Footer({children: [new Paragraph({alignment: AlignmentType.CENTER,
      children: [new TextRun({children: [PageNumber.CURRENT], font: FONT, size: 18})]})]})},
    children,
  }],
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(outFile, buffer);
  console.log(`wrote ${outFile} (${tableNo} tables, ${figureNo} figures)`);
});
