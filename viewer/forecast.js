'use strict';

// Tab 4: archived test-set forecasts as surfaces over (horizon, issue time), plus a fan chart per issue time.
const ForecastTab = (() => {
  const MODEL_LABELS = {persistence: 'Persistence', lstm_point: 'LSTM điểm', energy_lstm: 'Energy-LSTM', gan_lstm: 'GAN-LSTM'};
  let runs = [];
  let data = null;
  let view = '3d';

  const status = (text, kind) => setStatus('fc-status', text, kind);
  const mode = () => document.querySelector('input[name=fc-mode]:checked').value;
  const currentRun = () => runs.find(r => r.run === $('fc-run').value);

  async function init() {
    runs = await getJSON('/api/forecast_runs');
    if (!runs.length) throw new Error('Chưa có outputs/*/evaluation/*_test.npz. Chạy huấn luyện trước.');
    fillSelect($('fc-run'), runs, r => r.run, r => `${r.run} (${r.task === 'loads' ? 'tải' : 'môi trường'})`);
    // The environmental forecast is the paper's main experiment; show it first.
    const environment = runs.find(r => r.task === 'environment');
    if (environment) $('fc-run').value = environment.run;
    syncRun();
    await load();
  }

  function syncRun() {
    const run = currentRun();
    const model = $('fc-model').value;
    fillSelect($('fc-model'), run.models, m => m, m => MODEL_LABELS[m] || m);
    $('fc-model').value = run.models.includes(model) ? model : 'gan_lstm';
    fillSelect($('fc-scenario'), run.scenarios, s => s, s => `scenario ${String(s).padStart(3, '0')}`);
    fillSelect($('fc-variable'), run.variables, v => v.index, v => `${v.label} (${v.unit})`);
    // Default to the variable the paper warns on: reflected Hm0, or sway RMS for loads.
    $('fc-variable').value = run.task === 'loads' ? 1 : 3;
  }

  async function load() {
    const query = new URLSearchParams({run: $('fc-run').value, model: $('fc-model').value,
                                       scenario: $('fc-scenario').value, variable: $('fc-variable').value});
    status('Đang đọc dự báo đã lưu…');
    try {
      data = await getJSON(`/api/forecast?${query}`);
      $('fc-origin').max = data.origin_min.length - 1;
      $('fc-origin').value = Math.min(+$('fc-origin').value, data.origin_min.length - 1);
      render();
      status(data.samples > 1 ? `${data.origin_min.length} lần phát hành × ${data.samples} mẫu ensemble.`
                              : `${data.origin_min.length} lần phát hành; mô hình tất định, không có dải tin cậy.`);
    } catch (error) {
      status(`Lỗi: ${error.message}`, 'error');
    }
  }

  function surfaceValues() {
    const m = mode();
    if (m === 'error') return data.mean.map((row, i) => row.map((v, j) => v - data.target[i][j]));
    if (m === 'width') return data.q95 ? data.q95.map((row, i) => row.map((v, j) => v - data.q05[i][j])) : null;
    return data.mean;
  }

  function render() {
    const unit = data.variable.unit;
    let z = surfaceValues();
    if (!z) {
      status('Mô hình tất định không có độ rộng khoảng 90%; hiển thị giá trị dự báo.', 'warn');
      document.querySelector('input[name=fc-mode][value=mean]').checked = true;
      z = data.mean;
    }
    const m = mode();
    const symmetric = m === 'error';
    const limit = finiteMax(z);
    const titles = {mean: `Dự báo trung bình (${unit})`, error: `Sai số dự báo − thực (${unit})`, width: `Độ rộng 90% (${unit})`};
    const traces = [{
      type: 'surface', x: data.horizon_min, y: data.origin_min, z, name: titles[m],
      colorscale: COLORMAPS[symmetric ? 'balance' : $('fc-colormap').value],
      ...(symmetric ? {cmin: -limit, cmax: limit} : {}),
      colorbar: {title: {text: unit}, thickness: 14, len: .75},
      hovertemplate: `chân trời %{x} phút<br>phát hành %{y:.0f} phút<br>%{z:.3f} ${unit}<extra></extra>`,
    }];
    if (m === 'mean' && $('fc-target').checked) {
      // Wireframe of the observed future, like MATLAB mesh() over surf().
      traces.push({type: 'surface', x: data.horizon_min, y: data.origin_min, z: data.target, name: 'Quan trắc',
                   showscale: false, hidesurface: true, hoverinfo: 'skip',
                   contours: {x: {show: true, color: '#000', width: 1}, y: {show: true, color: '#000', width: 1}}});
    }
    const i = +$('fc-origin').value;
    traces.push({type: 'scatter3d', mode: 'lines', name: 'Lần phát hành chọn', x: data.horizon_min,
                 y: data.horizon_min.map(() => data.origin_min[i]), z: z[i], line: {color: '#a2142f', width: 7}});
    Plotly.react('fc-surface', traces, {
      margin: {l: 0, r: 0, t: 30, b: 0}, uirevision: view, legend: {x: 0, y: 1},
      title: {text: `${MODEL_LABELS[data.model]} — ${data.variable.label}, scenario ${data.scenario} (tập test)`, font: {size: 14}},
      scene: {
        xaxis: {...AXIS, title: {text: 'Chân trời (phút)'}}, yaxis: {...AXIS, title: {text: 'Thời điểm phát hành (phút)'}},
        zaxis: {...AXIS, title: {text: view === 'top' ? '' : unit}, showticklabels: view !== 'top'},
        aspectmode: 'manual', aspectratio: {x: 1.1, y: 1.6, z: .6}, camera: VIEWS[view],
      },
    }, CONFIG);
    renderFan();
    renderSkill();
  }

  function renderFan() {
    const i = +$('fc-origin').value;
    $('fc-origin-value').textContent = `${data.origin_min[i].toFixed(0)} phút`;
    const history = data.history[i];
    const past = history.map((_, k) => k - history.length + 1);
    const traces = [];
    if (data.q05) {
      traces.push({x: data.horizon_min, y: data.q95[i], line: {width: 0}, showlegend: false, hoverinfo: 'skip'},
                  {x: data.horizon_min, y: data.q05[i], fill: 'tonexty', fillcolor: 'rgba(0,114,189,.2)',
                   line: {width: 0}, name: 'Khoảng 90%'});
    }
    traces.push(
      {x: past, y: history, name: 'Lịch sử quan trắc', line: {color: '#000', width: 1.5}},
      {x: data.horizon_min, y: data.target[i], name: 'Tương lai quan trắc', line: {color: '#000', dash: 'dash'}},
      {x: data.horizon_min, y: data.mean[i], name: 'Dự báo trung bình', line: {color: COLORS.total, width: 2.5}},
    );
    Plotly.react('fc-fan', traces, {
      title: {text: `Phát hành lúc ${data.origin_min[i].toFixed(0)} phút`, font: {size: 13}},
      margin: {l: 60, r: 15, t: 35, b: 90}, xaxis: axis2d('Phút so với lúc phát hành'),
      yaxis: axis2d(`${data.variable.label} (${data.variable.unit})`),
      shapes: [{type: 'line', xref: 'x', yref: 'paper', x0: 0, x1: 0, y0: 0, y1: 1, line: {color: '#8c8c8c', width: 1}}],
      legend: {orientation: 'h', x: 0, y: -0.32, yanchor: 'top'},
    }, CONFIG);
    const surfaceTraces = $('fc-surface').data;
    if (surfaceTraces) {
      const z = surfaceValues() || data.mean;
      Plotly.restyle('fc-surface', {y: [data.horizon_min.map(() => data.origin_min[i])], z: [z[i]]},
                     [surfaceTraces.length - 1]);
    }
  }

  function renderSkill() {
    const traces = [{x: data.horizon_min, y: data.rmse_by_horizon, name: `RMSE (${data.variable.unit})`,
                     line: {color: COLORS.total, width: 2}}];
    const layout = {
      title: {text: `Theo chân trời, riêng scenario ${data.scenario}`, font: {size: 13}},
      margin: {l: 60, r: 60, t: 35, b: 90}, xaxis: axis2d('Chân trời (phút)'),
      yaxis: axis2d(`RMSE (${data.variable.unit})`, {rangemode: 'tozero'}),
      legend: {orientation: 'h', x: 0, y: -0.32, yanchor: 'top'},
    };
    if (data.coverage_by_horizon) {
      traces.push({x: data.horizon_min, y: data.coverage_by_horizon.map(v => 100 * v), name: 'Coverage 90% (%)',
                   yaxis: 'y2', line: {color: COLORS.incident, width: 2}});
      layout.yaxis2 = {title: {text: 'Coverage (%)'}, overlaying: 'y', side: 'right', range: [0, 105],
                       tickvals: [0, 25, 50, 75, 90, 100],
                       showline: true, linecolor: '#262626', ticks: 'outside', zeroline: false};
      layout.shapes = [{type: 'line', xref: 'paper', yref: 'y2', x0: 0, x1: 1, y0: 90, y1: 90,
                        line: {color: COLORS.incident, dash: 'dot', width: 1}}];
    }
    Plotly.react('fc-skill', traces, layout, CONFIG);
  }

  $('fc-run').addEventListener('change', () => { syncRun(); load(); });
  for (const id of ['fc-model', 'fc-scenario', 'fc-variable']) $(id).addEventListener('change', load);
  $('fc-origin').addEventListener('input', () => data && renderFan());
  for (const input of document.querySelectorAll('input[name=fc-mode], #fc-target, #fc-colormap')) {
    input.addEventListener('change', () => data && render());
  }
  viewButtons($('page-forecast'), name => { view = name; if (data) render(); });

  return {init, plots: ['fc-surface', 'fc-fan', 'fc-skill']};
})();
