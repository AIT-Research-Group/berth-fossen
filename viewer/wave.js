'use strict';

// Tab 1: free surface eta(x, t). Globals from common.js: COLORMAPS, COLORS, AXIS, CONFIG, VIEWS.
const WaveTab = (() => {
const LABELS = {total: 'Tổng', incident: 'Sóng tới', reflected: 'Phản xạ'};
let sources = [];
let data = null;
let view = '3d';

function status(text, kind = '') { setStatus('status', text, kind); }

function component() {
  return document.querySelector('input[name=component]:checked').value;
}

async function loadSources() {
  const response = await fetch('/api/sources');
  if (!response.ok) throw new Error(`Không đọc được danh sách nguồn (${response.status})`);
  sources = await response.json();
  const select = $('source');
  const groups = new Map();
  for (const source of sources) {
    if (!groups.has(source.run)) {
      const group = document.createElement('optgroup');
      group.label = source.run;
      groups.set(source.run, group);
      select.append(group);
    }
    groups.get(source.run).append(new Option(source.label, source.id));
  }
  if (!sources.length) throw new Error('Chưa có kết quả trong outputs/. Chạy mô phỏng demo trước.');
  select.value = sources.some(s => s.id === 'demo') ? 'demo' : sources[0].id;
  syncLimits();
}

function currentSource() {
  return sources.find(s => s.id === $('source').value);
}

function syncLimits() {
  const source = currentSource();
  const maximum = Math.max(0, source.duration_s - 5);
  $('t0range').max = maximum;
  $('t0').max = maximum;
  if (+$('t0').value > maximum) $('t0').value = Math.min(600, maximum);
  $('t0range').value = $('t0').value;
}

async function draw() {
  const [nx, nt] = $('resolution').value.split(',');
  const query = new URLSearchParams({source: $('source').value, t0: $('t0').value, span: $('span').value,
                                     x_min: $('xmin').value, nx, nt});
  $('draw').disabled = true;
  status('Đang tính mặt sóng…');
  try {
    const response = await fetch(`/api/surface?${query}`);
    if (!response.ok) throw new Error(await response.text() || response.statusText);
    data = await response.json();
    $('slice').max = data.t.length - 1;
    $('slice').value = Math.floor(data.t.length / 2);
    render();
    info();
    status(data.aliased ? `Cảnh báo: bước thời gian ${data.dt_s.toFixed(3)} s thô hơn Nyquist của dải sóng; tăng độ phân giải hoặc giảm độ dài.`
                        : `Đã vẽ lưới ${data.x.length} × ${data.t.length}.`, data.aliased ? 'warn' : '');
  } catch (error) {
    status(`Lỗi: ${error.message}`, 'error');
  } finally {
    $('draw').disabled = false;
  }
}

function probeIndices() {
  return data.source.probe_x_m.map(p => data.x.reduce((best, x, i) => Math.abs(x - p) < Math.abs(data.x[best] - p) ? i : best, 0));
}

function render() {
  const key = component();
  const z = data[key];
  const limit = Math.max(1e-6, ...z.map(row => Math.max(...row.map(Math.abs))));
  const t0 = data.t[0], t1 = data.t[data.t.length - 1];
  const traces = [{
    type: 'surface', x: data.x, y: data.t, z, name: LABELS[key],
    colorscale: COLORMAPS[$('colormap').value], cmin: -limit, cmax: limit,
    colorbar: {title: {text: 'η (m)'}, thickness: 14, len: .75},
    contours: {z: {show: false}},
    lighting: {ambient: .75, diffuse: .6, specular: .08, roughness: .9},
    hovertemplate: 'x = %{x:.2f} m<br>t = %{y:.2f} s<br>η = %{z:.3f} m<extra></extra>',
  }, {
    type: 'surface', name: 'Tường (x = 0)', showscale: false, opacity: .35, hoverinfo: 'skip',
    x: [[0, 0], [0, 0]], y: [[t0, t1], [t0, t1]], z: [[-limit, -limit], [limit, limit]],
    colorscale: [[0, '#7f7f7f'], [1, '#7f7f7f']],
  }];
  for (const [n, j] of probeIndices().entries()) {
    traces.push({type: 'scatter3d', mode: 'lines', name: n ? undefined : 'Đầu đo', showlegend: n === 0,
                 legendgroup: 'probe', x: data.t.map(() => data.x[j]), y: data.t, z: z.map(row => row[j]),
                 line: {color: COLORS.probe, width: 5}, hoverinfo: 'skip'});
  }
  const i = +$('slice').value;
  traces.push({type: 'scatter3d', mode: 'lines', name: 't*', x: data.x, y: data.x.map(() => data.t[i]),
               z: z[i], line: {color: '#000000', width: 6}, hoverinfo: 'skip'});
  const scale = +$('zscale').value || 1;
  const layout = {
    margin: {l: 0, r: 0, t: 30, b: 0}, paper_bgcolor: '#ffffff',
    title: {text: `${LABELS[key]} — ${data.source.id}`, font: {size: 14}},
    legend: {x: 0, y: 1, bgcolor: 'rgba(255,255,255,.8)'},
    scene: {
      xaxis: {...AXIS, title: {text: 'x (m)'}}, yaxis: {...AXIS, title: {text: 't (s)'}},
      // The top view looks straight down the eta axis, whose labels would pile up.
      zaxis: {...AXIS, title: {text: view === 'top' ? '' : 'η (m)'}, showticklabels: view !== 'top',
              range: [-limit * 1.05, limit * 1.05]},
      aspectmode: 'manual', aspectratio: {x: 1.4, y: 1.4, z: .55 * scale},
      camera: VIEWS[view],
    },
    // Keeps a user-rotated camera across redraws until another preset is chosen.
    uirevision: view,
  };
  Plotly.react('surface', traces, layout, CONFIG);
  renderSlice();
  renderProfile();
}

function renderSlice() {
  const i = +$('slice').value;
  $('sliceValue').textContent = data.t[i].toFixed(2);
  const all = ['incident', 'reflected', 'total'];
  const limit = Math.max(...all.flatMap(k => data[k].map(row => Math.max(...row.map(Math.abs)))));
  const traces = all.map(k => ({x: data.x, y: data[k][i], name: LABELS[k], mode: 'lines',
                                line: {color: COLORS[k], width: k === 'total' ? 2.5 : 1.5}}));
  Plotly.react('sliceplot', traces, {
    title: {text: `η(x) tại t* = ${data.t[i].toFixed(2)} s`, font: {size: 13}},
    margin: {l: 55, r: 15, t: 35, b: 90}, xaxis: {title: {text: 'x (m)'}, mirror: true, showline: true, linecolor: '#262626'},
    yaxis: {title: {text: 'η (m)'}, range: [-limit * 1.05, limit * 1.05], mirror: true, showline: true, linecolor: '#262626'},
    shapes: wallAndProbes(), legend: {orientation: 'h', x: 0, y: -0.32, yanchor: 'top'},
  }, CONFIG);
  const surfaceTraces = document.getElementById('surface').data;
  if (surfaceTraces) {
    const last = surfaceTraces.length - 1;
    Plotly.restyle('surface', {y: [data.x.map(() => data.t[i])], z: [data[component()][i]]}, [last]);
  }
}

function renderProfile() {
  const p = data.profile;
  const traces = [
    {x: data.x, y: p.hm0_total, name: 'Tổng (phổ, dừng)', mode: 'lines', line: {color: COLORS.total, width: 2.5}},
    {x: data.x, y: p.hm0_total_sample, name: 'Tổng (4σ trong cửa sổ)', mode: 'markers',
     marker: {color: COLORS.total, size: 4, symbol: 'circle-open'}},
    {x: data.x, y: data.x.map(() => p.hm0_incident), name: 'Sóng tới', mode: 'lines', line: {color: COLORS.incident, dash: 'dash'}},
    {x: data.x, y: data.x.map(() => p.hm0_reflected), name: 'Phản xạ', mode: 'lines', line: {color: COLORS.reflected, dash: 'dash'}},
  ];
  Plotly.react('profile', traces, {
    title: {text: 'Hm0(x): sóng đứng gần tường', font: {size: 13}},
    margin: {l: 55, r: 15, t: 35, b: 90}, xaxis: {title: {text: 'x (m)'}, mirror: true, showline: true, linecolor: '#262626'},
    yaxis: {title: {text: 'Hm0 (m)'}, rangemode: 'tozero', mirror: true, showline: true, linecolor: '#262626'},
    shapes: wallAndProbes(), legend: {orientation: 'h', x: 0, y: -0.32, yanchor: 'top'},
  }, CONFIG);
}

function wallAndProbes() {
  const line = (x, color, dash) => ({type: 'line', xref: 'x', yref: 'paper', x0: x, x1: x, y0: 0, y1: 1,
                                     line: {color, width: 1.5, dash}});
  return [line(0, '#404040', 'solid'), ...data.source.probe_x_m.map(x => line(x, COLORS.probe, 'dot'))];
}

function info() {
  const s = data.source;
  const reflection = typeof s.reflection === 'object' && !Array.isArray(s.reflection)
    ? `${s.reflection.model}` : Number(s.reflection).toFixed(2);
  const rows = [
    ['Hs thiết kế', `${s.hs_m.toFixed(2)} m`], ['Tp', `${s.tp_s.toFixed(2)} s`], ['γ', s.gamma.toFixed(2)],
    ['Kr', reflection], ['Độ sâu h', `${s.depth_m} m`],
    ['Bước sóng đỉnh Lp', `${data.peak_wavelength_m.toFixed(1)} m`],
    ['Khoảng nút Lp/2', `${(data.peak_wavelength_m / 2).toFixed(1)} m`],
    ['Hướng trung bình', `${(data.mean_direction_rad * 180 / Math.PI).toFixed(1)}°`],
    ['Bao biên bão', `${data.envelope.min.toFixed(2)}–${data.envelope.max.toFixed(2)}`],
    ['Năng lượng trong dải', `${(100 * data.band_energy_fraction).toFixed(1)}%`],
    ['Tập', s.split], ['Δt lưới', `${data.dt_s.toFixed(3)} s`],
  ];
  $('info').replaceChildren(...rows.flatMap(([k, v]) => {
    const dt = document.createElement('dt'); dt.textContent = k;
    const dd = document.createElement('dd'); dd.textContent = v;
    return [dt, dd];
  }));
}

function setView(name) {
  // A projection change needs a full react; relayout alone leaves a stale scene.
  view = name;
  if (data) render();
}

$('draw').addEventListener('click', draw);
$('source').addEventListener('change', () => { syncLimits(); draw(); });
$('t0range').addEventListener('input', () => { $('t0').value = $('t0range').value; });
$('t0range').addEventListener('change', draw);
$('t0').addEventListener('change', () => { $('t0range').value = $('t0').value; draw(); });
$('slice').addEventListener('input', () => data && renderSlice());
for (const input of document.querySelectorAll('input[name=component], #colormap, #zscale')) {
  input.addEventListener('change', () => data && render());
}
for (const button of $('page-wave').querySelectorAll('[data-view]')) {
  button.addEventListener('click', () => setView(button.dataset.view));
}

return {init: () => loadSources().then(draw).catch(error => status(error.message, 'error')),
        plots: ['surface', 'sliceplot', 'profile']};
})();
