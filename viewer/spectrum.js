'use strict';

// Tab 2: Welch-separated spectra S(f, t) of every feature window, as the training features saw them.
const SpectrumTab = (() => {
  const LABELS = {incident: 'Sóng tới', reflected: 'Phản xạ'};
  let data = null;
  let view = '3d';

  const status = (text, kind) => setStatus('sp-status', text, kind);
  const component = () => document.querySelector('input[name=sp-component]:checked').value;
  const logScale = () => $('sp-scale').value === 'log';
  const transform = v => v === null ? null : (logScale() ? Math.log10(Math.max(v, 1e-8)) : v);

  async function init() {
    const sources = await getJSON('/api/sources');
    const select = $('sp-source');
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
    const test = sources.find(s => s.split === 'test');
    select.value = test ? test.id : sources[0].id;
    await load();
  }

  async function load() {
    status('Đang mô phỏng lại và tách phổ từng cửa sổ… (kịch bản 4 giờ mất vài giây)');
    try {
      data = await getJSON(`/api/spectrogram?source=${encodeURIComponent($('sp-source').value)}`);
      $('sp-window').max = data.time_min.length - 1;
      $('sp-window').value = peakWindow();
      render();
      info();
      status(`${data.time_min.length} cửa sổ × ${data.frequency_hz.length} tần số.`);
    } catch (error) {
      status(`Lỗi: ${error.message}`, 'error');
    }
  }

  // Start the slice where reflected energy is largest, i.e. the storm peak.
  function peakWindow() {
    let best = 0;
    data.hm0_truth.forEach((v, i) => { if (v !== null && v > (data.hm0_truth[best] ?? -1)) best = i; });
    return best;
  }

  function render() {
    const key = component();
    const z = data[`${key}_psd`].map(row => row.map(transform));
    const values = z.flat().filter(v => v !== null);
    const lo = Math.min(...values), hi = Math.max(...values);
    const i = +$('sp-window').value;
    const traces = [{
      type: 'surface', x: data.frequency_hz, y: data.time_min, z, name: LABELS[key],
      colorscale: COLORMAPS[$('sp-colormap').value], cmin: logScale() ? Math.max(lo, hi - 5) : 0, cmax: hi,
      colorbar: {title: {text: logScale() ? 'log₁₀ S (m²/Hz)' : 'S (m²/Hz)'}, thickness: 14, len: .75},
      hovertemplate: 'f = %{x:.3f} Hz<br>t = %{y:.0f} min<br>S = %{customdata:.3g} m²/Hz<extra></extra>',
      customdata: data[`${key}_psd`],
    }, {
      type: 'scatter3d', mode: 'lines', name: 'Cửa sổ chọn', x: data.frequency_hz,
      y: data.frequency_hz.map(() => data.time_min[i]), z: z[i], line: {color: '#000', width: 6}, hoverinfo: 'skip',
    }];
    Plotly.react('sp-surface', traces, {
      margin: {l: 0, r: 0, t: 30, b: 0}, uirevision: view,
      title: {text: `Phổ ${LABELS[key].toLowerCase()} tách được — ${data.source.id}`, font: {size: 14}},
      legend: {x: 0, y: 1},
      scene: {
        xaxis: {...AXIS, title: {text: 'f (Hz)'}}, yaxis: {...AXIS, title: {text: 't (phút)'}},
        zaxis: {...AXIS, title: {text: view === 'top' ? '' : (logScale() ? 'log₁₀ S' : 'S (m²/Hz)')},
                showticklabels: view !== 'top'},
        aspectmode: 'manual', aspectratio: {x: 1.3, y: 1.5, z: .6}, camera: VIEWS[view],
      },
    }, CONFIG);
    // Hm0 plot must exist before renderWindow moves its cursor.
    renderHm0();
    renderWindow();
  }

  function renderWindow() {
    const i = +$('sp-window').value;
    $('sp-window-value').textContent = `${data.time_min[i].toFixed(0)} phút`;
    const scale = data.envelope_power[i];
    const design = data.design_incident_psd.map(v => v * scale);
    const kr2 = data.reflection ** 2;
    const traces = [
      {x: data.frequency_hz, y: data.incident_psd[i], name: 'Sóng tới (tách)', line: {color: COLORS.incident, width: 2}},
      {x: data.frequency_hz, y: data.reflected_psd[i], name: 'Phản xạ (tách)', line: {color: COLORS.reflected, width: 2}},
      {x: data.frequency_hz, y: design, name: 'JONSWAP thiết kế', line: {color: COLORS.design, dash: 'dash'}},
      {x: data.frequency_hz, y: design.map(v => v * kr2), name: 'Kr² × thiết kế',
       line: {color: COLORS.design, dash: 'dot'}},
    ];
    Plotly.react('sp-slice', traces, {
      title: {text: `Phổ cửa sổ kết thúc t = ${data.time_min[i].toFixed(0)} phút (Welch ${data.segment_s} s)`, font: {size: 13}},
      margin: {l: 60, r: 15, t: 35, b: 90}, xaxis: axis2d('f (Hz)'),
      yaxis: axis2d('S (m²/Hz)', {type: logScale() ? 'log' : 'linear', exponentformat: 'power'}),
      legend: {orientation: 'h', x: 0, y: -0.32, yanchor: 'top'},
    }, CONFIG);
    const surfaceTraces = $('sp-surface').data;
    if (surfaceTraces) {
      const z = data[`${component()}_psd`][i].map(transform);
      Plotly.restyle('sp-surface', {y: [data.frequency_hz.map(() => data.time_min[i])], z: [z]}, [1]);
    }
    Plotly.relayout('sp-hm0', {shapes: [cursor(data.time_min[i])]});
  }

  function cursor(x) {
    return {type: 'line', xref: 'x', yref: 'paper', x0: x, x1: x, y0: 0, y1: 1, line: {color: '#000', width: 1}};
  }

  function renderHm0() {
    const bad = data.quality_ok.map((ok, i) => ok ? null : data.hm0_truth[i]);
    const traces = [
      {x: data.time_min, y: data.hm0_truth, name: 'Thật (4σ phản xạ sạch)', line: {color: COLORS.truth, width: 2}},
      {x: data.time_min, y: data.hm0_estimate, name: 'Ước lượng từ 3 đầu đo', mode: 'markers',
       marker: {color: COLORS.reflected, size: 5}},
      {x: data.time_min, y: bad, name: 'Cửa sổ bị loại (quality_ok = 0)', mode: 'markers',
       marker: {color: COLORS.probe, symbol: 'x', size: 8}},
    ];
    const i = +$('sp-window').value;
    Plotly.react('sp-hm0', traces, {
      title: {text: 'Hm0 phản xạ theo thời gian', font: {size: 13}},
      margin: {l: 60, r: 15, t: 35, b: 90}, xaxis: axis2d('t (phút)'), yaxis: axis2d('Hm0 (m)', {rangemode: 'tozero'}),
      legend: {orientation: 'h', x: 0, y: -0.32, yanchor: 'top'}, shapes: [cursor(data.time_min[i])],
    }, CONFIG);
  }

  function info() {
    const s = data.source;
    const errors = data.hm0_estimate.map((v, i) => v === null || data.hm0_truth[i] === null ? null : v / data.hm0_truth[i] - 1)
      .filter(v => v !== null);
    const mean = errors.reduce((a, b) => a + b, 0) / errors.length;
    const std = Math.sqrt(errors.reduce((a, b) => a + (b - mean) ** 2, 0) / errors.length);
    const rows = [
      ['Hs thiết kế', `${s.hs_m.toFixed(2)} m`], ['Tp', `${s.tp_s.toFixed(2)} s`], ['Kr', data.reflection.toFixed(2)],
      ['Cửa sổ / đoạn Welch', `${data.window_s} s / ${data.segment_s} s`],
      ['Sai số Hm0 (TB ± σ)', `${(100 * mean).toFixed(1)} ± ${(100 * std).toFixed(1)} %`],
      ['Cửa sổ hợp lệ', `${data.quality_ok.filter(Boolean).length}/${data.quality_ok.length}`],
    ];
    $('sp-info').replaceChildren(...rows.flatMap(([k, v]) => {
      const dt = document.createElement('dt'); dt.textContent = k;
      const dd = document.createElement('dd'); dd.textContent = v;
      return [dt, dd];
    }));
  }

  $('sp-source').addEventListener('change', load);
  $('sp-window').addEventListener('input', () => data && renderWindow());
  for (const input of document.querySelectorAll('input[name=sp-component], #sp-scale, #sp-colormap')) {
    input.addEventListener('change', () => data && render());
  }
  viewButtons($('page-spectrum'), name => { view = name; if (data) render(); });

  return {init, plots: ['sp-surface', 'sp-slice', 'sp-hm0']};
})();
