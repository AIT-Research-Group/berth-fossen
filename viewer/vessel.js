'use strict';

// Tab 3: replay of the simulated 3-DOF ship beside the quay, with BODY loads drawn in global axes.
const VesselTab = (() => {
  let data = null;
  let index = 0;
  let playing = false;
  let lastFrame = 0;
  let view = '3d';

  const status = (text, kind) => setStatus('vs-status', text, kind);
  const exaggeration = () => +$('vs-exaggeration').value || 1;

  async function init() {
    const runs = await getJSON('/api/vessel_runs');
    if (!runs.length) throw new Error('Chưa có outputs/*/vessel.npz. Chạy mô phỏng với vessel.enabled = true.');
    fillSelect($('vs-run'), runs);
    await load();
  }

  async function load() {
    stop();
    status('Đang đọc chuyển động tàu…');
    try {
      data = await getJSON(`/api/vessel?run=${encodeURIComponent($('vs-run').value)}`);
      data.body = shipGeometry(data.geometry);
      data.draft = data.geometry.draft_m;
      data.freeboard = data.geometry.freeboard_m;
      data.total = data.wind.map((w, i) => w.map((v, k) => v + data.wave[i][k]));
      // Arrow scale: the largest total planar load maps to 0.4 ship lengths.
      data.forceScale = Math.max(...data.total.map(f => Math.hypot(f[0], f[1]))) / (0.4 * data.length_m) || 1;
      $('vs-time').max = data.time_s.length - 1;
      index = 0;
      $('vs-time').value = 0;
      render();
      status(data.wave_rao_zero_axes.length
        ? `RAO minh họa của lần chạy này cho lực sóng ${data.wave_rao_zero_axes.join(', ')} = 0.` : '');
    } catch (error) {
      status(`Lỗi: ${error.message}`, 'error');
    }
  }

  // Exaggerated pose: motions of a few metres are invisible against a 350 m hull otherwise.
  function pose(i) {
    const k = exaggeration();
    return {x: data.hold[0] + k * (data.x[i] - data.hold[0]), y: data.hold[1] + k * (data.y[i] - data.hold[1]),
            psi: data.hold[2] + k * (data.psi[i] - data.hold[2])};
  }

  // Both meshes of the shared Wigley ship model, at the exaggerated pose of sample i.
  function shipAt(i) {
    const p = pose(i);
    return [placeShip(data.body.hull, p), placeShip(data.body.box, p)];
  }

  function arrow(i, force) {
    const {x, y, psi} = pose(i);
    const c = Math.cos(psi), s = Math.sin(psi);
    const gx = (c * force[0] - s * force[1]) / data.forceScale;
    const gy = (s * force[0] + c * force[1]) / data.forceScale;
    // Arrows sit just above the deck so the hull does not hide them.
    const z = data.freeboard + 4;
    return {x: [x, x + gx], y: [y, y + gy], z: [z, z]};
  }

  function sceneTraces(i) {
    const L = data.length_m;
    const [hx, hy] = data.hold;
    const yLo = hy - 0.9 * L, yHi = hy + 0.9 * L, xLo = hx - 0.9 * L;
    const k = exaggeration();
    const traces = [
      {type: 'surface', name: 'Mặt nước', x: [xLo, 0], y: [yLo, yHi], z: [[0, 0], [0, 0]], showscale: false,
       opacity: .25, colorscale: [[0, '#4da6ff'], [1, '#4da6ff']], hoverinfo: 'skip'},
      {type: 'surface', name: 'Tường cầu (x = 0)', x: [[0, 0], [0, 0]], y: [[yLo, yHi], [yLo, yHi]],
       z: [[-data.draft - 5, -data.draft - 5], [data.freeboard + 5, data.freeboard + 5]], showscale: false,
       colorscale: [[0, '#8c8c8c'], [1, '#8c8c8c']], opacity: .8, hoverinfo: 'skip'},
      {type: 'scatter3d', mode: 'lines', name: `Quỹ đạo tâm (×${k})`,
       x: data.x.map(v => hx + k * (v - hx)), y: data.y.map(v => hy + k * (v - hy)), z: data.x.map(() => data.freeboard),
       line: {color: data.time_s, colorscale: COLORMAPS.parula, width: 4}, hoverinfo: 'skip'},
      {type: 'scatter3d', mode: 'markers', name: 'Đầu đo sóng', x: data.probe_x_m, y: data.probe_x_m.map(() => hy),
       z: data.probe_x_m.map(() => 0), marker: {color: COLORS.probe, size: 4}},
      ...shipTraces(data.geometry, pose(i)),
    ];
    for (const [key, label, color] of [['wind', 'Lực gió', COLORS.wind], ['wave', 'Lực sóng', COLORS.wave],
                                        ['total', 'Tổng', COLORS.sum]]) {
      traces.push({type: 'scatter3d', mode: 'lines', name: label, ...arrow(i, data[key][i]),
                   line: {color, width: 9}, hoverinfo: 'skip'});
    }
    return traces;
  }

  function render() {
    const L = data.length_m;
    Plotly.react('vs-scene', sceneTraces(index), {
      margin: {l: 0, r: 0, t: 30, b: 0}, uirevision: view,
      title: {text: `Tàu tại cầu — ${data.run}`, font: {size: 14}}, legend: {x: 0, y: 1},
      scene: {
        xaxis: {...AXIS, title: {text: 'x (m) → tường'}}, yaxis: {...AXIS, title: {text: 'y dọc cầu (m)'}},
        zaxis: {...AXIS, title: {text: 'z (m)'}, range: [-data.draft - 5, data.freeboard * 3.5 + 10]},
        aspectmode: 'data', camera: VIEWS[view],
      },
    }, CONFIG);
    renderSeries();
    frame(index, true);
  }

  function subplotLayout(title, rows) {
    // Title on the top line, legend on its own line below it, so the two never overlap.
    const layout = {title: {text: title, font: {size: 13}, y: .98, yanchor: 'top'}, margin: {l: 65, r: 15, t: 62, b: 45},
                    showlegend: true, legend: {orientation: 'h', x: 0, y: 1.0, yanchor: 'bottom'},
                    grid: {rows: rows.length, columns: 1}};
    rows.forEach((label, r) => {
      const suffix = r ? r + 1 : '';
      layout[`yaxis${suffix}`] = axis2d(label, {domain: [1 - (r + 1) / rows.length + .06, 1 - r / rows.length - .02]});
      layout[`xaxis${suffix}`] = axis2d(r === rows.length - 1 ? 't (s)' : '', {anchor: `y${suffix}`, matches: 'x',
                                                                               showticklabels: r === rows.length - 1});
    });
    return layout;
  }

  function renderSeries() {
    const t = data.time_s;
    const loadTraces = [];
    const units = [['X (kN)', 1e3], ['Y (kN)', 1e3], ['N (MNm)', 1e6]];
    units.forEach(([, scale], r) => {
      const axes = r ? {xaxis: `x${r + 1}`, yaxis: `y${r + 1}`} : {};
      for (const [key, label, color] of [['wind', 'Gió', COLORS.wind], ['wave', 'Sóng', COLORS.wave],
                                          ['total', 'Tổng', COLORS.sum]]) {
        loadTraces.push({x: t, y: data[key].map(f => f[r] / scale), name: label, legendgroup: key, showlegend: r === 0,
                         line: {color, width: key === 'total' ? 1.6 : 1}, ...axes});
      }
    });
    Plotly.react('vs-loads', loadTraces, subplotLayout('Tải môi trường, hệ BODY', units.map(u => u[0])), CONFIG);
    const [hx, hy, hpsi] = data.hold;
    const motion = [
      {x: t, y: data.x.map(v => v - hx), name: 'Δx', line: {color: COLORS.wind}},
      {x: t, y: data.y.map(v => v - hy), name: 'Δy', line: {color: COLORS.wave}, xaxis: 'x2', yaxis: 'y2'},
      {x: t, y: data.psi.map(v => (v - hpsi) * 180 / Math.PI), name: 'Δψ', line: {color: COLORS.reflected},
       xaxis: 'x3', yaxis: 'y3'},
    ];
    Plotly.react('vs-motion', motion, subplotLayout('Lệch khỏi vị trí giữ (chưa phóng đại)', ['Δx (m)', 'Δy (m)', 'Δψ (°)']), CONFIG);
  }

  function frame(i, updateCharts) {
    index = i;
    $('vs-time').value = i;
    const meshes = shipAt(i);
    const scene = $('vs-scene').data;
    const meshIndex = scene.findIndex(t => t.type === 'mesh3d');
    Plotly.restyle('vs-scene', {x: meshes.map(m => m.x), y: meshes.map(m => m.y), z: meshes.map(m => m.z)},
                   [meshIndex, meshIndex + 1]);
    const arrows = ['wind', 'wave', 'total'].map(key => arrow(i, data[key][i]));
    Plotly.restyle('vs-scene', {x: arrows.map(a => a.x), y: arrows.map(a => a.y), z: arrows.map(a => a.z)},
                   [meshIndex + 2, meshIndex + 3, meshIndex + 4]);
    info(i);
    if (updateCharts) {
      const line = {type: 'line', xref: 'x', yref: 'paper', x0: data.time_s[i], x1: data.time_s[i], y0: 0, y1: 1,
                    line: {color: '#a2142f', width: 1}};
      Plotly.relayout('vs-loads', {shapes: [line]});
      Plotly.relayout('vs-motion', {shapes: [line]});
    }
  }

  function info(i) {
    const [hx, hy, hpsi] = data.hold;
    const rows = [
      ['t', `${data.time_s[i].toFixed(0)} s`], ['Δx, Δy', `${(data.x[i] - hx).toFixed(2)}, ${(data.y[i] - hy).toFixed(2)} m`],
      ['Δψ', `${((data.psi[i] - hpsi) * 180 / Math.PI).toFixed(3)}°`],
      ['Lực gió X, Y', `${(data.wind[i][0] / 1e3).toFixed(0)}, ${(data.wind[i][1] / 1e3).toFixed(0)} kN`],
      ['Lực sóng X, Y', `${(data.wave[i][0] / 1e3).toFixed(0)}, ${(data.wave[i][1] / 1e3).toFixed(0)} kN`],
      ['Mômen N (gió+sóng)', `${(data.total[i][2] / 1e6).toFixed(2)} MNm`],
      ['Thân tàu', `Wigley ${data.length_m} × ${data.geometry.beam_m.toFixed(0)} × ${data.draft.toFixed(1)} m`],
    ];
    $('vs-info').replaceChildren(...rows.flatMap(([k, v]) => {
      const dt = document.createElement('dt'); dt.textContent = k;
      const dd = document.createElement('dd'); dd.textContent = v;
      return [dt, dd];
    }));
  }

  function tick(now) {
    if (!playing) return;
    // Advance by real elapsed time times the chosen speed, independent of frame rate.
    const dtData = data.time_s[1] - data.time_s[0];
    const steps = Math.max(1, Math.round((now - lastFrame) / 1000 * +$('vs-speed').value / dtData));
    lastFrame = now;
    const next = index + steps;
    if (next >= data.time_s.length) { stop(); return; }
    frame(next, next % 10 === 0);
    requestAnimationFrame(tick);
  }

  function play() {
    if (!data) return;
    if (index >= data.time_s.length - 1) index = 0;
    playing = true;
    $('vs-play').textContent = '⏸ Dừng';
    lastFrame = performance.now();
    requestAnimationFrame(tick);
  }

  function stop() {
    playing = false;
    $('vs-play').textContent = '▶ Phát lại';
    if (data) frame(index, true);
  }

  $('vs-run').addEventListener('change', load);
  $('vs-play').addEventListener('click', () => playing ? stop() : play());
  $('vs-time').addEventListener('input', () => data && frame(+$('vs-time').value, true));
  for (const input of [$('vs-exaggeration')]) input.addEventListener('change', () => data && render());
  viewButtons($('page-vessel'), name => { view = name; if (data) render(); });

  return {init, plots: ['vs-scene', 'vs-loads', 'vs-motion'], leave: () => playing && stop()};
})();
