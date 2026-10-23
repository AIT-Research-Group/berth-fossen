'use strict';

// Tab: plan-view sea surface eta(x, y, t) from the scenario spectrum, with the moored ship at true scale.
const SeaTab = (() => {
  let data = null;
  let frame = 0;
  let playing = false;
  let lastTime = 0;
  let view = '3d';

  const status = (text, kind) => setStatus('se-status', text, kind);
  const component = () => document.querySelector('input[name=se-component]:checked').value;
  const etaScale = () => +$('se-eta').value || 1;

  async function init() {
    const sources = await getJSON('/api/sources');
    const select = $('se-source');
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
    select.value = sources.some(s => s.id === 'demo') ? 'demo' : sources[0].id;
    await load();
  }

  async function load() {
    stop();
    const query = new URLSearchParams({source: $('se-source').value, component: component(), t0: $('se-t0').value,
                                       step: $('se-step').value, frames: $('se-frames').value, n: $('se-n').value});
    status('Đang tính mặt biển quanh tàu…');
    $('se-load').disabled = true;
    try {
      data = await getJSON(`/api/seastate?${query}`);
      data.limit = finiteMax(data.frames.flat());
      data.lineLimit = finiteMax(data.centerline);
      frame = 0;
      $('se-frame').max = data.t.length - 1;
      $('se-frame').value = 0;
      render();
      info();
      status(`Lưới ${data.x.length} × ${data.y.length}, ${data.t.length} khung hình. Sóng ngắn hơn ${data.min_wavelength_m.toFixed(1)} m không vẽ được trên lưới này (giữ ${(100 * data.kept_variance_fraction).toFixed(0)}% phương sai).`);
    } catch (error) {
      status(`Lỗi: ${error.message}`, 'error');
    } finally {
      $('se-load').disabled = false;
    }
  }

  function pose() {
    return {x: data.ship.x_m, y: data.ship.y_m, psi: data.ship.heading_rad};
  }

  function surfaceTrace(i) {
    const eta = data.frames[i];
    return {type: 'surface', name: 'Mặt biển', x: data.x, y: data.y, z: eta.map(row => row.map(v => v * etaScale())),
            surfacecolor: eta, colorscale: COLORMAPS[$('se-colormap').value], cmin: -data.limit, cmax: data.limit,
            colorbar: {title: {text: 'η (m)'}, thickness: 14, len: .7},
            lighting: {ambient: .7, diffuse: .6, specular: .15, roughness: .6},
            hovertemplate: 'x = %{x:.0f} m<br>y = %{y:.0f} m<br>η = %{surfacecolor:.2f} m<extra></extra>'};
  }

  function render() {
    const g = data.ship;
    const [xLo, xHi] = [data.x[0], data.x[data.x.length - 1]];
    const [yLo, yHi] = [data.y[0], data.y[data.y.length - 1]];
    const zLo = -g.draft_m - 4, zHi = g.freeboard_m * 3.5 + 4;
    const traces = [
      surfaceTrace(frame),
      ...shipTraces(g, pose()),
      {type: 'surface', name: 'Tường cầu (x = 0)', x: [[0, 0], [0, 0]], y: [[yLo, yHi], [yLo, yHi]],
       z: [[zLo, zLo], [g.freeboard_m + 4, g.freeboard_m + 4]], showscale: false, opacity: .85,
       colorscale: [[0, '#8c8c8c'], [1, '#8c8c8c']], hoverinfo: 'skip', showlegend: true},
      {type: 'scatter3d', mode: 'markers', name: 'Đầu đo sóng', x: data.probe_x_m,
       y: data.probe_x_m.map(() => 0), z: data.probe_x_m.map(() => 1), marker: {color: COLORS.probe, size: 4}},
    ];
    // Vertical exaggeration of the whole scene, like MATLAB daspect; ticks stay in true metres.
    const span = Math.max(xHi - xLo, yHi - yLo);
    const vertical = +$('se-vertical').value || 1;
    Plotly.react('se-scene', traces, {
      margin: {l: 0, r: 0, t: 30, b: 0}, uirevision: view, legend: {x: 0, y: 1},
      title: {text: `Mặt biển quanh tàu — ${data.source.id}, t = ${data.t[frame].toFixed(1)} s`, font: {size: 14}},
      scene: {
        xaxis: {...AXIS, title: {text: 'x (m) → tường'}, range: [xLo, xHi]},
        yaxis: {...AXIS, title: {text: 'y dọc cầu (m)'}, range: [yLo, yHi]},
        zaxis: {...AXIS, title: {text: view === 'top' ? '' : 'z (m)'}, range: [zLo, zHi], showticklabels: view !== 'top'},
        aspectmode: 'manual',
        aspectratio: {x: (xHi - xLo) / span * 1.6, y: (yHi - yLo) / span * 1.6, z: vertical * (zHi - zLo) / span * 1.6},
        camera: VIEWS[view],
      },
    }, CONFIG);
    renderSpectrum();
    renderCenterline();
  }

  function renderSpectrum() {
    const peak = Math.max(...data.spectrum);
    const marker = (f, color, dash) => [
      {type: 'line', xref: 'x', yref: 'paper', x0: f, x1: f, y0: 0, y1: 1, line: {color, width: 1.5, dash}},
    ];
    const shapes = [
      // Waves longer than the ship: the band that can pitch or heave a hull of this length as a whole.
      {type: 'rect', xref: 'x', yref: 'paper', x0: 0, x1: data.f_ship_length_hz, y0: 0, y1: 1,
       fillcolor: 'rgba(0,114,189,.12)', line: {width: 0}},
      // Waves shorter than 3 grid cells, missing from the 3D surface.
      {type: 'rect', xref: 'x', yref: 'paper', x0: data.f_min_drawn_hz, x1: data.spectrum_f_hz.at(-1), y0: 0, y1: 1,
       fillcolor: 'rgba(128,128,128,.15)', line: {width: 0}},
      ...marker(data.f_ship_length_hz, COLORS.total, 'solid'),
      ...marker(data.f_ship_beam_hz, COLORS.reflected, 'dash'),
      ...marker(data.f_peak_hz, COLORS.incident, 'dot'),
    ];
    const g = data.ship;
    const label = (x, text, y) => ({x, y, xref: 'x', yref: 'paper', text, showarrow: false, xanchor: 'left',
                                    font: {size: 11}, bgcolor: 'rgba(255,255,255,.8)'});
    Plotly.react('se-spectrum', [
      {x: data.spectrum_f_hz, y: data.spectrum, name: 'JONSWAP thiết kế', line: {color: COLORS.design, width: 2.5},
       fill: 'tozeroy', fillcolor: 'rgba(126,47,142,.12)'},
    ], {
      title: {text: 'Phổ sóng so với kích thước tàu', font: {size: 13}},
      margin: {l: 60, r: 15, t: 35, b: 50}, showlegend: false,
      xaxis: axis2d('f (Hz)', {range: [0, data.spectrum_f_hz.at(-1)]}),
      yaxis: axis2d('S (m²/Hz)', {range: [0, peak * 1.15]}), shapes,
      annotations: [
        label(data.f_ship_length_hz, `λ = L = ${g.length_m.toFixed(0)} m`, .97),
        label(data.f_ship_beam_hz, `λ = B = ${g.beam_m.toFixed(0)} m`, .86),
        label(data.f_peak_hz, `fp: λp = ${data.peak_wavelength_m.toFixed(0)} m`, .75),
        label(data.f_min_drawn_hz, `không vẽ trên lưới (λ < ${data.min_wavelength_m.toFixed(0)} m)`, .6),
      ],
    }, CONFIG);
  }

  // Two stacked profiles: along the keel line, and athwartships from the ship to the wall.
  function renderCenterline() {
    const g = data.ship;
    const limit = Math.max(data.lineLimit, finiteMax(data.transverse)) * 1.1;
    const hull = (xref, yref, x0, x1) => ({type: 'rect', xref, yref, x0, x1, y0: -limit, y1: limit,
                                            fillcolor: 'rgba(60,74,87,.12)', line: {color: '#3c4a57', width: 1}});
    Plotly.react('se-centerline', [
      {x: data.centerline_s, y: data.centerline[frame], name: 'Dọc trục tàu', line: {color: COLORS.total, width: 2}},
      {x: data.transverse_r, y: data.transverse[frame], name: 'Ngang tàu → tường', xaxis: 'x2', yaxis: 'y2',
       line: {color: COLORS.incident, width: 2}},
    ], {
      title: {text: profileTitle(), font: {size: 13}, y: .98, yanchor: 'top'},
      margin: {l: 60, r: 15, t: 40, b: 50}, showlegend: false,
      grid: {rows: 2, columns: 1, pattern: 'independent', ygap: .45},
      xaxis: axis2d('s dọc tàu (m), mũi phía +'), yaxis: axis2d('η (m)', {range: [-limit, limit]}),
      xaxis2: axis2d('r ngang tàu (m), 0 = tâm tàu, tường ở phía +'), yaxis2: axis2d('η (m)', {range: [-limit, limit]}),
      shapes: [
        hull('x', 'y', -g.length_m / 2, g.length_m / 2),
        hull('x2', 'y2', -g.beam_m / 2, g.beam_m / 2),
        {type: 'line', xref: 'x2', yref: 'y2', x0: data.wall_r, x1: data.wall_r, y0: -limit, y1: limit,
         line: {color: '#404040', width: 3}},
      ],
      annotations: [
        {x: g.length_m / 2, y: limit, xref: 'x', yref: 'y', text: 'mũi', showarrow: false, yanchor: 'bottom'},
        {x: -g.length_m / 2, y: limit, xref: 'x', yref: 'y', text: 'lái', showarrow: false, yanchor: 'bottom'},
        {x: data.wall_r, y: limit, xref: 'x2', yref: 'y2', text: 'tường', showarrow: false, yanchor: 'bottom', xanchor: 'right'},
        {x: 0, y: limit, xref: 'x2', yref: 'y2', text: `thân tàu B = ${g.beam_m.toFixed(0)} m`, showarrow: false, yanchor: 'bottom'},
      ],
    }, CONFIG);
  }

  function profileTitle() {
    return `η dọc và ngang thân tàu, t = ${data.t[frame].toFixed(1)} s (không cắt bước sóng)`;
  }

  function show(i) {
    frame = i;
    $('se-frame').value = i;
    $('se-frame-value').textContent = `${data.t[i].toFixed(1)} s`;
    const eta = data.frames[i];
    Plotly.restyle('se-scene', {z: [eta.map(row => row.map(v => v * etaScale()))], surfacecolor: [eta]}, [0]);
    Plotly.relayout('se-scene', {'title.text': `Mặt biển quanh tàu — ${data.source.id}, t = ${data.t[i].toFixed(1)} s`});
    Plotly.restyle('se-centerline', {y: [data.centerline[i], data.transverse[i]]}, [0, 1]);
    Plotly.relayout('se-centerline', {'title.text': profileTitle()});
  }

  function info() {
    const g = data.ship;
    const rows = [
      ['Tàu (L × B × T)', `${g.length_m.toFixed(0)} × ${g.beam_m.toFixed(0)} × ${g.draft_m.toFixed(1)} m`],
      ['Tỉ lệ thân', `theo tàu dầu MSS (${g.proportions_from})`],
      ['Hướng mũi', `${(g.heading_rad * 180 / Math.PI).toFixed(0)}° (dọc cầu)`],
      ['Khoảng cách tới tường', `${(-g.x_m - g.beam_m / 2).toFixed(0)} m (mạn tới tường)`],
      ['Bước sóng đỉnh λp', `${data.peak_wavelength_m.toFixed(1)} m`],
      ['L / λp', (g.length_m / data.peak_wavelength_m).toFixed(1)],
      ['Hs thiết kế, Tp', `${data.source.hs_m.toFixed(2)} m, ${data.source.tp_s.toFixed(2)} s`],
      ['Bao biên bão', `${Math.min(...data.envelope).toFixed(2)}–${Math.max(...data.envelope).toFixed(2)}`],
    ];
    $('se-info').replaceChildren(...rows.flatMap(([k, v]) => {
      const dt = document.createElement('dt'); dt.textContent = k;
      const dd = document.createElement('dd'); dd.textContent = v;
      return [dt, dd];
    }));
  }

  function tick(now) {
    if (!playing) return;
    const step = data.t[1] - data.t[0];
    // Real-time playback: advance one frame per `step` seconds of wall time.
    if (now - lastTime >= step * 1000 / +$('se-speed').value) {
      lastTime = now;
      show((frame + 1) % data.t.length);
    }
    requestAnimationFrame(tick);
  }

  function play() {
    if (!data) return;
    playing = true;
    $('se-play').textContent = '⏸ Dừng';
    lastTime = performance.now();
    requestAnimationFrame(tick);
  }

  function stop() {
    playing = false;
    $('se-play').textContent = '▶ Phát';
  }

  $('se-load').addEventListener('click', load);
  $('se-source').addEventListener('change', load);
  for (const input of document.querySelectorAll('input[name=se-component]')) input.addEventListener('change', load);
  $('se-play').addEventListener('click', () => playing ? stop() : play());
  $('se-frame').addEventListener('input', () => data && show(+$('se-frame').value));
  for (const input of [$('se-eta'), $('se-vertical'), $('se-colormap')]) {
    input.addEventListener('change', () => data && render());
  }
  viewButtons($('page-sea'), name => { view = name; if (data) render(); });

  return {init, plots: ['se-scene', 'se-spectrum', 'se-centerline'], leave: () => playing && stop()};
})();
