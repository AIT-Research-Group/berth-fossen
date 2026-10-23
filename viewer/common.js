'use strict';

// Approximate MATLAB colormaps (parula sampled at 11 points).
const COLORMAPS = {
  parula: [[0, '#352a87'], [.1, '#0363e1'], [.2, '#1485d4'], [.3, '#06a7c6'], [.4, '#38b99e'],
           [.5, '#92bf73'], [.6, '#d9ba56'], [.7, '#fcc238'], [.8, '#f5d72a'], [.9, '#f5ec1c'], [1, '#f9fb0e']],
  jet: [[0, '#00007f'], [.125, '#0000ff'], [.375, '#00ffff'], [.625, '#ffff00'], [.875, '#ff0000'], [1, '#7f0000']],
  viridis: 'Viridis',
  balance: [[0, '#08306b'], [.25, '#4292c6'], [.5, '#f7f7f7'], [.75, '#ef6548'], [1, '#7f0000']],
};
// MATLAB line colors, in default order.
const COLORS = {total: '#0072bd', incident: '#d95319', reflected: '#77ac30', probe: '#a2142f',
                wind: '#0072bd', wave: '#d95319', sum: '#000000', design: '#7e2f8e', truth: '#000000'};
const AXIS = {showbackground: true, backgroundcolor: '#ffffff', gridcolor: '#d9d9d9',
              zerolinecolor: '#8c8c8c', linecolor: '#262626', showline: true, ticks: 'outside'};
const CONFIG = {responsive: true, displaylogo: false,
                toImageButtonOptions: {format: 'png', scale: 3, filename: 'berth_viewer'}};

const $ = id => document.getElementById(id);



// MATLAB view(az, el) -> Plotly camera eye; az measured from -y toward +x.
function matlabEye(az, el, distance = 2.1) {
  const a = az * Math.PI / 180, e = el * Math.PI / 180;
  return {x: distance * Math.sin(a) * Math.cos(e), y: -distance * Math.cos(a) * Math.cos(e), z: distance * Math.sin(e)};
}

const VIEWS = {
  '3d': {eye: matlabEye(-37.5, 30), up: {x: 0, y: 0, z: 1}, projection: {type: 'perspective'}},
  // Tiny offsets keep the eye off an axis, where the orbit is degenerate.
  top: {eye: {x: 1e-4, y: -1e-4, z: 2.4}, up: {x: 0, y: 1, z: 0}, projection: {type: 'orthographic'}},
  side: {eye: {x: 1e-4, y: -2.4, z: 1e-4}, up: {x: 0, y: 0, z: 1}, projection: {type: 'orthographic'}},
};

function setStatus(id, text, kind = '') {
  $(id).textContent = text;
  $(id).className = `status ${kind}`;
}

async function getJSON(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error((await response.text()) || `${response.status} ${response.statusText}`);
  return response.json();
}

function fillSelect(select, items, value = item => item, label = item => item) {
  select.replaceChildren(...items.map(item => new Option(label(item), value(item))));
}

// Boxed 2D axes like a MATLAB figure.
function axis2d(title, extra = {}) {
  return {title: {text: title}, mirror: true, showline: true, linecolor: '#262626', ticks: 'outside',
          gridcolor: '#e5e5e5', zeroline: false, ...extra};
}

function finiteMax(rows) {
  let best = 0;
  for (const row of rows) for (const v of row) if (v !== null && Math.abs(v) > best) best = Math.abs(v);
  return best || 1e-6;
}

// Wigley hull y = (B/2)(1 - (2x/L)^2)(1 - (z/T)^2), the standard analytic test hull,
// with vertical topsides to the deck and an aft accommodation block. Body axes: x forward, y to port.
function shipGeometry(g, stations = 41, levels = 8) {
  const L = g.length_m, B = g.beam_m, T = g.draft_m, F = g.freeboard_m;
  const rows = [];
  for (let r = 0; r <= levels; r++) {
    const zeta = 1 - r / levels;
    rows.push({z: -T * zeta, width: 1 - zeta * zeta});
  }
  rows.push({z: F, width: 1});
  const hull = {x: [], y: [], z: [], i: [], j: [], k: [], color: []};
  const index = (side, r, c) => side * rows.length * stations + r * stations + c;
  for (const sign of [1, -1]) {
    for (const row of rows) {
      for (let c = 0; c < stations; c++) {
        const xi = -1 + 2 * c / (stations - 1);
        hull.x.push(xi * L / 2);
        hull.y.push(sign * B / 2 * (1 - xi * xi) * row.width);
        hull.z.push(row.z);
        // Antifouling red below the waterline, dark grey topsides.
        hull.color.push(row.z < 0 ? '#8b2a1f' : '#3c4a57');
      }
    }
  }
  const quad = (a, b, c, d) => { hull.i.push(a, b); hull.j.push(b, d); hull.k.push(c, c); };
  for (let side = 0; side < 2; side++) {
    for (let r = 0; r < rows.length - 1; r++) {
      for (let c = 0; c < stations - 1; c++) {
        quad(index(side, r, c), index(side, r, c + 1), index(side, r + 1, c), index(side, r + 1, c + 1));
      }
    }
  }
  const deck = rows.length - 1;
  for (let c = 0; c < stations - 1; c++) {
    quad(index(0, deck, c), index(0, deck, c + 1), index(1, deck, c), index(1, deck, c + 1));
  }
  // Accommodation block aft, where the Wigley hull is still wide enough to carry it.
  const x0 = -0.36 * L, x1 = -0.26 * L, halfWidth = 0.22 * B, top = F + 2.5 * F;
  const box = {x: [], y: [], z: [], i: [], j: [], k: []};
  for (const bx of [x0, x1]) for (const by of [-halfWidth, halfWidth]) for (const bz of [F, top]) {
    box.x.push(bx); box.y.push(by); box.z.push(bz);
  }
  return {hull, box};
}

// Place body-axis vertices at a planar pose (x, y, heading).
function placeShip(body, pose) {
  const c = Math.cos(pose.psi), s = Math.sin(pose.psi);
  return {x: body.x.map((bx, n) => pose.x + c * bx - s * body.y[n]),
          y: body.x.map((bx, n) => pose.y + s * bx + c * body.y[n]), z: body.z};
}

function shipTraces(geometry, pose) {
  const {hull, box} = shipGeometry(geometry);
  return [
    {type: 'mesh3d', name: 'Tàu (thân Wigley)', ...placeShip(hull, pose), i: hull.i, j: hull.j, k: hull.k,
     vertexcolor: hull.color, flatshading: false, lighting: {ambient: .6, diffuse: .8, specular: .2},
     hoverinfo: 'skip', showlegend: true},
    {type: 'mesh3d', name: 'Thượng tầng', ...placeShip(box, pose), alphahull: 0, color: '#f2f2f2',
     flatshading: true, hoverinfo: 'skip', showlegend: false},
  ];
}

function viewButtons(container, onView) {
  for (const button of container.querySelectorAll('[data-view]')) {
    button.addEventListener('click', () => onView(button.dataset.view));
  }
}
