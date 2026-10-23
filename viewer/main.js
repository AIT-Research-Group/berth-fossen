'use strict';

// Tab switching. Each tab loads its data the first time it is shown.
(() => {
  const TABS = {wave: WaveTab, spectrum: SpectrumTab, sea: SeaTab, vessel: VesselTab, forecast: ForecastTab};
  const started = new Set();
  const statusIds = {wave: 'status', spectrum: 'sp-status', sea: 'se-status', vessel: 'vs-status', forecast: 'fc-status'};
  let active = null;

  function show(name) {
    if (active && TABS[active].leave) TABS[active].leave();
    active = name;
    for (const button of document.querySelectorAll('[data-tab]')) {
      button.setAttribute('aria-selected', String(button.dataset.tab === name));
    }
    for (const key of Object.keys(TABS)) $(`page-${key}`).hidden = key !== name;
    try { localStorage.setItem('berth-viewer-tab', name); } catch (error) { /* storage may be blocked */ }
    if (!started.has(name)) {
      started.add(name);
      Promise.resolve(TABS[name].init()).catch(error => setStatus(statusIds[name], error.message, 'error'));
    } else {
      // Plots drawn while hidden have zero size; resize them now they are visible.
      for (const id of TABS[name].plots) if ($(id).data) Plotly.Plots.resize(id);
    }
  }

  for (const button of document.querySelectorAll('[data-tab]')) {
    button.addEventListener('click', () => show(button.dataset.tab));
  }
  let initial = 'wave';
  try { initial = localStorage.getItem('berth-viewer-tab') || 'wave'; } catch (error) { /* default tab */ }
  const requested = new URLSearchParams(location.search).get('tab');
  show(TABS[requested] ? requested : (TABS[initial] ? initial : 'wave'));
})();
