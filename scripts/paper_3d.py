"""Publication 3D figures (English labels, Arial) rendered with the viewer's Plotly code.

Data come from the same functions as the web viewer (scripts/wave_viewer.py): the
Welch-separated reflected spectra of every feature window, and the plan-view sea
surface around the moored tanker. Rendering runs in headless Edge/Chromium via
Playwright; panels are then assembled with matplotlib.
"""
import argparse
import base64
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import wave_viewer  # noqa: E402

plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                     "mathtext.it": "Arial:italic", "font.size": 8.5, "savefig.dpi": 300, "savefig.bbox": "tight"})
# Same approximate parula stops as viewer/common.js.
PARULA = [(0, "#352a87"), (.1, "#0363e1"), (.2, "#1485d4"), (.3, "#06a7c6"), (.4, "#38b99e"), (.5, "#92bf73"),
          (.6, "#d9ba56"), (.7, "#fcc238"), (.8, "#f5d72a"), (.9, "#f5ec1c"), (1, "#f9fb0e")]
# view(-37.5, 30); the camera sits a little further out so the front corner is not clipped.
MATLAB_EYE = {"x": -1.107 * 1.3, "y": -1.443 * 1.3, "z": 1.05 * 1.3}

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<script src="{plotly}"></script><script src="{common}"></script></head>
<body style="margin:0;background:#fff"><div id="plot" style="width:{w}px;height:{h}px"></div>
<script>
const spec = {spec};
{build}
Plotly.newPlot('plot', traces, layout, {{staticPlot: true}}).then(() =>
  Plotly.toImage('plot', {{format: 'png', width: {w}, height: {h}, scale: 3}})).then(url => {{ window.png = url; }});
</script></body></html>"""

SPECTRUM_JS = """
// Clamp to the colour floor: values below it would otherwise be cut out of the axis range as holes.
const z = spec.psd.map(row => row.map(v => v === null ? null : Math.max(Math.log10(Math.max(v, 1e-8)), spec.cmin)));
const axis = {showbackground: true, backgroundcolor: '#fff', gridcolor: '#d9d9d9', linecolor: '#262626',
              showline: true, ticks: 'outside', tickfont: {size: 17}, title: {font: {size: 20}}};
const traces = [{type: 'surface', x: spec.f, y: spec.t, z, colorscale: spec.colorscale, cmin: spec.cmin,
                 cmax: spec.cmax, showscale: false, lighting: {ambient: .75, diffuse: .6, specular: .1}}];
const layout = {font: {family: 'Arial', size: 13, color: '#000'}, margin: {l: 0, r: 0, t: 0, b: 0},
  scene: {xaxis: {...axis, title: {text: 'f (Hz)', font: {size: 20}}}, yaxis: {...axis, title: {text: 't (min)', font: {size: 20}}},
          zaxis: {...axis, title: {text: 'log<sub>10</sub> S<sub>R</sub>', font: {size: 20}}, range: [spec.cmin, spec.cmax + .3]},
          aspectmode: 'manual', aspectratio: {x: 1.2, y: 1.5, z: .65}, camera: {eye: spec.eye}}};
"""

SEA_JS = """
const eta = spec.frame;
const axis = {showbackground: true, backgroundcolor: '#fff', gridcolor: '#d9d9d9', linecolor: '#262626',
              showline: true, ticks: 'outside', tickfont: {size: 12}, titlefont: {size: 14}};
const g = spec.ship;
const ship = shipTraces(g, {x: g.x_m, y: g.y_m, psi: g.heading_rad});
ship[0].name = 'Tanker (Wigley hull, true scale)';
const yLo = spec.y[0], yHi = spec.y[spec.y.length - 1], xLo = spec.x[0];
const zLo = -g.draft_m - 4, zHi = g.freeboard_m * 3.5 + 4;
const traces = [
  {type: 'surface', name: 'Sea surface', x: spec.x, y: spec.y, z: eta.map(r => r.map(v => v * spec.exaggeration)),
   surfacecolor: eta, colorscale: spec.colorscale, cmin: -spec.limit, cmax: spec.limit,
   colorbar: {title: {text: 'η (m)', font: {size: 14}}, tickfont: {size: 12}, thickness: 16, len: .6, x: .93},
   lighting: {ambient: .7, diffuse: .6, specular: .15}},
  ...ship,
  {type: 'surface', name: 'Quay wall (x = 0)', x: [[0, 0], [0, 0]], y: [[yLo, yHi], [yLo, yHi]],
   z: [[zLo, zLo], [g.freeboard_m + 4, g.freeboard_m + 4]], showscale: false, opacity: .85,
   colorscale: [[0, '#8c8c8c'], [1, '#8c8c8c']], showlegend: true},
  {type: 'scatter3d', mode: 'markers', name: 'Wave probes', x: spec.probes, y: spec.probes.map(() => 0),
   z: spec.probes.map(() => 1), marker: {color: '#a2142f', size: 4}},
];
const span = Math.max(0 - xLo, yHi - yLo);
const layout = {font: {family: 'Arial', size: 13, color: '#000'}, margin: {l: 0, r: 0, t: 0, b: 0},
  legend: {x: .02, y: .98, font: {size: 13}, bgcolor: 'rgba(255,255,255,.85)'}, showlegend: true,
  scene: {xaxis: {...axis, title: {text: 'x (m), toward wall'}, range: [xLo, 0]},
          yaxis: {...axis, title: {text: 'y along quay (m)'}, range: [yLo, yHi]},
          zaxis: {...axis, title: {text: 'z (m)'}, range: [zLo, zHi]}, aspectmode: 'manual',
          aspectratio: {x: -xLo / span * 1.6, y: (yHi - yLo) / span * 1.6, z: 3 * (zHi - zLo) / span * 1.6},
          camera: {eye: spec.eye}}};
"""


def render(pages, out_dir):
    """Render each (name, html) page and return the PNG paths."""
    from playwright.sync_api import sync_playwright
    paths = []
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="msedge")
        except Exception:  # fall back to Playwright's own Chromium
            browser = p.chromium.launch()
        page = browser.new_page()
        for name, html in pages:
            source = out_dir / f"{name}.html"
            source.write_text(html, encoding="utf-8")
            page.goto(source.as_uri())
            page.wait_for_function("window.png !== undefined", timeout=120000)
            png = out_dir / f"{name}.png"
            png.write_bytes(base64.b64decode(page.evaluate("window.png").split(",", 1)[1]))
            source.unlink()
            paths.append(png)
        browser.close()
    return paths


def trim(image, pad=8):
    """Crop white margins from an RGBA/RGB array."""
    mask = (image[..., :3] < .98).any(axis=2)
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    return image[max(rows[0] - pad, 0):rows[-1] + pad, max(cols[0] - pad, 0):cols[-1] + pad]


def page(build, spec, w, h):
    return PAGE.format(plotly=(ROOT / "viewer/vendor/plotly.min.js").as_uri(), common=(ROOT / "viewer/common.js").as_uri(),
                       spec=json.dumps(spec), build=build, w=w, h=h)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="v3_data", help="dataset run name as listed by the viewer catalog")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/paper_results")
    args = parser.parse_args()
    out = args.output
    work = out / "_3d"
    work.mkdir(parents=True, exist_ok=True)
    sources = wave_viewer.catalog()
    tests = [key for key, s in sources.items() if s["run"] == args.run and s["split"] == "test"]
    tables = out / "tables.json"
    if tables.exists():
        # Same representative subset as the 2D test-scenario figure.
        selected = set(json.loads(tables.read_text(encoding="utf-8")).get("selected_tests") or [])
        tests = [key for key in tests if not selected or int(key.split("_")[-1]) in selected]
    spectra = {key: wave_viewer.spectrogram(key) for key in tests}
    logs = [np.log10(np.maximum(np.array(s["reflected_psd"], dtype=float), 1e-8)) for s in spectra.values()]
    cmax = float(np.nanmax([np.nanmax(v) for v in logs]))
    cmin = cmax - 4.5
    pages = []
    for key, s in spectra.items():
        spec = {"f": s["frequency_hz"], "t": s["time_min"], "psd": s["reflected_psd"], "cmin": cmin, "cmax": cmax,
                "colorscale": [[a, c] for a, c in PARULA], "eye": MATLAB_EYE}
        pages.append((key.replace("/", "_"), page(SPECTRUM_JS, spec, 700, 560)))
    # Sea surface at the reflected-wave peak of the most energetic test scenario shown.
    sea_key = max(spectra, key=lambda k: np.nanmax(np.array(spectra[k]["hm0_truth"], dtype=float)))
    peak = spectra[sea_key]["time_min"][int(np.nanargmax(np.array(spectra[sea_key]["hm0_truth"], dtype=float)))]
    sea = wave_viewer.seastate(sea_key, {"t0": [str(60 * peak)], "frames": ["2"], "n": ["180"]})
    frame = np.array(sea["frames"][0])
    pages.append(("sea_ship", page(SEA_JS, {
        "x": sea["x"], "y": sea["y"], "frame": sea["frames"][0], "ship": sea["ship"], "probes": sea["probe_x_m"],
        "limit": float(np.abs(frame).max()), "exaggeration": 5, "eye": MATLAB_EYE,
        "colorscale": [[0, "#08306b"], [.25, "#4292c6"], [.5, "#f7f7f7"], [.75, "#ef6548"], [1, "#7f0000"]]}, 1100, 700)))
    rendered = dict(zip([name for name, _ in pages], render(pages, work)))

    # Composite of the eight test spectra with one shared colour bar.
    keys = list(spectra)
    figure, axes = plt.subplots(4, 2, figsize=(6.5, 9.4), gridspec_kw={"hspace": .22, "wspace": .04})
    for n, (axis, key) in enumerate(zip(axes.flat, keys)):
        s = sources[key]["config"]["waves"]
        axis.imshow(trim(plt.imread(rendered[key.replace("/", "_")])))
        axis.set_axis_off()
        sid = key.split("_")[-1]
        axis.set_title(f"({'abcdefgh'[n]}) Scenario {sid}\n$H_s$ = {s['hs_m']:.2f} m, $T_p$ = {s['tp_s']:.1f} s, "
                       f"$K_r$ = {s['reflection']:.2f}", fontsize=8, loc="left")
    colormap = LinearSegmentedColormap.from_list("parula", PARULA)
    bar = figure.colorbar(plt.cm.ScalarMappable(Normalize(cmin, cmax), colormap), ax=axes, orientation="horizontal",
                          fraction=.025, pad=.02, aspect=50)
    bar.set_label(r"$\log_{10} S_R$ (m$^2$/Hz), reflected spectral density from three-probe Welch separation")
    figure.savefig(out / "fig02_test_spectra_3d.png")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.imshow(trim(plt.imread(rendered["sea_ship"])))
    axis.set_axis_off()
    figure.savefig(out / "fig00_berth_sea_ship_3d.png")
    plt.close(figure)
    for png in rendered.values():
        png.unlink()
    work.rmdir()
    summary = {"spectra_scenarios": keys, "log10_range": [cmin, cmax], "sea_scenario": sea_key, "sea_time_min": peak,
               "sea_kept_variance": sea["kept_variance_fraction"], "sea_min_wavelength_m": sea["min_wavelength_m"]}
    (out / "figures_3d.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
