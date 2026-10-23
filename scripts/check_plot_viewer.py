"""Optional browser smoke check (requires playwright and installed Chrome)."""
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='chrome', headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1200}, device_scale_factor=1)
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto('http://127.0.0.1:8080', wait_until='networkidle')
        page.wait_for_function("document.getElementById('forecastNote').textContent.includes('100 mẫu')")
        assert not page.locator('body').evaluate("el=>el.classList.contains('dark')")
        assert 'TÀU CỐ ĐỊNH' in page.locator('#sceneTag').inner_text()
        page.locator('#tx').fill('0.01')
        assert 'vượt ngưỡng' in page.locator('#alertState').inner_text()
        page.locator('#tx').fill('100')
        page.locator('#truth').check()
        page.select_option('#stat', 'mean')
        page.select_option('#source', 'wave')
        page.locator('#scrub').fill('4000')
        assert page.locator('#count').inner_text().startswith('66 /')
        page.select_option('#axis', '2')
        page.locator('#play').click()
        page.wait_for_function("parseFloat(document.getElementById('scrub').value)>4005")
        page.locator('#play').click()
        page.select_option('#stat', 'rms')
        page.select_option('#source', 'environment')
        page.select_option('#axis', '1')
        page.locator('#reset').click()
        page.wait_for_timeout(300)
        Path('outputs/threejs').mkdir(exist_ok=True)
        page.screenshot(path='outputs/threejs/viewer.png', full_page=True)
        with page.expect_download() as download:
            page.locator('#save').click()
        download.value.save_as('outputs/threejs/loads-3d.png')
        page.locator('#theme').click()
        assert page.locator('body').evaluate("el=>el.classList.contains('dark')")
        page.locator('#theme').click()
        page.select_option('#dataset', 'outputs/force_training_v1/dataset/scenario_000.csv')
        page.wait_for_function("document.getElementById('status').textContent.includes('tập train')")
        assert 'Chưa có dự đoán' in page.locator('#alertState').inner_text()
        page.select_option('#dataset', 'outputs/loads_demo/load_features.csv')
        page.wait_for_function("document.getElementById('sceneTag').textContent.includes('REPLAY')")
        page.locator('#scrub').fill('1000')
        page.screenshot(path='outputs/threejs/vessel-replay.png', full_page=True)
        page.set_viewport_size({'width': 390, 'height': 844})
        page.wait_for_timeout(200)
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        page.screenshot(path='outputs/threejs/mobile.png', full_page=True)
        assert not errors, errors
        print('PASS: vessel/waves, archived forecast, no-forecast state, thresholds, replay, light/dark, timeline, playback, PNG and mobile layout; no page errors')
        browser.close()


if __name__ == '__main__':
    main()
