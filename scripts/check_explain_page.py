"""Browser smoke check of the explanation page, using local Chrome."""
from pathlib import Path
from playwright.sync_api import sync_playwright


with sync_playwright() as p:
    browser = p.chromium.launch(channel='chrome', headless=True)
    page = browser.new_page(viewport={'width': 1440, 'height': 1100})
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.goto('http://127.0.0.1:8080/explain.html', wait_until='networkidle')
    page.wait_for_function("document.getElementById('status').textContent.includes('408 cửa sổ')")
    assert page.locator('.matrix').count() == 4
    page.locator('#train').click()
    page.locator('#flow button').nth(2).click()
    assert 'logit' in page.locator('#stepDetail').inner_text()
    page.locator('#infer').click()
    page.locator('[data-axis="1"][data-cell="fn"]').click()
    page.wait_for_function("document.getElementById('caseInfo').textContent.includes('FN (bỏ sót)')")
    assert page.locator('#case option').count() == 49
    Path('outputs/threejs').mkdir(exist_ok=True)
    page.screenshot(path='outputs/threejs/explain.png', full_page=True)
    page.locator('[data-axis="0"][data-cell="fp"]').click()
    assert page.locator('#case').is_disabled()
    page.locator('#clear').click()
    assert page.locator('#case option').count() == 408
    page.select_option('#model', 'energy_lstm')
    page.select_option('#h', '10')
    page.locator('#p').fill('0.3')
    page.locator('#apply').click()
    page.wait_for_function("document.getElementById('status').textContent.includes('Energy–LSTM')")
    assert '10 phút' in page.locator('#status').inner_text()
    page.set_viewport_size({'width': 390, 'height': 844})
    page.wait_for_timeout(200)
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.screenshot(path='outputs/threejs/explain-mobile.png', full_page=True)
    assert not errors, errors
    print('PASS: matrices, architecture tabs, cell drill-down, empty cell, model/threshold/horizon, mobile layout')
    browser.close()
