#!/usr/bin/env python3
"""Validate model-gated LoRA controls while preserving TensorAgent's side menu.

Run TensorAgentHost with no model first. Catalog/sidebar checks use the real HTTP
API. Image-model controls use clearly labeled fixtures without model downloads or
inference. Requires playwright and a Chromium executable. Outputs belong in ignored
artifacts/ or docs/validation/ directories.
"""
import argparse
import copy
import json
from pathlib import Path
import statistics
import time

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connection', type=Path, required=True)
    parser.add_argument('--browser', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=5)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    connection = json.loads(args.connection.read_text(encoding='utf-8-sig'))
    report = {'status': 'failed', 'surfaces': [], 'limitations': [
        'Real AgentAppHost catalog/sidebar in headless Chromium; native MAUI and iOS WebKit are not exercised.',
        'Loaded-model and adapter controls use HTTP fixtures; no inference quality or device throughput is measured.',
        'Timings include browser automation; these are repeated UX latency samples, not inference benchmarks.',
    ]}
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=args.browser, headless=True)
            try:
                for name, width, height, touch in [('desktop', 1280, 900, False), ('touch-390', 390, 844, True), ('touch-320', 320, 568, True)]:
                    context = browser.new_context(viewport={'width': width, 'height': height}, has_touch=touch, is_mobile=touch)
                    page = context.new_page()
                    errors, lora_requests = [], []
                    page.on('pageerror', lambda error: errors.append(str(error)))
                    page.on('request', lambda request: lora_requests.append(request.url) if '/api/agent/loras' in request.url else None)
                    page.goto(connection['entryUrl'])
                    expect(page.locator('#send')).to_be_disabled()
                    expect(page.locator('#composer-loras')).to_be_hidden()
                    catalog_response = page.request.get(connection['baseUrl'] + '/api/agent/loras')
                    assert catalog_response.ok
                    catalog = catalog_response.json()
                    assert not catalog['supportsLoras'] and not catalog['loadedModelIsImage']
                    assert catalog['loras']
                    page.locator('#model').click()
                    expect(page.locator('#open-loras')).to_be_hidden()
                    page.locator('#sheet-bg').click(position={'x': width - 5, 'y': 5})
                    page.locator('#menu').click()
                    expect(page.locator('#nav-chats')).to_be_visible()
                    routes = page.locator('#nav-sheet [data-route]').evaluate_all('(items) => items.map(e => e.getAttribute("data-route"))')
                    assert routes == ['sessions', 'models', 'settings', 'about'], routes
                    expect(page.locator('#nav-sheet [data-sheet="skills-sheet"]')).to_be_visible()
                    expect(page.locator('#nav-new')).to_be_visible()
                    assert page.locator('#nav-loras').count() == 0, 'Do not modify the side menu.'
                    page.locator('#sheet-bg').click(position={'x': width - 5, 'y': 5})

                    # Explicit fixture: real catalog data, a supported image model, and
                    # installed Film Stills. The host has no model/adapter weights.
                    fixture = copy.deepcopy(catalog)
                    fixture.update({'loadedModel': 'qwen-image-2.1-q4km', 'loadedModelName': 'Qwen-Image 2.1',
                                    'loadedModelFile': 'qwen.gguf', 'loadedModelIsImage': True, 'supportsLoras': True,
                                    'chosen': [{'id': 'qwen-image-2.1-film-stills', 'strength': .7}]})
                    film = next(row for row in fixture['loras'] if row['id'] == 'qwen-image-2.1-film-stills')
                    film.update({'state': 'Installed', 'chosen': True, 'strength': .7})
                    model = {'loaded': 'qwen.gguf', 'architecture': 'qwen_image'}
                    page.route('**/api/models', lambda route: route.fulfill(json=model))
                    page.route('**/api/agent/loras', lambda route: route.fulfill(json=fixture))
                    page.reload()
                    shortcut = page.locator('#composer-loras')
                    expect(shortcut).to_be_visible()
                    expect(shortcut).to_contain_text('Film Stills')
                    target = shortcut.bounding_box()
                    assert target['height'] >= 44 and 0 <= target['x'] and target['x'] + target['width'] <= width
                    times = []
                    for _ in range(args.repetitions):
                        started = time.perf_counter()
                        shortcut.click()
                        expect(page.get_by_role('dialog', name='LoRA plug-ins', exact=True)).to_be_visible()
                        expect(page.locator('#lora-list')).to_have_attribute('aria-busy', 'false')
                        expect(page.locator('.lorarow').first).to_be_visible()
                        times.append((time.perf_counter() - started) * 1000)
                        page.locator('#lora-close').click()
                        expect(shortcut).to_be_focused()
                    shortcut.click()
                    expect(page.locator('#lora-context')).to_have_text('Compatible with Qwen-Image 2.1')
                    count = page.locator('.lorarow').count()
                    assert count == len(fixture['loras'])
                    reads = len(lora_requests)
                    page.locator('#lora-search').fill('Film Stills')
                    expect(page.locator('.lorarow')).to_have_count(1)
                    expect(page.get_by_role('checkbox', name='Use Film Stills', exact=True)).to_be_checked()
                    expect(page.get_by_role('slider', name='Strength for Film Stills', exact=True)).to_have_value('0.7')
                    assert page.locator('#lora-sheet').evaluate('(e) => e.scrollWidth <= e.clientWidth')
                    page.screenshot(path=str(args.out / f'{name}-search-fixture.png'))
                    page.locator('#lora-search').fill('no-such-lora')
                    expect(page.locator('#lora-list')).to_contain_text('No matching LoRA')
                    page.locator('#lora-search').fill('')
                    expect(page.locator('.lorarow')).to_have_count(count)
                    assert len(lora_requests) == reads, 'Search made an unnecessary HTTP request.'
                    page.screenshot(path=str(args.out / f'{name}-browse-fixture.png'))
                    page.keyboard.press('Escape')
                    expect(page.get_by_role('dialog', name='LoRA plug-ins', exact=True)).to_be_hidden()
                    expect(shortcut).to_be_focused()
                    page.locator('#model').click()
                    page.locator('#open-loras').click()
                    expect(page.locator('#lora-sheet')).to_be_visible()
                    page.keyboard.press('Escape')
                    expect(page.locator('#model')).to_be_focused()
                    # Switching to unsupported image and text models removes all entries.
                    for loaded, architecture in [('unsupported.gguf', 'qwen_image'), ('text.gguf', 'gemma3')]:
                        model.update({'loaded': loaded, 'architecture': architecture})
                        fixture.update({'supportsLoras': False, 'loadedModelFile': loaded})
                        page.evaluate('document.dispatchEvent(new Event("visibilitychange"))')
                        expect(shortcut).to_be_hidden()
                        page.locator('#model').click()
                        expect(page.locator('#open-loras')).to_be_hidden()
                        page.locator('#sheet-bg').click(position={'x': width - 5, 'y': 5})
                    assert not errors, errors
                    report['surfaces'].append({'name': name, 'status': 'passed', 'real_catalog_rows': count,
                                               'unchanged_menu_routes': routes, 'image_fixture_target': target,
                                               'open_to_loaded_ms': times, 'median_ms': statistics.median(times),
                                               'max_ms': max(times), 'page_errors': errors})
                    context.close()
                report['status'] = 'passed'
            finally:
                browser.close()
    finally:
        (args.out / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
