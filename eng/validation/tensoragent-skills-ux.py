#!/usr/bin/env python3
"""Validate TensorAgent's in-list skill management against an isolated app host.

Start eng/validation/TensorAgentHost with --root under artifacts/ or
docs/validation/ and --skills TensorAgent/skills, without loading a model. This
probe installs temporary skills and removes bundled skill IDs ONLY in that
host's isolated app data. Original bundled files must remain unchanged.

The catalog, installs, removals, rescans and global setting use the real HTTP
API. Loaded-model state and chat responses use explicit browser fixtures to
inspect outgoing selection payloads without model inference. Requires Python
Playwright and a Chromium executable. Generated evidence must stay ignored.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import statistics
import time
from urllib.parse import quote, urlparse
import uuid
import zipfile

from playwright.sync_api import expect, sync_playwright


def ignored_path(path, repo):
    path = path.resolve()
    assert any(path.is_relative_to(root) for root in
               (repo / 'artifacts', repo / 'docs' / 'validation')), (
        f'Validation data must be inside artifacts/ or docs/validation/: {path}')
    return path


def roster(page, base_url):
    response = page.request.get(base_url + '/api/skills')
    assert response.ok, response.text()
    return response.json()


def row_for(page, name):
    # Skill names are validated manifest IDs, and JSON quoting also prevents
    # punctuation in a fixture from changing the selector's meaning.
    return page.locator('.skillrow[data-skill=' + json.dumps(name) + ']')


def same_sheet(page):
    expect(page.locator('#skills-sheet')).to_have_class(re.compile(r'\bon\b'))
    expect(page.locator('#skills-sheet')).to_be_visible()
    expect(page.locator('#skill-sheet')).not_to_have_class(re.compile(r'\bon\b'))


def open_skills(page, expected_count):
    page.locator('#menu').click()
    page.locator('#nav-sheet [data-sheet="skills-sheet"]').click()
    same_sheet(page)
    expect(page.locator('.skillrow[data-skill]')).to_have_count(expected_count)


def close_sheet(page, width):
    page.locator('#sheet-bg').click(position={'x': width - 5, 'y': 5})
    expect(page.locator('#skills-sheet')).not_to_have_class(re.compile(r'\bon\b'))


def send_fixture(page, width, text):
    close_sheet(page, width)
    page.locator('#text').fill(text)
    expect(page.locator('#send')).to_be_enabled()
    with page.expect_request(lambda request:
                             urlparse(request.url).path == '/api/chat'
                             and request.method == 'POST') as sent:
        page.locator('#send').click()
    payload = sent.value.post_data_json
    expect(page.locator('#send')).to_be_enabled()
    expect(page.locator('#send')).to_have_class('round send')
    return payload


def bundle_bytes(name):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name + '/SKILL.md',
                         '---\nname: ' + name + '\n'
                         'description: Temporary browser validation fixture for in-list skill controls.\n'
                         '---\n\nThis fixture contains no executable scripts.\n')
    return buffer.getvalue()


def check_controls_fit(page, row, width):
    row.scroll_into_view_if_needed()
    bounds = {}
    for selector in ('.skill-choice', 'button.skill-remove'):
        control = row.locator(selector)
        expect(control).to_be_visible()
        box = control.bounding_box()
        assert box and box['height'] >= 44, (selector, box)
        assert box['x'] >= 0 and box['x'] + box['width'] <= width + 1, (selector, box, width)
        bounds[selector] = box
    assert page.locator('#skills-sheet').evaluate('(e) => e.scrollWidth <= e.clientWidth')
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    return bounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connection', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True,
                        help='Exact isolated --root passed to TensorAgentHost.')
    parser.add_argument('--browser', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=5)
    args = parser.parse_args()
    assert args.repetitions > 0
    repo = Path(__file__).resolve().parents[2]
    root = ignored_path(args.root, repo)
    out = ignored_path(args.out, repo)
    assert args.connection.resolve() == root / 'connection.json'
    connection = json.loads(args.connection.read_text(encoding='utf-8-sig'))
    base_url = connection['baseUrl']
    assert urlparse(base_url).hostname in ('localhost', '127.0.0.1', '::1')
    installed_root = root / 'data' / 'skills'
    assert installed_root.is_dir(), 'Start a fresh isolated TensorAgentHost first.'
    out.mkdir(parents=True, exist_ok=True)
    report = {'status': 'failed', 'surfaces': [], 'limitations': [
        'Real AgentAppHost HTTP API and headless Chromium; native MAUI/iOS WebKit are not exercised.',
        'Model availability and chat responses are browser fixtures; no inference or model quality is measured.',
        'Toggle timings include browser automation and are UX latency samples, not inference benchmarks.',
        'Bundled removals persist as tombstones only in the disposable host root.',
    ]}
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=args.browser, headless=True)
            try:
                for surface, width, height, touch in [
                    ('desktop', 1280, 900, False), ('touch-390', 390, 844, True),
                    ('touch-320', 320, 568, True),
                ]:
                    context = browser.new_context(viewport={'width': width, 'height': height},
                                                  has_touch=touch, is_mobile=touch)
                    page = context.new_page()
                    errors, skill_requests = [], []
                    temporary = None
                    page.on('pageerror', lambda error: errors.append(str(error)))
                    page.on('request', lambda request: skill_requests.append(
                        (request.method, request.url)) if '/api/skills' in request.url else None)
                    try:
                        page.goto(connection['entryUrl'])
                        initial = roster(page, base_url)
                        assert installed_root in [Path(p).resolve() for p in initial['roots']], (
                            'The connected host does not use the declared isolated app data.')
                        model = page.request.get(base_url + '/api/models')
                        assert model.ok and not model.json().get('loaded'), 'Do not use a host with a loaded model.'
                        settings = page.request.get(base_url + '/api/agent/settings')
                        assert settings.ok
                        defaults = settings.json()
                        defaults.update({'skillsEnabled': True, 'defaultSkills': [], 'uiLanguage': 'en'})
                        saved = page.request.post(base_url + '/api/agent/settings', data=defaults)
                        assert saved.ok, saved.text()

                        temporary = 'ux-probe-' + uuid.uuid4().hex[:12]
                        installed = page.request.post(base_url + '/api/skills', multipart={
                            'file': {'name': temporary + '.zip', 'mimeType': 'application/zip',
                                     'buffer': bundle_bytes(temporary)}})
                        assert installed.ok, installed.text()
                        assert (installed_root / temporary / 'SKILL.md').is_file()
                        catalog = roster(page, base_url)
                        count = len(catalog['skills'])
                        bundled = next(skill for skill in catalog['skills'] if skill['origin'] == 'discovered')
                        name = bundled['name']
                        source = next(Path(p) / name / 'SKILL.md' for p in catalog['roots']
                                      if (Path(p) / name / 'SKILL.md').is_file()
                                      and Path(p).resolve() != installed_root)
                        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

                        # These two fixtures only permit inspecting outgoing chat
                        # choices. Catalog/settings/install/remove stay real.
                        page.route('**/api/models', lambda route: route.fulfill(json={
                            'loaded': 'skills-ui-fixture.gguf', 'architecture': 'qwen35'}))
                        page.route('**/api/chat', lambda route: route.fulfill(
                            content_type='text/event-stream',
                            body='data: {"token":"Selection fixture acknowledged."}\n\n'
                                 'data: {"done":true}\n\n'))
                        page.reload()
                        expect(page.locator('#send')).to_be_enabled()
                        page.locator('#menu').click()
                        routes = page.locator('#nav-sheet [data-route]').evaluate_all(
                            '(items) => items.map(e => e.getAttribute("data-route"))')
                        assert routes == ['sessions', 'models', 'settings', 'about'], routes
                        expect(page.locator('#nav-chats')).to_be_visible()
                        expect(page.locator('#nav-new')).to_be_visible()
                        page.locator('#nav-sheet [data-sheet="skills-sheet"]').click()
                        same_sheet(page)
                        expect(page.locator('.skillrow[data-skill]')).to_have_count(count)
                        selected = row_for(page, temporary)
                        checkbox = selected.locator('input.skill-toggle')
                        expect(checkbox).not_to_be_checked()
                        bounds = check_controls_fit(page, selected, width)
                        reads = len(skill_requests)
                        timings = []
                        for _ in range(args.repetitions):
                            start = time.perf_counter()
                            selected.locator('.skill-choice').click()
                            expect(checkbox).to_be_checked()
                            same_sheet(page)
                            selected.locator('.skill-choice').click()
                            expect(checkbox).not_to_be_checked()
                            same_sheet(page)
                            timings.append((time.perf_counter() - start) * 1000)
                        assert len(skill_requests) == reads, 'Local selection made unnecessary skill API requests.'
                        checkbox.focus()
                        page.keyboard.press('Space')
                        expect(checkbox).to_be_checked()
                        same_sheet(page)
                        page.screenshot(path=str(out / (surface + '-selected.png')), animations='disabled')
                        selected_payload = send_fixture(page, width, 'Validate selected skill payload.')
                        assert selected_payload.get('skills') == [temporary], selected_payload
                        open_skills(page, count)
                        expect(checkbox).to_be_checked()
                        selected.locator('.skill-choice').click()
                        empty_payload = send_fixture(page, width, 'Validate explicit empty skill payload.')
                        assert empty_payload.get('skills') == [] and empty_payload.get('skills_discovery') is False
                        open_skills(page, count)
                        selected.locator('.skill-choice').click()

                        rejection_url = base_url + '/api/skills/' + quote(temporary)
                        page.route(rejection_url, lambda route: route.fulfill(
                            status=403, json={'error': 'Intentional validation delete rejection.'}))
                        selected.locator('button.skill-remove').click()
                        expect(page.locator('#skills-error')).to_contain_text('Intentional validation delete rejection.')
                        expect(selected).to_be_visible()
                        expect(checkbox).to_be_checked()
                        expect(selected.locator('button.skill-remove')).to_be_enabled()
                        same_sheet(page)
                        assert (installed_root / temporary / 'SKILL.md').is_file()
                        page.screenshot(path=str(out / (surface + '-delete-rejected.png')), animations='disabled')
                        page.unroute(rejection_url)
                        selected.locator('button.skill-remove').click()
                        expect(selected).to_have_count(0)
                        same_sheet(page)
                        assert not (installed_root / temporary).exists()
                        assert temporary not in [skill['name'] for skill in roster(page, base_url)['skills']]
                        count -= 1
                        removed_payload = send_fixture(page, width, 'Validate removed skill leaves selection.')
                        assert removed_payload.get('skills') == [] and removed_payload.get('skills_discovery') is False
                        open_skills(page, count)

                        master = page.locator('#skills-master')
                        page.locator('label.skills-master').click()
                        expect(master).to_be_enabled()
                        expect(master).not_to_be_checked()
                        assert roster(page, base_url)['enabled'] is False
                        for box in page.locator('input.skill-toggle').all():
                            expect(box).to_be_disabled()
                        discovered = row_for(page, name)
                        expect(discovered.locator('button.skill-remove')).to_be_enabled()
                        off_bounds = check_controls_fit(page, discovered, width)
                        page.screenshot(path=str(out / (surface + '-global-off.png')), animations='disabled')
                        discovered.locator('button.skill-remove').click()
                        expect(discovered).to_have_count(0)
                        same_sheet(page)
                        assert source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
                        tombstone = root / 'data' / 'removed-skills.txt'
                        assert name in tombstone.read_text().splitlines()
                        rescanned = page.request.post(base_url + '/api/skills/rescan')
                        assert rescanned.ok and name not in [skill['name'] for skill in rescanned.json()['skills']]
                        count -= 1
                        off_payload = send_fixture(page, width, 'Validate global skills off payload.')
                        assert off_payload.get('skills_discovery') is False and not off_payload.get('skills')
                        open_skills(page, count)
                        page.locator('label.skills-master').click()
                        expect(master).to_be_enabled()
                        expect(master).to_be_checked()
                        assert roster(page, base_url)['enabled'] is True
                        for box in page.locator('input.skill-toggle').all():
                            expect(box).to_be_enabled()
                        assert not errors, errors
                        report['surfaces'].append({
                            'name': surface, 'status': 'passed', 'initial_real_catalog_count': len(initial['skills']),
                            'unchanged_menu_routes': routes, 'control_bounds': bounds,
                            'global_off_control_bounds': off_bounds, 'installed_and_removed': temporary,
                            'bundled_tombstoned': name, 'bundled_source_unchanged': str(source),
                            'payloads': {'selected': selected_payload, 'explicit_empty': empty_payload,
                                         'after_remove': removed_payload, 'global_off': off_payload},
                            'enable_disable_pair_ms': timings, 'median_ms': statistics.median(timings),
                            'max_ms': max(timings), 'page_errors': errors,
                        })
                    except Exception:
                        page.screenshot(path=str(out / (surface + '-failure.png')), animations='disabled')
                        raise
                    finally:
                        if temporary and (installed_root / temporary).exists():
                            # Failed assertions must not accumulate fixtures on
                            # a rerun. Only this run's exact generated ID is ours.
                            page.request.delete(base_url + '/api/skills/' + quote(temporary))
                        context.close()
                report['status'] = 'passed'
            finally:
                browser.close()
    finally:
        (out / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
