#!/usr/bin/env python3
"""Check TensorAgent's knowledge UI in Chromium using real page HTML/CSS/JS.

Populates explicit browser fixtures without a host or inference model. Exercises
long collections, chat selection, thumbnail results and source previews. The
keyboard case sets the same --vh property maintained by visualViewport; it does
not emulate a native keyboard or validate iOS WebKit. Requires Python Playwright
and a Chromium executable. Generated evidence stays in ignored directories.
"""
import argparse
import json
from pathlib import Path
import re
import struct
from urllib.parse import parse_qs, urlparse
import zlib

from playwright.sync_api import expect, sync_playwright


def bounds(locator):
    box = locator.bounding_box()
    assert box, f'No layout box for {locator}'
    return box


def fits(box, width, height):
    assert box['x'] >= -1 and box['x'] + box['width'] <= width + 1, box
    assert box['y'] >= -1 and box['y'] + box['height'] <= height + 1, box


def fixture_png():
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data))
    pixels = (b'\0' + b'\x3b\x82\xf6' * 160) * 120
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!IIBBBBB', 160, 120, 8, 2, 0, 0, 0)) \
        + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b'')


def interactive(browser, repo, html, out):
    strings = {}
    for table in (repo / 'TensorAgent/src/TensorAgent.Core/Localization/en').glob('*.json'):
        strings.update(json.loads(table.read_text()))
    i18n = 'window.TensorAgentI18n = ' + json.dumps({'lang': 'en', 'strings': strings}) + ';\n' \
        + (repo / 'TensorAgent/src/TensorAgent.Core/WebUi/i18n.js').read_text()
    script = (repo / 'TensorAgent/src/TensorAgent.Core/WebUi/tensoragent.js').read_text()
    page_html = html.replace('</body>', '<script src="/i18n.js"></script><script src="/tensoragent.js"></script></body>')
    reports = []
    for name, width, height, touch in [('desktop', 1280, 900, False), ('touch-390', 390, 844, True),
                                       ('touch-320', 320, 568, True)]:
        context = browser.new_context(viewport={'width': width, 'height': height}, has_touch=touch, is_mobile=touch)
        page = context.new_page()
        errors, imports, searches = [], [], []
        page.on('pageerror', lambda error: errors.append(str(error)))
        chats = [{'id': f'{i + 1:032x}', 'title': f'Historical chat {i + 1}: design notes with images',
                  'messageCount': 4, 'updatedAt': '2026-10-10T12:00:00Z'} for i in range(40)]
        second = {'id': 'b' * 32, 'title': 'Earlier design discussion', 'messageCount': 2}
        documents = []
        photo = {'id': 'photo', 'documentId': 'photo', 'collectionId': 'alpha', 'fileName': 'photo.png',
                 'relativePath': 'Chats/' + chats[0]['id'] + '/photo.png', 'version': 1, 'modality': 'image',
                 'chunkCount': 1, 'status': 'Ready', 'warnings': []}
        def route_fixture(route):
            request = route.request
            parsed = urlparse(request.url)
            path, query = parsed.path, parse_qs(parsed.query)
            if path == '/':
                route.fulfill(content_type='text/html', body=page_html)
                return
            if path in ('/i18n.js', '/tensoragent.js'):
                route.fulfill(content_type='application/javascript', body=i18n if path == '/i18n.js' else script)
                return
            if path.endswith('/source'):
                route.fulfill(content_type='image/png', body=fixture_png())
                return
            data = {}
            if path == '/api/agent/engine':
                data = {'model': {'id': 'fixture', 'name': 'Browser fixture', 'state': 'ready'}}
            elif path == '/api/models':
                data = {'loaded': 'browser-fixture.gguf', 'architecture': 'qwen35', 'visionReady': True}
            elif path == '/api/agent/settings':
                data = {'thinkByDefault': False, 'defaultSkills': [], 'skillsEnabled': True, 'maxTokens': 2048}
            elif path == '/api/skills':
                data = {'enabled': True, 'installable': True, 'skills': []}
            elif path == '/api/agent/conversations':
                data = {'conversations': []}
            elif path == '/api/agent/launch':
                data = {'cold': True}
            elif path == '/api/sessions':
                data = {'sessionId': 's1', 'conversationId': 'c1', 'messages': [], 'think': False, 'skills': []}
            elif path == '/api/knowledge/collections':
                data = {'collections': [{'id': 'alpha', 'name': 'Research', 'documentCount': len(documents)}]}
            elif path == '/api/knowledge/collections/alpha/documents':
                data = {'documents': documents, 'total': len(documents), 'collectionTotal': len(documents)}
            elif path == '/api/knowledge/chats':
                data = {'chats': [second] if query.get('offset') == ['100'] else chats, 'total': 101,
                        'nextOffset': None if query.get('offset') == ['100'] else 100}
            elif path == '/api/knowledge/collections/alpha/chats':
                imports.append(request.post_data_json)
                documents.append(photo)
                data = {'documents': documents, 'importedChats': [{'id': chats[0]['id']}, {'id': second['id']}], 'errors': []}
            elif path == '/api/knowledge/search':
                searches.append(request.post_data_json)
                data = {'results': [dict(photo, reference='knowledge:alpha/photo/v1/0', chunkIndex=0,
                                        text='Retrieved design image from the indexed chat.', score=.9)]}
            elif path.endswith('/preview'):
                data = {'document': photo, 'inlineContentType': 'image/png', 'text': 'Indexed chat image description.'}
            elif path == '/api/knowledge/ocr/status':
                data = {'managed': False, 'status': None}
            route.fulfill(json=data)
        page.route('**/*', route_fixture)
        try:
            page.goto('http://127.0.0.1:7357/')
            expect(page.locator('#send')).to_be_enabled()
            page.locator('#menu').click()
            page.locator('#nav-sheet [data-sheet="knowledge-sheet"]').click()
            expect(page.locator('#knowledge-add-chats')).to_be_enabled()
            page.locator('#knowledge-collections .knowledge-browse').click()
            page.locator('#knowledge-add-chats').click()
            expect(page.locator('.knowledge-chat-row')).to_have_count(40)
            page.locator('.knowledge-chat-toggle').first.check()
            page.locator('.knowledge-chat-next').click()
            expect(page.locator('.knowledge-chat-row')).to_have_count(1)
            page.locator('.knowledge-chat-toggle').check()
            page.locator('.knowledge-chat-previous').click()
            expect(page.locator('.knowledge-chat-toggle').first).to_be_checked()
            page.locator('#knowledge-chat-import').click()
            expect(page.locator('#knowledge-status')).to_have_text(
                strings['page.knowledge.chatImportComplete'].replace('{added}', '2').replace('{failed}', '0'))
            expect(page.locator('#knowledge-chat-count')).to_have_text('0 selected')
            assert imports == [{'conversationIds': [chats[0]['id'], second['id']]}], imports
            page.locator('#knowledge-chat-close').click()
            page.locator('#knowledge-query').fill('design image')
            page.locator('#knowledge-search').click()
            image = page.locator('#knowledge-results img.knowledge-thumbnail')
            expect(image).to_be_visible()
            image.scroll_into_view_if_needed()
            page.wait_for_function("document.querySelector('#knowledge-results img.knowledge-thumbnail').naturalWidth > 0")
            fits(bounds(image), width, height)
            fits(bounds(page.locator('#knowledge-close')), width, height)
            assert searches == [{'query': 'design image', 'collectionIds': ['alpha'], 'limit': 8, 'maxCharacters': 12000}], searches
            page.screenshot(path=str(out / (name + '-interactive-search.png')), animations='disabled')
            page.locator('#knowledge-results .knowledge-thumbnail-link').click()
            dialog = page.locator('.knowledge-source-dialog')
            expect(dialog.locator('img')).to_be_visible()
            expect(dialog.locator('.knowledge-source-text')).to_have_text('Indexed chat image description.')
            fits(bounds(dialog), width, height)
            dialog.get_by_role('button', name=strings['page.knowledge.sourceClose'], exact=True).click()
            expect(dialog).to_have_count(0)
            page.locator('#knowledge-close').click()
            expect(page.locator('#knowledge-sheet')).not_to_have_class(re.compile(r'\bon\b'))
            assert not errors, errors
            reports.append({'name': name, 'status': 'passed', 'selectedChats': 2, 'imports': len(imports),
                            'searches': len(searches), 'thumbnailLoaded': True, 'pageErrors': errors})
        finally:
            context.close()
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--html', type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    out = args.out.resolve()
    assert any(out.is_relative_to(root) for root in
               (repo / 'artifacts', repo / 'docs' / 'validation')), out
    out.mkdir(parents=True, exist_ok=True)
    html = (args.html or repo / 'TensorAgent/src/TensorAgent.Maui/wwwroot/index.html').read_text()
    # The fixtures exercise only layout, avoiding host API calls and native code.
    html = re.sub(r'<script\b[^>]*>.*?</script>', '', html, flags=re.S | re.I)
    report = {'status': 'failed', 'surfaces': [], 'limitations': [
        'Headless Chromium with real page HTML/CSS and explicit long-content fixtures.',
        'Picker/import/search/preview interactions run real JavaScript against intercepted API fixtures; backend correctness needs separate tests.',
        'Native MAUI/iOS WebKit, actual keyboard events, host APIs and inference are not exercised.',
        'Keyboard coverage supplies --vh directly, matching the property maintained by visualViewport.',
        'No model/device scenario or performance benchmark is counted as validated.',
    ]}
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=args.browser, headless=True)
            try:
                if args.html is None:
                    report['interactions'] = interactive(browser, repo, html, out)
                for name, width, height, visible, touch, safe_bottom in [
                    ('keyboard-390', 390, 844, 340, True, 34),
                    ('desktop', 1280, 900, 900, False, 0),
                    ('touch-390', 390, 844, 844, True, 34),
                    ('touch-320', 320, 568, 568, True, 20),
                    ('desktop-short', 1280, 340, 340, False, 0),
                    ('landscape', 844, 390, 390, True, 20),
                ]:
                    context = browser.new_context(viewport={'width': width, 'height': height},
                                                  has_touch=touch, is_mobile=touch)
                    page = context.new_page()
                    try:
                        page.set_content(html)
                        page.evaluate('''({visible, safeBottom}) => {
                          document.documentElement.style.setProperty('--vh', visible + 'px');
                          document.documentElement.style.setProperty('--safe-bottom', safeBottom + 'px');
                          document.getElementById('knowledge-sheet').classList.add('on');
                          document.getElementById('sheet-bg').classList.add('on');
                        }''', {'visible': visible, 'safeBottom': safe_bottom})
                        sheet = page.locator('#knowledge-sheet')
                        # This assertion catches the previous bottom:0 / 88vh
                        # behavior before any new body or chat-picker markup is used.
                        page.wait_for_timeout(250)
                        fits(bounds(sheet), width, visible)
                        page.evaluate('''() => {
                          const add = (parent, tag, cls, text) => {
                            const node = document.createElement(tag);
                            node.className = cls; node.textContent = text;
                            document.getElementById(parent).appendChild(node); return node;
                          };
                          for (let i = 0; i < 25; i++) {
                            add('knowledge-collections', 'div', 'knowledge-collection', 'Collection ' + i + ': research notes');
                            add('knowledge-documents', 'div', 'knowledge-document', 'Indexed source ' + i + ': historical chat and media attachments');
                          }
                          document.getElementById('knowledge-active-name').textContent = 'Research collection';
                          document.getElementById('knowledge-chat-picker').hidden = false;
                          for (let i = 0; i < 40; i++) {
                            const label = add('knowledge-chat-list', 'label', 'knowledge-chat-row', '');
                            const checkbox = document.createElement('input'); checkbox.type = 'checkbox'; label.appendChild(checkbox);
                            const name = document.createElement('span'); name.className = 'knowledge-chat-label'; label.appendChild(name);
                            name.textContent = 'Historical chat ' + i + ': a long title with notes and images';
                          }
                          document.getElementById('knowledge-chat-import').disabled = false;
                          for (let i = 0; i < 8; i++) {
                            const row = add('knowledge-results', 'div', 'knowledge-result', '');
                            const link = document.createElement('a'); link.className = 'knowledge-source knowledge-source-thumbnail'; row.appendChild(link);
                            const thumb = document.createElement(i % 2 ? 'img' : 'span');
                            thumb.className = 'knowledge-thumbnail';
                            if (i % 2) thumb.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="200" height="200"%3E%3Crect width="200" height="200" fill="%233b82f6"/%3E%3C/svg%3E';
                            else { thumb.classList.add('knowledge-thumbnail-placeholder'); thumb.textContent = '♫'; }
                            link.appendChild(thumb);
                            const name = document.createElement('span'); name.className = 'knowledge-source-label'; name.textContent = 'Retrieved source ' + i + ': a multimodal historical chat'; link.appendChild(name);
                          }
                          add('knowledge-read', 'pre', 'knowledge-text', 'Source preview\\n'.repeat(35));
                          add('knowledge-read', 'p', '', 'End of retrieved source').id = 'validation-source-end';
                          document.getElementById('knowledge-search').disabled = false;
                        }''')
                        body = sheet.locator('.knowledge-body')
                        assert body.evaluate('e => e.scrollHeight > e.clientHeight')
                        close = page.locator('#knowledge-close')
                        fits(bounds(close), width, visible)
                        for selector in ('#knowledge-add-chats', '#knowledge-chat-import', '#knowledge-search',
                                         '#knowledge-results .knowledge-source-thumbnail',
                                         '#validation-source-end'):
                            control = page.locator(selector).last
                            control.scroll_into_view_if_needed()
                            fits(bounds(control), width, visible)
                            fits(bounds(close), width, visible)
                            if selector == '#knowledge-chat-import':
                                page.screenshot(path=str(out / (name + '-picker.png')), animations='disabled')
                            if selector == '#knowledge-results .knowledge-source-thumbnail':
                                thumb = control.locator('.knowledge-thumbnail')
                                thumb_box = bounds(thumb)
                                assert abs(thumb_box['width'] - 96) <= 1 and abs(thumb_box['height'] - 72) <= 1, thumb_box
                                page.screenshot(path=str(out / (name + '-thumbnail.png')), animations='disabled')
                        assert sheet.evaluate('e => e.scrollWidth <= e.clientWidth')
                        assert body.evaluate('e => e.scrollWidth <= e.clientWidth')
                        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
                        assert sheet.evaluate('e => e.scrollTop === 0')
                        page.screenshot(path=str(out / (name + '-bottom.png')), animations='disabled')
                        surface = {'name': name, 'viewport': [width, height], 'visibleHeight': visible,
                                   'sheet': bounds(sheet), 'close': bounds(close),
                                   'bodyScrollHeight': body.evaluate('e => e.scrollHeight'), 'status': 'passed'}
                        page.evaluate('''() => {
                          const viewer = document.createElement('div'); viewer.className = 'knowledge-source-viewer';
                          const dialog = document.createElement('section'); dialog.className = 'knowledge-source-dialog'; viewer.appendChild(dialog);
                          const header = document.createElement('div'); header.className = 'knowledge-source-header'; dialog.appendChild(header);
                          const title = document.createElement('h2'); title.textContent = 'Historical chat source'; header.appendChild(title);
                          const close = document.createElement('button'); close.className = 'mini quiet'; close.textContent = 'Done'; header.appendChild(close);
                          const content = document.createElement('div'); content.className = 'knowledge-source-content'; dialog.appendChild(content);
                          const text = document.createElement('pre'); text.className = 'knowledge-source-text'; text.textContent = 'Retrieved historical message\\n'.repeat(100); content.appendChild(text);
                          const end = document.createElement('p'); end.id = 'validation-modal-end'; end.textContent = 'End of original source'; content.appendChild(end);
                          document.body.appendChild(viewer);
                        }''')
                        dialog = page.locator('.knowledge-source-dialog')
                        fits(bounds(dialog), width, visible)
                        page.locator('#validation-modal-end').scroll_into_view_if_needed()
                        fits(bounds(page.locator('#validation-modal-end')), width, visible)
                        fits(bounds(dialog.locator('button')), width, visible)
                        page.screenshot(path=str(out / (name + '-source.png')), animations='disabled')
                        surface['sourceDialog'] = bounds(dialog)
                        report['surfaces'].append(surface)
                    finally:
                        context.close()
            finally:
                browser.close()
        report['status'] = 'passed'
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
