#!/usr/bin/env python3
"""Create original deterministic PDF OCR fixtures. Validation-only PIL/reportlab.

Committed PDFs are the byte-stable test inputs. Regeneration with different fonts
or package versions can change pixels/bytes; manifest records these dependencies.
No hidden text or answer-bearing PDF metadata is written into scanned pages.
"""
import argparse
import hashlib
import io
import json
import pathlib
import platform
import PIL
import reportlab
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from pypdf import PdfReader


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='InferenceWeb.Tests/Fixtures/Knowledge')
    parser.add_argument('--preview', default='artifacts/knowledge/pdf-fixture-preview')
    parser.add_argument('--latin-font', required=True)
    parser.add_argument('--cjk-font', required=True)
    args = parser.parse_args()
    out, preview = pathlib.Path(args.output), pathlib.Path(args.preview)
    out.mkdir(parents=True, exist_ok=True)
    preview.mkdir(parents=True, exist_ok=True)
    latin, cjk = pathlib.Path(args.latin_font), pathlib.Path(args.cjk_font)
    f = lambda size: ImageFont.truetype(str(latin), size)
    zh = ImageFont.truetype(str(cjk), 36)
    pages = [
        {'title': 'CEDAR FIELD LOG', 'code': 'CEDAR-4821', 'subject': 'Cedar observation code',
         'rows': [('Sensor', 'Reading'), ('North', '17'), ('South', '29'), ('Total', '46')],
         'description': 'Record the four-digit identifier exactly.\nThe sensor total is 46 units.'},
        {'title': 'ORBIT INVENTORY', 'code': 'ORBIT-7356', 'subject': 'Orbit inventory code',
         'rows': [('Item', 'Quantity'), ('Amber bolts', '12'), ('Blue clamps', '31'), ('Total', '43')],
         'description': 'Inventory belongs to the Orbit workshop.\nThe listed stock total is 43 items.'},
        {'title': 'HARBOR DISPATCH', 'code': 'HARBOR-9064', 'subject': 'Harbor dispatch code',
         'rows': [('Route', 'Crates'), ('West pier', '24'), ('East pier', '18'), ('Total', '42')],
         'description': 'Two routes share one dispatch code.\nThe shipment contains 42 crates.'},
        {'title': 'BAMBOO CHECKLIST', 'code': 'BAMBOO-2187', 'subject': 'Bamboo inspection code',
         'rows': [('Check', 'Count'), ('Passed', '36'), ('Pending', '9'), ('Total', '45')],
         'description': 'The checklist covers 45 inspections.\nNine checks remain pending.',
         'chinese': '竹林项目：蓝色箱子有二十七个。', 'chineseExpected': ['竹林', '蓝色', '二十七']},
    ]
    images = []
    for i, page in enumerate(pages, 1):
        im = Image.new('RGB', (1200, 1600), '#ffffff')
        d = ImageDraw.Draw(im)
        d.rectangle((64, 62, 1136, 164), fill='#17364d')
        d.text((92, 83), page['title'], font=f(43), fill='white')
        d.text((80, 218), f"Page {i} of 4 | Original synthetic test record", font=f(28), fill='#404850')
        d.text((80, 310), page['subject'], font=f(38), fill='#17364d')
        d.text((80, 377), page['code'], font=f(57), fill='#000000')
        d.multiline_text((80, 493), page['description'], font=f(32), spacing=18, fill='#202020')
        y, widths = 655, (80, 730, 1120)
        for row_index, row in enumerate(page['rows']):
            fill = '#e6eef4' if row_index == 0 else ('#f3f5f7' if row_index % 2 else '#ffffff')
            d.rectangle((80, y, 1120, y+96), fill=fill, outline='#52606a', width=2)
            d.line((730, y, 730, y+96), fill='#52606a', width=2)
            d.text((104, y+26), row[0], font=f(32), fill='#111111')
            d.text((756, y+26), row[1], font=f(32), fill='#111111')
            y += 96
        if page.get('chinese'):
            d.text((80, 1115), 'Supplementary Chinese OCR case', font=f(31), fill='#17364d')
            d.text((80, 1180), page['chinese'], font=zh, fill='#111111')
        d.line((80, 1450, 1120, 1450), fill='#52606a', width=2)
        d.text((80, 1480), 'No real customer or operational data. OCR validation only.', font=f(27), fill='#404850')
        image_path = preview / f'scan-page-{i}.png'
        im.save(image_path)
        images.append(im)
    scan = out / 'scanned-four-pages.pdf'
    c = canvas.Canvas(str(scan), pagesize=(600, 800), invariant=1, pageCompression=1)
    c.setTitle('Original synthetic image-only PDF fixture')
    c.setAuthor('TensorSharp test fixtures')
    for im in images:
        c.drawImage(ImageReader(im), 0, 0, width=600, height=800)
        c.showPage()
    c.save()
    mixed = out / 'mixed-two-pages.pdf'
    c = canvas.Canvas(str(mixed), pagesize=(600, 800), invariant=1, pageCompression=1)
    c.setTitle('Original synthetic mixed-content PDF fixture')
    c.setAuthor('TensorSharp test fixtures')
    c.setFont('Helvetica-Bold', 24)
    c.drawString(40, 724, 'SELECTABLE CONTROL PAGE')
    c.setFont('Helvetica', 18)
    c.drawString(40, 655, 'The Juniper selectable code is JUNIPER-5643.')
    c.drawString(40, 610, 'This first page has a real PDF text layer.')
    c.drawString(40, 570, 'The second page is an image-only Orbit record.')
    c.showPage()
    # Two separate raster blocks, both needed for code and table coverage.
    c.setPageSize((600, 860))
    c.setFont('Helvetica', 12)
    c.drawString(40, 838, 'SCANNED ATTACHMENT WITH SELECTABLE HEADING')
    c.drawImage(ImageReader(images[1].crop((0, 0, 1200, 600))), 0, 520, width=600, height=300)
    c.drawImage(ImageReader(images[1].crop((0, 600, 1200, 1600))), 0, 0, width=600, height=500)
    c.showPage()
    c.save()
    for path, flags in ((scan, [False]*4), (mixed, [True, True])):
        reader = PdfReader(path)
        assert len(reader.pages) == len(flags)
        for index, (page, has_text) in enumerate(zip(reader.pages, flags)):
            assert bool((page.extract_text() or '').strip()) == has_text
            assert len(page.images) == (1 if path == scan else 0 if index == 0 else 2), 'Unexpected raster block count'
    manifest = {
        'schemaVersion': 1,
        'provenance': 'Original synthetic content authored for TensorSharp validation; no external documents, personal data, or runtime-service dependencies.',
        'generator': 'eng/validation/knowledge/create_pdf_fixtures.py',
        'generatorEnvironment': {'python': platform.python_version(), 'pillow': PIL.__version__, 'reportlab': reportlab.Version,
            'latinFont': {'name': latin.name, 'sha256': sha(latin)}, 'cjkFont': {'name': cjk.name, 'sha256': sha(cjk)}},
        'pageRaster': {'width':1200, 'height':1600, 'mode':'RGB', 'pdfPagePoints':[600,800]},
        'files': [
            {'fileName':scan.name,'sha256':sha(scan),'pages':[
                {'page':i, 'selectableText':False, 'imageCount':1, 'query':p['subject'], 'expectedCode':p['code'],
                 'expectedTotal':p['rows'][-1][1], 'expectedText':p['title']+'\n'+p['subject']+'\n'+p['code']+'\n'+p['description']+'\n'+'\n'.join(' | '.join(r) for r in p['rows']) + ('\n'+p['chinese'] if p.get('chinese') else ''),
                 **({'chineseExpected':p['chineseExpected']} if p.get('chinese') else {})}
                for i,p in enumerate(pages,1)]},
            {'fileName':mixed.name,'sha256':sha(mixed),'pages':[
                {'page':1,'selectableText':True,'imageCount':0,'query':'Juniper selectable code','expectedCode':'JUNIPER-5643'},
                {'page':2,'selectableText':True,'selectableExpected':'SCANNED ATTACHMENT WITH SELECTABLE HEADING','imageCount':2,'pdfPagePoints':[600,860],'query':'Orbit inventory code','expectedCode':'ORBIT-7356','expectedTotal':'43'}]}
        ],
        'limitations':['Clean, upright, high-contrast synthetic raster pages; not a representative real-world scanning/OCR benchmark.',
            'Chinese is a separately reported supplementary case, never assumed covered by English success.',
            'Regeneration bytes depend on recorded fonts and dependency versions; committed PDF hashes identify the actual fixtures.']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'files':[{'path':str(out/x['fileName']),'sha256':x['sha256']} for x in manifest['files']]},indent=2))


if __name__ == '__main__':
    main()
