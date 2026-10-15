#!/usr/bin/env python3
"""Render an audited offline HTML artifact with an already installed Chrome.

Requires no npm packages, network resources, or installation. Screenshots at two
viewport sizes need visual review; they do not prove physical phone execution.
"""
import argparse
import hashlib
import html as html_lib
import json
import re
import struct
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chrome", type=Path, default=Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"))
    args = parser.parse_args()
    if not args.chrome.is_file():
        parser.error("Chrome is unavailable; rendering cannot be counted as passed.")
    source = args.html.read_text()
    if re.search(r'<(?:script|link|img)[^>]+(?:src|href)=["\']https?://|(?:url\(|@import)[^;\n]*https?://', source, re.I):
        parser.error("The artifact requires external resources; the offline fixture failed.")
    args.output.mkdir(parents=True, exist_ok=True)
    version = subprocess.check_output([str(args.chrome), "--version"], text=True).strip()
    instrumented = args.output / "instrumented.html"
    # DOM attributes do not change the rendered layout. Measure the actual page
    # viewport: Chrome's CLI can clamp a requested sub-500px window while still
    # producing a smaller, cropped PNG.
    probe = '<script>window.addEventListener("load",()=>document.documentElement.setAttribute("data-skills-validation",JSON.stringify({innerWidth,innerHeight,clientWidth:document.documentElement.clientWidth,scrollWidth:document.documentElement.scrollWidth})))</script>'
    instrumented.write_text(re.sub(r"</body>", probe + "</body>", source, count=1, flags=re.I))
    views = []
    for name, width, height in [("desktop", 1280, 900), ("narrow", 500, 844)]:
        screenshot = (args.output / (name + ".png")).resolve()
        command = [str(args.chrome), "--headless=new", "--disable-gpu", "--no-first-run",
            "--no-default-browser-check", "--disable-background-networking", "--force-device-scale-factor=1",
            "--hide-scrollbars", "--user-data-dir=" + str((args.output / (name + "-profile")).resolve()),
            "--window-size=" + str(width) + "," + str(height), "--dump-dom", "--virtual-time-budget=1000",
            "--screenshot=" + str(screenshot), instrumented.resolve().as_uri()]
        result = subprocess.run(command, capture_output=True, text=True, timeout=45)
        (args.output / (name + "-browser.log")).write_text(result.stdout + result.stderr)
        if result.returncode or not screenshot.is_file():
            raise RuntimeError("Chrome failed to render " + name)
        image = screenshot.read_bytes()
        actual = struct.unpack(">II", image[16:24])
        if actual != (width, height):
            raise RuntimeError("Unexpected screenshot dimensions: " + repr(actual))
        match = re.search(r'data-skills-validation="([^"]+)"', result.stdout)
        if not match:
            raise RuntimeError("Browser did not report actual layout dimensions.")
        layout = json.loads(html_lib.unescape(match.group(1)))
        if layout["innerWidth"] != width or layout["clientWidth"] != width:
            raise RuntimeError("Requested viewport differs from the actual layout: " + repr(layout))
        if layout["scrollWidth"] > width:
            raise RuntimeError("The artifact has horizontal overflow at " + name + ": " + repr(layout))
        views.append({"viewport": name, "width": width, "height": height,
            "actual_layout": layout, "png_sha256": hashlib.sha256(image).hexdigest(), "browser_exit_code": result.returncode})
    (args.output / "report.json").write_text(json.dumps({"chrome": version,
        "html_sha256": hashlib.sha256(args.html.read_bytes()).hexdigest(), "views": views,
        "visual_review": "pending", "limitations": ["Two desktop headless browser viewports at 1280/500px width, not physical mobile execution; a 390px layout is not measured.",
        "No user-study or subjective-design superiority claim."]}, indent=2) + "\n")
    print(json.dumps(views))


if __name__ == "__main__":
    main()
