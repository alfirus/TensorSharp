#!/usr/bin/env python3
"""Write a deterministic, original PDF crop/rotation fixture (standard library only).

The crop box is offset from the media box. Four colored corner regions identify
orientation and full raster coverage. Text is counter-rotated so every visible
page reads upright. The magenta margin and OUTSIDE text must be cropped away.
"""
import argparse
import pathlib


def create_pdf():
    objects = []

    def add(body):
        objects.append(body.encode("ascii"))
        return len(objects)

    add("<< /Type /Catalog /Pages 2 0 R >>")
    add("<< /Type /Pages /Count 4 /Kids [4 0 R 6 0 R 8 0 R 10 0 R] >>")
    add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    transforms = {
        0: "1 0 0 1 0 0",
        90: "0 1 -1 0 600 0",
        180: "-1 0 0 -1 600 400",
        270: "0 -1 1 0 0 400",
    }
    for rotation in transforms:
        page_id = len(objects) + 1
        add(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 720 540] "
            f"/CropBox [40 60 640 460] /Rotate {rotation} "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_id + 1} 0 R >>")
        width, height = (600, 400) if rotation % 180 == 0 else (400, 600)
        content = (
            "1 0 1 rg 0 0 720 540 re f\n"
            "0 0 0 rg BT /F1 24 Tf 5 15 Td (OUTSIDE 9463) Tj ET\n"
            "q 1 0 0 1 40 60 cm\n"
            "1 0 0 rg 0 0 300 200 re f\n"       # bottom left: red
            "0 1 0 rg 300 0 300 200 re f\n"     # bottom right: green
            "0 0 1 rg 0 200 300 200 re f\n"     # top left: blue
            "1 1 0 rg 300 200 300 200 re f\n"   # top right: yellow
            "1 1 1 rg 60 60 480 280 re f\n"
            f"q {transforms[rotation]} cm\n"
            f"0 0 0 rg BT /F1 42 Tf {width // 2 - 110} {height // 2} Td "
            f"(CROP {rotation}) Tj ET Q Q\n"
        ).encode("ascii")
        add(f"<< /Length {len(content)} >>\nstream\n" + content.decode("ascii") + "endstream")
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, body in enumerate(objects, 1):
        offsets.append(len(output))
        output += f"{i} 0 obj\n".encode("ascii") + body + b"\nendobj\n"
    xref = len(output)
    output += f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode("ascii")
    for offset in offsets[1:]:
        output += f"{offset:010} 00000 n \n".encode("ascii")
    output += (f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n"
               f"{xref}\n%%EOF\n").encode("ascii")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="InferenceWeb.Tests/Fixtures/Knowledge/crop-rotation.pdf")
    target = pathlib.Path(parser.parse_args().output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(create_pdf())
    print(target)
