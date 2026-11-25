from collections.abc import Sequence
from pathlib import Path

from machine_fault_triage.testing import FakeOcrEngine


def write_pdf(path: Path, pages: Sequence[Sequence[str]]) -> Path:
    kids = " ".join(f"{3 + 2 * index} 0 R" for index in range(len(pages)))
    font = 3 + 2 * len(pages)
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>",
    ]
    for index, lines in enumerate(pages):
        shown = " ".join(f"({line}) Tj T*" for line in lines)
        body = f"BT /F1 12 Tf 72 720 Td 14 TL {shown} ET"
        objects.append(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
            f" /Contents {4 + 2 * index} 0 R /Resources << /Font << /F1 {font} 0 R >> >> >>"
        )
        objects.append(f"<< /Length {len(body)} >>\nstream\n{body}\nendstream")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    content = "%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(content))
        content += f"{number} 0 obj\n{body}\nendobj\n"
    table = "".join(f"{offset:010d} 00000 n \n" for offset in offsets)
    content += (
        f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n{table}"
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{len(content)}\n%%EOF\n"
    )
    path.write_bytes(content.encode("latin-1"))
    return path


def scripted_engine(
    text: str = "E7: flow sensor fault.", name: str = "fake engine"
) -> FakeOcrEngine:
    return FakeOcrEngine([text] * 20, fingerprint=name)
