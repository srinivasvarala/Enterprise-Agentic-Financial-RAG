"""Run inside the `eval` image: the renderer works offline and re-renders to the same page text.

    docker run --rm --network none -i finrag-eval:dev python - < infra/docker/check_render.py

Renders a small HTML document twice, then checks that the PDF has the expected pages and text and
that a second render extracts identical text per page (PDF bytes themselves may differ).
"""

import tempfile
from pathlib import Path

from finrag.evaluation.render import WkhtmltopdfRenderer, pdf_page_texts

HTML = """<html><body>
<p>PART I</p><p>Item 1. Business</p><p>Net sales were $391,035 million.</p>
<table><tr><td>Products</td><td>$ 294,866</td></tr>
<tr><td>Services</td><td>$ 96,169</td></tr></table>
<div style="page-break-before: always">Item 1A. Risk Factors</div>
</body></html>"""

renderer = WkhtmltopdfRenderer()
with tempfile.TemporaryDirectory() as tmp:
    html = Path(tmp) / "doc.htm"
    html.write_text(HTML)
    first, second = Path(tmp) / "a.pdf", Path(tmp) / "b.pdf"
    renderer.render(html, first)
    renderer.render(html, second)
    pages = pdf_page_texts(first)
    assert len(pages) == 2, pages
    assert "Net sales were $391,035 million." in pages[0], pages[0]
    assert "Products" in pages[0] and "$ 294,866" in pages[0], pages[0]
    assert "Item 1A. Risk Factors" in pages[1], pages[1]
    assert pdf_page_texts(second) == pages, "re-render changed the per-page text"
print(f"render ok: {renderer.name} {renderer.version}, 2 pages, identical text on re-render")
