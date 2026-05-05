"""PDF report export for explanation artifacts."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def export_explanation_pdf(
    output_path: Path,
    original: Image.Image,
    heatmap_overlay: Image.Image,
    landmark_overlay: Image.Image,
    reliability_score: float,
    clinical_flags: list[str],
) -> Path:
    """Export a compact explanation report PDF."""

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise ImportError("Install reportlab to export explanation PDFs.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(36, height - 40, "Diabetic Retinopathy Explanation Report")
    c.setFont("Helvetica", 11)
    c.drawString(36, height - 62, f"XAI Reliability Score: {reliability_score:.2f}")
    y = height - 82
    for flag in clinical_flags:
        c.drawString(36, y, f"- {flag}")
        y -= 14

    images = [("Original", original), ("Heatmap", heatmap_overlay), ("Clinical Overlay", landmark_overlay)]
    x_positions = [36, 220, 404]
    for (title, image), x in zip(images, x_positions):
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x, y - 10, title)
        c.drawImage(ImageReader(image.resize((160, 160))), x, y - 180, width=160, height=160)
    c.showPage()
    c.save()
    return output_path
