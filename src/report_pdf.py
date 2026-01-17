# src/report_pdf.py

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
import io

def build_compare_pdf(
    title: str,
    subtitle: str,
    profiles: list[str],
    table_df,
    radar_png_bytes: bytes,
):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=30,
    )

    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    if subtitle:
        story.append(Paragraph(subtitle, styles["Normal"]))
    story.append(Spacer(1, 12))

    # Player headers
    for p in profiles:
        story.append(Paragraph(p, styles["Heading3"]))
    story.append(Spacer(1, 12))

    # Table
    table_data = [ [table_df.index.name or "Metric"] + list(table_df.columns) ]
    for idx, row in table_df.iterrows():
        table_data.append([idx] + row.astype(str).tolist())

    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("FONT", (0,0), (-1,0), "Helvetica-Bold"),
        ("ALIGN", (1,1), (-1,-1), "CENTER"),
    ]))

    story.append(table)
    story.append(Spacer(1, 16))

    # Radar
    img = Image(io.BytesIO(radar_png_bytes), width=400, height=400)
    story.append(img)

    doc.build(story)
    buffer.seek(0)
    return buffer
