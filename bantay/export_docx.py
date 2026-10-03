"""Word (.docx) export laid out like the barangay's paper BLOTTER FORM
(Template/Template Blotter.png), one form per record.

The slots come from narrative.parse() - the stored narrative is already the
form-shaped template, so reading it back is the inverse of what the encoder
saved. Records typed as free prose (the plain New Blotter Record form) have
no form layout to parse; for those the record's own columns fill what they
can and the prose goes under E. SALAYSAY.
"""
import io
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

from . import narrative as narrative_tpl

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "Template"
_LOGOS = (_TEMPLATE_DIR / "logo_barangay.png", _TEMPLATE_DIR / "logo_city.png")

PROVINCE, CITY, BARANGAY = "Bulacan", "Angeles", "Anunas"
_BLANK_LINE = "_" * 22


def record_fields(record, text):
    """Slot dict for one record. `text` is the narrative as the caller may see
    it (redacted for non-admins), so the export never shows more than the page."""
    entries = narrative_tpl.parse(text)
    if entries:
        return entries[0]
    fields = dict.fromkeys(narrative_tpl.SLOTS, "")
    fields.update(date=record.date or "", time=record.time or "",
                  location_purok=record.location_purok or "",
                  complaint=record.incident_type_primary or "",
                  incident_summary=text or "")
    return fields


def _no_borders(table):
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "nil")
        borders.append(el)
    table._tbl.tblPr.append(borders)


def _bottom_rule(paragraph):
    pbdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    for key, val in (("val", "single"), ("sz", "6"), ("space", "1"), ("color", "5B9BD5")):
        bottom.set(qn(f"w:{key}"), val)
    pbdr.append(bottom)
    paragraph._p.get_or_add_pPr().append(pbdr)


def _para(container, space_after=0):
    p = container.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(0)
    return p


def _field(p, label, value, bold=False):
    """`label: value` with the value underlined like a filled-in blank."""
    if label:
        p.add_run(f"{label} ").bold = bold
    p.add_run((value or "").strip() or _BLANK_LINE).underline = True


def _line(doc, label, value, bold=False, right=None, space_after=0):
    """One form line; `right` is an optional (label, value) at the right tab -
    the Edad / Oras column on the paper form. label=None leaves the left empty."""
    p = _para(doc, space_after)
    if label is not None:
        _field(p, label, value, bold)
    if right:
        p.paragraph_format.tab_stops.add_tab_stop(Inches(4.3))
        p.add_run("\t")
        _field(p, *right)
    return p


def _header(doc):
    table = doc.add_table(rows=1, cols=3)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    _no_borders(table)
    left, mid, right = table.rows[0].cells
    for cell, logo in ((left, _LOGOS[0]), (right, _LOGOS[1])):
        cell.width = Inches(1.4)
        if logo.exists():
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run().add_picture(str(logo), width=Inches(1.15))
    mid.width = Inches(4.2)
    for i, (text, value) in enumerate((("Republic of the Philippines", None),
                                       (f"Province of {PROVINCE}", None),
                                       ("CITY / MUNICIPALITY OF ", CITY),
                                       ("Barangay ", BARANGAY))):
        p = mid.paragraphs[0] if i == 0 else mid.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        if i == 0:
            p.paragraph_format.space_before = Pt(12)
        p.add_run(text)
        if value:
            run = p.add_run(value)
            run.bold = run.underline = True

    p = _para(doc, 6)
    p.paragraph_format.space_before = Pt(8)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("OFFICE OF THE PUNONG BARANGAY /LUPONG TAGAPAMAYAPA").bold = True
    _bottom_rule(p)


def _thumb_box(cell):
    box = cell.add_table(rows=1, cols=1)
    box.style = "Table Grid"
    inner = box.rows[0].cells[0]
    inner.width = Inches(1.1)
    p = inner.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(8)
    p.add_run("THUMB\nMARK")


def _form(doc, f):
    _header(doc)

    p = _para(doc, 12)
    p.paragraph_format.space_before = Pt(14)
    stops = p.paragraph_format.tab_stops
    stops.add_tab_stop(Inches(3.25), WD_TAB_ALIGNMENT.CENTER)
    stops.add_tab_stop(Inches(4.9))
    p.add_run("\tBLOTTER FORM\tNO: ").bold = True
    p.add_run(f["blotter_no"] or "_" * 18).underline = True

    _line(doc, "Petsa ng Pagblotter:", f["date_reported"], bold=True, space_after=10)

    for letter, title, who, addr, contact, age in (
            ("A", "Pangalan ng Nag-blo-blotter/Nagrereklamo:", "reporting_party",
             "complainant_address", "complainant_contact", "complainant_age"),
            ("B", "Pangalan ng Inirereklamo:", "respondent",
             "respondent_address", "respondent_contact", "respondent_age")):
        _line(doc, f"{letter}. {title}", f[who], bold=True)
        _line(doc, "Tirahan:", f[addr])
        _line(doc, "Contact No.:", f[contact], right=("Edad:", f[age]), space_after=10)

    _line(doc, "C. REKLAMO", f["complaint"], bold=True, space_after=10)

    _para(doc).add_run("D. KAGANAPAN NG PANGYAYARI:").bold = True
    _line(doc, "Petsa:", f["date"], right=("Oras:", f["time"]))
    _line(doc, None, "", right=("Lugar:", f["location_purok"]), space_after=10)

    _para(doc, 4).add_run("E. SALAYSAY").bold = True
    for chunk in (f["incident_summary"] or "").split("\n"):
        p = _para(doc)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.add_run(chunk)
    _para(doc, 4)

    sig = doc.add_table(rows=1, cols=2)
    _no_borders(sig)
    left, right = sig.rows[0].cells
    left.width, right.width = Inches(4.3), Inches(2.2)
    left.paragraphs[0].text = "LAGDA NG SUMULAT NG SALAYSAY:"
    _field(left.add_paragraph(), "Pangalan ng Sumulat ng Salaysay:", f["statement_writer"])
    _field(left.add_paragraph(), "Petsa:", f["statement_date"])
    right.paragraphs[0].text = "☐ Walang kakayahang magsulat"
    _field(right.add_paragraph(), "Oras:", f["statement_time"])
    _thumb_box(right)

    p = _para(doc)
    p.paragraph_format.space_before = Pt(6)
    p.add_run("(Mga) Testigo: (Maaaring magsulat sa likod)")
    _line(doc, "Pangalan:", f["witness"])
    _line(doc, "Edad:", f["witness_age"])
    _line(doc, "Tirahan:", f["witness_address"])
    _line(doc, "Contact No.:", f["witness_contact"], space_after=12)

    _line(doc, "Pangalan ng Barangay Desk Officer / Imbestigador:", f["desk_officer"], space_after=12)
    _line(doc, "Lagda ng Barangay Desk Officer / Imbestigador:", "")


def build(fields_list):
    """A .docx (bytes) with one BLOTTER FORM per slot dict, each on its own page."""
    doc = Document()
    for section in doc.sections:
        section.top_margin = section.bottom_margin = Inches(0.5)
        section.left_margin = section.right_margin = Inches(0.8)
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)

    for i, fields in enumerate(fields_list):
        if i:
            doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
        _form(doc, fields)

    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()
