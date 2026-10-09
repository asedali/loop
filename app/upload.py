"""
Text extraction from uploaded files.

An upload is read into memory, turned into text, and **discarded**. Nothing is
written to disk and no path is ever constructed, so there is nothing to traverse,
clean up, or scan later. `MAX_UPLOAD_BYTES` is enforced while reading rather than
trusted from `Content-Length`, because the header is attacker-controlled.

Format detection is by magic bytes on the actual content. The browser's
`Content-Type` and the filename extension are both supplied by whoever is
uploading, so treating either as evidence would mean the check is only as good
as the attacker's willingness to lie.

Every failure is converted into a plain-English sentence. A malformed PDF is a
user problem, not a bug report, and a parser traceback in the UI tells them
nothing they can act on while telling an attacker which library is in use.

The one genuinely subtle piece is the deadline. `MAX_PARSE_SECONDS` is checked
*between PDF pages* rather than passed to a `future.result(timeout=...)`: a thread
in CPython cannot be cancelled, so a future timeout bounds only the response,
not the work, and a slow document would keep burning a worker after the user had
already been answered. Checking per page is the only thing that actually stops
`pypdf`, which is also the parser that decompresses content streams.

Residual risk, stated rather than hidden: a document engineered to be slow *per
page* can still consume up to the deadline. That is a bounded window rather than
an unbounded one. Making the bound hard would need a subprocess, which is not
worth it until uploads are a real attack surface.
"""
import io
import time
import zipfile

import docx
import pypdf

from . import config

# Magic bytes. `%PDF-` per the spec (a PDF may legally have up to 1024 bytes of
# junk before it, which readers tolerate; we accept the normal case and let
# pypdf decide).
_PDF = b"%PDF-"
# A ZIP local file header. Necessary but not sufficient for DOCX — see _sniff.
_ZIP = b"PK\x03\x04"

# An OOXML package always contains this part; a plain .zip does not. That is how
# the two are told apart. Compared against namelist() entries, which are str.
_OOXML_MARKER = "[Content_Types].xml"


class UploadError(Exception):
    """A problem with the upload the user can act on. The message is shown to
    them verbatim, so it must be a sentence, not a diagnostic."""


def sniff(head: bytes) -> str:
    """One of `pdf`, `docx`, `text`, or `unsupported` — from the bytes alone."""
    if head.startswith(_PDF):
        return "pdf"
    if head.startswith(_ZIP):
        # A ZIP claiming to be OOXML. Verified properly in _extract_docx, because
        # an attacker controls these bytes.
        return "docx"
    # Binary-ish content is not text. Checked up front so a renamed binary is
    # refused rather than decoded into mojibake.
    if b"\x00" in head[:2048]:
        return "unsupported"
    return "text"


def extract(data: bytes, filename: str = "", deadline_s: float = None) -> str:
    """Bytes in, text out. Raises `UploadError` with a sentence a user can act on."""
    if not data:
        raise UploadError("That file is empty.")
    limit = config.max_upload_bytes()
    if len(data) > limit:
        raise UploadError(
            f"That file is {len(data) // 1048576} MB. The limit is "
            f"{limit // 1048576} MB — try the relevant chapter or section.")

    kind = sniff(data[:8])
    if kind == "unsupported":
        raise UploadError(
            "That doesn't look like a PDF, a Word document, or a text file. "
            "If it is a scan or an image, there is no text to read — paste the "
            "text instead.")

    # One absolute deadline, computed here and checked by the parsers. A module
    # global would be wrong: it would carry the first request's start time into
    # every later one, so the budget would shrink over the life of the process.
    budget = config.max_parse_seconds() if deadline_s is None else deadline_s
    deadline = time.monotonic() + budget
    if kind == "pdf":
        text = _extract_pdf(data, deadline)
    elif kind == "docx":
        text = _extract_docx(data, deadline)
    else:
        text = _extract_text(data)

    text = text.strip()
    if not text.strip():
        raise UploadError(
            "No text came out of that file. If it is a scanned document, there is "
            "no text layer to read — open it, select the text and paste it here.")
    return _truncate(text)


def _extract_pdf(data: bytes, deadline: float) -> str:
    try:
        reader = pypdf.PdfReader(io.BytesIO(data), strict=False)
    except Exception:
        raise UploadError(
            "That PDF could not be read — it may be damaged or not really a PDF.")

    if getattr(reader, "is_encrypted", False):
        # pypdf can often open these with an empty password, but a PDF with a real
        # password is not something to guess at.
        raise UploadError(
            "That PDF is password-protected. Open it, save an unprotected copy, "
            "and upload that.")

    max_pages = config.max_pdf_pages()
    pages = len(reader.pages)
    if pages == 0:
        raise UploadError("That PDF has no pages.")
    if pages > max_pages:
        raise UploadError(
            f"That PDF has {pages} pages; the limit is {max_pages}. Upload the "
            "chapter or section you care about.")

    out = []
    for index, page in enumerate(reader.pages):
        # The deadline that actually does something — see the module docstring.
        if _over_deadline(deadline):
            out.append(f"\n\n[extraction stopped after {index} pages: the file "
                       f"took too long to read]")
            break
        try:
            out.append(page.extract_text() or "")
        except Exception:
            # One unreadable page should not lose the other 199.
            out.append(f"\n[page {index + 1} could not be read]\n")
    return "\n\n".join(out)


def _extract_docx(data: bytes, deadline: float) -> str:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise UploadError("That Word file is damaged and could not be opened.")

    # A ZIP is not a Word document. Without this, any archive would be reported as
    # one and then fail confusingly further in — or, worse, be read as a document.
    if not any(_OOXML_MARKER in name for name in archive.namelist()):
        raise UploadError(
            "That is a zip archive, not a Word document. If it is a .docx, "
            "re-save it as .docx and try again.")
    # A decompression bomb is bounded here rather than at the request boundary.
    if sum(i.file_size for i in archive.infolist()) > config.max_upload_bytes() * 8:
        raise UploadError("That Word file expands to an implausible size.")

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception:
        raise UploadError("That Word file could not be read.")

    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
        if _over_deadline(deadline):
            break
    return "\n".join(parts)


def _extract_text(data: bytes) -> str:
    # errors="replace" so a stray non-UTF-8 byte produces U+FFFD rather than a
    # 500 on a file that is otherwise readable.
    return data.decode("utf-8", errors="replace")


def _over_deadline(deadline: float) -> bool:
    return time.monotonic() > deadline


def _truncate(text: str) -> str:
    cap = config.max_extracted_chars()
    if len(text) <= cap:
        return text
    return (text[:cap]
            + f"\n\n[extraction truncated at {cap} characters — the file was longer. "
            "The text above is what will be analysed.]")