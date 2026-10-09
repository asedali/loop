"""
Identifier import for Phase 1 — turn an ORCID iD, a DOI, or an arXiv ID into
editable material, without the user writing a prompt or opening a PDF.

This is the same round-trip M1.1's file upload takes: it produces *text* in the
Phase 1 textarea, and submitting that text is an ordinary `POST /phase1/extract`.
That matters for three reasons. There is no new AI-call path, so there is no
quota change and nothing new for the model prompt to get wrong. The user sees and
edits the result before any money is spent. And because the text arrives by the
one route that already exists, it is wrapped as untrusted data on the way to the
model exactly like a paste — fetched metadata is third-party text that may contain
instructions, and this module does not get a pass.

SSRF is the risk that decides the design
-----------------------------------------
Fetching a URL built from what the user typed is the classic server-side request
forgery hole, and "validate the URL" does not fix it — allowlists, private-IP
blocklists and redirect limits are each bypassable, and a blocklist is worse than
useless because it implies a safety that is not there. So the user never builds a
URL:

    * An adapter recognises the input by matching a **strict regex anchored to the
      whole string**, and only the captured group is used. Anchoring matters: a
      pattern matched with `search` would accept a URL carrying a payload in its
      query string, which is exactly the shape being avoided.
    * That captured group is **percent-encoded** before it goes anywhere near a
      URL, and percent-encoding is the actual gate. `quote(..., safe="")` can only
      emit unreserved characters and `%XX`, so by construction nothing that could
      change the request survives: no `/` (so no extra path segment), no `:`
      (no port or scheme change), no `@` (no userinfo), no `?` or `#`, no
      whitespace. A whitelist here would be strictly weaker *and* would reject
      valid DOIs, which legitimately contain most of printable ASCII.
    * The URL is a **hardcoded template with a hardcoded host**. There is no path
      by which user input chooses the host, the scheme, or the port.
    * **Redirects are not followed.** A 302 from an allowlisted host to
      `169.254.169.254` is the whole attack, and refusing to follow redirects
      removes it without needing to inspect a Location header.

Given that, the fetch is a plain synchronous GET with a timeout and a response
cap. There is deliberately no async client and no connection pool: these are
three independent public APIs, called a handful of times per session, by a
token-bucket-limited route.

Failures are sentences, not diagnostics. A wrong DOI, a rate-limited arXiv and a
DNS failure are all things the user can act on, and none of them is a bug report.
A httpx traceback in the UI tells them nothing and tells an attacker which client
library is installed.
"""
import json
import re
import threading
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from urllib.parse import quote

import httpx

from . import config

# Cap on cache entries, same reasoning as ratelimit._MAX_BUCKETS: the key is
# attacker-influenced (it is derived from what the user typed), so the map must be
# bounded or a loop over distinct identifiers grows it for the life of the process.
_MAX_CACHE = 1_000

# The invariant percent-encoding guarantees. Asserted on the encoded form rather
# than used as a filter on the raw one — see the module docstring for why a
# whitelist on the raw identifier would be both weaker and wrong for DOIs.
_ENCODED_SAFE = re.compile(r"^[A-Za-z0-9._~%-]+$")


class SourceError(Exception):
    """A problem with the identifier the user can act on. The message is shown to
    them verbatim, so it must be a sentence, not a diagnostic."""


class Record:
    """One fetched record, normalised across the three providers.

    `body` is the material text. `label` names the thing for the preview banner.
    """

    def __init__(self, source: str, identifier: str, label: str, body: str,
                 url: str = ""):
        self.source = source
        self.identifier = identifier
        self.label = label
        self.body = body
        self.url = url


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_cache: "OrderedDict[tuple, tuple]" = OrderedDict()


def _cache_get(key):
    with _lock:
        hit = _cache.get(key)
        if hit is None:
            return None
        expires, value = hit
        if expires <= time.monotonic():
            _cache.pop(key, None)
            return None
        _cache.move_to_end(key)
        return value


def _cache_put(key, value):
    ttl = config.import_cache_ttl_seconds()
    if ttl <= 0:
        return
    with _lock:
        while len(_cache) >= _MAX_CACHE:
            _cache.popitem(last=False)
        _cache[key] = (time.monotonic() + ttl, value)


def cache_clear() -> None:
    """Drop every cached record. Used by the tests, and the lever to pull if a bad
    record turns out to have been cached."""
    with _lock:
        _cache.clear()


# ---------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------

def _encode(identifier: str) -> str:
    """Percent-encode a captured identifier so it can be interpolated safely."""
    encoded = quote(identifier, safe="")
    assert _ENCODED_SAFE.match(encoded), "percent-encoding produced an unexpected character"
    return encoded


def _get(url: str, accept: str) -> bytes:
    """GET with a hard timeout, a streaming response cap, and no redirects.

    The cap is applied while iterating the body rather than by trusting
    Content-Length, which is the same reasoning as MAX_UPLOAD_BYTES: the header is
    controlled by whoever is answering.
    """
    limit = config.import_max_response_bytes()
    timeout = httpx.Timeout(config.import_timeout_seconds())
    try:
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            with client.stream("GET", url, headers={"Accept": accept,
                                                    "User-Agent": "LaunchLoop/1.0"}) as response:
                if response.status_code >= 400:
                    raise SourceError(_http_error_sentence(response.status_code))
                data = bytearray()
                for chunk in response.iter_bytes():
                    data.extend(chunk)
                    if len(data) > limit:
                        raise SourceError(
                            "That source returned an unexpectedly large response, so "
                            "the import was stopped. Paste the material instead.")
    except SourceError:
        raise
    except httpx.TimeoutException:
        raise SourceError(
            "That source took too long to answer. Try again in a moment, or paste "
            "the material directly.")
    except httpx.RequestError:
        # Deliberately names neither the exception class nor the URL. Which client
        # is in use is not the user's business, and echoing a URL back into a page
        # is how reflected content gets into a screenshot.
        raise SourceError(
            "Couldn't reach that source — it's either down or blocked from this "
            "network. Paste the material instead.")
    return bytes(data)


def _http_error_sentence(status: int) -> str:
    """A 404 from a metadata API means "no such record", by far the most common
    failure and worth its own sentence. Everything else is one sentence. The status
    is not leaked: a 401/403 here would mean *our* contact details are wrong, which
    is our problem and nothing the user did."""
    if status == 404:
        return ("No record found for that identifier. Check it for typos — a DOI "
                "is case-insensitive but the rest has to be exact.")
    if status == 429:
        return "That source is rate-limiting us right now. Wait a minute and try again."
    if status in (401, 403):
        return "That source refused the request. Paste the material instead."
    if 500 <= status < 600:
        return "That source is having problems at its end. Try again shortly."
    return ("That source wouldn't answer that request. Check the identifier, or "
            "paste the material directly.")


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------

# Registration order is match order, and it is deliberate. The DOI pattern is the
# loosest of the three, so it is registered last: a near-miss is better reported as
# "not recognised" than silently fetched from the wrong provider.
_ADAPTERS = []


def _adapter(name, pattern):
    """Register a matcher. `pattern` must be anchored to the whole string and
    expose exactly one capturing group: that group, and nothing else, is what gets
    encoded and interpolated."""
    compiled = re.compile(pattern, re.IGNORECASE)

    def register(fn):
        _ADAPTERS.append((name, compiled, fn))
        return fn

    return register


@_adapter("ORCID", r"^\s*(?:https?://orcid\.org/)?(\d{4}-\d{4}-\d{4}-\d{3}[\dXx])\s*$")
def _fetch_orcid(identifier: str) -> Record:
    # The iD's check digit is verified by ORCID itself, so a mistyped one comes back
    # as 404 and _http_error_sentence already explains that. Encoding is a no-op
    # here (digits and dashes are unreserved) and is applied anyway so that all
    # three adapters are defended by the same mechanism.
    url = f"https://pub.orcid.org/v3.0/{_encode(identifier)}/record"
    data = _get(url, "application/json")

    try:
        payload = json.loads(data)
    except ValueError:
        raise SourceError("ORCID sent back something unreadable. Try again shortly.")

    person = payload.get("person") or {}
    name = _orcid_name(person)

    # A work group is found by its *path* in the document, not by array position,
    # so ordering is never assumed.
    groups = []
    summary = (payload.get("activities-summary") or {}).get("works") or {}
    for group in (summary.get("group") or []):
        entries = group.get("work-summary") or [{}]
        entry = entries[0] or {}
        title = ((entry.get("title") or {}).get("title") or {}).get("value")
        if not title:
            continue
        year = None
        for key in ("publication-date", "created-date"):
            date = ((entry.get(key) or {}).get("year") or {}).get("value")
            if date:
                year = str(date)
                break
        journal = _orcid_source_name(entry.get("journal-title"))
        groups.append((title, year, journal))

    if not groups:
        raise SourceError(
            "That ORCID iD has no public works on it. Only public records can be "
            "read — check the visibility settings, or paste your material.")

    return Record(
        source="ORCID",
        identifier=identifier,
        label=name or identifier,
        url=f"https://orcid.org/{identifier}",
        body=_render_works(identifier, name, groups),
    )


def _orcid_name(person) -> str:
    for key in ("credit-name", "other-names"):
        entries = (person.get(key) or {}).get("given-and-family-names") or []
        for entry in entries:
            if isinstance(entry, dict) and entry.get("value"):
                return entry["value"].strip()
    return ""


def _orcid_source_name(value) -> str:
    if isinstance(value, dict):
        return (value.get("value") or "").strip()
    return ""


def _render_works(identifier, name, groups) -> str:
    """An ORCID iD describes a person, not a work, so the material is the work list
    — which is what Phase 1 actually needs, since the titles are the claims."""
    limit = config.import_max_works()
    shown = groups[:limit]
    lines = [
        f"Publication record imported from ORCID iD {identifier}"
        + (f" ({name})." if name else "."),
        "",
        (f"Titles of {len(groups)} public works" if len(groups) <= limit
         else f"Titles of {len(groups)} public works (showing the first {limit})")
        + ":",
        "",
    ]
    for index, (title, year, journal) in enumerate(shown, start=1):
        suffix = f" ({year})" if year else ""
        if journal:
            suffix += f", {journal}"
        lines.append(f"{index}. {title.strip()}{suffix}")
    if len(groups) > limit:
        lines += ["", f"[{len(groups) - limit} further works were not included.]"]
    return "\n".join(lines)


_ARXIV_ID = r"\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?"


@_adapter("arXiv", rf"^\s*(?:https?://arxiv\.org/abs/|arxiv:\s*)?({_ARXIV_ID})\s*$")
def _fetch_arxiv(identifier: str) -> Record:
    # The export API rather than resolving an /abs/ URL: it answers in Atom, and
    # passing the identifier as an encoded query *parameter* means even the `/` in
    # an old-style ID cannot be read as a path separator.
    url = f"https://export.arxiv.org/api/query?id_list={_encode(identifier)}&max_results=1"
    data = _get(url, "application/atom+xml")

    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        raise SourceError("arXiv sent back something unreadable. Try again shortly.")

    # Atom namespaces are fixed by the spec, so the exact URIs are matched rather
    # than guessed. ElementTree does not resolve external entities and raises on
    # undefined ones, and the body is already capped by _get.
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entry = root.find("a:entry", ns)
    if entry is None:
        raise SourceError(_arxiv_missing())
    title = _text(entry.find("a:title", ns))
    # For an unknown ID arXiv returns a well-formed feed whose single entry is
    # titled "Error", so a missing title is not the only signal of failure.
    if not title or title.lower().startswith("error"):
        raise SourceError(_arxiv_missing())

    abstract = _text(entry.find("a:summary", ns))
    authors = [a for a in (_text(n.find("a:name", ns))
                            for n in entry.findall("a:author", ns)) if a]
    published = _text(entry.find("a:published", ns))[:10]
    doi = _text(entry.find("{http://arxiv.org/schemas/atom}doi", ns))
    link = _text(entry.find("a:id", ns))

    cap = config.max_extracted_chars()
    body = _join([
        f"Preprint imported from arXiv ({identifier}).",
        f"Title: {title}",
        f"Authors: {', '.join(authors) if authors else 'not listed'}",
        f"Published: {published}" if published else "",
        f"DOI: {doi}" if doi else "",
        "",
        "Abstract:",
        abstract or "[no abstract supplied]",
    ])[:cap]
    return Record(source="arXiv", identifier=identifier,
                  label=f"{identifier} — {title}", url=link, body=body)


def _arxiv_missing() -> str:
    return ("No arXiv paper found for that ID. New-style IDs look like 2301.01234; "
            "old-style ones like hep-th/9901001.")


@_adapter("DOI", r"^\s*(?:https?://(?:dx\.)?doi\.org/|doi:\s*)?(10\.\d{4,9}/\S+)\s*$")
def _fetch_doi(identifier: str) -> Record:
    # Everything after the slash is encoded with safe="", including any `#`: a DOI
    # suffix can legitimately contain one, and stripping it would silently change
    # which paper is fetched. The slash becomes %2F, which is what a path segment
    # needs.
    url = "https://api.crossref.org/works/" + _encode(identifier)
    data = _get(url, "application/json")

    try:
        message = json.loads(data)["message"]
    except (ValueError, KeyError, TypeError):
        raise SourceError("Crossref sent back something unreadable. Try again shortly.")

    titles = message.get("title") or []
    title = (titles[0] if titles else "").strip()
    if not title:
        raise SourceError("That DOI has no title on record, so there is nothing to analyse.")

    authors = []
    for author in (message.get("author") or []):
        name = " ".join(p for p in ((author.get("given") or "").strip(),
                                    (author.get("family") or "").strip()) if p)
        name = name or (author.get("name") or "").strip()
        if name:
            authors.append(name)

    container = (message.get("container-title") or [""])[0].strip()
    year = None
    for key in ("published-print", "published-online", "issued", "created"):
        parts = (((message.get(key) or {}).get("date-parts")) or [[]])[0]
        if parts:
            year = str(parts[0])
            break

    # Only some publishers register an abstract, and those that do send it as a
    # JATS XML fragment.
    abstract = _strip_jats(message.get("abstract") or "")

    cap = config.max_extracted_chars()
    body = _join([
        f"Journal article imported via Crossref (DOI {identifier}).",
        f"Title: {title}",
        f"Authors: {', '.join(authors) if authors else 'not listed'}",
        f"Published: {year}" if year else "",
        f"Venue: {container}" if container else "",
        "",
        "Abstract:",
        abstract or "[no abstract on record — the title and authors are all there is]",
    ])[:cap]
    return Record(source="DOI", identifier=identifier,
                  label=f"{title} ({year})" if year else title,
                  url=f"https://doi.org/{identifier}", body=body)


def _text(node) -> str:
    """Flattened text of an element. arXiv folds abstracts across lines, so the
    internal whitespace is collapsed or the prompt gets ragged text."""
    if node is None or not node.text:
        return ""
    return re.sub(r"\s+", " ", node.text).strip()


_TAG = re.compile(r"<[^>]+>")
_NUMERIC_ENTITY = re.compile(r"&#(x?)([0-9A-Fa-f]+);")

# XML's five predefined entities, then numeric character references. The named
# ones are replaced in this order, and exactly once, on purpose: `&amp;lt;` has no
# `&lt;` substring to match on the first pass, so it correctly survives as `&lt;`
# rather than being decoded twice into a `<`.
_ENTITIES = (("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'),
             ("&apos;", "'"), ("&amp;", "&"))


def _numeric_entity(match) -> str:
    """`&#181;` or `&#xB5;` to the character. Numeric references are how XML spells
    anything outside ASCII, and a JATS abstract is full of them — leaving them
    literal would put "3&#181;m-pitch" in front of the model."""
    digits = match.group(2)
    try:
        code = int(digits, 16 if match.group(1) else 10)
    except ValueError:
        return match.group(0)
    # Bounds-checked rather than trusted: chr() on a huge code point is a ValueError,
    # and surrogates would produce a string the model would only see as noise.
    return chr(code) if 0 < code < 0x110000 and not 0xD800 <= code <= 0xDFFF else ""


def _strip_jats(fragment: str) -> str:
    """Crossref abstracts arrive as JATS XML, reduced to readable prose.

    Tags are removed with a regex rather than by parsing. The case for a parser is
    weak — this is display text going into a prompt, not a security boundary, and
    publishers use JATS's tags inconsistently enough that a strict parse would fail
    on the ones we do not handle. The case against is the one that matters:
    ElementTree is the thing that would have to defend against entity expansion, so
    not using it removes the question entirely.

    Entities are decoded *before* tags are stripped, and that order is not
    arbitrary: `&lt;em&gt;` is an escaped tag, and decoding after stripping would
    leave the reader looking at live `<em>` markup. Decoding first means escaped
    markup is stripped too, which is the outcome wanted here — the goal is prose,
    not a faithful round-trip of the publisher's XML.
    """
    if not fragment:
        return ""
    text = _NUMERIC_ENTITY.sub(_numeric_entity, fragment)
    for entity, char in _ENTITIES:
        text = text.replace(entity, char)
    text = _TAG.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()[: config.max_extracted_chars()]


def _join(parts) -> str:
    return "\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def import_material(raw: str) -> Record:
    """Turn a pasted identifier into editable Phase 1 material.

    Raises `SourceError` — a sentence — for anything the user can act on. The cache
    is consulted after identification but before the network, so re-importing the
    same identifier costs nothing and cannot fail differently on the second try.
    """
    raw = (raw or "").strip()
    if not raw:
        raise SourceError("Paste an identifier first.")
    if not config.source_import_enabled():
        raise SourceError("Identifier import is turned off on this deployment.")

    match = _match(raw)
    if match is None:
        raise SourceError(
            "That doesn't look like an ORCID iD, a DOI, or an arXiv ID. An ORCID is "
            "0000-0002-1825-0097, a DOI is 10.1000/xyz123, an arXiv ID is "
            "2301.01234 or hep-th/9901001.")
    name, compiled, fetch = match
    identifier = compiled.match(raw).group(1).strip()

    key = (name, identifier.lower())
    cached = _cache_get(key)
    if cached is not None:
        return cached

    record = fetch(identifier)
    _cache_put(key, record)
    return record


def _match(raw: str):
    """First adapter whose anchored pattern matches, else None."""
    for adapter in _ADAPTERS:
        if adapter[1].match(raw):
            return adapter
    return None


def supported() -> list:
    """For the preview hint in the template."""
    return ["ORCID iD", "DOI", "arXiv ID"]