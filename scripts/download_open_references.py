"""Download openly licensed (CC0) reference images for the Klein style experiments.

All images come from the Art Institute of Chicago (AIC) Open Access programme:
artworks that AIC marks as public domain are released under CC0 1.0, and the
images are served through AIC's public IIIF API (https://api.artic.edu/docs/).

For every artwork the script
  1. fetches the record from the AIC API and REFUSES to download it unless
     `is_public_domain` is true (so a licensing change on AIC's side is caught);
  2. downloads the image through IIIF;
  3. writes provenance next to the images:
       data/reference_images/open/sources.json   machine-readable metadata
       data/reference_images/open/SOURCES.md     human-readable credit table
       data/reference_images/open/references.bib BibTeX entries for the paper
       data/reference_images/open/style_tags.tsv text-SHIFT baseline tags

Run once from the repository root on a machine with internet access:

    python scripts/download_open_references.py
    python scripts/download_open_references.py --width 1686   # larger copies

Image licence: CC0 1.0 (https://creativecommons.org/publicdomain/zero/1.0/).
Attribution is not legally required for CC0, but credit the museum anyway
(SOURCES.md) and cite the artworks in the paper (references.bib).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

API = "https://api.artic.edu/api/v1/artworks/{id}?fields={fields}"
IIIF = "{iiif}/{image_id}/full/{width},/0/default.jpg"
FIELDS = ",".join([
    "id", "title", "artist_title", "artist_display", "date_display", "medium_display",
    "dimensions", "main_reference_number", "credit_line", "is_public_domain", "image_id",
    "style_title", "classification_title",
])
# AIC asks API users to identify themselves; adjust the contact if you redistribute the script.
HEADERS = {"AIC-User-Agent": "SHIFT-klein-style-steering (research; contact: gugmelik@gmail.com)",
           "User-Agent": "SHIFT-klein-style-steering/1.0"}

# AIC artwork ids, chosen to span distinct, visually separable styles.
# slug -> (aic_id, style label used in the paper, text tag for the text-SHIFT baseline)
# V2_REFERENCES: the seven references of the second manuscript version (kept for reproducibility).
V2_REFERENCES = {
    "hartley_movement":        (65916, "cubist-influenced abstraction",
                                ", in a cubist style with fragmented geometric planes"),
    "kandinsky_improvisation30": (8991, "expressionist abstraction",
                                ", in the style of an expressionist abstract painting by Wassily Kandinsky"),
    "vangogh_self_portrait":   (80607, "post-impressionism, impasto brushwork",
                                ", in the style of Vincent van Gogh with thick visible brushstrokes"),
    "seurat_grande_jatte":     (27992, "pointillism",
                                ", in pointillist style made of small dots of colour"),
    "monet_water_lilies":      (16568, "impressionism",
                                ", as an impressionist painting by Claude Monet"),
    "hokusai_great_wave":      (24645, "ukiyo-e woodblock print",
                                ", as a Japanese ukiyo-e woodblock print"),
    "degas_jockey":            (7157, "drawing / sketch",
                                ", as a loose hand-drawn sketch"),
}

# Revision 3: four references.
#  * two widely known styles (the model knows them by name): Monet, Van Gogh  - AIC, CC0;
#  * two styles of little-known painters, unlikely to be represented in the training data by name
#    or by many images: M. K. Ciurlionis (Lithuanian symbolism, d. 1911) and N. Pirosmani (Georgian
#    naive painting, d. 1918). Both are public domain worldwide (author died > 100 years ago);
#    images come from Wikimedia Commons (PD-Art). "Unlikely to be seen" is checked empirically by
#    the name probe (stage nameprobe): prompting the model with the artist's name.
# Commons entries: dict(query=Commons search, artist=substring that must occur in the Artist field).
REFERENCES = {
    "monet_water_lilies":      (16568, "impressionism",
                                ", as an impressionist painting by Claude Monet"),
    "vangogh_self_portrait":   (80607, "post-impressionism, impasto brushwork",
                                ", in the style of Vincent van Gogh with thick visible brushstrokes"),
    "ciurlionis_sonata_sun":   (dict(query="Čiurlionis Sonata of the Sun Allegro", artist="ciurlion"),
                                "symbolism, tempera on paper (little-known)",
                                ", as a symbolist tempera painting by Mikalojus Konstantinas Ciurlionis"),
    "pirosmani_giraffe":       (dict(query="Pirosmani Giraffe", artist="pirosman"),
                                "naive painting on oilcloth (little-known)",
                                ", as a naive painting on black oilcloth by Niko Pirosmani"),
}
# Artist name only (no style words): the name probe checks whether the model knows the style by name.
NAME_TAGS = {
    "monet_water_lilies": ", in the style of Claude Monet",
    "vangogh_self_portrait": ", in the style of Vincent van Gogh",
    "ciurlionis_sonata_sun": ", in the style of Mikalojus Konstantinas Ciurlionis",
    "pirosmani_giraffe": ", in the style of Niko Pirosmani",
    "monet_water_lily_pond": ", in the style of Claude Monet",
    "vangogh_bedroom": ", in the style of Vincent van Gogh",
}

# Second image of the same artist / style for the stability check of the extracted shift
# (review item 3). Each twin is compared with the main reference named in TWIN_OF.
TWIN_REFERENCES = {
    "monet_water_lily_pond":   (87088, "impressionism",
                                ", as an impressionist painting by Claude Monet"),
    "vangogh_bedroom":         (28560, "post-impressionism, impasto brushwork",
                                ", in the style of Vincent van Gogh with thick visible brushstrokes"),
}
TWIN_OF = {"monet_water_lily_pond": "monet_water_lilies", "vangogh_bedroom": "vangogh_self_portrait"}
SETS = {"main": (REFERENCES, "data/reference_images/open"),
        "twins": (TWIN_REFERENCES, "data/reference_images/twins"),
        "v2": (V2_REFERENCES, "data/reference_images/v2")}
COMMONS_API = "https://commons.wikimedia.org/w/api.php"


def get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def download(url: str, path: Path) -> None:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=120) as resp:
        path.write_bytes(resp.read())


def _ascii(text: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()


def _strip_html(text: str) -> str:
    import html
    import re
    return html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip()


def commons_lookup(spec: dict, width: int, exact_file: str = None) -> dict:
    """Find a public-domain bitmap on Wikimedia Commons: the first search hit whose licence is public
    domain and whose Artist field contains spec['artist']. --commons_file pins an exact file."""
    import urllib.parse
    common = {"action": "query", "format": "json", "prop": "imageinfo",
              "iiprop": "url|size|mime|extmetadata", "iiurlwidth": str(width)}
    if exact_file:
        common["titles"] = exact_file if exact_file.startswith("File:") else f"File:{exact_file}"
    else:
        common.update(generator="search", gsrnamespace="6", gsrlimit="20",
                      gsrsearch=f"{spec['query']} filetype:bitmap")
    pages = get_json(f"{COMMONS_API}?{urllib.parse.urlencode(common)}").get("query", {}).get("pages", {})
    for page in sorted(pages.values(), key=lambda p: p.get("index", 0)):
        info = (page.get("imageinfo") or [{}])[0]
        meta = {k: v.get("value") for k, v in info.get("extmetadata", {}).items()}
        licence = _ascii(meta.get("LicenseShortName", "") + " " + meta.get("License", ""))
        artist = _ascii(_strip_html(meta.get("Artist", "")))
        ok_lic = "public domain" in licence or licence.startswith("pd") or " pd" in licence
        if not (ok_lic and spec["artist"] in artist and info.get("mime") in ("image/jpeg", "image/png")
                and info.get("width", 0) >= 600):
            continue
        return {"file_title": page["title"], "image_url": info.get("thumburl") or info["url"],
                "page_url": info.get("descriptionurl"), "title": _strip_html(meta.get("ObjectName", "")) or page["title"],
                "artist": _strip_html(meta.get("Artist", "")), "date": _strip_html(meta.get("DateTimeOriginal", "")),
                "credit": _strip_html(meta.get("Credit", "")), "license_short": meta.get("LicenseShortName", ""),
                "license_url": meta.get("LicenseUrl") or "https://commons.wikimedia.org/wiki/Commons:Licensing"}
    raise RuntimeError(f"no public-domain Commons image matched {spec} (pin one with --commons_file)")


def bibtex_key(slug: str) -> str:
    return "ref_" + slug.replace("-", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=sorted(SETS), default="main",
                        help="main: the four revision-3 references; twins: same-style second images; "
                             "v2: the seven references of manuscript version 2")
    parser.add_argument("--out_dir", default=None, help="default: data/reference_images/{open,twins}")
    parser.add_argument("--width", type=int, default=843,
                        help="IIIF width in px (AIC recommends 843; 1686 for larger copies)")
    parser.add_argument("--only", nargs="*", default=None, help="Subset of slugs to download")
    parser.add_argument("--commons_file", nargs="*", default=[],
                        help="Pin Commons files: slug=File:Exact_name.jpg (otherwise the first PD search hit)")
    args = parser.parse_args()

    refs, default_dir = SETS[args.set]
    out = Path(args.out_dir or default_dir)
    out.mkdir(parents=True, exist_ok=True)
    records, failures = [], []

    pinned = dict(x.split("=", 1) for x in args.commons_file)
    for slug, (aic_id, style, tag) in refs.items():
        if args.only and slug not in args.only:
            continue
        if isinstance(aic_id, dict):  # Wikimedia Commons entry
            try:
                c = commons_lookup(aic_id, args.width, pinned.get(slug))
                path = out / f"{slug}.jpg"
                download(c["image_url"], path)
                record = {"slug": slug, "file": path.name, "style": style, "text_tag": tag,
                          "name_tag": NAME_TAGS.get(slug, ""), "aic_id": None,
                          "title": c["title"], "artist": c["artist"], "artist_display": c["artist"],
                          "date": c["date"], "medium": "", "dimensions": "", "reference_number": c["file_title"],
                          "credit_line": c["credit"], "classification": "painting",
                          "institution": "Wikimedia Commons", "page_url": c["page_url"], "image_url": c["image_url"],
                          "license": f"Public domain ({c['license_short']}; author died more than 100 years ago)",
                          "license_url": c["license_url"], "retrieved": time.strftime("%Y-%m-%d")}
                records.append(record)
                print(f"ok   {slug:<28} {record['artist']}, {record['title']} ({record['date']})  <- {c['file_title']}")
                print(f"     CHECK the image visually: {c['page_url']}")
            except Exception as exc:  # noqa: BLE001
                failures.append((slug, str(exc)))
                print(f"FAIL {slug:<28} {exc}", file=sys.stderr)
            time.sleep(0.5)
            continue
        try:
            payload = get_json(API.format(id=aic_id, fields=FIELDS))
            meta, config = payload["data"], payload.get("config", {})
            if not meta.get("is_public_domain"):
                raise RuntimeError("AIC no longer marks this artwork as public domain; skipped")
            if not meta.get("image_id"):
                raise RuntimeError("no image available")
            iiif = config.get("iiif_url", "https://www.artic.edu/iiif/2")
            image_url = IIIF.format(iiif=iiif, image_id=meta["image_id"], width=args.width)
            path = out / f"{slug}.jpg"
            download(image_url, path)
            record = {
                "slug": slug,
                "file": path.name,
                "style": style,
                "text_tag": tag,
                "name_tag": NAME_TAGS.get(slug, ""),
                "aic_id": aic_id,
                "title": meta["title"],
                "artist": meta.get("artist_title") or meta.get("artist_display", "").split("\n")[0],
                "artist_display": meta.get("artist_display"),
                "date": meta.get("date_display"),
                "medium": meta.get("medium_display"),
                "dimensions": meta.get("dimensions"),
                "reference_number": meta.get("main_reference_number"),
                "credit_line": meta.get("credit_line"),
                "classification": meta.get("classification_title"),
                "institution": "The Art Institute of Chicago",
                "page_url": f"https://www.artic.edu/artworks/{aic_id}",
                "image_url": image_url,
                "license": "CC0 1.0 Universal (public-domain artwork, AIC Open Access)",
                "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                "retrieved": time.strftime("%Y-%m-%d"),
            }
            records.append(record)
            print(f"ok   {slug:<28} {record['artist']}, {record['title']} ({record['date']})")
        except Exception as exc:  # noqa: BLE001 - report and continue with the others
            failures.append((slug, str(exc)))
            print(f"FAIL {slug:<28} {exc}", file=sys.stderr)
        time.sleep(0.5)  # be polite to the API

    (out / "sources.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Open-access reference images",
        "",
        "Images: The Art Institute of Chicago Open Access (**CC0 1.0**) or Wikimedia Commons "
        "(public domain, see the licence column). "
        "Generated by `scripts/download_open_references.py`; do not edit by hand.",
        "",
        "| File | Artwork | Style | Ref. | Licence | Source |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in records:
        lines.append(f"| `{r['file']}` | {r['artist']}, *{r['title']}*, {r['date']}. {r['medium']} "
                     f"| {r['style']} | {r['reference_number']} | {r['license']} | <{r['page_url']}> |")
    lines += ["", "Credit line format used in figure captions:", "",
              "> Artist, *Title*, date. The Art Institute of Chicago, ref. no. — CC0 (AIC Open Access).", ""]
    (out / "SOURCES.md").write_text("\n".join(lines), encoding="utf-8")

    bib = []
    for r in records:
        bib.append(
            f"@misc{{{bibtex_key(r['slug'])},\n"
            f"  author       = {{{r['artist']}}},\n"
            f"  title        = {{{r['title']}}},\n"
            f"  year         = {{{r['date']}}},\n"
            f"  howpublished = {{{r['institution']}, {r['reference_number']}. "
            f"{(r['medium'] + '. ') if r['medium'] else ''}Image: {r['license']}}},\n"
            f"  url          = {{{r['page_url']}}},\n"
            f"  note         = {{Accessed {r['retrieved']}}}\n}}\n")
    (out / "references.bib").write_text("\n".join(bib), encoding="utf-8")

    (out / "style_tags.tsv").write_text(
        "".join(f"{r['slug']}\t{r['text_tag']}\n" for r in records), encoding="utf-8")
    (out / "name_tags.tsv").write_text(
        "".join(f"{r['slug']}\t{r['name_tag']}\n" for r in records if r.get("name_tag")), encoding="utf-8")

    print(f"\n{len(records)} downloaded, {len(failures)} failed -> {out}/")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
