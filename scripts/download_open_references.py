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
REFERENCES = {
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


def get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def download(url: str, path: Path) -> None:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=120) as resp:
        path.write_bytes(resp.read())


def bibtex_key(slug: str) -> str:
    return "ref_" + slug.replace("-", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", default="data/reference_images/open")
    parser.add_argument("--width", type=int, default=843,
                        help="IIIF width in px (AIC recommends 843; 1686 for larger copies)")
    parser.add_argument("--only", nargs="*", default=None, help="Subset of slugs to download")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records, failures = [], []

    for slug, (aic_id, style, tag) in REFERENCES.items():
        if args.only and slug not in args.only:
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
        "All images: The Art Institute of Chicago, Open Access, **CC0 1.0** "
        "(<https://creativecommons.org/publicdomain/zero/1.0/>). "
        "Generated by `scripts/download_open_references.py`; do not edit by hand.",
        "",
        "| File | Artwork | Style | AIC ref. | Source |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in records:
        lines.append(f"| `{r['file']}` | {r['artist']}, *{r['title']}*, {r['date']}. {r['medium']} "
                     f"| {r['style']} | {r['reference_number']} | <{r['page_url']}> |")
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
            f"  howpublished = {{The Art Institute of Chicago, {r['reference_number']}. "
            f"{r['medium']}. Image: CC0 1.0, AIC Open Access}},\n"
            f"  url          = {{{r['page_url']}}},\n"
            f"  note         = {{Accessed {r['retrieved']}}}\n}}\n")
    (out / "references.bib").write_text("\n".join(bib), encoding="utf-8")

    (out / "style_tags.tsv").write_text(
        "".join(f"{r['slug']}\t{r['text_tag']}\n" for r in records), encoding="utf-8")

    print(f"\n{len(records)} downloaded, {len(failures)} failed -> {out}/")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
