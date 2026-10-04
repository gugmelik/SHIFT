# Open-access reference images (CC0)

This folder is populated by:

```bash
python scripts/download_open_references.py
```

The script downloads public-domain artworks from the Art Institute of Chicago Open Access collection
(CC0 1.0). It checks AIC's `is_public_domain` flag for every work, then writes:
- `SOURCES.md` — credits;
- `references.bib` — citations;
- `sources.json` — machine-readable metadata;
- `style_tags.tsv` — text tags for the text-SHIFT baseline.

See `KLEIN_STYLE_README.md`, section 3.
