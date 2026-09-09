"""
Rebuilds the published tuner page from tuner_template.html +
data/processed/client_data.json. Writes two copies of the same output:

  - docs/index.html         the GitHub Pages source (live site, PWA manifest
                             + icons + service worker live alongside it in
                             docs/) -- this is the primary output.
  - build/tuner_published.html   kept for the legacy manual-republish-to-
                             Artifact workflow; harmless to ignore otherwise.

Run this any time the template changes OR the data is refreshed:

    python -m nfl_edge.export_client_data   # refresh data/processed/client_data.json
    python build_tuner.py                   # embed it into docs/index.html

Kept as a real script (not a one-off shell command) so the weekly
scheduled refresh can call it the same way a human would, with no
hand-typed sed/python one-liners to get subtly wrong.
"""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).parent
TEMPLATE = ROOT / "tuner_template.html"
DATA = ROOT / "data" / "processed" / "client_data.json"
OUTPUTS = [ROOT / "docs" / "index.html", ROOT / "build" / "tuner_published.html"]


def main() -> None:
    template = TEMPLATE.read_text()
    data_json = DATA.read_text()
    if "__DATA_JSON__" not in template:
        raise SystemExit(f"{TEMPLATE} has no __DATA_JSON__ placeholder -- did it already get built?")
    out = template.replace("__DATA_JSON__", data_json)
    for path in OUTPUTS:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(out)
        print(f"Wrote {path} ({len(out):,} bytes) from {TEMPLATE.name} + {DATA.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
