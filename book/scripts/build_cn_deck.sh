#!/usr/bin/env bash
# Build one Chinese slide deck (HTML + slides.pdf) with colloquium.
#   usage: build_cn_deck.sh teach/cn/<kind>/<name>.md
# Output: build/html/teach/cn/<kind>/<name>/{index.html,slides.pdf,assets/}
# (standalone talks named talk.md/slides.md go to build/html/teach/cn/<kind>/).
# Mirrors the English rules in the Makefile: outputs are content-addressed so CI
# can cache build/html/teach + build/.teach-stamps; on a stamp miss the old
# outputs are deleted first so a failed export can never ship stale files.
set -euo pipefail
src=$1
kind=$(basename "$(dirname "$src")")
name=$(basename "$src" .md)
case "$name" in talk|slides) rel="$kind" ;; *) rel="$kind/$name" ;; esac
out="build/html/teach/cn/$rel"
stamp_dir=build/.teach-stamps
stamp="$stamp_dir/cn-$(echo "$rel" | tr '/' '-').sha"
assets="teach/$kind/assets"
refs="teach/$kind/refs.bib"
mkdir -p "$out" "$stamp_dir"
cur=$({ shasum -a 256 "$src" uv.lock Makefile book/scripts/build_cn_deck.sh
        find "$assets" "$refs" -type f -exec shasum -a 256 {} + 2>/dev/null | LC_ALL=C sort; } | shasum -a 256 | cut -d' ' -f1)
if [ "$(cat "$stamp" 2>/dev/null)" = "$cur" ] && [ -s "$out/index.html" ] && [ -s "$out/slides.pdf" ]; then
  echo "Cached teach/cn/$rel (inputs unchanged)"; exit 0
fi
rm -f "$stamp" "$out/index.html" "$out/slides.pdf"; rm -rf "$out/assets"
uv run --extra teach colloquium build "$src" -o "$out/"
( cd "$out" && for f in *.html; do [ "$f" != index.html ] && mv "$f" index.html || true; done )
[ -s "$out/index.html" ] || { echo "FAILED teach/cn/$rel: no index.html" >&2; exit 1; }
uv run --extra teach colloquium export "$src" -o "$out/slides.pdf"
[ -s "$out/slides.pdf" ] || { echo "FAILED teach/cn/$rel: PDF export failed" >&2; exit 1; }
[ ! -d "$assets" ] || cp -rL "$assets" "$out/"
echo "$cur" > "$stamp"
echo "Built teach/cn/$rel"
