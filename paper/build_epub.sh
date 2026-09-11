#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# build_epub.sh -- ePub versions of the ICRA paper, both languages.
#
# There is ONE source of truth: icra_en.tex and icra_fr.tex. This script
# derives a pandoc-friendly copy from them rather than maintaining a third
# manuscript, so the ePub can never drift from the PDF.
#
# Two things do not survive the trip and are handled here:
#   - the IEEEtran two-column layout, which is meaningless in a reflowable
#     format: we fall back to `article`;
#   - the TikZ/pgfplots figures, which pandoc cannot execute: they are
#     pre-rendered to PNG at ~3x and referenced as images.
#
# Usage:  ./build_epub.sh
# Output: icra_en.epub, icra_fr.epub
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

TECTONIC=${TECTONIC:-tectonic}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# --- 1. render each TikZ figure to PNG --------------------------------------
render_fig () {           # $1 = figure file stem, $2 = source paper (for labels)
  local stem=$1 src=$2
  # reuse the calling paper's preamble so labels and colours stay identical
  awk '/^\\usepackage\[T1\]\{fontenc\}/,/^\\begin\{document\}/' "$src" \
    | grep -v '^\\begin{document}' > "$WORK/pre.tex"
  {
    echo '\documentclass[border=4pt]{standalone}'
    cat "$WORK/pre.tex"
    echo '\begin{document}'
    echo "\\resizebox{78cm}{!}{\\input{$stem}}"
    echo '\end{document}'
  } > "$WORK/${stem}_r.tex"
  cp "$stem.tex" "$WORK/"
  (cd "$WORK" && $TECTONIC -X compile "${stem}_r.tex" --outdir . >/dev/null 2>&1)
  sips -s format png --out "${stem}.png" "$WORK/${stem}_r.pdf" >/dev/null
  echo "  $stem.png  $(sips -g pixelWidth "${stem}.png" | tail -1 | tr -d ' ')"
}

# --- 2. derive a pandoc-friendly manuscript ---------------------------------
derive () {               # $1 = paper stem
  local p=$1
  python3 - "$p" <<'PY'
import re, sys
p = sys.argv[1]
s = open(f"{p}.tex", encoding="utf-8").read()

s = s.replace(r"\documentclass[conference]{IEEEtran}", r"\documentclass{article}")
s = s.replace(r"\IEEEoverridecommandlockouts", "")
# packages that only make sense for the print layout
for pkg in ["balance", "tikz", "pgfplots", "multirow", "cite"]:
    s = re.sub(r"\\usepackage(\[[^\]]*\])?\{" + pkg + r"\}\n", "", s)
s = re.sub(r"\\usetikzlibrary\{[^}]*\}\n", "", s)
s = re.sub(r"\\pgfplotsset\{[^}]*\}\n", "", s)
s = s.replace(r"\balance", "")
# author block -> plain author
s = re.sub(r"\\IEEEauthorblockN\{([^}]*)\}", r"\1", s)
s = re.sub(r"\\IEEEauthorblockA\{(.*?)\}\%", r"\\\\ \1", s, flags=re.S)
# figures: TikZ -> pre-rendered image, and no starred floats in a reflow
s = s.replace(r"\resizebox{\textwidth}{!}{\input{fig_method}}",
              r"\includegraphics[width=\textwidth]{fig_method.png}")
s = s.replace(r"\resizebox{\textwidth}{!}{\input{fig_results}}",
              r"\includegraphics[width=\textwidth]{fig_results.png}")
s = s.replace(r"\input{fig_method}",
              r"\includegraphics[width=\textwidth]{fig_method.png}")
s = s.replace(r"\input{fig_results}",
              r"\includegraphics[width=\textwidth]{fig_results.png}")
s = s.replace(r"\begin{figure*}", r"\begin{figure}").replace(r"\end{figure*}", r"\end{figure}")
# multirow is gone: keep the cell text, drop the spanning
s = re.sub(r"\\multirow\{\d+\}\{\*\}\{(.*?)\}", r"\1", s, flags=re.S)
# emphasis inside a heading truncates pandoc's table of contents at that point
s = re.sub(r"\\(sub)?section\{([^}]*?)\\emph\{([^}]*)\}([^}]*)\}",
           r"\\\1section{\2\3\4}", s)
# the bibliography needs its own heading, or it lands silently inside the
# conclusion and never appears in the table of contents
heading = "Références" if p.endswith("_fr") else "References"
s = s.replace(r"\begin{thebibliography}{99}",
              "\\section*{%s}\n\\begin{thebibliography}{99}" % heading, 1)
open(f"{p}_epub.tex", "w", encoding="utf-8").write(s)
PY
}

echo "rendu des figures :"
render_fig fig_method  icra_en.tex
render_fig fig_results icra_en.tex

for lang in en fr; do
  derive "icra_$lang"
done

# --- 3. pandoc ---------------------------------------------------------------
LANG_en="en-GB"; LANG_fr="fr-FR"
TITLE_en="Sparse Metric Anchors for a Single-View 3D Generative Prior"
TITLE_fr="Ancres métriques éparses pour un a priori génératif 3D mono-vue"

for lang in en fr; do
  eval "l=\$LANG_$lang; t=\$TITLE_$lang"
  pandoc "icra_${lang}_epub.tex" \
    -f latex -t epub3 \
    --mathml \
    --toc --toc-depth=2 \
    --resource-path=. \
    --metadata title="$t" \
    --metadata lang="$l" \
    --metadata author="ISIR, Sorbonne Université" \
    -o "icra_${lang}.epub"
  echo "  icra_${lang}.epub  $(du -h "icra_${lang}.epub" | cut -f1)"
done

rm -f icra_en_epub.tex icra_fr_epub.tex
echo "fait."
