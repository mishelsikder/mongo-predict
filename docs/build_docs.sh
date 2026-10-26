#!/usr/bin/env bash
# Build the API docs from the docstrings into docs/ (pdoc): the package page and one page per API
# (embeddings API, predictions API), with only the user-facing classes and functions (the list is in
# docs/_template/module.html.jinja2), then styled by docs/_postprocess.py.
#
#     bash docs/build_docs.sh          # then open docs/index.html
#
# The pages are generated: edit the docstrings in pymongopredict/, not the HTML, and rebuild.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf docs/pymongopredict docs/pymongopredict.html docs/index.html docs/search.js
uv run --group dev pdoc pymongopredict \
    --template-directory docs/_template \
    --docformat numpy \
    --no-show-source \
    --output-directory docs \
    --footer-text "mongo-predict $(uv run --group dev python -c 'import pymongopredict; print(pymongopredict.__version__)')"
# highlighting of docstring examples, the theme (light and dark) and copy buttons: docs/_postprocess.py
uv run --group dev python docs/_postprocess.py docs
echo "built $(find docs -name '*.html' | wc -l) pages in docs/"
