#!/bin/sh
rm -f dist/spotpython*; python -m build; python -m pip install dist/spotpython*.tar.gz
python -m mkdocs build
