#!/bin/sh
cd ~/workspace/spotPython
rm -f dist/spotPython*; python -m build; python -m pip install dist/spotPython*.tar.gz
python -m mkdocs build
pytest
