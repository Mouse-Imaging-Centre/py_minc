py_minc
=======

py_minc is a python module for working with MINC data format files, and was developed by John G. Sled.


Installation
============

Requirements:
1) the MINC library (can be retrieved from GitHub)

2) the numpy library (python)

3) (recommended) swig (Simplified Wrapper and Interface Generator)

General installation:

# build
python setup.py build_ext -I /where/minc/headers/are/ -L /where/minc/libraries/are/

python setup.py build

# install
python setup.py install



