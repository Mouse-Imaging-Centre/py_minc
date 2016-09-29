py_minc
=======

py_minc is a python module for working with MINC data format files, and was developed by John G. Sled.


Installation
============

Requirements:
* the MINC library (can be retrieved from GitHub)
* the numpy library (python)
* swig (Simplified Wrapper and Interface Generator, needed to create src/VolumeIO_a.c)

General installation:

# build
<pre><code>
python setup.py build_ext -I /where/minc/headers/are/ -L /where/minc/libraries/are/
python setup.py build
</code></pre>
# install
<pre><code>
python setup.py install
</code></pre>


