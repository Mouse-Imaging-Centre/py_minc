#!/usr/bin/env python
#
#  Setup script for py_minc
#  Copyright 2002-2008,  John G. Sled
#
from distutils.core import setup, Extension
import os, commands

include_dirs=None #["/usr/local/mni/include"]
library_dirs=None #["/usr/local/mni/lib"]
libraries=["volume_io2", "minc2", "netcdf", "hdf5"]

if commands.getoutput("which swig") != "":
	VolumeIO_a = ["src/VolumeIO_a.i"]
else:
    print "Warning: swig is not present, using existing interface" + \
          " for VolumeIO_a\n"
    VolumeIO_a = ["src/VolumeIO_a.c"]
    

setup(name="py_minc",
      version="0.89.1",
      description="python access to the MINC file format using " + \
      "the volume io library",
      author="John G. Sled",
      author_email="jgsled@phenogenomics.ca",
      packages=['py_minc'],
      package_dir = {'': 'lib'},
      ext_modules = [Extension("py_minc.VolumeIO", ["src/volume_io_wrapper.c"],
                               include_dirs=include_dirs,
                               library_dirs=library_dirs,
                               libraries=libraries),
                     Extension("py_minc._VolumeIO_a", VolumeIO_a,
                               include_dirs=include_dirs,
                               library_dirs=library_dirs,
                               libraries=libraries),
                     Extension("py_minc.VolumeIO_constants",
                               ["src/VolumeIO_constants.c"],
                               include_dirs=include_dirs,
                               library_dirs=library_dirs,
                               libraries=libraries)],
     )

