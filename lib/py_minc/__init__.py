# py_minc/__init__.py
#
#
# Copyright 2002, John G. Sled


from VolumeIO_constants import *
import VolumeIO
import _VolumeIO_a

from py_minc import Volume, ArrayVolume, set_default_cache_block_sizes, get_default_dim_names, FILE_ORDER_DIMENSION_NAMES, VolumeTags, akindof_Volume, GeneralTransform

from _VolumeIO_a import get_n_bytes_cache_threshold, set_n_bytes_cache_threshold, set_default_max_bytes_in_cache, get_default_max_bytes_in_cache, set_cache_block_sizes_hint




