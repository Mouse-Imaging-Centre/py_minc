# py_minc/py_minc.py
#
#  A Python interface to the minc file format using the volume_io library 
#
#   by John G. Sled  
#
#   Created: March 21, 2001
#   Last revised:
#    September 13, 2008
#
#   Copyright 2002, John G. Sled

import types, copy
import numpy
from VolumeIO_constants import *
import VolumeIO, _VolumeIO_a
VolumeIO_a = _VolumeIO_a  # work around new SWIG naming convention
#import VolumeIO, VolumeIO_a

# useful helper functions

class RealArray:
    def __init__(self, length):
        self.length = length
        self.ptr = VolumeIO_a.new_RealArray(length)
    def __del__(self):
        VolumeIO_a.delete_RealArray(self.ptr)
    def __getitem__(self, index):
        return VolumeIO_a.RealArray_getitem(self.ptr, index)
    def __setitem__(self, index, value):
        VolumeIO_a.RealArray_setitem(self.ptr, index, value)
    def __len__(self):
        return self.length
    def asnumpyarray(self):
        a = numpy.zeros([self.length], dtype = numpy.float_)
        for i in range(self.length):
            a[i] = VolumeIO_a.RealArray_getitem(self.ptr, i)
        return a
    
class intArray:
    def __init__(self, length):
        self.length = length
        self.ptr = VolumeIO_a.new_intArray(length)
    def __del__(self):
        VolumeIO_a.delete_intArray(self.ptr)
    def __getitem__(self, index):
        return VolumeIO_a.intArray_getitem(self.ptr, index)
    def __setitem__(self, index, value):
        VolumeIO_a.intArray_setitem(self.ptr, index, value)
    def __len__(self):
        return self.length
    def asnumpyarray(self):
        a = numpy.zeros([self.length], dtype = numpy.int_)
        for i in range(self.length):
            a[i] = VolumeIO_a.intArray_getitem(self.ptr, i)
        return a

# additional wrappers for VolumeIO_a

def set_default_cache_block_sizes(sizes):
    VolumeIO_a.set_default_cache_block_sizes(tuple(sizes))

def get_default_dim_names(n_dimensions):
    if(n_dimensions > 5 or n_dimensions < 1):
        raise ValueError, "number of dimensions must be at least \
one and at most five"
    return ('fifth', 'forth', MIzspace, MIyspace, MIxspace)[5-n_dimensions:5]
#    ptr = VolumeIO_a.get_default_dim_names(n_dimensions)
#    return tuple(_build_list(VolumeIO_a.ptrcast(ptr, "char **"), n_dimensions))

FILE_ORDER_DIMENSION_NAMES = ("", "", "", "", "")

def akindof_Volume(volume):
    return type(volume) is types.InstanceType and \
           volume.__class__ in [Volume, ArrayVolume] 

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
# definition of base class for MINC volumes
class Volume:
    """a class for MINC volumes

Volume object are constructed differently depending on the
type of the arguments as follows:

Volume(StringType [, ...])                => input_volume

Volume(Volume)                            => copy_volume

Volume(Volume, copy='all', [, ...])       => copy_volume

Volume(Volume, copy='defintion' [, ...])  => copy_volume_definition

Volume(TupleType [, ...])                 => create_volume
    """

#---------------------------------------------------------------------------
    def __init__(self, arg1, *args, **kwargs):
        # determine which constructor to use based on arg1
        if type(arg1) is types.StringType:
            apply(Volume.input_volume, (self, arg1) + args, kwargs)

        elif akindof_Volume(arg1):
            if len(args) + len(kwargs) == 0 or \
               len(args) > 0 and args[0] == 'all':
                apply(Volume.copy_volume, (self, arg1) + args, kwargs)
            elif kwargs.has_key('copy') and kwargs['copy'] == 'all':
                del kwargs['copy']
                apply(Volume.copy_volume, (self, arg1) + args, kwargs)
            elif len(args) > 0 and args[0] == 'definition':
                apply(Volume.copy_volume_definition, (self, arg1) + args, kwargs)
            elif kwargs.has_key('copy') and kwargs['copy'] == 'definition':
                del kwargs['copy']
                apply(Volume.copy_volume_definition,
                      (self, arg1) + args, kwargs)
            else:
                raise TypeError

        elif type(arg1) is types.TupleType:
            apply(Volume.create_volume, (self, arg1) + args, kwargs)
        else:
            raise TypeError
        
    def __del__(self):
        if hasattr(self, 'input_info'):
            VolumeIO._delete_volume_input(self.input_info)
        if hasattr(self, 'volume_io_obj'):
            if VolumeIO_a.volume_is_cached(self.volume_io_obj) and \
               not VolumeIO_a.volume_is_alloced(self.volume_io_obj):
                # this is memory leak; however, there is a bug in
                #  delete_volume that can't handle this case
                pass
            else:
                VolumeIO_a.delete_volume(self.volume_io_obj)

#---------------------------------------------------------------------------
    def input_volume(self, filename, n_dimensions=MI_ORIGINAL_TYPE,
                 dim_names=None,
                 nc_data_type=MI_ORIGINAL_TYPE,
                 signed_flag=MI_ORIGINAL_TYPE, voxel_min=MI_ORIGINAL_TYPE,
                 voxel_max=MI_ORIGINAL_TYPE, create_flag=1):
        """construct Volume object from existing MINC file

Usage: input_volume(filename, n_dimensions, dim_names, nc_data_type,
                      signed_flag, voxel_min, voxel_max, create_flag)

Arguments:                      
    string                    filename         (required)
    integer                   n_dimensions
    tuple of strings | None   dim_names    
    integer                   nc_data_type
    integer                   signed_flag
    floating point            voxel_min
    floating point            voxel_max
    integer                   create_flag
"""
        if hasattr(self, 'volume_io_obj'):
            raise RuntimeError, \
                  'volume_io_obj can only be created by a constructor'
        self.volume_io_obj, self.input_info, self.attributes = \
                            VolumeIO._start_volume_input(filename, n_dimensions,
                                                   dim_names, nc_data_type,
                                                   signed_flag, float(voxel_min),
                                                   float(voxel_max), create_flag)
        if self.volume_io_obj is None:
            raise IOError, "input of volume %s failed" % filename

        Volume._prune_header(self)

        VolumeIO._finish_volume_input(self.volume_io_obj, self.input_info)
        del self.input_info

#---------------------------------------------------------------------------
    def copy_volume(self, existing_volume, starts=None, sizes=None):
        """create a Volume object by copying an existing Volume

Usage: copy_volume(existing_volume [, start=(s0,s1, ...), sizes=(n0,n1,...)])

Arguments:                      
    Volume        :  existing_volume
    start         :  extract a subvolume starting at given voxel (optional)
    size          :  extract a subvolume of given dimensions (optional)
"""
        if hasattr(self, 'volume_io_obj'):
            raise RuntimeError, \
                  'volume_io_obj can only be created by a constructor'

        if existing_volume.__class__ is not Volume:
            raise TypeError, ("Cannot create volume of type %s from type Volume."
                              + "  Try copy='definition' instead.") \
                              % existing_volume.__class
            
        # copy the whole volume
        if starts is None and sizes is None:
            self.volume_io_obj = VolumeIO_a.copy_volume(existing_volume.volume_io_obj)
        # or copy only a subvolume
        else:
            Volume._copy_sub_volume_definition_no_alloc(self, existing_volume, starts, sizes)
            VolumeIO_a.alloc_volume_data(self.volume_io_obj)
            hyperslab = VolumeIO._get_real_subvolume(existing_volume.volume_io_obj,
                                                     tuple(starts), tuple(sizes))
            VolumeIO._set_real_subvolume(self.volume_io_obj,
                                         (0,) * existing_volume.get_n_dimensions(),
                                         hyperslab)

        if self.volume_io_obj is None:
            raise RuntimeError

        self._copy_attributes(existing_volume.attributes)

#---------------------------------------------------------------------------
    def copy_volume_definition(self, existing_volume,
                               nc_data_type=MI_ORIGINAL_TYPE,
                               signed_flag=MI_ORIGINAL_TYPE,
                               voxel_min=MI_ORIGINAL_TYPE,
                               voxel_max=MI_ORIGINAL_TYPE,
                               starts=None, sizes=None):
        """create a Volume object based on the definition of an existing Volume

Usage: copy_volume_definition(existing_volume, nc_data_type,
                      signed_flag, voxel_min, voxel_max, start, size

Arguments:                      
    Volume                    existing_volume :  (required)
    integer                   nc_data_type
    integer                   signed_flag
    floating point            voxel_min
    floating point            voxel_max
    tuple                     starts          : voxel coordinate of subvolume starts
    tuple                     size            : voxel dimensions of subvolume
"""

        if hasattr(self, 'volume_io_obj'):
            raise RuntimeError,\
                  'volume_io_obj can only be created by a constructor'
        # copy the whole volume
        if starts is None and sizes is None:
            self.volume_io_obj = VolumeIO._copy_volume_definition(
                existing_volume.volume_io_obj, nc_data_type, signed_flag,
                float(voxel_min), float(voxel_max))
        else:
            Volume._copy_sub_volume_definition_no_alloc(self, existing_volume, starts, sizes,
                                                 nc_data_type, signed_flag,
                                                 voxel_min, voxel_max)
            VolumeIO_a.alloc_volume_data(self.volume_io_obj)
            
        if self.volume_io_obj is None:
            raise RuntimeError

        self.attributes = {}

#---------------------------------------------------------------------------
    def create_volume(self, sizes, dimension_names=None,
                      nc_data_type=NC_SHORT,
                      signed_flag=MI_ORIGINAL_TYPE,
                      voxel_min=MI_ORIGINAL_TYPE,
                      voxel_max=MI_ORIGINAL_TYPE,
                      fill_value=None):
        """create a Volume object from scratch

Usage: create_volume(sizes, dimension_names, nc_data_type,
                     signed_flag, voxel_min, voxel_max)

Arguments:
    tuple of integers sizes
    tuple of strings  dimension_names
    integer           nc_data_type
    integer (0 or 1)  signed_flag
    floating point    voxel_min
    floating point    voxel_max

Returns: 
    Volume           newly created MINC volume
"""

        if hasattr(self, 'volume_io_obj'):
            raise RuntimeError, \
                  'volume_io_obj can only be created by a constructor'

        # create volume_io structure
        n_dimensions = len(sizes)
        if dimension_names == None:
            dimension_names = get_default_dim_names(n_dimensions)
            
        self.volume_io_obj = VolumeIO_a.create_volume(
            n_dimensions, dimension_names, nc_data_type, signed_flag,
            float(voxel_min), float(voxel_max))

        if self.volume_io_obj is None:
            raise RuntimeError

        # set dimensions of volume
        VolumeIO_a.set_volume_sizes(self.volume_io_obj, tuple(sizes))
        # allocate memory for the data
        VolumeIO_a.alloc_volume_data(self.volume_io_obj)

        # optionally set fill value
        if fill_value != None:
            self.fill(fill_value)

        self.attributes = {}

#---------------------------------------------------------------------------

    def output(self, filename, history="", nc_data_type=MI_ORIGINAL_TYPE,
               signed_flag=MI_ORIGINAL_TYPE, voxel_min=MI_ORIGINAL_TYPE,
               voxel_max=MI_ORIGINAL_TYPE):
        """output(filename, history, nc_data_type, signed_flag, voxel_min, voxel_max)
        
Arguments:
    string              filename         
    string              history
    integer             nc_data_type
    integer             signed_flag
    floating point      voxel_min
    floating point      voxel_max

   Returns:
    None
"""

        VolumeIO._output_volume(filename, self.volume_io_obj,
                                history, nc_data_type, signed_flag,
                                float(voxel_min), float(voxel_max))

#---------------------------------------------------------------------------

    def set_cache_output_parameters(self, filename, history="",
                                    nc_data_type=MI_ORIGINAL_TYPE,
                                    signed_flag=MI_ORIGINAL_TYPE,
                                    voxel_min=MI_ORIGINAL_TYPE,
                                    voxel_max=MI_ORIGINAL_TYPE,
                                    original_filename=None):
        VolumeIO._set_cache_output_volume_parameters(self.volume_io_obj, 
                 filename, nc_data_type, signed_flag, float(voxel_min), float(voxel_max),
                 original_filename, history)
    

    def _copy_sub_volume_definition_no_alloc(self, existing_volume,
                                             voxel_starts, sizes,
                                             nc_data_type=MI_ORIGINAL_TYPE,
                                             signed_flag=MI_ORIGINAL_TYPE,
                                             voxel_min=MI_ORIGINAL_TYPE,
                                             voxel_max=MI_ORIGINAL_TYPE,
                                             inside_flag=1):
        if hasattr(self, 'volume_io_obj'):
            raise RuntimeError, \
                  'volume_io_obj can only be created by a constructor'

        n_dimensions = existing_volume.get_n_dimensions()
        if voxel_starts is None:
            voxel_starts = (0,) * n_dimensions
        elif sizes is None:
            sizes = existing_volume.get_sizes()

        if inside_flag:  # check that volume contains subvolume
            if not numpy.alltrue(numpy.greater_equal(voxel_starts, (0,) * n_dimensions)) or \
                   not numpy.alltrue(numpy.less_equal(sizes, existing_volume.get_sizes())):
                raise ValueError, \
                      'subvolume must be contained within existing volume'

        self.volume_io_obj = \
              VolumeIO_a.copy_volume_definition_no_alloc(existing_volume.volume_io_obj,
                                                         MI_ORIGINAL_TYPE,
                                                         MI_ORIGINAL_TYPE,
                                                         MI_ORIGINAL_TYPE,
                                                         MI_ORIGINAL_TYPE)
        VolumeIO_a.set_volume_sizes(self.volume_io_obj, tuple(sizes))

        steps = existing_volume.get_separations()
        starts = existing_volume.get_starts() + \
                 steps*numpy.array(voxel_starts)
        VolumeIO_a.set_volume_starts(self.volume_io_obj, tuple(starts))


    def _requires_rounding(self, source, target):
        return not ( source == target or \
           target in [numpy.float64, numpy.float32] or \
                      source in [numpy.int0, numpy.int8, \
                                 numpy.int16, numpy.int32, numpy.int64] )

    def _copy_and_cast(self, value, typecode):
        "round and cast if necessary.  value is copied either way"
        if self._requires_rounding(value.dtype, typecode):
            return numpy.floor(value + 0.5).astype(typecode)
        else:
            return value.astype(typecode)

    def _cast(self, value, typecode):
        "round and cast if necessary.  value is copied only if needed"
        if value.dtype == typecode:
            return value
        elif self._requires_rounding(value.dtype, typecode):
            return numpy.floor(value + 0.5).astype(typecode)
        else:
            return value.astype(typecode)

    def _prune_header(self):
        "remove attributes from the MINC file header that are handled by volume io"

        if self.attributes is None:
            return
        
        # store header information in attributes variable
        exclude_variables = [MIrootvariable, MIimage, MIimagemax,
                             MIimagemin] + list(self.get_dimension_names())
        exclude_attributes = ['typecode', MIparent, MIvartype]
        for variable in self.attributes.keys():
            if variable in exclude_variables:
                del self.attributes[variable]
            else:
                for attribute in self.attributes[variable].keys():
                    if attribute in exclude_attributes:
                        del self.attributes[variable][attribute]

    def _copy_attributes(self, attributes):
        # this is needed because older versions of Numeric do not support deepcopy
        self.attributes = {}
        for variable in attributes.keys():
            self.attributes[variable] = {}
            for attribute in attributes[variable].keys():
                self.attributes[variable][attribute] = \
                  copy.copy(attributes[variable][attribute])

    def __copy__(self):
        return self.__class__(self, copy='all')

    def __deepcopy__(self, memo):
        return self.__class__(self, copy='all')

    def set_range(self, min, max):
        "set_range(min, max)"
        VolumeIO_a.set_volume_real_range(self.volume_io_obj,
                                        float(min), float(max));
    def get_dimension_names(self):
        "get_dimension_names()"
        return VolumeIO._get_volume_dimension_names(self.volume_io_obj)

    def get_value(self, location):
        return apply(VolumeIO_a.get_volume_real_value,
                     (self.volume_io_obj,) + tuple(map(int,location)))

    def get_voxel_value(self, location):
        return apply(VolumeIO_a.get_volume_voxel_value,
                     (self.volume_io_obj,) + tuple(map(int,location)))

    def set_value(self, location, value):
        def expand(v0, v1=0, v2=0, v3=0, v4=0):
            return v0, v1, v2, v3, v4

        v0, v1, v2, v3, v4 = apply(expand,location)
        VolumeIO_a.set_volume_real_value(self.volume_io_obj,
                                         int(v0), int(v1), int(v2), int(v3), int(v4), float(value))

    def set_voxel_value(self, location, value):
        def expand(v0, v1=0, v2=0, v3=0, v4=0):
            return v0, v1, v2, v3, v4

        v0, v1, v2, v3, v4 = apply(expand,location)
        VolumeIO_a.set_volume_voxel_value(self.volume_io_obj,
                                          int(v0), int(v1), int(v2), int(v3), int(v4), float(value))

    def get_all_values(self, typecode=numpy.float_):
        """Return a numpy array of the specified type with all values from the volume.
Note that integer types are produced by rounding using floor(x+0.5).
"""
        return VolumeIO._get_volume_all_real_values(self.volume_io_obj, numpy.dtype(typecode).char)
        
    def get_hyperslab(self, start, size, typecode=numpy.float_):
        "get an n dimensional array of real values from the volume"
        return self._cast(VolumeIO._get_real_subvolume(self.volume_io_obj,
                                            tuple(start), tuple(size)), typecode)
    def set_hyperslab(self, start, data):
        "set an n dimensional array of real values in the volume."
        VolumeIO._set_real_subvolume(self.volume_io_obj,
                                            tuple(start), data)
#    def set_all_values(self, data):
#        VolumeIO._set_volume_all_real_values(self.volume_io_obj, data)

    def set_all_values(self, data):
        self.set_hyperslab(numpy.zeros(len(numpy.shape(data)), numpy.int_), data) 

    def convert_voxel_to_value(self, voxel):
        "convert voxel value to real value"
        return VolumeIO_a.convert_voxel_to_value(self.volume_io_obj, float(voxel))

    def convert_value_to_voxel(self, value):
        "convert real value to voxel value"
        return VolumeIO_a.convert_value_to_voxel(self.volume_io_obj, float(value))

    def get_total_n_voxels(self):
        return VolumeIO_a.get_volume_total_n_voxels(self.volume_io_obj)
        
    def get_voxel_to_world_transform(self):
        return VolumeIO_a.get_voxel_to_world_transform(self.volume_io_obj)

    def set_voxel_to_world_transform(self, transform):
        VolumeIO_a.set_voxel_to_world_transform(self.volume_io_obj, transform)

    def get_n_dimensions(self):
        return VolumeIO_a.get_volume_n_dimensions(self.volume_io_obj)

    def get_space_type(self):
        return VolumeIO_a.get_volume_space_type(self.volume_io_obj)

    def set_space_type(self, spacetype):
        VolumeIO_a.set_volume_space_type(self.volume_io_obj, spacetype)

    def get_voxel_min(self):
        "get minimum allowed value for VolumeIO internal representation (voxel value)"
        return VolumeIO_a.get_volume_voxel_min(self.volume_io_obj)
        
    def get_voxel_max(self):
        "get maximum allowed value for VolumeIO internal representation (voxel value)"
        return VolumeIO_a.get_volume_voxel_max(self.volume_io_obj)

    def set_voxel_range(self, min, max):
        "get minimum and maximum allowed values for VolumeIO internal representation (voxel values)"
        VolumeIO_a.set_volume_voxel_range(self.volume_io_obj, float(min), float(max))

    def get_min(self):
        "get real value corresponding to the minimum voxel value"
        return VolumeIO_a.get_volume_real_min(self.volume_io_obj)
        
    def get_max(self):
        "get real value corresponding to the maximum voxel value"
        return VolumeIO_a.get_volume_real_max(self.volume_io_obj)

    def set_type(self, nc_data_type, signed_flag, voxel_min, voxel_max):
        VolumeIO_a.set_volume_type(self.volume_io_obj, nc_data_type, signed_flag,
                                   float(voxel_min), float(voxel_max))
    def set_sizes(self, sizes):
        "set number of voxels along each dimension"
        VolumeIO_a.set_volume_sizes(self.volume_io_obj,
                                    tuple(sizes))

    def get_sizes(self):
        "get number of voxels along each dimension"
        c_array = intArray(self.get_n_dimensions())
        VolumeIO_a.get_volume_sizes(self.volume_io_obj, c_array.ptr)
        return c_array.asnumpyarray()
        
    def get_separations(self):
        c_array = RealArray(self.get_n_dimensions())
        VolumeIO_a.get_volume_separations(self.volume_io_obj, c_array.ptr)
        return c_array.asnumpyarray()

    def set_separations(self, separations):
        VolumeIO_a.set_volume_separations(self.volume_io_obj,
                                          tuple(separations))
        
    def set_starts(self, starts):
        VolumeIO_a.set_volume_starts(self.volume_io_obj,
                                          tuple(starts))
        
    def get_starts(self):
        c_array = RealArray(self.get_n_dimensions())
        VolumeIO_a.get_volume_starts(self.volume_io_obj, c_array.ptr)
        return c_array.asnumpyarray()

    def set_direction_unit_cosine(self, axis, cosine):
        VolumeIO_a.set_volume_direction_unit_cosine(self.volume_io_obj, axis,
                                          tuple(cosine))

    def set_direction_cosine(self, axis, cosine):
        VolumeIO_a.set_volume_direction_cosine(self.volume_io_obj, axis,
                                          tuple(cosine))

    def get_direction_cosine(self, axis):
        c_array = RealArray(3)  # is this alway 3?
        VolumeIO_a.get_volume_direction_cosine(self.volume_io_obj,
                                               axis, c_array.ptr)
        return c_array.asnumpyarray()

    def set_translation(self, voxel, world_space_voxel_maps_to):
        VolumeIO_a.set_volume_translation(self.volume_io_obj, tuple(voxel),
                                          tuple(world_space_voxel_maps_to))

    def get_range(self):
        return VolumeIO_a.get_volume_real_range(self.volume_io_obj)

    def get_voxel_range(self):
        return VolumeIO_a.get_volume_voxel_range(self.volume_io_obj)

    def get_nc_data_type(self):
        "returns: nc_type, signed_flag"
        return tuple(VolumeIO_a.get_volume_nc_data_type(self.volume_io_obj))

    def set_cache_size(self, size):
        VolumeIO_a.set_volume_cache_size(self.volume_io_obj, size)

    def is_cached(self):
        return VolumeIO_a.volume_is_cached(self.volume_io_obj)

    def set_cache_block_sizes(self, sizes):
        VolumeIO_a.set_volume_cache_block_sizes(self.volume_io_obj,
                                                tuple(sizes))

    def convert_voxel_to_world(self,voxel):
        v = VolumeIO_a.convert_voxel_to_world(self.volume_io_obj, tuple(voxel))
        return (numpy.array(v))

    def convert_world_to_voxel(self, world):
        c_array = RealArray(self.get_n_dimensions())
        VolumeIO_a.convert_world_to_voxel(self.volume_io_obj, float(world[0]),
                                          float(world[1]), float(world[2]), c_array.ptr)
        return c_array.asnumpyarray()

    def fill(self, value):
        "fill all voxels with given value"
        VolumeIO._fill_volume_real_value(self.volume_io_obj, value)

#     def __getitem__(self, indices):
#         try: # suppose that each indice is an integer
#             if type(indices) == types.TupleType:
#                 args = (self.volume_io_obj,) + indices
#             else:
#                 args = (self.volume_io_obj, indices)
                
#             return apply(VolumeIO._get_volume_real_value, args)
                
#         except TypeError:  # otherwise perhaps some indices are slices
#             return self.__getslice__(indices)

#     def __getslice__(self, arg1, arg2=None):
#         return (arg1, arg2)


#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
# definition of ArrayVolume class for MINC volumes
class ArrayVolume (Volume):
    """a class for MINC volumes in which the data is stored in a Numerical Python array

ArrayVolume objects are constructed differently depending on the
type of the arguments as follows:

ArrayVolume(StringType [, ...])                        => input_volume

ArrayVolume(Volume)                                    => copy_volume

ArrayVolume(Volume, copy='all')                        => copy_volume

ArrayVolume(Volume, copy='definition' [, ...])         => copy_volume_definition

ArrayVolume(TupleType [, ...])                         => create_volume

Note that all of these constructors accept a keyword argument 'typecode' which specifies
the type of the numpy array.  If the typecode is not specified then the default type
for the numpy array if float_.  Note that this typecode specification is independent
of the NetCDF type specified for the ArrayVolume.  The latter is used when the volume
is written to a file using the output method.
    """

#---------------------------------------------------------------------------
    def __init__(self, arg1, *args, **kwargs):
        if kwargs.has_key('typecode'):
            typecode = kwargs['typecode']
            del kwargs['typecode']
        else:
            typecode = numpy.float_

        # determine which constructor to use based on arg1
        if type(arg1) is types.StringType:
            apply(ArrayVolume.input_volume, (self, typecode, arg1) + args, kwargs)

        elif (type(arg1) is types.InstanceType) and \
             arg1.__class__ in [Volume, ArrayVolume]:
            if len(args) + len(kwargs) == 0 or \
               len(args) > 0 and args[0] == 'all':
                apply(ArrayVolume.copy_volume, (self, typecode, arg1) + args, kwargs)
            elif kwargs.has_key('copy') and kwargs['copy'] == 'all':
                del kwargs['copy']
                apply(ArrayVolume.copy_volume, (self, typecode, arg1) + args, kwargs)
            elif kwargs.has_key('copy') and kwargs['copy'] == 'all':
                del kwargs['copy']
                apply(Volume.copy_volume, (self, typecode, arg1) + args, kwargs)
            elif len(args) > 0 and args[0] == 'definition':
                apply(ArrayVolume.copy_volume_definition,
                      (self, typecode, arg1) + args[2:], kwargs)
            elif kwargs.has_key('copy') and kwargs['copy'] == 'definition':
                del kwargs['copy']
                apply(ArrayVolume.copy_volume_definition,
                      (self, typecode, arg1) + args, kwargs)
            else:
                raise TypeError

        elif type(arg1) is types.TupleType:
            apply(ArrayVolume.create_volume, (self, typecode, arg1) + \
                  args, kwargs)

        else:
            raise TypeError
        
#---------------------------------------------------------------------------
    def __del__(self):
        Volume.__del__(self)
        
#---------------------------------------------------------------------------
    def input_volume(self, typecode, *args, **kwargs):
        "see Volume.input_volume for usage information"

        # read in MINC volume
        apply(Volume.input_volume, (self,) + args, kwargs)

        # copy value to a python array
        self.array = Volume.get_all_values(self,typecode)

        # delete the data component of the volume_io object
        VolumeIO_a.free_volume_data(self.volume_io_obj)

#---------------------------------------------------------------------------
    def copy_volume(self, typecode, existing_volume, starts=None, sizes=None):
        "see Volume.copy_volume for usage information"

        if hasattr(self, 'volume_io_obj'):
            raise RuntimeError, \
                  'volume_io_obj can only be created by a constructor'

        # copy the whole volume
        if starts is None and sizes is None:
            self.volume_io_obj = \
                VolumeIO_a.copy_volume_definition_no_alloc(existing_volume.volume_io_obj,
                                                           MI_ORIGINAL_TYPE,
                                                           MI_ORIGINAL_TYPE,
                                                           MI_ORIGINAL_TYPE,
                                                           MI_ORIGINAL_TYPE)
            self.array = existing_volume.get_all_values(typecode)

                # or copy only a subvolume
        else:
            Volume._copy_sub_volume_definition_no_alloc(self, existing_volume, starts, sizes)
            self.array = existing_volume.get_hyperslab(starts, sizes, typecode)

        if self.volume_io_obj is None:
            raise RuntimeError

        self._copy_attributes(existing_volume.attributes)

#---------------------------------------------------------------------------
    def copy_volume_definition(self, typecode, existing_volume,
                               nc_data_type=MI_ORIGINAL_TYPE,
                               signed_flag=MI_ORIGINAL_TYPE,
                               voxel_min=MI_ORIGINAL_TYPE,
                               voxel_max=MI_ORIGINAL_TYPE,
                               starts=None, sizes=None):
        "see Volume.copy_volume_definition for usage information"


        if hasattr(self, 'volume_io_obj'):
            raise RuntimeError,\
                  'volume_io_obj can only be created by a constructor'
        # copy the whole volume
        if starts is None and sizes is None:
            self.volume_io_obj = VolumeIO_a.copy_volume_definition_no_alloc(
                existing_volume.volume_io_obj, nc_data_type, signed_flag,
                float(voxel_min), float(voxel_max))
        else:
            Volume._copy_sub_volume_definition_no_alloc(self, existing_volume, starts, sizes,
                                                 nc_data_type, signed_flag,
                                                 float(voxel_min), float(voxel_max),
                                                        inside_flag = 0)

        if self.volume_io_obj is None:
            raise RuntimeError

        self.array = numpy.zeros(self.get_sizes(),typecode)

        self.attributes = {}

#---------------------------------------------------------------------------
    def create_volume(self, typecode, sizes, dimension_names=None,
                      nc_data_type=NC_SHORT,
                      signed_flag=MI_ORIGINAL_TYPE,
                      voxel_min=MI_ORIGINAL_TYPE,
                      voxel_max=MI_ORIGINAL_TYPE,
                      fill_value=None):
        "see Volume.create_volume for usage information"

        if hasattr(self, 'volume_io_obj'):
            raise RuntimeError, \
                  'volume_io_obj can only be created by a constructor'

        # create volume_io structure
        n_dimensions = len(sizes)
        if dimension_names == None:
            dimension_names = get_default_dim_names(n_dimensions)
            
        self.volume_io_obj = VolumeIO_a.create_volume(
            n_dimensions, dimension_names, nc_data_type, signed_flag,
            float(voxel_min), float(voxel_max))

        if self.volume_io_obj is None:
            raise RuntimeError

        # set dimensions of volume
        VolumeIO_a.set_volume_sizes(self.volume_io_obj, tuple(sizes))

        # allocate an array for the data
        self.array = numpy.zeros(sizes, typecode)

        # optionally set fill value
        if fill_value != None:
            self.fill(fill_value)

        self.attributes = {}

#---------------------------------------------------------------------------
    def output(self, *args, **kwargs):
        "see Volume.output for usage information"

        VolumeIO_a.alloc_volume_data(self.volume_io_obj)
        Volume.set_hyperslab(self, numpy.zeros(len(self.array.shape), numpy.int_), self.array)
        apply(Volume.output, (self,) + args, kwargs)

        # delete the data component of the volume_io object
        VolumeIO_a.free_volume_data(self.volume_io_obj)

#---------------------------------------------------------------------------

    def get_value(self, location):
        return float(self.array[tuple(location)])

    def get_voxel_value(self, location):
        value = self.array[tuple(location)]
        return self.convert_value_to_voxel(value)

    def set_value(self, location, value):
        self.array[tuple(location)] = self._cast(numpy.array(value), self.array.dtype)

    def set_voxel_value(self, location, value):
        self.array[tuple(location)] = self._cast(
            array(self.convert_value_to_voxel(value)),
            self.array.typecode)

    def get_all_values(self, typecode=numpy.float_):
        return self._copy_and_cast(self.array, typecode)

    def get_hyperslab(self, start, size, typecode=numpy.float_):
        "get an n dimensional array of real values from the volume"
        slices = map(lambda x, y: slice(x, x+y), start, size)
        return self._copy_and_cast(self.array[slices], typecode)
        
    def set_hyperslab(self, start, data):
        "set an n dimensional array of real values in the volume."
        size = data.shape
        slices = map(lambda x, y: slice(x, x+y), start, size)
        self.array[slices] = self._cast(data, self.array.dtype)

    def set_all_values(self, data):
        self.array[:] = self._cast(data, self.array.dtype)

    def fill(self, value):
        self.array[:] = self._cast(numpy.array(value), self.array.dtype)


#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
# definition of MNI tag file class

class VolumeTags:
    "class for accessing the MNI tag file format"

    def __init__(self, arg0):
        "If arg0 is a filename, read tags from file.  Otherwise, if arg0 is 1, 2 create an empty tags structure appropriate for the specified number of volumes"
        if type(arg0) is types.StringType:
            r = VolumeIO._input_tag_file(arg0)

            self.n_volumes = r[0]
            if(self.n_volumes == 1):
                self.locations = r[1]
                r = r[2:]
            else:
                self.locations = [r[1], r[2]]
                r = r[3:]
            self.weights, self.structure_ids, self.patient_ids, self.labels = r

        elif arg0 in [1, 2]:
            self.n_volumes = arg0
            self.locations = ([], [[], []])[arg0 == 2]
            self.weights = []
            self.structure_ids = []
            self.patient_ids = []
            self.labels = []
        else:
            raise TypeError, "Unknown argument type for VolumeTags"

    def append(self, location, weight, structure_id, patient_id, label):
        if self.n_volumes == 2:
            self.locations[0].append(location[0])
            self.locations[1].append(location[1])
        else:
            self.locations.append(location)
        self.weights.append(weight)
        self.structure_ids.append(structure_id)
        self.patient_ids.append(patient_id)
        self.labels.append(label)

    def number(self):
        return len(self.weights)

    def output(self, filename, comment = "Tag file created by py_minc"):
        if self.n_volumes == 2:
            VolumeIO._output_tag_file(filename, comment, self.n_volumes, self.number(),
                                      numpy.array(self.locations[0]), numpy.array(self.locations[1]),
                                      self.weights, self.structure_ids, self.patient_ids,
                                      self.labels)
        else:
            VolumeIO._output_tag_file(filename, comment, self.n_volumes, len(self.weights),
                                      numpy.array(self.locations), numpy.array([]),
                                      self.weights, self.structure_ids, self.patient_ids,
                                      self.labels)
            


#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
# definition of General Transform class


class GeneralTransform:
    "class for accessing the MNI xfm file format"

    def __init__(self, filename):
        "read transform from file"

        self.transform = VolumeIO_a.new_General_transform()
        status = VolumeIO_a.input_transform_file(filename, self.transform)
        if status != 0:
            raise IOError, "reading of transform file %s failed" % filename

    def transform_point(self, point):
        "apply general transform to given point (x, y, z)"
        return numpy.array(VolumeIO_a.general_transform_point(self.transform, \
                                                        float(point[0]), float(point[1]), float(point[2])))

    def inverse_transform_point(self, point):
        "apply inverse of general transform to given point (x, y, z)"
        return numpy.array(VolumeIO_a.general_inverse_transform_point(self.transform, \
                                                        float(point[0]), float(point[1]), float(point[2])))


    def __del__(self):
        "not yet implemented"
        pass
