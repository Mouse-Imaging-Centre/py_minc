/* A Python interface to the volume_io library 

   by John G. Sled  

   Created: March 21, 2001
   Last revised: September 13, 2008

   Copyright 2002-2008, John G. Sled  
*/

#include <Python.h>
#include <numpy/arrayobject.h>
#include <volume_io.h>
#include <math.h>
#include <string.h>
#include <minc_compat.h>
#include <minc.h>

# define MAX_VOLUME_DIMENSIONS   5

#define DISABLE_ATTR

# define DECLARE_VARG_PYTHON_FUNC(wrapper)  \
 static PyObject *wrapper (PyObject *, PyObject *)

/* Declarations of Python callable methods */
static PyObject *start_volume_input_wrapper (PyObject *, PyObject *, PyObject *);
DECLARE_VARG_PYTHON_FUNC(finish_volume_input_wrapper);
DECLARE_VARG_PYTHON_FUNC(delete_volume_input_wrapper);
DECLARE_VARG_PYTHON_FUNC(output_volume_wrapper);
DECLARE_VARG_PYTHON_FUNC(set_cache_output_volume_parameters_wrapper);
DECLARE_VARG_PYTHON_FUNC(copy_volume_definition_wrapper);
DECLARE_VARG_PYTHON_FUNC(get_volume_all_real_values_wrapper);
DECLARE_VARG_PYTHON_FUNC(get_real_subvolume_wrapper);
DECLARE_VARG_PYTHON_FUNC(set_volume_all_real_values_wrapper);
DECLARE_VARG_PYTHON_FUNC(set_real_subvolume_wrapper);
DECLARE_VARG_PYTHON_FUNC(get_volume_dimension_names_wrapper);
DECLARE_VARG_PYTHON_FUNC(fill_volume_real_value_wrapper);
DECLARE_VARG_PYTHON_FUNC(input_tag_file_wrapper);
DECLARE_VARG_PYTHON_FUNC(output_tag_file_wrapper);


# define VARG_METHOD_TABLE_ENTRY(name) \
  {"_"#name, name##_wrapper, METH_VARARGS}

/* Method Table */
static PyMethodDef VolumeIOMethods[] = {
  {"_start_volume_input", (PyCFunction) start_volume_input_wrapper,
   METH_VARARGS | METH_KEYWORDS},
  VARG_METHOD_TABLE_ENTRY(finish_volume_input),
  VARG_METHOD_TABLE_ENTRY(delete_volume_input),
  VARG_METHOD_TABLE_ENTRY(output_volume),
  VARG_METHOD_TABLE_ENTRY(copy_volume_definition),
  /*  VARG_METHOD_TABLE_ENTRY(get_volume_real_value_wrapper), */
  VARG_METHOD_TABLE_ENTRY(get_volume_all_real_values),
  VARG_METHOD_TABLE_ENTRY(get_real_subvolume),
  VARG_METHOD_TABLE_ENTRY(set_volume_all_real_values),
  VARG_METHOD_TABLE_ENTRY(set_real_subvolume),
  VARG_METHOD_TABLE_ENTRY(get_volume_dimension_names),
  VARG_METHOD_TABLE_ENTRY(fill_volume_real_value),
  VARG_METHOD_TABLE_ENTRY(input_tag_file),
  VARG_METHOD_TABLE_ENTRY(output_tag_file),
  VARG_METHOD_TABLE_ENTRY(set_cache_output_volume_parameters),
  { NULL,    NULL}
};


static PyObject *read_variable_attributes(int file_id);
static void collect_attributes(int fileid, int varid, 
			       PyObject *attributes, int nattrs);


/* Module initialization function */
void
initVolumeIO(void)
{
  PyObject *m;

  m = Py_InitModule("VolumeIO", VolumeIOMethods);

  /* required for NumPy library */
  import_array();

}


/* Method: start_volume_input */
/* Arguments:
    string                    filename         
    integer                   n_dimensions
    tuple of strings | None   dim_names      (need to implement File_order_dimension_names)
    integer                   volume_nc_data_type
    integer                   volume_signed_flag
    floating point            volume_voxel_min
    floating point            volume_voxel_max
    integer                   create_volume_flag
    (not implemented)         minc_input_options

  Returns:
    PyObject            volume
    PyObject            input_info
    int                 netcdfid
*/
static PyObject *
start_volume_input_wrapper (PyObject *self, PyObject *args, PyObject *keywds)
{
  char      *filename;
  volume_input_struct *input_info;
  Minc_file minc_file;
  int       n_dimensions;
  int       volume_nc_data_type;
  int       volume_signed_flag, create_volume_flag;  
  double    volume_voxel_min, volume_voxel_max;
  PyObject  *dim_names_tuple, *dim_name;
  STRING    *dim_names;
  int       i, size;
  Volume    volume;
  PyObject  *variables;

  /* define named arguments */
  static char *kwlist[] = {"filename", "n_dimensions", "dim_names",
			   "volume_nc_data_type", "volume_signed_flag",
			   "volume_voxel_min", "volume_voxel_max",
			   "create_volume_flag", NULL};


  /* set default values for arguments */
  n_dimensions = 0;
  dim_names_tuple = Py_None;
  volume_nc_data_type = NC_UNSPECIFIED;
  volume_signed_flag = NC_UNSPECIFIED;
  volume_voxel_min = NC_UNSPECIFIED;
  volume_voxel_max = NC_UNSPECIFIED;
  create_volume_flag = 1;

  /* translate arguments to C data types */
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "s|iOiiddi", kwlist,
				   &filename, &n_dimensions, &dim_names_tuple,
				   &volume_nc_data_type, &volume_signed_flag,
				   &volume_voxel_min, &volume_voxel_max,
				   &create_volume_flag))
    return NULL;

  /* parse the dimension names tuple */
  if (PyTuple_Check(dim_names_tuple)) {
    size = PyTuple_Size(dim_names_tuple);
    ALLOC(dim_names, size);
    for (i = 0; i < size; i++) {
      dim_name = PyTuple_GetItem(dim_names_tuple, i);
      if(!PyString_Check(dim_name)) {
	PyErr_SetString(PyExc_ValueError, "dim_names must be a tuple of strings");
	return NULL;  /* fail: not a string */
      }
      /* copy string from tuple */
      dim_names[i] = create_string(PyString_AsString(dim_name));
    }
  }
  else if(dim_names_tuple == Py_None) {
    dim_names = NULL;
  }
  else {
    PyErr_SetString(PyExc_ValueError, "dim_names must be a tuple or None");
     return NULL;  /* fail: not a tuple */
  }

  /* allocate space for recording volume input progress */
  input_info = malloc(sizeof(volume_input_struct));
  if (input_info == NULL) {
    return NULL;
  }

  /* call input_volume from volume_io library */
  if(start_volume_input((STRING) filename, n_dimensions, 
		  dim_names,
		  (nc_type) volume_nc_data_type,
		  (BOOLEAN) volume_signed_flag,
		  (Real) volume_voxel_min, (Real) volume_voxel_max,
		  (BOOLEAN) create_volume_flag, &volume, 
		  (minc_input_options *) NULL,
		  input_info) != OK) {
    PyErr_SetString(PyExc_IOError, "VolumeIO.start_volume_input");
    return NULL;
  }
  
#ifdef DISABLE_ATTR
  variables = PyDict_New();
#else
  /* get the minc_file info as a means to access header attributes */
  minc_file = get_volume_input_minc_file(input_info);

  variables = read_variable_attributes(minc_file->cdfid);
  if (variables == NULL) {
    PyErr_SetString(PyExc_IOError, "VolumeIO.start_volume_input");
    return NULL;
  }
#endif
  
  /* return volume and input_info as opaque PyCObjects */
  return Py_BuildValue("NNN", PyCObject_FromVoidPtr((void *) volume, NULL),
		       PyCObject_FromVoidPtr((void *) input_info, NULL),
		       variables);
}


/* Method: finish_volume_input */
/* Arguments:
    PyObject            volume
    PyObject            input_info
   Returns:
    None
*/
static PyObject *
finish_volume_input_wrapper (PyObject *self, PyObject *args)
{
  volume_input_struct *input_info;
  Volume volume;
  PyObject  *arg0, *arg1;
  Real      fraction_done;
  progress_struct      progress;
  static const int     FACTOR = 1000;

  /* translate arguments to C data types */
  if (!PyArg_ParseTuple(args, "O!O!", &PyCObject_Type, &arg0,
			&PyCObject_Type, &arg1))
    return NULL;
  
  volume = (Volume) PyCObject_AsVoidPtr(arg0);
  input_info = (volume_input_struct *) PyCObject_AsVoidPtr(arg1);

  initialize_progress_report( &progress, FALSE, FACTOR, "Reading Volume");

  /* load the remainder of the volume */
  while(input_more_of_volume(volume, input_info, &fraction_done)) 
  {
    update_progress_report( &progress,
			    ROUND( (Real) FACTOR * fraction_done));
  }

  terminate_progress_report( &progress );

  delete_volume_input( input_info );

  /* free memory allocated by start_volume_input */
  free(input_info);

  /* return None */
  Py_INCREF(Py_None);
  return Py_None;
}

/* Method: delete_volume_input */
/* Arguments:
    PyObject            input_info
   Returns:
    None
*/
static PyObject *
delete_volume_input_wrapper (PyObject *self, PyObject *args)
{
  volume_input_struct *input_info;
  PyObject  *arg0;

  /* translate arguments to C data types */
  if (!PyArg_ParseTuple(args, "O!", &PyCObject_Type, &arg0))
    return NULL;
  
  input_info = (volume_input_struct *) PyCObject_AsVoidPtr(arg0);

  /* load the remainder of the volume */
  delete_volume_input(input_info);

  /* free memory allocated by start_volume_input */
  free(input_info);

  /* return None */
  Py_INCREF(Py_None);
  return Py_None;
}


/* Method: output_volume */
/* Arguments:
    string              filename         
    PyCObject           volume
    string              history
    integer             nc_data_type
    integer             signed_flag
    floating point      voxel_min
    floating point      voxel_max
    (not implemented)   minc_output_options

   Returns:
    None
*/
static PyObject *
output_volume_wrapper (PyObject *self, PyObject *args)
{
  char      *filename;
  PyObject  *cobj;
  int       nc_data_type;
  int       signed_flag;  
  double    voxel_min, voxel_max;
  char      *history;

  Volume volume;

  /* set default values for arguments */
  nc_data_type = NC_UNSPECIFIED;
  signed_flag = NC_UNSPECIFIED;
  voxel_min = NC_UNSPECIFIED;
  voxel_max = NC_UNSPECIFIED;
  history = NULL;
  
  /* translate arguments to C data types */
  if (!PyArg_ParseTuple(args, "sO!|siidd", &filename,
			&PyCObject_Type, &cobj,
			&history,
			&nc_data_type, &signed_flag,
                        &voxel_min, &voxel_max))
    return NULL;

  volume = (Volume) PyCObject_AsVoidPtr(cobj);

  /* call output_volume from volume_io library */
  if(output_volume((STRING) filename, (nc_type) nc_data_type,
		   (BOOLEAN) signed_flag,
		   (Real) voxel_min, (Real) voxel_max,
		   volume, history,
		   (minc_output_options *) NULL) != OK)
    {
      PyErr_SetString(PyExc_IOError, "Failed to output minc volume to file.");
      return NULL;
    }

  /* return None */
  Py_INCREF(Py_None);
  return Py_None;
}


/* Method: set_cache_output_volume_parameters( */
/* Arguments:
    PyCObject           volume
    string              filename         
    integer             nc_data_type
    integer             signed_flag
    floating point      voxel_min
    floating point      voxel_max
    string              original_filename or None
    string              history
    (not implemented)   minc_output_options

   Returns:
    None
*/
static PyObject *
set_cache_output_volume_parameters_wrapper (PyObject *self, PyObject *args)
{
  char      *filename, *original_filename;
  PyObject  *cobj, *original_obj;
  int       nc_data_type;
  int       signed_flag;  
  double    voxel_min, voxel_max;
  char      *history;

  Volume volume;

  /* translate arguments to C data types */
  if (!PyArg_ParseTuple(args, "O!siiddOs", &PyCObject_Type, &cobj,
			&filename, &nc_data_type, &signed_flag,
                        &voxel_min, &voxel_max, &original_obj, &history))
    return NULL;

  volume = (Volume) PyCObject_AsVoidPtr(cobj);
  if (original_obj == Py_None) {
    original_filename = NULL;
  }
  else if (PyString_Check(original_obj)) {
    original_filename = PyString_AsString(original_obj);
  }
  else {
    return NULL;
  }


  set_cache_output_volume_parameters(volume, (STRING) filename,
					(nc_type) nc_data_type,
					(BOOLEAN) signed_flag,
					(Real) voxel_min, (Real) voxel_max,
					original_filename, history,
				     (minc_output_options *) NULL);
  /* return None */
  Py_INCREF(Py_None);
  return Py_None;
}

/* Method: copy_volume_definition_wrapper */
/* Arguments:
    PyCObject           existing_volume
    integer             nc_data_type
    integer             signed_flag
    floating point      voxel_min
    floating point      voxel_max

  Returns:
    PyCObject           volume
*/
static PyObject *
copy_volume_definition_wrapper (PyObject *self, PyObject *args)
{
  PyObject  *cobj;
  Volume    existing_volume, volume;
  int       nc_data_type;
  int       signed_flag;  
  double    voxel_min, voxel_max;

  /* set default values for arguments */
  nc_data_type = NC_UNSPECIFIED;
  signed_flag = NC_UNSPECIFIED;
  voxel_min = NC_UNSPECIFIED;
  voxel_max = NC_UNSPECIFIED;
  
  /* translate arguments to C data types */
  if (!PyArg_ParseTuple(args, "O!|iidd", &PyCObject_Type, &cobj,
			&nc_data_type, &signed_flag,
                        &voxel_min, &voxel_max))
    return NULL;
  
  existing_volume = (Volume) PyCObject_AsVoidPtr(cobj);

  /* call copy_volume_definition from volume_io library */
  volume = copy_volume_definition(existing_volume, 
				  (nc_type) nc_data_type, 
				  (BOOLEAN) signed_flag,
				  (Real) voxel_min, (Real) voxel_max);

  /* return volume as an opaque PyCObject */
  return PyCObject_FromVoidPtr((void *) volume, NULL);
}

/* Method: set_volume_real_range */
/* Arguments:
    PyCObject           volume
    floating point      real_min
    floating point      real_max

  Returns:
    None
*/
static PyObject *
set_volume_real_range_wrapper (PyObject *self, PyObject *args)
{
  PyObject  *volume;
  Real      real_min, real_max;

  /* translate arguments to C data types */
  if (!PyArg_ParseTuple(args, "O!dd", &PyCObject_Type, &volume,
			&real_min, &real_max))
    return NULL;

  /* call get_volume_real_value from volume_io library */
  set_volume_real_range((Volume) PyCObject_AsVoidPtr(volume), 
				real_min, real_max);

  /* return None */
  Py_INCREF(Py_None);
  return Py_None;
}



/* Method: get_volume_real_value */
/* Arguments:
    PyCObject           volume
    integer             v0
    integer             v1  (optional)
    integer             v2  (optional)
    integer             v3  (optional)
    integer             v4  (optional)

  Returns:
    floating point      value
*/
#if 0
static PyObject *
get_volume_real_value_wrapper (PyObject *self, PyObject *args)
{
  PyObject  *volume;
  int       v0, v1, v2, v3, v4;
  Real      value;

  v0 = v1 = v2 = v3 = v4 = 0;
  /* translate arguments to C data types */
  if (!PyArg_ParseTuple(args, "O!i|iiii", &PyCObject_Type, &volume,
			&v0, &v1, &v2, &v3, &v4))
    return NULL;

  /* call get_volume_real_value from volume_io library */
  value = get_volume_real_value((Volume) PyCObject_AsVoidPtr(volume), 
				v0, v1, v2, v3, v4);

  /* return value */
  return Py_BuildValue("d", (double) value);
}
#endif



#define get_volume_all_real_values_inner_loop(type,rhs) \
  ptr0 = array->data; \
  v0 = v1 = v2 = v3 = v4 = 0; \
  switch(n_dimensions) { \
  case 5: \
    for (v0 = 0; v0 < sizes[0]; v0++) { \
      ptr1 = ptr0; \
      for (v1 = 0; v1 < sizes[1]; v1++) { \
        ptr2 = ptr1; \
        for (v2 = 0; v2 < sizes[2]; v2++) { \
          ptr3 = ptr2; \
          for (v3 = 0; v3 < sizes[3]; v3++) { \
            ptr4 = ptr3; \
            for (v4 = 0; v4 < sizes[4]; v4++) { \
              *(type) ptr4 = rhs; \
              ptr4 += array->strides[4]; \
            } \
            ptr3 += array->strides[3]; \
          } \
          ptr2 += array->strides[2]; \
        } \
        ptr1 += array->strides[1]; \
      } \
      ptr0 += array->strides[0]; \
    } \
    break; \
  case 4: \
    for (v0 = 0; v0 < sizes[0]; v0++) { \
      ptr1 = ptr0; \
      for (v1 = 0; v1 < sizes[1]; v1++) { \
        ptr2 = ptr1; \
        for (v2 = 0; v2 < sizes[2]; v2++) { \
          ptr3 = ptr2; \
          for (v3 = 0; v3 < sizes[3]; v3++) { \
	    *(type) ptr3 = rhs; \
            ptr3 += array->strides[3]; \
          } \
          ptr2 += array->strides[2]; \
        } \
        ptr1 += array->strides[1]; \
      } \
      ptr0 += array->strides[0]; \
    } \
    break; \
  case 3: \
    for (v0 = 0; v0 < sizes[0]; v0++) { \
      ptr1 = ptr0; \
      for (v1 = 0; v1 < sizes[1]; v1++) { \
        ptr2 = ptr1; \
        for (v2 = 0; v2 < sizes[2]; v2++) { \
	  *(type) ptr2 = rhs; \
          ptr2 += array->strides[2]; \
        } \
        ptr1 += array->strides[1]; \
      } \
      ptr0 += array->strides[0]; \
    } \
    break; \
  case 2: \
    for (v0 = 0; v0 < sizes[0]; v0++) { \
      ptr1 = ptr0; \
      for (v1 = 0; v1 < sizes[1]; v1++) { \
	*(type) ptr1 = rhs; \
        ptr1 += array->strides[1]; \
      } \
      ptr0 += array->strides[0]; \
    } \
    break; \
  case 1: \
    for (v0 = 0; v0 < sizes[0]; v0++) { \
      *(type) ptr0 = rhs; \
      ptr0 += array->strides[0]; \
    } \
    break; \
  }


/* Method: get_volume_all_real_values */
/* Arguments:
    PyCObject           volume

  Returns:
    NumPy array         values
*/
static PyObject *
get_volume_all_real_values_wrapper (PyObject *self, PyObject *args)
{
  PyObject       *cobj;
  char           *typecode;
  Volume         volume;
  int            n_dimensions;
  int            sizes[MAX_VOLUME_DIMENSIONS];
  int            v0, v1, v2, v3, v4;
  int            i;
  PyArray_Descr  *descr;
  PyArrayObject  *array;
  char           *ptr0, *ptr1, *ptr2, *ptr3, *ptr4;

  /* translate arguments to C data types */
  typecode = NULL;
  if (!PyArg_ParseTuple(args, "O!|s", &PyCObject_Type, &cobj, &typecode))
    return NULL;

  /* obtain volume object */
  volume = (Volume) PyCObject_AsVoidPtr(cobj);

  n_dimensions = get_volume_n_dimensions(volume);
  get_volume_sizes(volume, sizes);

  /* set dimension of remaining dimensions to one  */
  for (i = n_dimensions; i < MAX_VOLUME_DIMENSIONS; i++)
    sizes[i] = 1;

  descr = (typecode == NULL)? PyArray_DescrFromType(PyArray_DOUBLE):
    PyArray_DescrFromType(typecode[0]);

  /* create NumPy array */
  array = (PyArrayObject *) 
    PyArray_FromDims(n_dimensions, sizes, descr->type_num); 

  /* call get_volume_real_value from volume_io library */
  switch (descr->type_num) {
  case NPY_DOUBLE:
    get_volume_all_real_values_inner_loop(npy_double *, 
		    get_volume_real_value(volume,  v0, v1, v2, v3, v4))
    break;
  case NPY_FLOAT:
    get_volume_all_real_values_inner_loop(npy_float *, 
		    get_volume_real_value(volume,  v0, v1, v2, v3, v4))
    break;
  case NPY_LONG:
    get_volume_all_real_values_inner_loop(npy_long *, 
		    floor(get_volume_real_value(volume,  v0, v1, v2, v3, v4)+0.5))
    break;
  case NPY_INT:
    get_volume_all_real_values_inner_loop(npy_int *, 
		    floor(get_volume_real_value(volume,  v0, v1, v2, v3, v4)+0.5))
    break;
  case NPY_SHORT:
    get_volume_all_real_values_inner_loop(npy_short *, 
		    floor(get_volume_real_value(volume,  v0, v1, v2, v3, v4)+0.5))
    break;
  case NPY_BYTE:
    get_volume_all_real_values_inner_loop(npy_byte *, 
			floor(get_volume_real_value(volume,  v0, v1, v2, v3, v4)+0.5))
    break;
  case NPY_UBYTE:
    get_volume_all_real_values_inner_loop(npy_ubyte *, 
			floor(get_volume_real_value(volume,  v0, v1, v2, v3, v4)+0.5))
    break;
  default:
    PyErr_SetString(PyExc_TypeError, "Specified typecode is not supported");
    return NULL;
  }
  /* return volume as an opaque PyCObject */
  return (PyObject *) array;
}

/* Method: get_real_subvolume_wrapper */
/* Arguments:
    PyCObject           volume
    tuple               starts
    tuple               sizes

  Returns:
    NumPy array         values
*/
static PyObject *
get_real_subvolume_wrapper (PyObject *self, PyObject *args)
{
  PyObject       *cobj, *tuple1, *tuple2;
  Volume         volume;
  int            n_dimensions;
  int            starts[MAX_VOLUME_DIMENSIONS];
  int            sizes[MAX_VOLUME_DIMENSIONS];
  int            volume_sizes[MAX_VOLUME_DIMENSIONS];
  int            v0, v1, v2, v3, v4;
  int            i;
  PyArrayObject  *array;
  char           *ptr0, *ptr1, *ptr2, *ptr3, *ptr4;

  /* translate arguments to C data types */
  if (!PyArg_ParseTuple(args, "O!O!O!", &PyCObject_Type, &cobj, 
			&PyTuple_Type, &tuple1, &PyTuple_Type, &tuple2))
    return NULL;

  /* obtain volume object */
  volume = (Volume) PyCObject_AsVoidPtr(cobj);

  n_dimensions = get_volume_n_dimensions(volume);
  get_volume_sizes(volume, volume_sizes);

  /* set size of remaining dimensions to one and start to zero */
  for (i = n_dimensions; i < MAX_VOLUME_DIMENSIONS; i++) {
    sizes[i] = 1;
    starts[i] = 0;
  }

  /* read starts and sizes tuple */
  if (!PyArg_ParseTuple(tuple1, "i|iiii", starts, starts+1, starts+2,
			starts+3, starts+4))
    return NULL;

  if (!PyArg_ParseTuple(tuple2, "i|iiii", sizes, sizes+1, sizes+2,
			sizes+3, sizes+4))
    return NULL;

  /* test that hyperslab is within volume */
  for (i = 0; i < n_dimensions; i++) {
    if((starts[i] < 0) || (starts[i] + sizes[i] > volume_sizes[i])) {
      PyErr_SetString(PyExc_ValueError, 
	   "Requested sub-volume is not within the given volume.");
      return NULL;
    } 
  }

  /* create NumPy array */
  array = (PyArrayObject *) 
    PyArray_FromDims(n_dimensions, sizes, PyArray_DOUBLE); 
  
  /* copy values into NumPy array */
  ptr0 = array->data;
  switch (n_dimensions) {
  case 5:
    for (v0 = starts[0]; v0 < starts[0]+sizes[0]; v0++) {
      ptr1 = ptr0;
      for (v1 = starts[1]; v1 < starts[1]+sizes[1]; v1++) {
	ptr2 = ptr1;
	for (v2 = starts[2]; v2 < starts[2]+sizes[2]; v2++) {
	  ptr3 = ptr2;
	  for (v3 = starts[3]; v3 < starts[3]+sizes[3]; v3++) {
	    ptr4 = ptr3;
	    for (v4 = starts[4]; v4 < starts[4]+sizes[4]; v4++) {
	      /* call get_volume_real_value from volume_io library */
	      *(double *) ptr4 = 
		get_volume_real_value(volume,  v0, v1, v2, v3, v4);
	      ptr4 += array->strides[4];
	    }
	    ptr3 += array->strides[3];
	  }
	  ptr2 += array->strides[2];
	}
	ptr1 += array->strides[1];  
      }
      ptr0 += array->strides[0];
    }
    break;
  case 4:
    for (v0 = starts[0]; v0 < starts[0]+sizes[0]; v0++) {
      ptr1 = ptr0;
      for (v1 = starts[1]; v1 < starts[1]+sizes[1]; v1++) {
	ptr2 = ptr1;
	for (v2 = starts[2]; v2 < starts[2]+sizes[2]; v2++) {
	  ptr3 = ptr2;
	  for (v3 = starts[3]; v3 < starts[3]+sizes[3]; v3++) {
	    /* call get_volume_real_value from volume_io library */
	    *(double *) ptr3 = 
		get_volume_real_value(volume,  v0, v1, v2, v3, 0);
	    ptr3 += array->strides[3];
	  }
	  ptr2 += array->strides[2];
	}
	ptr1 += array->strides[1];  
      }
      ptr0 += array->strides[0];
    }
    break;
  case 3:
    for (v0 = starts[0]; v0 < starts[0]+sizes[0]; v0++) {
      ptr1 = ptr0;
      for (v1 = starts[1]; v1 < starts[1]+sizes[1]; v1++) {
	ptr2 = ptr1;
	for (v2 = starts[2]; v2 < starts[2]+sizes[2]; v2++) {
	  /* call get_volume_real_value from volume_io library */
	  *(double *) ptr2 = 
	    get_volume_real_value(volume,  v0, v1, v2, 0, 0);
	  ptr2 += array->strides[2];
	}
	ptr1 += array->strides[1];  
      }
      ptr0 += array->strides[0];
    }
    break;
  case 2:
    for (v0 = starts[0]; v0 < starts[0]+sizes[0]; v0++) {
      ptr1 = ptr0;
      for (v1 = starts[1]; v1 < starts[1]+sizes[1]; v1++) {
	  /* call get_volume_real_value from volume_io library */
	  *(double *) ptr1 = 
	    get_volume_real_value(volume,  v0, v1, 0, 0, 0);
	ptr1 += array->strides[1];  
      }
      ptr0 += array->strides[0];
    }
    break;
  case 1:
    for (v0 = starts[0]; v0 < starts[0]+sizes[0]; v0++) {
      /* call get_volume_real_value from volume_io library */
      *(double *) ptr0 = 
	    get_volume_real_value(volume,  v0, 0, 0, 0, 0);
      ptr0 += array->strides[0];
    }
    break;
  }

  /* return array */
  return (PyObject *) array;
}

/* Method: set_volume_all_real_values */
/* Arguments:
    PyCObject           volume
    NumPy array         values
*/
static PyObject *
set_volume_all_real_values_wrapper (PyObject *self, PyObject *args)
{
  PyObject       *cobj;
  Volume         volume;
  int            n_dimensions;
  int            sizes[MAX_VOLUME_DIMENSIONS];
  int            v0, v1, v2, v3, v4;
  int            i;
  PyArrayObject  *array;
  char           *ptr0, *ptr1, *ptr2, *ptr3, *ptr4;

  /* translate arguments to C data types */
  if (!PyArg_ParseTuple(args, "O!O!", &PyCObject_Type, &cobj,
			&PyArray_Type, &array))
    return NULL;

  /* obtain volume object */
  volume = (Volume) PyCObject_AsVoidPtr(cobj);

  n_dimensions = get_volume_n_dimensions(volume);
  get_volume_sizes(volume, sizes);
  
  /* test that number of dimensions and volumes sizes match */
  if (n_dimensions != array->nd) {
    PyErr_SetString(PyExc_ValueError, 
        "MINC volume and NumPy array must have the same number of dimensions");
    return NULL;
  }
  if (array->descr->type_num != PyArray_DOUBLE) {
    PyErr_SetString(PyExc_ValueError, 
        "NumPy array must by of type Double");
    return NULL;
  }
  for (i = 0; i < n_dimensions; i++) {
    if (array->dimensions[i] != sizes[i]) {
      PyErr_SetString(PyExc_ValueError, 
         "MINC volume and NumPy array must have the same dimensions");
      return NULL;
    } 
  }

  /* set dimension of remaining dimensions to one  */
  for (i = n_dimensions; i < MAX_VOLUME_DIMENSIONS; i++)
    sizes[i] = 1;

  /* copy values from NumPy array */
  ptr0 = array->data;
  switch (n_dimensions) {
  case 5:
    for (v0 = 0; v0 < sizes[0]; v0++) {
      ptr1 = ptr0;
      for (v1 = 0; v1 < sizes[1]; v1++) {
	ptr2 = ptr1;
	for (v2 = 0; v2 < sizes[2]; v2++) {
	  ptr3 = ptr2;
	  for (v3 = 0; v3 < sizes[3]; v3++) {
	    ptr4 = ptr3;
	    for (v4 = 0; v4 < sizes[4]; v4++) {
	      set_volume_real_value(volume,  v0, v1, v2, v3, v4,
				    *(double *) ptr4);
	      ptr4 += array->strides[4];
	    }
	    ptr3 += array->strides[3];
	  }
	  ptr2 += array->strides[2];
	}
	ptr1 += array->strides[1];  
      }
      ptr0 += array->strides[0];
    }
    break;
  case 4:
    for (v0 = 0; v0 < sizes[0]; v0++) {
      ptr1 = ptr0;
      for (v1 = 0; v1 < sizes[1]; v1++) {
	ptr2 = ptr1;
	for (v2 = 0; v2 < sizes[2]; v2++) {
	  ptr3 = ptr2;
	  for (v3 = 0; v3 < sizes[3]; v3++) {
	    set_volume_real_value(volume,  v0, v1, v2, v3, 0,
				    *(double *) ptr3);
	    ptr3 += array->strides[3];
	  }
	  ptr2 += array->strides[2];
	}
	ptr1 += array->strides[1];  
      }
      ptr0 += array->strides[0];
    }
    break;
  case 3:
    for (v0 = 0; v0 < sizes[0]; v0++) {
      ptr1 = ptr0;
      for (v1 = 0; v1 < sizes[1]; v1++) {
	ptr2 = ptr1;
	for (v2 = 0; v2 < sizes[2]; v2++) {
	  set_volume_real_value(volume,  v0, v1, v2, 0, 0,
				*(double *) ptr2);
	  ptr2 += array->strides[2];
	}
	ptr1 += array->strides[1];  
      }
      ptr0 += array->strides[0];
    }
    break;
  case 2:
    for (v0 = 0; v0 < sizes[0]; v0++) {
      ptr1 = ptr0;
      for (v1 = 0; v1 < sizes[1]; v1++) {
	  set_volume_real_value(volume,  v0, v1, 0, 0, 0,
				*(double *) ptr1);
	ptr1 += array->strides[1];  
      }
      ptr0 += array->strides[0];
    }
    break;
  case 1:
    for (v0 = 0; v0 < sizes[0]; v0++) {
      set_volume_real_value(volume,  v0, 0, 0, 0, 0,
			    *(double *) ptr0);
      ptr0 += array->strides[0];
    }
    break;
  }

  /* return None */
  Py_INCREF(Py_None);
  return Py_None;
}


#define set_real_subvolume_inner_loop(type) \
  ptr0 = array->data; \
  /* copy values into NumPy array */ \
  switch (n_dimensions) { \
  case 5: \
    for (v0 = starts[0]; v0 < starts[0]+sizes[0]; v0++) { \
      ptr1 = ptr0; \
      for (v1 = starts[1]; v1 < starts[1]+sizes[1]; v1++) { \
        ptr2 = ptr1; \
        for (v2 = starts[2]; v2 < starts[2]+sizes[2]; v2++) { \
          ptr3 = ptr2; \
          for (v3 = starts[3]; v3 < starts[3]+sizes[3]; v3++) { \
            ptr4 = ptr3; \
            for (v4 = starts[4]; v4 < starts[4]+sizes[4]; v4++) { \
              /* call set_volume_real_value from volume_io library */ \
              set_volume_real_value(volume,  v0, v1, v2, v3, v4, *(type) ptr4); \
              ptr4 += array->strides[4]; \
            } \
            ptr3 += array->strides[3]; \
          } \
          ptr2 += array->strides[2]; \
        } \
        ptr1 += array->strides[1]; \
      } \
      ptr0 += array->strides[0]; \
    } \
    break; \
  case 4: \
    for (v0 = starts[0]; v0 < starts[0]+sizes[0]; v0++) { \
      ptr1 = ptr0; \
      for (v1 = starts[1]; v1 < starts[1]+sizes[1]; v1++) { \
        ptr2 = ptr1; \
        for (v2 = starts[2]; v2 < starts[2]+sizes[2]; v2++) { \
          ptr3 = ptr2; \
          for (v3 = starts[3]; v3 < starts[3]+sizes[3]; v3++) { \
              /* call set_volume_real_value from volume_io library */ \
            set_volume_real_value(volume,  v0, v1, v2, v3, 0, *(type) ptr3); \
            ptr3 += array->strides[3]; \
          } \
          ptr2 += array->strides[2]; \
        } \
        ptr1 += array->strides[1]; \
      } \
      ptr0 += array->strides[0]; \
    } \
    break; \
  case 3: \
    for (v0 = starts[0]; v0 < starts[0]+sizes[0]; v0++) { \
      ptr1 = ptr0; \
      for (v1 = starts[1]; v1 < starts[1]+sizes[1]; v1++) { \
        ptr2 = ptr1; \
        for (v2 = starts[2]; v2 < starts[2]+sizes[2]; v2++) { \
            /* call set_volume_real_value from volume_io library */ \
          set_volume_real_value(volume,  v0, v1, v2, 0, 0, *(type) ptr2); \
          ptr2 += array->strides[2]; \
        } \
        ptr1 += array->strides[1]; \
      } \
      ptr0 += array->strides[0]; \
    } \
    break; \
  case 2: \
    for (v0 = starts[0]; v0 < starts[0]+sizes[0]; v0++) { \
      ptr1 = ptr0; \
      for (v1 = starts[1]; v1 < starts[1]+sizes[1]; v1++) { \
	  /* call set_volume_real_value from volume_io library */ \
	set_volume_real_value(volume,  v0, v1, 0, 0, 0, *(type) ptr1); \
        ptr1 += array->strides[1]; \
      } \
      ptr0 += array->strides[0]; \
    } \
    break; \
  case 1: \
    for (v0 = starts[0]; v0 < starts[0]+sizes[0]; v0++) { \
	  /* call set_volume_real_value from volume_io library */ \
      set_volume_real_value(volume,  v0, 0, 0, 0, 0, *(type) ptr0); \
      ptr0 += array->strides[0]; \
    } \
    break; \
}


/* Method: set_real_subvolume_wrapper */
/* Arguments:
    PyCObject           volume
    tuple               starts
    sequence type       data

  Returns:
    None
*/
static PyObject *
set_real_subvolume_wrapper (PyObject *self, PyObject *args)
{
  PyObject       *cobj, *tuple1;
  Volume         volume;
  int            n_dimensions;
  int            starts[MAX_VOLUME_DIMENSIONS];
  int            sizes[MAX_VOLUME_DIMENSIONS];
  int            volume_sizes[MAX_VOLUME_DIMENSIONS];
  int            v0, v1, v2, v3, v4;
  int            i;
  PyArrayObject  *array;
  char           *ptr0, *ptr1, *ptr2, *ptr3, *ptr4;

  /* translate arguments to C data types */
  if (!PyArg_ParseTuple(args, "O!O!O!", &PyCObject_Type, &cobj, 
			&PyTuple_Type, &tuple1, &PyArray_Type, &array))
    return NULL;

  /* obtain volume object */
  volume = (Volume) PyCObject_AsVoidPtr(cobj);

  n_dimensions = get_volume_n_dimensions(volume);
  get_volume_sizes(volume, volume_sizes);

  /* set size of remaining dimensions to one and start to zero */
  for (i = n_dimensions; i < MAX_VOLUME_DIMENSIONS; i++) {
    sizes[i] = 1;
    starts[i] = 0;
  }

  /* read starts tuple */
  if (!PyArg_ParseTuple(tuple1, "i|iiii", starts, starts+1, starts+2,
			starts+3, starts+4))
    return NULL;

  /* check dimensions */
  if(array->nd != n_dimensions) {
    PyErr_SetString(PyExc_ValueError, 
		    "dimensions of subvolume and volume do not match");
    return NULL;
  }

  /* test that hyperslab is within volume */
  for (i = 0; i < n_dimensions; i++) {
    sizes[i] = array->dimensions[i];
    if((starts[i] < 0) || 
       (starts[i] + sizes[i] > volume_sizes[i])) 
    {
      PyErr_SetString(PyExc_ValueError, 
	   "Requested sub-volume is not within the given volume.");
      return NULL;
    } 
  }

  /* call set_volume_real_value from volume_io library */
  switch (array->descr->type_num) {
  case NPY_DOUBLE:
    set_real_subvolume_inner_loop(npy_double *)
    break;
  case NPY_FLOAT:
    set_real_subvolume_inner_loop(npy_float *)
    break;
  case NPY_LONG:
    set_real_subvolume_inner_loop(npy_long *)
    break;
  case NPY_INT:
    set_real_subvolume_inner_loop(npy_int *)
    break;
  case NPY_SHORT:
    set_real_subvolume_inner_loop(npy_short *)
    break;
  case NPY_BYTE:
    set_real_subvolume_inner_loop(npy_byte *)
    break;
  case NPY_UBYTE:
    set_real_subvolume_inner_loop(npy_ubyte *)
    break;
  default:
    PyErr_SetString(PyExc_TypeError, "Specified typecode is not supported");
    return NULL;
  }

  /* return None */
  Py_INCREF(Py_None);
  return Py_None;
}


/* Method: get_volume_dimension_names_wrapper */
/* Arguments:
    PyCObject           volume

  Returns:
    tuple              dimension_names
*/
static PyObject *
get_volume_dimension_names_wrapper (PyObject *self, PyObject *args)
{
  PyObject  *volume, *tuple;
  STRING    *dimension_names;
  int       n_dimensions, i;

  /* translate arguments to C data types */
  if (!PyArg_ParseTuple(args, "O!", &PyCObject_Type, &volume))
    return NULL;

  dimension_names = get_volume_dimension_names((Volume) PyCObject_AsVoidPtr(volume));
  n_dimensions = get_volume_n_dimensions((Volume) PyCObject_AsVoidPtr(volume));
  tuple = PyTuple_New (n_dimensions); 

  for(i = 0; i < n_dimensions; i++) {
    PyTuple_SetItem(tuple, i, PyString_FromString(dimension_names[i]));
  }
  
  delete_dimension_names((Volume) PyCObject_AsVoidPtr(volume), dimension_names);
  return tuple;
}


/* Method: fill_volume_real_value */
/* fill all image elements with given value */
/* Arguments:
    PyCObject           volume
    floating point      value
*/
static PyObject *
fill_volume_real_value_wrapper (PyObject *self, PyObject *args)
{
  PyObject       *cobj;
  Volume         volume;
  int            n_dimensions;
  int            sizes[MAX_VOLUME_DIMENSIONS];
  int            v0, v1, v2, v3, v4;
  int            i;
  Real           value;
  Real           voxel;

  /* translate arguments to C data types */
  if (!PyArg_ParseTuple(args, "O!d", &PyCObject_Type, &cobj,
			&value))
    return NULL;

  /* obtain volume object */
  volume = (Volume) PyCObject_AsVoidPtr(cobj);

  n_dimensions = get_volume_n_dimensions(volume);
  get_volume_sizes(volume, sizes);
  
  /* set dimension of remaining dimensions to one  */
  for (i = n_dimensions; i < MAX_VOLUME_DIMENSIONS; i++)
    sizes[i] = 1;

  voxel = convert_value_to_voxel( volume, value );

  for (v0 = 0; v0 < sizes[0]; v0++) {
    for (v1 = 0; v1 < sizes[1]; v1++) {
      for (v2 = 0; v2 < sizes[2]; v2++) {
	for (v3 = 0; v3 < sizes[3]; v3++) {
	  for (v4 = 0; v4 < sizes[4]; v4++) {
	    set_volume_voxel_value(volume,  v0, v1, v2, v3, v4, voxel);
	  }
	}
      }
    }
  }

  /* return None */
  Py_INCREF(Py_None);
  return Py_None;
}


/* Method: input_tag_points */
/* fill all image elements with given value */
/* Arguments:
   string               filename

*/
static PyObject *
input_tag_file_wrapper (PyObject *self, PyObject *args)
{
  char *filename;
  int n_volumes;
  int n_tag_points;
  Real **tags_volume[2];
  Real *weights;
  int  *structure_ids, *patient_ids;
  STRING *labels;
  PyObject *tags, *w, *s, *p, *l;
  PyObject *result;
  PyArrayObject *tag;
  int i, index;
  int dimensions;


  /* translate arguments to C data types */
  if (!PyArg_ParseTuple(args, "s", &filename))
    return NULL;

  if(input_tag_file(filename, &n_volumes, &n_tag_points, &tags_volume[0],
		    &tags_volume[1], &weights, &structure_ids,
		    &patient_ids, &labels) != OK) {
    PyErr_SetString(PyExc_IOError, "Failed to read tag file.");
    return NULL;
  }

  result = PyTuple_New((n_volumes == 1)? 6 : 7);

  PyTuple_SetItem(result, 0, PyInt_FromLong(n_volumes));
  dimensions = 3;
  for (index = 0; index < n_volumes; index++) {
    tags = PyList_New(n_tag_points);
    for (i = 0; i < n_tag_points; i++) {
      tag = (PyArrayObject *) PyArray_FromDims(1, &dimensions, PyArray_DOUBLE);
      memcpy(tag->data, tags_volume[index][i], sizeof(Real)*dimensions);
      PyList_SetItem(tags, i, (PyObject *) tag); 
    }
    PyTuple_SetItem(result, 1+index, tags);
  }

  w     = PyList_New(n_tag_points);
  s     = PyList_New(n_tag_points);
  p     = PyList_New(n_tag_points);
  l     = PyList_New(n_tag_points);

  for (i = 0; i < n_tag_points; i++) {
      PyList_SetItem(w, i, PyFloat_FromDouble(weights[i]));
      PyList_SetItem(s, i, PyInt_FromLong(structure_ids[i]));
      PyList_SetItem(p, i, PyInt_FromLong(patient_ids[i]));
      PyList_SetItem(l, i, PyString_FromString((labels[i] != NULL)? 
					       labels[i]: ""));
  }

  PyTuple_SetItem(result, n_volumes+1, w);
  PyTuple_SetItem(result, n_volumes+2, s);
  PyTuple_SetItem(result, n_volumes+3, p);
  PyTuple_SetItem(result, n_volumes+4, l);

  free_tag_points(n_volumes, n_tag_points, tags_volume[0], tags_volume[1],
		  weights, structure_ids, patient_ids, labels);

  return result;
}

/* Method: input_tag_points */
/* fill all image elements with given value */
/* Arguments:
   string               filename

*/
static PyObject *
output_tag_file_wrapper (PyObject *self, PyObject *args)
{
  char *filename, *comments;
  FILE *file;
  int n_volumes;
  int n_tag_points, structure_id, patient_id;
  double weight;
  double *tag1, *tag2;
  int  structure_ids, patient_ids;
  char *label;
  PyObject *w, *s, *p, *l, *arg4, *arg5;
  PyArrayObject *tags1, *tags2;
  int i;


  /* translate arguments to C data types */
  if (!PyArg_ParseTuple(args, "ssiiO!O!O!O!O!O!", &filename, &comments, 
			&n_volumes, &n_tag_points,
			&PyArray_Type, &arg4, &PyArray_Type, &arg5, 
			&PyList_Type, &w, &PyList_Type, &s, &PyList_Type, &p, &PyList_Type, &l))
    return NULL;

  tags1 = (PyArrayObject *) PyArray_ContiguousFromObject(arg4, PyArray_DOUBLE, 2, 2);
  if (tags1 == NULL) {
    return NULL;
  }
  if (n_volumes == 2) {
    tags2 = (PyArrayObject *) PyArray_ContiguousFromObject(arg5, PyArray_DOUBLE, 2, 2);
    if (tags2 == NULL) {
      return NULL;
    }
  }
  
  file = fopen(filename, "w");
  if (file == NULL) {
    PyErr_SetString(PyExc_IOError, "Failed to open new tag file.");
    return NULL;
  }

  if(initialize_tag_file_output(file, comments, n_volumes) != OK) {
    PyErr_SetString(PyExc_IOError, "Failed to initialize tag file ouput.");
    return NULL;
  }

  for (i = 0; i < n_tag_points; i++) {
    tag1 = (double *) (tags1->data + i*tags1->strides[0]);
    weight = PyFloat_AsDouble(PyList_GetItem(w, i));
    structure_id = (int) PyInt_AsLong(PyList_GetItem(s, i));
    patient_id = (int) PyInt_AsLong(PyList_GetItem(p, i));
    label = PyString_AsString(PyList_GetItem(l, i));
    if (n_volumes == 2) {
      tag2 = (double *) (tags2->data + i*tags2->strides[0]);
    }

    if(output_one_tag(file, n_volumes, tag1, tag2, &weight, &structure_id, &patient_id, label) 
       != OK) {
      fclose(file);
      PyErr_SetString(PyExc_IOError, "Failed to output tags.");
      return NULL;
    }
    
  }

  terminate_tag_file_output(file);
  fclose(file);

  /* release numpy arrays */
  Py_DECREF(tags1);
  if (n_volumes == 2)
    { Py_DECREF(tags2); }

  /* return None */
  Py_INCREF(Py_None);
  return Py_None;
}

