/*
 * Objects representing netCDF files and variables.
 *
 * Written by Konrad Hinsen
 * last revision: 2001-1-8
 */

/* Hacked for use with py_minc by John Sled 
   December 4, 2001  */


#include "Python.h"
#include "Numeric/arrayobject.h"
#include "netcdf.h"

#define _PY_MINC_NETCDF_MODULE
#include "py_minc_netcdfmodule.h"


staticforward int
netcdf_file_init(PyNetCDFFileObject *self);
staticforward PyNetCDFVariableObject *
netcdf_variable_new(PyNetCDFFileObject *file, char *name, int id, int type,
		    int ndims, int *dimids, int nattrs);

/* Lock which grants access to netCDF routines (netCDF isn't thread-safe) */
#undef WITH_THREAD
#ifdef WITH_THREAD

#include "pythread.h"
PyThread_type_lock netCDF_lock;
#define acquire_netCDF_lock() { PyThread_acquire_lock(netCDF_lock, 1); }
#define release_netCDF_lock() { PyThread_release_lock(netCDF_lock); }

#else

#define acquire_netCDF_lock() {}
#define release_netCDF_lock() {}

#endif

/* Error object and error messages for netcdf-specific errors */

static char *netcdf_errors[] = {
  "No Error",
  "Not a netcdf id",
  "Too many netcdfs open",
  "netcdf file exists && NC_NOCLOBBER",
  "Invalid Argument",
  "Write to read only",
  "Operation not allowed in data mode",
  "Operation not allowed in define mode",
  "Coordinates out of Domain",
  "MAX_NC_DIMS exceeded",
  "String match to name in use",
  "Attribute not found",
  "MAX_NC_ATTRS exceeded",
  "Not a netcdf data type",
  "Invalid dimension id",
  "NC_UNLIMITED in the wrong index",
  "MAX_NC_VARS exceeded",
  "Variable not found",
  "Action prohibited on NC_GLOBAL varid",
  "Not a netcdf file",
  "In Fortran, string too short",
  "MAX_NC_NAME exceeded",
  "NC_UNLIMITED size already in use", /* 22 */
  "", "", "", "", "", "", "", "", "",
  "XDR error" /* 32 */
};

/* Set error string */
static void
netcdf_seterror(void)
{
  if (ncerr != 0) {
    char *error = "Unknown error";
    if (ncerr > 0 && ncerr <= 32)
      error = netcdf_errors[ncerr];
    PyErr_SetString(PyExc_IOError, error);
  }
}

static void
netcdf_signalerror(int code)
{
  static char buffer[200];
  if (code != NC_NOERR) {
    Py_BEGIN_ALLOW_THREADS;
    acquire_netCDF_lock();
    sprintf(buffer, "netcdf: %s", nc_strerror(code));
    release_netCDF_lock();
    Py_END_ALLOW_THREADS;
    PyErr_SetString(PyExc_IOError, buffer);
 }
}

/*
 * Python equivalents to netCDF data types
 *
 * Caution: the following specification may not be fully portable.
 * The comments indicate the correct netCDF specification. The assignment
 * of Python types assumes that 'short' is 16-bit and 'int' is 32-bit.
 */

int data_types[] = {-1,  /* not used */
		    PyArray_SBYTE,  /* signed 8-bit int */
		    PyArray_CHAR,   /* 8-bit character */
		    PyArray_SHORT,  /* 16-bit signed int */
		    PyArray_INT,    /* 32-bit signed int */
		    PyArray_FLOAT,  /* 32-bit IEEE float */
		    PyArray_DOUBLE  /* 64-bit IEEE float */
};

/* Utility functions */

static void
define_mode(PyNetCDFFileObject *file, int define_flag)
{
  if (file->define != define_flag) {
    Py_BEGIN_ALLOW_THREADS;
    acquire_netCDF_lock();
    if (file->define)
      ncendef(file->id);
    else
      ncredef(file->id);
    release_netCDF_lock();
    file->define = define_flag;
    Py_END_ALLOW_THREADS;
  }
}

static char
typecode(int type)
{
  char t;
  switch(type) {
  case PyArray_CHAR:
    t = 'c';
    break;
  case PyArray_UBYTE:
    t = 'b';
    break;
  case PyArray_SBYTE:
    t = '1';
    break;
  case PyArray_SHORT:
    t = 's';
    break;
  case PyArray_INT:
    t = 'i';
    break;
  case PyArray_LONG:
    t = 'l';
    break;
  case PyArray_FLOAT:
    t = 'f';
    break;
  case PyArray_DOUBLE:
    t = 'd';
    break;
  default: t = ' ';
  }
  return t;
}

static int
netcdf_type_from_code(char code)
{
  int type;
  switch(code) {
  case 'c':
    type = NC_CHAR;
    break;
  case 'b':
  case '1':
    type = NC_BYTE;
    break;
  case 's':
    type = NC_SHORT;
    break;
  case 'i':
  case 'l':
    type = NC_LONG;
    break;
  case 'f':
    type = NC_FLOAT;
    break;
  case 'd':
    type = NC_DOUBLE;
    break;
  default:
    type = 0;
  }
  return type;
}


static void
collect_attributes(int fileid, int varid, PyObject *attributes, int nattrs)
{
  char name[MAX_NC_NAME];
  nc_type type;
  int length;
  int py_type;
  int i;
  for (i = 0; i < nattrs; i++) {
    Py_BEGIN_ALLOW_THREADS;
    acquire_netCDF_lock();
    ncattname(fileid, varid, i, name);
    ncattinq(fileid, varid, name, &type, &length);
    release_netCDF_lock();
    Py_END_ALLOW_THREADS;
    py_type = data_types[type];
    if (py_type == PyArray_CHAR) {
      char *s = (char *)malloc((length+1)*sizeof(char));
      if (s != NULL) {
	PyObject *string;
	Py_BEGIN_ALLOW_THREADS;
	acquire_netCDF_lock();
	ncattget(fileid, varid, name, s);
	release_netCDF_lock();
	Py_END_ALLOW_THREADS;
	s[length] = '\0';
	string = PyString_FromString(s);
	free(s);
	if (string != NULL) {
	  PyDict_SetItemString(attributes, name, string);
	  Py_DECREF(string);
	}
      }
    }
    else {
      PyObject *array = PyArray_FromDims(1, &length, py_type);
      if (array != NULL) {
	Py_BEGIN_ALLOW_THREADS;
	acquire_netCDF_lock();
	ncattget(fileid, varid, name, ((PyArrayObject *)array)->data);
	release_netCDF_lock();
	Py_END_ALLOW_THREADS;
	array = PyArray_Return((PyArrayObject *)array);
	if (array != NULL) {
	  PyDict_SetItemString(attributes, name, array);
	  Py_DECREF(array);
	}
      }
    }
  }
}

static int
set_attribute(int fileid, int varid, PyObject *attributes,
	      char *name, PyObject *value)
{
  if (PyString_Check(value)) {
    int len = PyString_Size(value);
    char *string = PyString_AsString(value);
    int ret;
    Py_BEGIN_ALLOW_THREADS;
    acquire_netCDF_lock();
    ret = nc_put_att_text(fileid, varid, name, len, string);
    release_netCDF_lock();
    Py_END_ALLOW_THREADS;
    if (ret != NC_NOERR) {
      netcdf_signalerror(ret);
      return -1;
    }
    PyDict_SetItemString(attributes, name, value);
    return 0;
  }
  else {
    int ret;
    PyArrayObject *array =
    (PyArrayObject *)PyArray_ContiguousFromObject(value, PyArray_NOTYPE, 0, 1);
    if (array != NULL) {
      int len = (array->nd == 0) ? 1 : array->dimensions[0];
      int type = netcdf_type_from_code(array->descr->type);
      if (data_types[type] != array->descr->type_num) {
	PyArrayObject *array2 = (PyArrayObject *)
	  PyArray_ContiguousFromObject((PyObject *)array,
				       data_types[type], 0, 1);
	Py_DECREF(array);
	array = array2;
	if (array == NULL)
	  return -1;
      }
      Py_BEGIN_ALLOW_THREADS;
      acquire_netCDF_lock();
      ret = ncattput(fileid, varid, name, type, len, array->data);
      release_netCDF_lock();
      Py_END_ALLOW_THREADS;
      if (ret == -1) {
	netcdf_seterror();
	return -1;
      }
      PyDict_SetItemString(attributes, name, (PyObject *)array);
      return 0;
    }
    else
      return -1;
  }
}

static int
check_if_open(PyNetCDFFileObject *file, int mode)
{
  /* mode: -1 read, 1 write, 0 other */
  if (file == NULL || !file->open) {
    PyErr_SetString(PyExc_IOError, "netcdf: file has been closed");
    return 0;
  }
  else {
    if (mode != 1 || file->write) {
      return 1;
    }
    else {
      PyErr_SetString(PyExc_IOError, "netcdf: write access to read-only file");
      return 0;
    }
  }
}

/*
 * NetCDFFile object
 * (type declaration in netcdfmodule.h)
 */

/* Destroy file object */

static void
PyNetCDFFileObject_dealloc(PyNetCDFFileObject *self)
{
  /*  if (self->open)
      PyNetCDFFile_Close(self); */  /* <---------- Changed */
  Py_XDECREF(self->dimensions);
  Py_XDECREF(self->variables);
  Py_XDECREF(self->attributes);
  Py_XDECREF(self->name);
  Py_XDECREF(self->mode);
  PyMem_DEL(self);
}

/* Create file object */

static PyNetCDFFileObject *
PyNetCDFFile_Open(char *filename, char *mode)
{
}
/*
  PyNetCDFFileObject *self = PyObject_NEW(PyNetCDFFileObject,
					  &PyNetCDFFile_Type);
  int rw, share, ret;
  if (self == NULL)
    return NULL;
  self->dimensions = NULL;
  self->variables = NULL;
  self->attributes = NULL;
  self->name = NULL;
  self->mode = NULL;
  rw = share = ret = 0;
  if (strlen(mode) > 1) {
    if (mode[1] == '+') rw = 1;
    else if (mode[1] == 's') share = NC_SHARE;
    else ret = -1;
  }
  if (strlen(mode) > 2) {
    if (mode[2] == '+') rw = 1;
    else if (mode[2] == 's') share = NC_SHARE;
    else ret = -1;
  }
  if (ret == -1 || strlen(mode) > 3 ||
      (mode[0] != 'r' && mode[0] != 'w' && mode[0] != 'a')) {
    PyErr_SetString(PyExc_IOError, "illegal mode specification");
    PyNetCDFFileObject_dealloc(self);
    return NULL;
  }
  self->open = 0;
  if (mode[0] == 'w') {
    Py_BEGIN_ALLOW_THREADS;
    acquire_netCDF_lock();
    ret = nc_create(filename, NC_CLOBBER|share, &self->id);
    release_netCDF_lock();
    Py_END_ALLOW_THREADS;
    self->define = 1;
    self->write = 1;
    if (ret == NC_NOERR) {
      self->open = 1;
      netcdf_file_init(self);
    }
  }
  else if (mode[0] == 'a') {
    Py_BEGIN_ALLOW_THREADS;
    acquire_netCDF_lock();
    ret = nc_open(filename, NC_WRITE|share, &self->id);
    self->define = 0;
    if (ret == ENOENT) {
      ret = nc_create(filename, NC_NOCLOBBER|share, &self->id);
      self->define = 1;
    }
    release_netCDF_lock();
    Py_END_ALLOW_THREADS;
    self->write = 1;
    if (ret == NC_NOERR) {
      self->open = 1;
      netcdf_file_init(self);
    }
  }
  else if (mode[0] == 'r') {
    Py_BEGIN_ALLOW_THREADS;
    acquire_netCDF_lock();
    ret = nc_open(filename, rw ? (NC_WRITE|share) : (NC_NOWRITE|share),
		  &self->id);
    release_netCDF_lock();
    Py_END_ALLOW_THREADS;
    self->define = 0;
    self->write = rw;
    if (ret == NC_NOERR) {
      self->open = 1;
      netcdf_file_init(self);
    }
  }
  else {
    PyNetCDFFileObject_dealloc(self);
    return NULL;
  }
  if (ret != NC_NOERR) {
    netcdf_signalerror(ret);
    PyNetCDFFileObject_dealloc(self);
    return NULL;
  }
  self->name = PyString_FromString(filename);
  self->mode = PyString_FromString(mode);
  return self;
}
*/

/* Create file object from existing open file */

PyNetCDFFileObject * 
PyNetCDFFile_from_id(int id)
{
  PyNetCDFFileObject *self = PyObject_NEW(PyNetCDFFileObject,
					  &PyNetCDFFile_Type);
  int rw, share, ret;
  if (self == NULL)
    return NULL;
  self->id = id;
  self->dimensions = NULL;
  self->variables = NULL;
  self->attributes = NULL;
  self->name = NULL;
  self->mode = NULL;
  rw = share = ret = 0;
  self->open = 1;
  self->define = 0;
  self->write = rw;
  self->open = 1;
  netcdf_file_init(self);
  return self;
}


/* Create variables from file */

static int
netcdf_file_init(PyNetCDFFileObject *self)
{
  int ndims, nvars, ngattrs, recdim, status;
  int i;
  self->dimensions = PyDict_New();
  self->variables = PyDict_New();
  self->attributes = PyDict_New();
  Py_BEGIN_ALLOW_THREADS;
  acquire_netCDF_lock();
  status = ncinquire(self->id, &ndims, &nvars, &ngattrs, &recdim);
  release_netCDF_lock();
  Py_END_ALLOW_THREADS;
  if (status != NC_NOERR) {
      netcdf_signalerror(status);
      return -1;
  }
  self->recdim = recdim;
  for (i = 0; i < ndims; i++) {
    char name[MAX_NC_NAME];
    long size;
    PyObject *size_ob;
    Py_BEGIN_ALLOW_THREADS;
    acquire_netCDF_lock();
    ncdiminq(self->id, i, name, &size);
    release_netCDF_lock();
    Py_END_ALLOW_THREADS;
    if (i == recdim)
      PyDict_SetItemString(self->dimensions, name, Py_None);
    else {
      size_ob = PyInt_FromLong(size);
      PyDict_SetItemString(self->dimensions, name, size_ob);
      Py_DECREF(size_ob);
    }
  }
  for (i = 0; i < nvars; i++) {
    char name[MAX_NC_NAME];
    nc_type datatype;
    int ndimensions, nattrs;
    int *dimids;
    PyNetCDFVariableObject *variable;
    Py_BEGIN_ALLOW_THREADS;
    acquire_netCDF_lock();
    ncvarinq(self->id, i, name, &datatype, &ndimensions, NULL, &nattrs);
    release_netCDF_lock();
    Py_END_ALLOW_THREADS;
    if (ndimensions > 0) {
      dimids = (int *)malloc(ndimensions*sizeof(int));
      if (dimids == NULL) {
	PyErr_NoMemory();
	return 0;
      }
      Py_BEGIN_ALLOW_THREADS;
      acquire_netCDF_lock();
      ncvarinq(self->id, i, NULL, NULL, NULL, dimids, NULL);
      release_netCDF_lock();
      Py_END_ALLOW_THREADS;
    }
    else
      dimids = NULL;
    variable = netcdf_variable_new(self, name, i, data_types[datatype],
				   ndimensions, dimids, nattrs);
    if (variable != NULL) {
      PyDict_SetItemString(self->variables, name, (PyObject *)variable);
      Py_DECREF(variable);
    }
    else
      free(dimids);
  }
  collect_attributes(self->id, NC_GLOBAL, self->attributes, ngattrs);
  return 1;
}

/* Create dimension */

static int
PyNetCDFFile_CreateDimension(PyNetCDFFileObject *file, char *name, long size)
{
  PyObject *size_ob;
  int id;
  if (check_if_open(file, 1)) {
    if (size == 0 && file->recdim != -1) {
      PyErr_SetString(PyExc_IOError,
		      "netcdf: there is already an unlimited dimension");
      return -1;
    }
    define_mode(file, 1);
    Py_BEGIN_ALLOW_THREADS;
    acquire_netCDF_lock();
    id = ncdimdef(file->id, name, (size == 0) ? NC_UNLIMITED : size);
    release_netCDF_lock();
    Py_END_ALLOW_THREADS;
    if (id == -1) {
      netcdf_seterror();
      return -1;
    }
    else {
      if (size == 0) {
	PyDict_SetItemString(file->dimensions, name, Py_None);
	file->recdim = id;
      }
      else {
	size_ob = PyInt_FromLong(size);
	PyDict_SetItemString(file->dimensions, name, size_ob);
	Py_DECREF(size_ob);
      }
      return 0;
    }
  }
  else
    return -1;
}

static PyObject *
PyNetCDFFileObject_new_dimension(PyNetCDFFileObject *self, PyObject *args)
{
  char *name;
  PyObject *size_ob;
  long size;
  if (!PyArg_ParseTuple(args, "sO", &name, &size_ob))
    return NULL;
  if (size_ob == Py_None)
    size = 0;
  else if (PyInt_Check(size_ob))
    size = PyInt_AsLong(size_ob);
  else {
    PyErr_SetString(PyExc_TypeError, "size must be None or integer");
    return NULL;
  }
  if (PyNetCDFFile_CreateDimension(self, name, size) == 0) {
    Py_INCREF(Py_None);
    return Py_None;
  }
  else
    return NULL;
}
static char createDimension_doc[] = "";

/* Create variable */

static PyNetCDFVariableObject *
PyNetCDFFile_CreateVariable(PyNetCDFFileObject *file, char *name, int typecode,
			    char **dimension_names, int ndim)
{
  int *dimids;
  PyNetCDFVariableObject *variable;
  int ntype, i, ret;
  if (check_if_open(file, 1)) {
    define_mode(file, 1);
    if (ndim == 0)
      dimids = NULL;
    else {
      dimids = (int *)malloc(ndim*sizeof(int));
      if (dimids == NULL)
	return (PyNetCDFVariableObject *)PyErr_NoMemory();
    }
    for (i = 0; i < ndim; i++) {
      Py_BEGIN_ALLOW_THREADS;
      acquire_netCDF_lock();
      dimids[i] = ncdimid(file->id, dimension_names[i]);
      release_netCDF_lock();
      Py_END_ALLOW_THREADS;
      if (dimids[i] == -1) {
	netcdf_seterror();
	free(dimids);
	return NULL;
      }
      if (dimids[i] == file->recdim && i > 0) {
	PyErr_SetString(PyExc_IOError,
			"netcdf: unlimited dimension must be first");
	free(dimids);
	return NULL;
      }
    }
    ntype = netcdf_type_from_code(typecode);
    Py_BEGIN_ALLOW_THREADS;
    acquire_netCDF_lock();
    ret = nc_def_var(file->id, name, ntype, ndim, dimids, &i);
    release_netCDF_lock();
    Py_END_ALLOW_THREADS;
    if (ret != NC_NOERR) {
      netcdf_signalerror(ret);
      if (dimids != NULL)
	free(dimids);
      return NULL;
    }
    variable = netcdf_variable_new(file, name, i, data_types[ntype],
				   ndim, dimids, 0);
    if (variable != NULL) {
      PyDict_SetItemString(file->variables, name, (PyObject *)variable);
      return variable;
    }
    else {
      free(dimids);
      return NULL;
    }
  }
  else
    return NULL;
}

static PyObject *
PyNetCDFFileObject_new_variable(PyNetCDFFileObject *self, PyObject *args)
{
  PyNetCDFVariableObject *var;
  char **dimension_names;
  PyObject *item, *dim;
  char *name;
  int ndim;
  char type;
  int i;
  if (!PyArg_ParseTuple(args, "scO!", &name, &type, &PyTuple_Type, &dim))
    return NULL;
  ndim = PyTuple_Size(dim);
  if (ndim == 0)
    dimension_names = NULL;
  else {
    dimension_names = (char **)malloc(ndim*sizeof(char *));
    if (dimension_names == NULL) {
      PyErr_SetString(PyExc_MemoryError, "out of memory");
      return NULL;
    }
  }
  for (i = 0; i < ndim; i++) {
    item = PyTuple_GetItem(dim, i);
    if (PyString_Check(item))
      dimension_names[i] = PyString_AsString(item);
    else {
      PyErr_SetString(PyExc_TypeError, "dimension name must be a string");
      free(dimension_names);
      return NULL;
    }
  }
  var = PyNetCDFFile_CreateVariable(self, name, type, dimension_names, ndim);
  free(dimension_names);
  return (PyObject *)var;
}
static char createVariable_doc[] = "";

/* Return a variable object referring to an existing variable */

static PyNetCDFVariableObject *
PyNetCDFFile_GetVariable(PyNetCDFFileObject *file, char *name)
{
  return (PyNetCDFVariableObject *)PyDict_GetItemString(file->variables, name);
}

/* Synchronize output */

static int
PyNetCDFFile_Sync(PyNetCDFFileObject *file)
{
  int ret;
  if (check_if_open(file, 0)) {
    define_mode(file, 0);
    Py_BEGIN_ALLOW_THREADS;
    acquire_netCDF_lock();
    ret = ncsync(file->id);
    release_netCDF_lock();
    Py_END_ALLOW_THREADS;
    if (ret == -1) {
      netcdf_seterror();
      return -1;
    }
    else
      return 0;
  }
  else
    return -1;
}

static PyObject *
PyNetCDFFileObject_sync(PyNetCDFFileObject *self, PyObject *args)
{
  if (!PyArg_ParseTuple(args, ""))
    return NULL;
  if (PyNetCDFFile_Sync(self) == 0) {
    Py_INCREF(Py_None);
    return Py_None;
  }
  else
    return NULL;
}
static char sync_doc[] = "";
static char flush_doc[] =
"ensures that all modified data is written to the file";

/* Close file */

static int
PyNetCDFFile_Close(PyNetCDFFileObject *file)
{
  PyObject *name;
  PyNetCDFVariableObject *variable;
  int pos, ret;

  if (!check_if_open(file, 0))
    return -1;
  Py_BEGIN_ALLOW_THREADS;
  acquire_netCDF_lock();
  ret = nc_close(file->id);
  release_netCDF_lock();
  Py_END_ALLOW_THREADS;
  if (ret != NC_NOERR) {
    netcdf_signalerror(ret);
    ret = -1;
  }
  else
    ret = 0;
  file->open = 0;
  pos = 0;
  while (PyDict_Next(file->variables, &pos, &name, (PyObject **)&variable)) {
    Py_DECREF(variable->file);
    variable->file = NULL;
  }
  return ret;
}

static PyObject *
PyNetCDFFileObject_close(PyNetCDFFileObject *self, PyObject *args)
{
  char *history = NULL;
  if (!PyArg_ParseTuple(args, "|s", &history))
    return NULL;
  if (history != NULL)
    PyNetCDFFile_AddHistoryLine(self, history);
  if (PyNetCDFFile_Close(self) == 0) {
    Py_INCREF(Py_None);
    return Py_None;
  }
  else
    return NULL;
}
static char close_doc[] = "";

/* Method table */

static PyMethodDef PyNetCDFFileObject_methods[] = {
  {"close", (PyCFunction)PyNetCDFFileObject_close, 1, close_doc},
  {"createDimension", (PyCFunction)PyNetCDFFileObject_new_dimension, 1,
                      createDimension_doc},
  {"createVariable", (PyCFunction)PyNetCDFFileObject_new_variable, 1,
                      createVariable_doc},
  {"sync", (PyCFunction)PyNetCDFFileObject_sync, 1, sync_doc},
  {"flush", (PyCFunction)PyNetCDFFileObject_sync, 1, flush_doc},
  {NULL, NULL}		/* sentinel */
};

/* Attribute access */

static PyObject *
PyNetCDFFile_GetAttribute(PyNetCDFFileObject *self, char *name)
{
  PyObject *value;
  if (check_if_open(self, -1)) {
    if (strcmp(name, "dimensions") == 0) {
      Py_INCREF(self->dimensions);
      return self->dimensions;
    }
    if (strcmp(name, "variables") == 0) {
      Py_INCREF(self->variables);
      return self->variables;
    }
    if (strcmp(name, "__dict__") == 0) {
      Py_INCREF(self->attributes);
      return self->attributes;
    }
    value = PyDict_GetItemString(self->attributes, name);
    if (value != NULL) {
      Py_INCREF(value);
      return value;
    }
    else {
      PyErr_Clear();
      return Py_FindMethod(PyNetCDFFileObject_methods, (PyObject *)self, name);
    }
  }
  else
    return NULL;
}

static int
PyNetCDFFile_SetAttribute(PyNetCDFFileObject *self, char *name,
			  PyObject *value)
{
  if (check_if_open(self, 1)) {
    if (strcmp(name, "dimensions") == 0 ||
	strcmp(name, "variables") == 0 ||
	strcmp(name, "__dict__") == 0) {
      PyErr_SetString(PyExc_TypeError, "object has read-only attributes");
      return -1;
    }
    define_mode(self, 1);
    return set_attribute(self->id, NC_GLOBAL, self->attributes, name, value);
  }
  else
    return -1;
}

static int
PyNetCDFFile_SetAttributeString(PyNetCDFFileObject *self,
				char *name, char *value)
{
  PyObject *string = PyString_FromString(value);
  if (string != NULL)
    return PyNetCDFFile_SetAttribute(self, name, string);
  else
    return -1;
}

static int
PyNetCDFFile_AddHistoryLine(PyNetCDFFileObject *self, char *text)
{
  static char *history = "history";
  int alloc, old, new, new_alloc;
  PyStringObject *new_string;
  PyObject *h = PyNetCDFFile_GetAttribute(self, history);
  if (h == NULL) {
    PyErr_Clear();
    alloc = 0;
    old = 0;
    new = strlen(text);
  }
  else {
    alloc = PyString_Size(h);
    old = strlen(PyString_AsString(h));
    new = old + strlen(text) + 1;
  }
  new_alloc = (new <= alloc) ? alloc : new + 500;
  new_string = (PyStringObject *)PyString_FromStringAndSize(NULL, new_alloc);
  if (new_string) {
    char *s = new_string->ob_sval;
    int len, ret;
    memset(s, 0, new_alloc+1);
    if (h == NULL)
      len = -1;
    else {
      strcpy(s, PyString_AsString(h));
      len = strlen(s);
      s[len] = '\n';
    }
    strcpy(s+len+1, text);
    ret = PyNetCDFFile_SetAttribute(self, history, (PyObject *)new_string);
    Py_XDECREF(h);
    Py_DECREF(new_string);
    return ret;
  }
  else
    return -1;
}

/* Printed representation */
static PyObject *
PyNetCDFFileObject_repr(PyNetCDFFileObject *file)
{
  char buf[300];
  sprintf(buf, "<%s netCDF file '%.256s', mode '%.10s' at %lx>",
	  file->open ? "open" : "closed",
	  PyString_AsString(file->name),
	  PyString_AsString(file->mode),
	  (long)file);
  return PyString_FromString(buf);
}

/* Type definition */

statichere PyTypeObject PyNetCDFFile_Type = {
  PyObject_HEAD_INIT(NULL)
  0,		/*ob_size*/
  "NetCDFFile",	/*tp_name*/
  sizeof(PyNetCDFFileObject),	/*tp_basicsize*/
  0,		/*tp_itemsize*/
  /* methods */
  (destructor)PyNetCDFFileObject_dealloc, /*tp_dealloc*/
  0,			/*tp_print*/
  (getattrfunc)PyNetCDFFile_GetAttribute, /*tp_getattr*/
  (setattrfunc)PyNetCDFFile_SetAttribute, /*tp_setattr*/
  0,			/*tp_compare*/
  (reprfunc)PyNetCDFFileObject_repr,   /*tp_repr*/
  0,			/*tp_as_number*/
  0,			/*tp_as_sequence*/
  0,			/*tp_as_mapping*/
  0,			/*tp_hash*/
};

/*
 * NetCDFVariable object
 * (type declaration in netcdfmodule.h)
 */

/* Destroy variable object */

static void
PyNetCDFVariableObject_dealloc(PyNetCDFVariableObject *self)
{
  if (self->dimids != NULL)
    free(self->dimids);
  if (self->dimensions != NULL)
    free(self->dimensions);
  if (self->name != NULL)
    free(self->name);
  Py_XDECREF(self->file);
  Py_XDECREF(self->attributes);
  PyMem_DEL(self);
}

/* Create variable object */

static PyNetCDFVariableObject *
netcdf_variable_new(PyNetCDFFileObject *file, char *name, int id, int type,
		    int ndims, int *dimids, int nattrs)
{
  PyNetCDFVariableObject *self;
  int recdim;
  int i;
  if (check_if_open(file, -1)) {
    self = PyObject_NEW(PyNetCDFVariableObject, &PyNetCDFVariable_Type);
    if (self == NULL)
      return NULL;
    self->file = file;
    Py_INCREF(file);
    self->id = id;
    self->type = type;
    self->nd = ndims;
    self->dimids = dimids;
    self->unlimited = 0;
    Py_BEGIN_ALLOW_THREADS;
    acquire_netCDF_lock();
    ncinquire(file->id, NULL, NULL, NULL, &recdim);
    self->dimensions = (long *)malloc(ndims*sizeof(long));
    if (self->dimensions != NULL) {
      for (i = 0; i < ndims; i++)
	ncdiminq(file->id, dimids[i], NULL, &self->dimensions[i]);
      if (ndims > 0 && self->dimids[0] == self->file->recdim)
	self->unlimited = 1;
    }
    release_netCDF_lock();
    Py_END_ALLOW_THREADS;
    self->name = (char *)malloc(strlen(name)+1);
    if (self->name != NULL)
      strcpy(self->name, name);
    self->attributes = PyDict_New();
    collect_attributes(file->id, self->id, self->attributes, nattrs);
    return self;
  }
  else
    return NULL;
}

/* Return value */

static PyObject *
PyNetCDFVariableObject_value(PyNetCDFVariableObject *self, PyObject *args)
{
  PyNetCDFIndex *indices;
  if (!PyArg_ParseTuple(args, ""))
    return NULL;
  if (self->nd == 0)
    indices = NULL;
  else
    indices = PyNetCDFVariable_Indices(self);
  return PyArray_Return(PyNetCDFVariable_ReadAsArray(self, indices));
}

/* Assign value */

static PyObject *
PyNetCDFVariableObject_assign(PyNetCDFVariableObject *self, PyObject *args)
{
  PyObject *value;
  PyNetCDFIndex *indices;
  if (!PyArg_ParseTuple(args, "O", &value))
    return NULL;
  if (self->nd == 0)
    indices = NULL;
  else
    indices = PyNetCDFVariable_Indices(self);
  PyNetCDFVariable_WriteArray(self, indices, value);
  Py_INCREF(Py_None);
  return Py_None;
}

/* Return typecode */

static PyObject *
PyNetCDFVariableObject_typecode(PyNetCDFVariableObject *self, PyObject *args)
{
  char t;
  if (!PyArg_ParseTuple(args, ""))
    return NULL;
  t = typecode(self->type);
  return PyString_FromStringAndSize(&t, 1);
}

/* Method table */

static PyMethodDef PyNetCDFVariableObject_methods[] = {
  {"assignValue", (PyCFunction)PyNetCDFVariableObject_assign, 1},
  {"getValue", (PyCFunction)PyNetCDFVariableObject_value, 1},
  {"typecode", (PyCFunction)PyNetCDFVariableObject_typecode, 1},
  {NULL, NULL}		/* sentinel */
};

/* Attribute access */

static int
PyNetCDFVariable_GetRank(PyNetCDFVariableObject *var)
{
  return var->nd;
}

static long *
PyNetCDFVariable_GetShape(PyNetCDFVariableObject *var)
{
  int i;
  if (check_if_open(var->file, -1)) {
    Py_BEGIN_ALLOW_THREADS;
    acquire_netCDF_lock();
    for (i = 0; i < var->nd; i++)
      ncdiminq(var->file->id, var->dimids[i], NULL, &var->dimensions[i]);
    release_netCDF_lock();
    Py_END_ALLOW_THREADS;
    return var->dimensions;
  }
  else
    return NULL;
}

static PyObject *
PyNetCDFVariable_GetAttribute(PyNetCDFVariableObject *self, char *name)
{
  PyObject *value;
  if (strcmp(name, "shape") == 0) {
    PyObject *tuple;
    int i;
    if (check_if_open(self->file, -1)) {
      PyNetCDFVariable_GetShape(self);
      tuple = PyTuple_New(self->nd);
      for (i = 0; i < self->nd; i++)
	PyTuple_SetItem(tuple, i, PyInt_FromLong(self->dimensions[i]));
      return tuple;
    }
    else
      return NULL;
  }
  if (strcmp(name, "dimensions") == 0) {
    PyObject *tuple;
    char name[MAX_NC_NAME];
    int i;
    if (check_if_open(self->file, -1)) {
      tuple = PyTuple_New(self->nd);
      for (i = 0; i < self->nd; i++) {
	Py_BEGIN_ALLOW_THREADS;
	acquire_netCDF_lock();
	ncdiminq(self->file->id, self->dimids[i], name, NULL);
	release_netCDF_lock();
	Py_END_ALLOW_THREADS;
	PyTuple_SetItem(tuple, i, PyString_FromString(name));
      }
      return tuple;
    }
    else
      return NULL;
  }
  if (strcmp(name, "__dict__") == 0) {
    Py_INCREF(self->attributes);
    return self->attributes;
  }
  value = PyDict_GetItemString(self->attributes, name);
  if (value != NULL) {
    Py_INCREF(value);
    return value;
  }
  else {
    PyErr_Clear();
    return Py_FindMethod(PyNetCDFVariableObject_methods, (PyObject *)self,name);
  }
}

static int
PyNetCDFVariable_SetAttribute(PyNetCDFVariableObject *self,
			      char *name, PyObject *value)
{
  if (check_if_open(self->file, 1)) {
    if (strcmp(name, "shape") == 0 ||
	strcmp(name, "dimensions") == 0 ||
	strcmp(name, "__dict__") == 0) {
      PyErr_SetString(PyExc_TypeError, "object has read-only attributes");
      return -1;
    }
    define_mode(self->file, 1);
    return set_attribute(self->file->id, self->id, self->attributes,
			 name, value);
  }
  else
    return -1;
}

static int
PyNetCDFVariable_SetAttributeString(PyNetCDFVariableObject *self,
				    char *name, char *value)
{
  PyObject *string = PyString_FromString(value);
  if (string != NULL)
    return PyNetCDFVariable_SetAttribute(self, name, string);
  else
    return -1;
}

/* Subscripting */

static int
PyNetCDFVariableObject_length(PyNetCDFVariableObject *self)
{
  if (self->nd > 0)
    return self->dimensions[0];
  else
    return 0;
}

static PyNetCDFIndex *
PyNetCDFVariable_Indices(PyNetCDFVariableObject *var)
{
  PyNetCDFIndex *indices = 
    (PyNetCDFIndex *)malloc(var->nd*sizeof(PyNetCDFIndex));
  int i;
  if (indices != NULL)
    for (i = 0; i < var->nd; i++) {
      indices[i].start = 0;
      indices[i].stop = var->dimensions[i];
      indices[i].stride = 1;
      indices[i].item = 0;
    }
  else
    PyErr_SetString(PyExc_MemoryError, "out of memory");
  return indices;
}

static PyArrayObject *
PyNetCDFVariable_ReadAsArray(PyNetCDFVariableObject *self,
			     PyNetCDFIndex *indices)
{
  int *dims;
  PyArrayObject *array;
  int i, d;
  int nitems;
  int error = 0;
  d = 0;
  nitems = 1;
  if (!check_if_open(self->file, -1)) {
    free(indices);
    return NULL;
  }
  define_mode(self->file, 0);
  if (self->nd == 0)
    dims = NULL;
  else {
    dims = (int *)malloc(self->nd*sizeof(int));
    if (dims == NULL) {
      free(indices);
      return (PyArrayObject *)PyErr_NoMemory();
    }
  }
  for (i = 0; i < self->nd; i++) {
    error = error || (indices[i].stride <= 0);
    if (indices[i].start < 0)
      indices[i].start += self->dimensions[i];
    if (indices[i].start < 0)
      indices[i].start = 0;
    if (indices[i].start > self->dimensions[i])
      indices[i].start = self->dimensions[i];
    if (indices[i].item != 0)
      indices[i].stop = indices[i].start + 1;
    else {
      if (indices[i].stop < 0)
	indices[i].stop += self->dimensions[i];
      if (indices[i].stop < 0)
	indices[i].stop = 0;
      if (indices[i].stop > self->dimensions[i])
	indices[i].stop = self->dimensions[i];
      dims[d] = (indices[i].stop-indices[i].start-1)/indices[i].stride+1;
      if (dims[d] < 0)
	dims[d] = 0;
      nitems *= dims[d];
      d++;
    }
  }
  if (error) {
    PyErr_SetString(PyExc_IndexError, "illegal index");
    if (dims != NULL)
      free(dims);
    if (indices != NULL)
      free(indices);
    return NULL;
  }
  array = (PyArrayObject *)PyArray_FromDims(d, dims, self->type);
  if (array != NULL && nitems > 0) {
    if (self->nd == 0) {
      long zero = 0;
      int ret;
      Py_BEGIN_ALLOW_THREADS;
      acquire_netCDF_lock();
      ret = ncvarget1(self->file->id, self->id, &zero, array->data);
      release_netCDF_lock();
      Py_END_ALLOW_THREADS;
      if (ret == -1) {
	netcdf_seterror();
	Py_DECREF(array);
	array = NULL;
      }
    }
    else {
      long *start;
      long *count;
      long *stride;
      start = (long *)malloc(self->nd*sizeof(long));
      count = (long *)malloc(self->nd*sizeof(long));
      stride = (long *)malloc(self->nd*sizeof(long));
      if (start != NULL && count != NULL && stride != NULL) {
	int ret;
	for (i = 0; i < self->nd; i++) {
	  start[i] = indices[i].start;
	  stride[i] = indices[i].stride;
	  count[i] = (indices[i].stop-indices[i].start-1)/indices[i].stride+1;
	}
	Py_BEGIN_ALLOW_THREADS;
	acquire_netCDF_lock();
	ret = ncvargetg(self->file->id, self->id, start, count, stride, NULL,
			array->data);
	release_netCDF_lock();
	Py_END_ALLOW_THREADS;
	if (ret == -1) {
	  netcdf_seterror();
	  Py_DECREF(array);
	  array = NULL;
	}
      }
      if (start != NULL)
	free(start);
      if (count != NULL)
	free(count);
      if (stride != NULL)
	free(stride);
    }
  }
  free(dims);
  free(indices);
  return array;
}

static PyStringObject *
PyNetCDFVariable_ReadAsString(PyNetCDFVariableObject *self)
{
  if (self->type != PyArray_CHAR || self->nd != 1) {
    PyErr_SetString(PyExc_IOError, "netcdf: not a string variable");
    return NULL;
  }
  if (check_if_open(self->file, -1)) {
    long zero = 0;
    int ret;
    char *temp;
    PyObject *string;
    define_mode(self->file, 0);
    temp = (char *)malloc((self->dimensions[0]+1)*sizeof(char));
    if (temp == NULL)
      return (PyStringObject *)PyErr_NoMemory();
    Py_BEGIN_ALLOW_THREADS;
    acquire_netCDF_lock();
    ret = ncvarget(self->file->id, self->id, &zero, self->dimensions, temp);
    release_netCDF_lock();
    Py_END_ALLOW_THREADS;
    if (ret == -1) {
      netcdf_seterror();
      string = NULL;
    }
    else {
      temp[self->dimensions[0]] = '\0';
      string = PyString_FromString(temp);
    }
    free(temp);
    return (PyStringObject *)string;
  }
  else
    return NULL;
}

static int
PyNetCDFVariable_WriteArray(PyNetCDFVariableObject *self,
			    PyNetCDFIndex *indices, PyObject *value)
{
  int *dims;
  PyArrayObject *array;
  int i, j, d;
  int nitems;
  int error = 0;
  int ret = 0;
  d = 0;
  nitems = 1;
  if (!check_if_open(self->file, 1)) {
    free(indices);
    return -1;
  }
  if (self->nd == 0)
    dims = NULL;
  else {
    dims = (int *)malloc(self->nd*sizeof(int));
    if (dims == NULL) {
      free(indices);
      PyErr_SetString(PyExc_MemoryError, "out of memory");
      return -1;
    }
  }
  define_mode(self->file, 0);
  for (i = 0; i < self->nd; i++) {
    error = error || (indices[i].stride <= 0);
    if (indices[i].start < 0)
      indices[i].start += self->dimensions[i];
    if (indices[i].start < 0)
      indices[i].start = 0;
    if (indices[i].stop < 0)
      indices[i].stop += self->dimensions[i];
    if (indices[i].stop < 0)
      indices[i].stop = 0;
    if (i > 0 || !self->unlimited) {
      if (indices[i].start > self->dimensions[i])
	indices[i].start = self->dimensions[i];
      if (indices[i].stop > self->dimensions[i])
	indices[i].stop = self->dimensions[i];
    }
    if (indices[i].item == 0) {
      dims[d] = (indices[i].stop-indices[i].start-1)/indices[i].stride+1;
      if (dims[d] < 0)
	dims[d] = 0;
      nitems *= dims[d];
      d++;
    }
    else
      indices[i].stop = indices[i].start + 1;
  }
  if (error) {
    PyErr_SetString(PyExc_IndexError, "illegal index");
    free(dims);
    free(indices);
    return -1;
  }
  array = (PyArrayObject *)PyArray_ContiguousFromObject(value, self->type,
							0, d);
  if (array != NULL) {
    if (self->nd == 0) {
      long zero = 0;
      int ret;
      Py_BEGIN_ALLOW_THREADS;
      acquire_netCDF_lock();
      ret = ncvarput1(self->file->id, self->id, &zero, array->data);
      release_netCDF_lock();
      Py_END_ALLOW_THREADS;
      if (ret == -1) {
	netcdf_seterror();
	ret = -1;
      }
    }
    else {
      long *start;
      long *count, *count1;
      long *stride;
      long *current;
      char *loop;
      long repeat = 1;
      int elim, lastloop;
      start = (long *)malloc(self->nd*sizeof(long));
      count = (long *)malloc(self->nd*sizeof(long));
      count1 = (long *)malloc(self->nd*sizeof(long));
      stride = (long *)malloc(self->nd*sizeof(long));
      current = (long *)malloc(self->nd*sizeof(long));
      loop = (char *)malloc(self->nd*sizeof(char));
      if (start != NULL && count != NULL && count1 != NULL
	  && stride != NULL && current != NULL && loop != NULL) {
	elim = 0;
	for (i = 0; i < self->nd; i++) {
	  start[i] = indices[i].start;
	  stride[i] = indices[i].stride;
	  count[i] = (indices[i].stop-indices[i].start-1)/indices[i].stride+1;
	  count1[i] = count[i];
	  current[i] = 0;
	  loop[i] = 0;
	  if (indices[i].item)
	    elim++;
	}
	for (i = array->nd-1, j = self->nd-1; i >= 0 && j >= 0; i--, j--) {
	  while (j >= 0 && indices[j].item)
	    j--;
	  if (j < 0) {
	    ret = -1;
	    break;
	  }
	  if (array->dimensions[i] != count[j])
	    ret = -1;
	}
	if (i == -1) {
	  lastloop = -1;
	  while (j >= 0) {
	    loop[j] = !indices[j].item;
	    if (loop[j]) {
	      if (lastloop < 0)
		lastloop = j;
	      repeat *= count[j];
	      count1[j] = 1;
	    }
	    j--;
	  }
	}
	else
	  ret = -1;
	if (ret == -1)
	  PyErr_SetString(PyExc_ValueError, "shapes are not aligned");
	while (repeat-- && ret == 0) {
	  int ret;
	  Py_BEGIN_ALLOW_THREADS;
	  acquire_netCDF_lock();
	  ret = ncvarputg(self->file->id, self->id, start, count1, stride,
			  NULL, array->data);
	  release_netCDF_lock();
	  Py_END_ALLOW_THREADS;
	  if (ret == -1) {
	    netcdf_seterror();
	    ret = -1;
	  }
	  if (lastloop >= 0) {
	    for (i = lastloop; i >= 0; i--) {
	      while (!loop[i] && i >= 0)
		i--;
	      if (i >= 0) {
		start[i] += stride[i];
		if (++current[i] != count[i])
		  break;
		start[i] -= count[i]*stride[i];
		current[i] = 0;
	      }
	    }
	  }
	}
	if (self->unlimited) {
	  Py_BEGIN_ALLOW_THREADS;
	  acquire_netCDF_lock();
	  ncdiminq(self->file->id, self->dimids[0], NULL,
		   &self->dimensions[0]);
	  release_netCDF_lock();
	  Py_END_ALLOW_THREADS;
	}
      }
      Py_DECREF(array);
      if (start != NULL)
	free(start);
      if (count != NULL)
	free(count);
      if (count1 != NULL)
	free(count1);
      if (stride != NULL)
	free(stride);
      if (current != NULL)
	free(current);
    }
    free(dims);
    free(indices);
    return ret;
  }
  else
    return -1;
}

static int
PyNetCDFVariable_WriteString(PyNetCDFVariableObject *self,
			     PyStringObject *value)
{
  long zero = 0;
  long len;
  if (self->type != PyArray_CHAR || self->nd != 1) {
    PyErr_SetString(PyExc_IOError, "netcdf: not a string variable");
    return -1;
  }
  len = PyString_Size((PyObject *)value);
  if (len > self->dimensions[0]) {
    PyErr_SetString(PyExc_ValueError, "string too long");
    return -1;
  }
  if (self->dimensions[0] > len)
    len++;
  if (check_if_open(self->file, 1)) {
    int ret;
    define_mode(self->file, 0);
    Py_BEGIN_ALLOW_THREADS;
    acquire_netCDF_lock();
    ret = ncvarput(self->file->id, self->id, &zero, &len,
		   PyString_AsString((PyObject *)value));
    release_netCDF_lock();
    Py_END_ALLOW_THREADS;
    if (ret == -1) {
      netcdf_seterror();
      return -1;
    }
    return 0;
  }
  else
    return -1;
}

static PyObject *
PyNetCDFVariableObject_item(PyNetCDFVariableObject *self, int i)
{
  PyNetCDFIndex *indices;
  if (self->nd == 0) {
    PyErr_SetString(PyExc_TypeError, "Not a sequence");
    return NULL;
  }
  indices = PyNetCDFVariable_Indices(self);
  if (indices != NULL) {
    indices[0].start = i;
    indices[0].stop = i+1;
    indices[0].item = 1;
    return PyArray_Return(PyNetCDFVariable_ReadAsArray(self, indices));
  }
  return NULL;
}

static PyObject *
PyNetCDFVariableObject_slice(PyNetCDFVariableObject *self, int low, int high)
{
  PyNetCDFIndex *indices;
  if (self->nd == 0) {
    PyErr_SetString(PyExc_TypeError, "Not a sequence");
    return NULL;
  }
  indices = PyNetCDFVariable_Indices(self);
  if (indices != NULL) {
    indices[0].start = low;
    indices[0].stop = high;
    return PyArray_Return(PyNetCDFVariable_ReadAsArray(self, indices));
  }
  return NULL;
}

static PyObject *
PyNetCDFVariableObject_subscript(PyNetCDFVariableObject *self, PyObject *index)
{
  PyNetCDFIndex *indices;
  if (PyInt_Check(index)) {
    int i = PyInt_AsLong(index);
    return PyNetCDFVariableObject_item(self, i);
  }
  if (self->nd == 0) {
    PyErr_SetString(PyExc_TypeError, "Not a sequence");
    return NULL;
  }
  indices = PyNetCDFVariable_Indices(self);
  if (indices != NULL) {
    if (PySlice_Check(index)) {
      PySlice_GetIndices((PySliceObject *)index, self->dimensions[0],
			 &indices->start, &indices->stop, &indices->stride);
      return PyArray_Return(PyNetCDFVariable_ReadAsArray(self, indices));
    }
    if (PyTuple_Check(index)) {
      int ni = PyTuple_Size(index);
      if (ni <= self->nd) {
	int i, d;
	d = 0;
	for (i = 0; i < ni; i++) {
	  PyObject *subscript = PyTuple_GetItem(index, i);
	  if (PyInt_Check(subscript)) {
	    int n = PyInt_AsLong(subscript);
	    indices[d].start = n;
	    indices[d].stop = n+1;
	    indices[d].item = 1;
	    d++;
	  }
	  else if (PySlice_Check(subscript)) {
	    PySlice_GetIndices((PySliceObject *)subscript, self->dimensions[d],
			       &indices[d].start, &indices[d].stop,
			       &indices[d].stride);
	    d++;
	  }
	  else if (subscript == Py_Ellipsis) {
	    d = self->nd - ni + i + 1;
	  }
	  else {
	    PyErr_SetString(PyExc_TypeError, "illegal subscript type");
	    free(indices);
	    return NULL;
	  }
	}
	return PyArray_Return(PyNetCDFVariable_ReadAsArray(self, indices));
      }
      else {
	PyErr_SetString(PyExc_IndexError, "too many subscripts");
	free(indices);
	return NULL;
      }
    }
    PyErr_SetString(PyExc_TypeError, "illegal subscript type");
    free(indices);
  }
  return NULL;
}

static int
PyNetCDFVariableObject_ass_item(PyNetCDFVariableObject *self,
				int i, PyObject *value)
{
  PyNetCDFIndex *indices;
  if (value == NULL) {
    PyErr_SetString(PyExc_ValueError, "Can't delete elements.");
    return -1;
  }
  if (self->nd == 0) {
    PyErr_SetString(PyExc_TypeError, "Not a sequence");
    return -1;
  }
  indices = PyNetCDFVariable_Indices(self);
  if (indices != NULL) {
    indices[0].start = i;
    indices[0].stop = i+1;
    indices[0].item = 1;
    return PyNetCDFVariable_WriteArray(self, indices, value);
  }
  return -1;
}

static int
PyNetCDFVariableObject_ass_slice(PyNetCDFVariableObject *self,
				 int low, int high, PyObject *value)
{
  PyNetCDFIndex *indices;
  if (value == NULL) {
    PyErr_SetString(PyExc_ValueError, "Can't delete elements.");
    return -1;
  }
  if (self->nd == 0) {
    PyErr_SetString(PyExc_TypeError, "Not a sequence");
    return -1;
  }
  indices = PyNetCDFVariable_Indices(self);
  if (indices != NULL) {
    indices[0].start = low;
    indices[0].stop = high;
    return PyNetCDFVariable_WriteArray(self, indices, value);
  }
  return -1;
}

static int
PyNetCDFVariableObject_ass_subscript(PyNetCDFVariableObject *self,
				     PyObject *index, PyObject *value)
{
  PyNetCDFIndex *indices;
  if (PyInt_Check(index)) {
    int i = PyInt_AsLong(index);
    return PyNetCDFVariableObject_ass_item(self, i, value);
  }
  if (value == NULL) {
    PyErr_SetString(PyExc_ValueError, "Can't delete elements.");
    return -1;
  }
  if (self->nd == 0) {
    PyErr_SetString(PyExc_TypeError, "Not a sequence");
    return -1;
  }
  indices = PyNetCDFVariable_Indices(self);
  if (indices != NULL) {
    if (PySlice_Check(index)) {
      PySlice_GetIndices((PySliceObject *)index, self->dimensions[0],
			 &indices->start, &indices->stop, &indices->stride);
      return PyNetCDFVariable_WriteArray(self, indices, value);
    }
    if (PyTuple_Check(index)) {
      int ni = PyTuple_Size(index);
      if (ni <= self->nd) {
	int i, d;
	d = 0;
	for (i = 0; i < ni; i++) {
	  PyObject *subscript = PyTuple_GetItem(index, i);
	  if (PyInt_Check(subscript)) {
	    int n = PyInt_AsLong(subscript);
	    indices[d].start = n;
	    indices[d].stop = n+1;
	    indices[d].item = 1;
	    d++;
	  }
	  else if (PySlice_Check(subscript)) {
	    PySlice_GetIndices((PySliceObject *)subscript, self->dimensions[d],
			       &indices[d].start, &indices[d].stop,
			       &indices[d].stride);
	    d++;
	  }
	  else if (subscript == Py_Ellipsis) {
	    d = self->nd - ni + i + 1;
	  }
	  else {
	    PyErr_SetString(PyExc_TypeError, "illegal subscript type");
	    free(indices);
	    return -1;
	  }
	}
	return PyNetCDFVariable_WriteArray(self, indices, value);
      }
      else {
	PyErr_SetString(PyExc_IndexError, "too many subscripts");
	free(indices);
	return -1;
      }
    }
    PyErr_SetString(PyExc_TypeError, "illegal subscript type");
    free(indices);
  }
  return -1;
}

/* Type definition */

static PyObject *
PyNetCDFVariableObject_error1(PyNetCDFVariableObject *self,
			      PyNetCDFVariableObject *other)
{
  PyErr_SetString(PyExc_TypeError, "can't add netCDF variables");
  return NULL;
}

static PyObject *
PyNetCDFVariableObject_error2(PyNetCDFVariableObject *self, int n)
{
  PyErr_SetString(PyExc_TypeError, "can't multiply netCDF variables");
  return NULL;
}


static PySequenceMethods PyNetCDFVariableObject_as_sequence = {
  (inquiry)PyNetCDFVariableObject_length,		/*sq_length*/
  (binaryfunc)PyNetCDFVariableObject_error1,       /*nb_add*/
  (intargfunc)PyNetCDFVariableObject_error2,       /*nb_multiply*/
  (intargfunc)PyNetCDFVariableObject_item,		/*sq_item*/
  (intintargfunc)PyNetCDFVariableObject_slice,	/*sq_slice*/
  (intobjargproc)PyNetCDFVariableObject_ass_item,	/*sq_ass_item*/
  (intintobjargproc)PyNetCDFVariableObject_ass_slice,   /*sq_ass_slice*/
};

static PyMappingMethods PyNetCDFVariableObject_as_mapping = {
  (inquiry)PyNetCDFVariableObject_length,		/*mp_length*/
  (binaryfunc)PyNetCDFVariableObject_subscript,	      /*mp_subscript*/
  (objobjargproc)PyNetCDFVariableObject_ass_subscript,   /*mp_ass_subscript*/
};

statichere PyTypeObject PyNetCDFVariable_Type = {
  PyObject_HEAD_INIT(NULL)
  0,		     /*ob_size*/
  "NetCDFVariable",  /*tp_name*/
  sizeof(PyNetCDFVariableObject),	     /*tp_basicsize*/
  0,		     /*tp_itemsize*/
  /* methods */
  (destructor)PyNetCDFVariableObject_dealloc, /*tp_dealloc*/
  0,			/*tp_print*/
  (getattrfunc)PyNetCDFVariable_GetAttribute, /*tp_getattr*/
  (setattrfunc)PyNetCDFVariable_SetAttribute, /*tp_setattr*/
  0,			/*tp_compare*/
  0,			/*tp_repr*/
  0,			/*tp_as_number*/
  &PyNetCDFVariableObject_as_sequence,	/*tp_as_sequence*/
  &PyNetCDFVariableObject_as_mapping,	/*tp_as_mapping*/
  0,			/*tp_hash*/
};


/* Creator for NetCDFFile objects */

static PyObject *
NetCDFFile(PyObject *self, PyObject *args)
{
  int netcdfid;
  PyNetCDFFileObject *file;

  if (!PyArg_ParseTuple(args, "i", &netcdfid))
    return NULL;
  file = PyNetCDFFile_from_id(netcdfid);
  if (file == NULL) {
    netcdf_seterror();
    return NULL;
  }
  return (PyObject *)file;
}

/* static PyObject *
NetCDFFile(PyObject *self, PyObject *args)
{
  char *filename;
  char *mode = NULL;
  char *history = NULL;
  PyNetCDFFileObject *file;

  if (!PyArg_ParseTuple(args, "s|ss:NetCDFFile", &filename, &mode, &history))
    return NULL;
  if (mode == NULL)
    mode = "r";
  file = PyNetCDFFile_Open(filename, mode);
  if (file == NULL) {
    netcdf_seterror();
    return NULL;
  }
  if (history != NULL)
    PyNetCDFFile_AddHistoryLine(file, history);
  return (PyObject *)file;
}
*/

/* static char netcdf_doc[] =
   "NetCDFFile(filename, mode, history_line)\n\nopens a netCDF file"; */

static char netcdf_doc[] =
"NetCDFFile(id)\n\nprovides access to an already open netcdf file";

/* Table of functions defined in the module */

static PyMethodDef netcdf_methods[] = {
  {"NetCDFFile",	NetCDFFile, 1, netcdf_doc},
  {NULL,		NULL}		/* sentinel */
};

/* Module initialization */

void
initpy_minc_netcdf(void)
{
  PyObject *m, *d;
  static void *PyNetCDF_API[PyNetCDF_API_pointers];

  /* Initialize netcdf variables */
  ncopts = 0;

  /* Initialize type object headers */
  PyNetCDFFile_Type.ob_type = &PyType_Type;
  PyNetCDFVariable_Type.ob_type = &PyType_Type;

  /* Create netCDF lock */
#ifdef WITH_THREAD
  netCDF_lock = PyThread_allocate_lock();
#endif

  /* Create the module and add the functions */
  m = Py_InitModule("py_minc_netcdf", netcdf_methods);

  /* Import the array module */
#ifdef import_array
  import_array(); 
#endif

  /* Initialize C API pointer array and store in module */
  PyNetCDF_API[PyNetCDFFile_Type_NUM] = (void *)&PyNetCDFFile_Type;
  PyNetCDF_API[PyNetCDFVariable_Type_NUM] = (void *)&PyNetCDFVariable_Type;
  PyNetCDF_API[PyNetCDFFile_Open_NUM] = (void *)&PyNetCDFFile_Open;
  PyNetCDF_API[PyNetCDFFile_from_id_NUM] = (void *)&PyNetCDFFile_from_id;
  PyNetCDF_API[PyNetCDFFile_Close_NUM] = (void *)&PyNetCDFFile_Close;
  PyNetCDF_API[PyNetCDFFile_Sync_NUM] = (void *)&PyNetCDFFile_Sync;
  PyNetCDF_API[PyNetCDFFile_CreateDimension_NUM] =
    (void *)&PyNetCDFFile_CreateDimension;
  PyNetCDF_API[PyNetCDFFile_CreateVariable_NUM] =
    (void *)&PyNetCDFFile_CreateVariable;
  PyNetCDF_API[PyNetCDFFile_GetVariable_NUM] =
    (void *)&PyNetCDFFile_GetVariable;
  PyNetCDF_API[PyNetCDFVariable_GetRank_NUM] =
    (void *)&PyNetCDFVariable_GetRank;
  PyNetCDF_API[PyNetCDFVariable_GetShape_NUM] =
    (void *)&PyNetCDFVariable_GetShape;
  PyNetCDF_API[PyNetCDFVariable_Indices_NUM] =
    (void *)&PyNetCDFVariable_Indices;
  PyNetCDF_API[PyNetCDFVariable_ReadAsArray_NUM] =
    (void *)&PyNetCDFVariable_ReadAsArray;
  PyNetCDF_API[PyNetCDFVariable_ReadAsString_NUM] =
    (void *)&PyNetCDFVariable_ReadAsString;
  PyNetCDF_API[PyNetCDFVariable_WriteArray_NUM] =
    (void *)&PyNetCDFVariable_WriteArray;
  PyNetCDF_API[PyNetCDFVariable_WriteString_NUM] =
    (void *)&PyNetCDFVariable_WriteString;
  PyNetCDF_API[PyNetCDFFile_GetAttribute_NUM] =
    (void *)&PyNetCDFFile_GetAttribute;
  PyNetCDF_API[PyNetCDFFile_SetAttribute_NUM] =
    (void *)&PyNetCDFFile_SetAttribute;
  PyNetCDF_API[PyNetCDFFile_SetAttributeString_NUM] =
    (void *)&PyNetCDFFile_SetAttributeString;
  PyNetCDF_API[PyNetCDFVariable_GetAttribute_NUM] =
    (void *)&PyNetCDFVariable_GetAttribute;
  PyNetCDF_API[PyNetCDFVariable_SetAttribute_NUM] =
    (void *)&PyNetCDFVariable_SetAttribute;
  PyNetCDF_API[PyNetCDFVariable_SetAttributeString_NUM] =
    (void *)&PyNetCDFVariable_SetAttributeString;
  PyNetCDF_API[PyNetCDFFile_AddHistoryLine_NUM] =
    (void *)&PyNetCDFFile_AddHistoryLine;
  d = PyModule_GetDict(m);
  PyDict_SetItemString(d, "_C_API",
		       PyCObject_FromVoidPtr((void *)PyNetCDF_API, NULL));
  
  /* Check for errors */
  if (PyErr_Occurred())
    Py_FatalError("can't initialize module py_minc_netcdf");
}
