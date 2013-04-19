/* A Python interface to the volume_io library 

   by John G. Sled  

   Created: March 21, 2001
   Last revised:

   Copyright 2002, John G. Sled
*/

#include <Python.h>
#include <volume_io.h>
#include <minc.h>

/* Method Table */
static PyMethodDef VolumeIO_constants_methods[] = {
  { NULL,    NULL}
};


enum module_constant_type { PyInt, PyString };

typedef struct {
  char *name;
  enum module_constant_type type;
  void *value;
} module_constant;


#define MODULE_CONSTANT(val, type)  {#val,  type, (void *) val},

static module_constant ModuleConstants[] = {
#ifdef MI_ORIGINAL_TYPE
  MODULE_CONSTANT(NC_NAT, PyInt)
  MODULE_CONSTANT(NC_INT, PyInt)
  MODULE_CONSTANT(MI_ORIGINAL_TYPE, PyInt)
#else
  {"MI_ORIGINAL_TYPE", PyInt, NC_UNSPECIFIED},
#endif
  MODULE_CONSTANT(NC_BYTE, PyInt)
  MODULE_CONSTANT(NC_SHORT, PyInt)
  MODULE_CONSTANT(NC_FLOAT, PyInt)
  MODULE_CONSTANT(NC_DOUBLE, PyInt)
  MODULE_CONSTANT(NC_UNSPECIFIED, PyInt)

/* standard dimension names */
  MODULE_CONSTANT(MIxspace, PyString)
  MODULE_CONSTANT(MIyspace, PyString)
  MODULE_CONSTANT(MIzspace, PyString)
  MODULE_CONSTANT(MItime, PyString)
  MODULE_CONSTANT(MItfrequency, PyString)
  MODULE_CONSTANT(MIxfrequency, PyString)
  MODULE_CONSTANT(MIyfrequency, PyString)
  MODULE_CONSTANT(MIzfrequency, PyString)

/*    MIspacetype values */
  MODULE_CONSTANT(MI_NATIVE, PyString)
  MODULE_CONSTANT(MI_TALAIRACH, PyString)
  MODULE_CONSTANT(MI_CALLOSAL, PyString)

/* The patient variable and its attributes */
  MODULE_CONSTANT(MIpatient, PyString)
  MODULE_CONSTANT(MIfull_name, PyString)
  MODULE_CONSTANT(MIother_names, PyString)
  MODULE_CONSTANT(MIidentification, PyString)
  MODULE_CONSTANT(MIother_ids, PyString)
  MODULE_CONSTANT(MIbirthdate, PyString)
  MODULE_CONSTANT(MIsex, PyString)
  MODULE_CONSTANT(MIage, PyString)
  MODULE_CONSTANT(MIweight, PyString)
  MODULE_CONSTANT(MIsize, PyString)
  MODULE_CONSTANT(MIaddress, PyString)
  MODULE_CONSTANT(MIinsurance_id, PyString)

/* Patient attribute constants */
  MODULE_CONSTANT(MI_MALE, PyString)
  MODULE_CONSTANT(MI_FEMALE, PyString)
  MODULE_CONSTANT(MI_OTHER, PyString)

/* The study variable and its attributes */
  MODULE_CONSTANT(MIstudy, PyString)
  MODULE_CONSTANT(MIstart_time, PyString)
  MODULE_CONSTANT(MIstart_year, PyString)
  MODULE_CONSTANT(MIstart_month, PyString)
  MODULE_CONSTANT(MIstart_day, PyString)
  MODULE_CONSTANT(MIstart_hour, PyString)
  MODULE_CONSTANT(MIstart_minute, PyString)
  MODULE_CONSTANT(MIstart_seconds, PyString)
  MODULE_CONSTANT(MImodality, PyString)
  MODULE_CONSTANT(MImanufacturer, PyString)
  MODULE_CONSTANT(MIdevice_model, PyString)
  MODULE_CONSTANT(MIinstitution, PyString)
  MODULE_CONSTANT(MIdepartment, PyString)
  MODULE_CONSTANT(MIstation_id, PyString)
  MODULE_CONSTANT(MIreferring_physician, PyString)
  MODULE_CONSTANT(MIattending_physician, PyString)
  MODULE_CONSTANT(MIradiologist, PyString)
  MODULE_CONSTANT(MIoperator, PyString)
  MODULE_CONSTANT(MIadmitting_diagnosis, PyString)
  MODULE_CONSTANT(MIprocedure, PyString)
  MODULE_CONSTANT(MIstudy_id, PyString)

/* Study attribute constants */
  MODULE_CONSTANT(MI_PET, PyString)
  MODULE_CONSTANT(MI_SPECT, PyString)
  MODULE_CONSTANT(MI_GAMMA, PyString)
  MODULE_CONSTANT(MI_MRI, PyString)
  MODULE_CONSTANT(MI_MRS, PyString)
  MODULE_CONSTANT(MI_MRA, PyString)
  MODULE_CONSTANT(MI_CT, PyString)
  MODULE_CONSTANT(MI_DSA, PyString)
  MODULE_CONSTANT(MI_DR, PyString)
  MODULE_CONSTANT(MI_LABEL, PyString)

/* The acquisition variable and its attributes */
  MODULE_CONSTANT(MIacquisition, PyString)
  MODULE_CONSTANT(MIprotocol, PyString)
  MODULE_CONSTANT(MIscanning_sequence, PyString)
  MODULE_CONSTANT(MIrepetition_time, PyString)
  MODULE_CONSTANT(MIecho_time, PyString)
  MODULE_CONSTANT(MIinversion_time, PyString)
  MODULE_CONSTANT(MInum_averages, PyString)
  MODULE_CONSTANT(MIimaging_frequency, PyString)
  MODULE_CONSTANT(MIimaged_nucleus, PyString)
  MODULE_CONSTANT(MIradionuclide, PyString)
  MODULE_CONSTANT(MIcontrast_agent, PyString)
  MODULE_CONSTANT(MIradionuclide_halflife, PyString)
  MODULE_CONSTANT(MItracer, PyString)
  MODULE_CONSTANT(MIinjection_time, PyString)
  MODULE_CONSTANT(MIinjection_year, PyString)
  MODULE_CONSTANT(MIinjection_month, PyString)
  MODULE_CONSTANT(MIinjection_day, PyString)
  MODULE_CONSTANT(MIinjection_hour, PyString)
  MODULE_CONSTANT(MIinjection_minute, PyString)
  MODULE_CONSTANT(MIinjection_seconds, PyString)
  MODULE_CONSTANT(MIinjection_length, PyString)
  MODULE_CONSTANT(MIinjection_dose, PyString)
  MODULE_CONSTANT(MIdose_units, PyString)
  MODULE_CONSTANT(MIinjection_volume, PyString)
  MODULE_CONSTANT(MIinjection_route, PyString)

  /* various header attributes which are hidden from the user */
  MODULE_CONSTANT(MIimage, PyString)    
  MODULE_CONSTANT(MIimagemax, PyString) 
  MODULE_CONSTANT(MIimagemin, PyString) 
  MODULE_CONSTANT(MIcomplete, PyString) 
  MODULE_CONSTANT(MIvartype, PyString) 
  MODULE_CONSTANT(MI_GROUP, PyString)   
  MODULE_CONSTANT(MIparent, PyString)   
  MODULE_CONSTANT(MIchildren, PyString) 
  MODULE_CONSTANT(MIrootvariable, PyString)


  /* Volume IO caching specific constants */
  MODULE_CONSTANT(SLICE_ACCESS, PyInt)
  MODULE_CONSTANT(RANDOM_VOLUME_ACCESS, PyInt)

  {NULL, 0}};


/* Module initialization function */
void
initVolumeIO_constants(void)
{
  PyObject *m, *d;
  module_constant *v;

  m = Py_InitModule("VolumeIO_constants", VolumeIO_constants_methods);

  /* add some useful constants to the module */
  d = PyModule_GetDict(m);
  for(v = ModuleConstants; v->name != NULL; v++) {
    switch (v->type) {
    case PyInt:
      PyDict_SetItemString(d, v->name, PyInt_FromLong ((long)v->value)); 
      break;
    case PyString:
      PyDict_SetItemString(d, v->name, PyString_FromString ((char*)v->value)); 
      break;
    default:
      printf("Warning constant %s has unknown type type: %d\n", 
	     v->name, v->type);
    }
  }
}
