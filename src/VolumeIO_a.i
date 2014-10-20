%module VolumeIO_a
%include typemaps.i
%{
#include <stdbool.h>
#include <volume_io.h>
#define MAX_DIM  5  
%}

%include carrays.i

%typemap(python,in) VIO_Volume volume {
  if (!PyCObject_Check($input)) { 
    PyErr_SetString(PyExc_TypeError, "VIO_Volume argument should be of type CObject");
    return NULL; 
  }
  $1 = (VIO_Volume) PyCObject_AsVoidPtr($input);
}

%typemap(python,out) VIO_Volume {
  if($1 == NULL) {   /* return NULL pointers as None */
    Py_INCREF(Py_None);
    $result = Py_None;
  }
  else {                  /* otherwise return CObject */
    $result = PyCObject_FromVoidPtr((void *) $1, NULL);
  }
}

/* create a MAX_DIM element array from a Python n-tuple */
%typemap(python,in) VIO_Real[MAX_DIM](double temp[MAX_DIM]) {
  /* temp[MAX_DIM] becomes a local variable */
  int i;
  for(i = 0; i < MAX_DIM; i++) {
    temp[i] = 0.0;  /* default values */
  }
  if (PyTuple_Check($input)) {
    if (!PyArg_ParseTuple($input, "d|dddd" ,temp,temp+1,temp+2,temp+3,temp+4)) {
      PyErr_SetString(PyExc_TypeError,"Expected a tuple of VIO_Real values");
      return NULL; 
    } 
    $1 = &temp[0];
  } else {
    PyErr_SetString(PyExc_TypeError,"expected a tuple");
    return NULL;
  }
}

/* create a MAX_DIM element array from a Python n-tuple */
%typemap(python,in) int[MAX_DIM](int temp[MAX_DIM]) {
  /* temp[MAX_DIM] becomes a local variable */
  int i;
  for(i = 0; i < MAX_DIM; i++) {
    temp[i] = 0;  /* default values */
  }
  if (PyTuple_Check($input)) {
    if (!PyArg_ParseTuple($input, "i|iiii" ,temp,temp+1,temp+2,temp+3,temp+4)) {
      PyErr_SetString(PyExc_TypeError,"Expected a tuple of integer values");
      return NULL; 
    } 
    $1 = &temp[0];
  } else {
    PyErr_SetString(PyExc_TypeError,"Expected a tuple");
    return NULL;
  }
}

/* create a MAX_DIM element array from a Python n-tuple */
%typemap(python,in) char *[MAX_DIM](char *temp[MAX_DIM]) {
  /* temp[MAX_DIM] becomes a local variable */
  int i;
  for(i = 0; i < MAX_DIM; i++) {
    temp[i] = NULL;  /* default values */
  }
  if (PyTuple_Check($input)) {
    if (!PyArg_ParseTuple($input, "s|ssss" ,
			  temp,temp+1,temp+2,temp+3,temp+4)) {
      PyErr_SetString(PyExc_TypeError,"Expected a tuple of string values");
      return NULL; 
    } 
    $1 = &temp[0];
  } else {
    PyErr_SetString(PyExc_TypeError,"Expected a tuple");
    return NULL;
  }
}


%apply double *OUTPUT { VIO_Real *voxel_min, VIO_Real *voxel_max, VIO_Real *max_value, 
	VIO_Real *min_value, VIO_Real *x_world, VIO_Real *y_world, VIO_Real *z_world,
	VIO_Real *voxel1,  VIO_Real *voxel2, VIO_Real *voxel3,
	VIO_Real *x_transformed, VIO_Real *y_transformed, VIO_Real *z_transformed};

%apply bool *OUTPUT { VIO_BOOL *signed_flag };

typedef double VIO_Real;
typedef char * VIO_STR;
typedef int   nc_type;
typedef bool VIO_BOOL;
typedef void *VIO_Volume;
typedef int VIO_Cache_block_size_hints;
typedef int VIO_Status;

%array_functions(VIO_Real, RealArray)
%array_functions(int, intArray)


%inline %{  // allocation functions for VIO_General_transform type
  VIO_General_transform *new_General_transform(void) {
    return (VIO_General_transform *) malloc(sizeof(VIO_General_transform));
  }
  
  void free_General_transform(VIO_General_transform *transform) {
    free(transform);
  }
%}


VIO_Real  convert_voxel_to_value(
    VIO_Volume   volume,
    VIO_Real     voxel );

VIO_Real  convert_value_to_voxel(
    VIO_Volume   volume,
    VIO_Real     value );

VIO_Real  get_volume_voxel_value(
    VIO_Volume   volume,
    int      v0,
    int      v1 = 0,
    int      v2 = 0,
    int      v3 = 0,
    int      v4 = 0 );

VIO_Real  get_volume_real_value(
    VIO_Volume   volume,
    int      v0,
    int      v1 = 0,
    int      v2 = 0,
    int      v3 = 0,
    int      v4 = 0 );

void  set_volume_voxel_value(
    VIO_Volume   volume,
    int      v0,
    int      v1,
    int      v2,
    int      v3,
    int      v4,
    VIO_Real     voxel );

void  set_volume_real_value(
    VIO_Volume   volume,
    int      v0,
    int      v1,
    int      v2,
    int      v3,
    int      v4,
    VIO_Real     value );

void  set_voxel_to_world_transform(
    VIO_Volume             volume,
    VIO_General_transform  *transform );

VIO_General_transform  *get_voxel_to_world_transform(
    VIO_Volume   volume );

unsigned int  get_volume_total_n_voxels(
    VIO_Volume    volume );

int  get_volume_n_dimensions(
    VIO_Volume   volume );

VIO_STR  get_volume_space_type(
    VIO_Volume   volume );

void  set_volume_space_type(
    VIO_Volume   volume,
    VIO_STR   name );

VIO_Real  get_volume_voxel_min(
    VIO_Volume   volume );

VIO_Real  get_volume_voxel_max(
    VIO_Volume   volume );

void  set_volume_voxel_range(
    VIO_Volume   volume,
    VIO_Real     voxel_min,
    VIO_Real     voxel_max );

VIO_Real  get_volume_real_min(
    VIO_Volume     volume );

VIO_Real  get_volume_real_max(
    VIO_Volume     volume );

void  set_volume_real_range(
    VIO_Volume   volume,
    VIO_Real     real_min,
    VIO_Real     real_max );

void  set_volume_type(
    VIO_Volume       volume,
    nc_type      nc_data_type,
    VIO_BOOL      signed_flag,
    VIO_Real         voxel_min,
    VIO_Real         voxel_max );

void  get_volume_sizes(
    VIO_Volume   volume,
    int      sizes[] );

void  set_volume_sizes(
    VIO_Volume       volume,
    int          sizes[MAX_DIM] );

void  get_volume_separations(
    VIO_Volume   volume,
    VIO_Real      separations[] );

void  set_volume_separations(
    VIO_Volume   volume,
    VIO_Real     separations[MAX_DIM] );

void  set_volume_starts(
    VIO_Volume  volume,
    VIO_Real    starts[MAX_DIM] );

void  get_volume_starts(
    VIO_Volume  volume,
    VIO_Real    starts[] );

void  set_volume_direction_unit_cosine(
    VIO_Volume   volume,
    int      axis,
    VIO_Real     dir[MAX_DIM] );

void  set_volume_direction_cosine(
    VIO_Volume   volume,
    int      axis,
    VIO_Real     dir[MAX_DIM] );

void  get_volume_direction_cosine(
    VIO_Volume   volume,
    int      axis,
    VIO_Real     dir[] );

void  set_volume_translation(
    VIO_Volume  volume,
    VIO_Real    voxel[MAX_DIM],
    VIO_Real    world_space_voxel_maps_to[MAX_DIM] );

void  get_volume_voxel_range(
    VIO_Volume     volume,
    VIO_Real       *voxel_min,
    VIO_Real       *voxel_max );

void  get_volume_real_range(
    VIO_Volume     volume,
    VIO_Real       *min_value,
    VIO_Real       *max_value );

void  set_n_bytes_cache_threshold(
    int  threshold );

int  get_n_bytes_cache_threshold( void );

VIO_Volume  copy_volume(
    VIO_Volume   volume );

void  set_volume_cache_size(
    VIO_Volume    volume,
    int       max_memory_bytes );

void  set_default_max_bytes_in_cache(
    int   max_bytes );

int  get_default_max_bytes_in_cache( void );


VIO_BOOL  volume_is_cached(
    VIO_Volume  volume );

void  set_default_cache_block_sizes(
    int                      block_sizes[MAX_DIM] );

void  set_volume_cache_block_sizes(
    VIO_Volume    volume,
    int       block_sizes[MAX_DIM] );

void  set_cache_block_sizes_hint(
    VIO_Cache_block_size_hints  hint );

VIO_Volume   create_volume(
    int         n_dimensions,
    VIO_STR      dimension_names[MAX_DIM],
    nc_type     nc_data_type,
    VIO_BOOL     signed_flag,
    VIO_Real        voxel_min,
    VIO_Real        voxel_max );


void  alloc_volume_data(
    VIO_Volume   volume );

void  convert_voxel_to_world(
    VIO_Volume   volume,
    VIO_Real     voxel[MAX_DIM],
    VIO_Real     *x_world,
    VIO_Real     *y_world,
    VIO_Real     *z_world );

void  convert_world_to_voxel(
    VIO_Volume   volume,
    VIO_Real     x_world,
    VIO_Real     y_world,
    VIO_Real     z_world,
    VIO_Real     voxel[]  );

nc_type  get_volume_nc_data_type(
    VIO_Volume       volume,
    VIO_BOOL      *signed_flag );

/*  ------------------------------------------ */
/*  Untested / disfunctional interfaces        */

void  set_volume_interpolation_tolerance(
    VIO_Real   tolerance );

int   evaluate_volume(
    VIO_Volume         volume,
    VIO_Real           voxel[],
    VIO_BOOL        interpolating_dimensions[],
    int            degrees_continuity,
    VIO_BOOL        use_linear_at_edge,
    VIO_Real           outside_value,
    VIO_Real           values[],
    VIO_Real           **first_deriv,
    VIO_Real           ***second_deriv );

void   evaluate_volume_in_world(
    VIO_Volume         volume,
    VIO_Real           x,
    VIO_Real           y,
    VIO_Real           z,
    int            degrees_continuity,
    VIO_BOOL        use_linear_at_edge,
    VIO_Real           outside_value,
    VIO_Real           values[],
    VIO_Real           deriv_x[],
    VIO_Real           deriv_y[],
    VIO_Real           deriv_z[],
    VIO_Real           deriv_xx[],
    VIO_Real           deriv_xy[],
    VIO_Real           deriv_xz[],
    VIO_Real           deriv_yy[],
    VIO_Real           deriv_yz[],
    VIO_Real           deriv_zz[] );

void  convert_voxels_to_values(
    VIO_Volume   volume,
    int      n_voxels,
    VIO_Real     voxels[],
    VIO_Real     values[] );

void  get_volume_value_hyperslab(
    VIO_Volume   volume,
    int      v0,
    int      v1,
    int      v2,
    int      v3,
    int      v4,
    int      n0,
    int      n1,
    int      n2,
    int      n3,
    int      n4,
    VIO_Real     values[] );

void  get_volume_value_hyperslab_5d(
    VIO_Volume   volume,
    int      v0,
    int      v1,
    int      v2,
    int      v3,
    int      v4,
    int      n0,
    int      n1,
    int      n2,
    int      n3,
    int      n4,
    VIO_Real     values[] );

void  get_volume_value_hyperslab_4d(
    VIO_Volume   volume,
    int      v0,
    int      v1,
    int      v2,
    int      v3,
    int      n0,
    int      n1,
    int      n2,
    int      n3,
    VIO_Real     values[] );

void  get_volume_value_hyperslab_3d(
    VIO_Volume   volume,
    int      v0,
    int      v1,
    int      v2,
    int      n0,
    int      n1,
    int      n2,
    VIO_Real     values[] );

void  get_volume_value_hyperslab_2d(
    VIO_Volume   volume,
    int      v0,
    int      v1,
    int      n0,
    int      n1,
    VIO_Real     values[] );

void  get_volume_value_hyperslab_1d(
    VIO_Volume   volume,
    int      v0,
    int      n0,
    VIO_Real     values[] );

void  get_voxel_values_5d(
    VIO_Data_types  data_type,
    void        *void_ptr,
    int         steps[],
    int         counts[],
    VIO_Real        values[] );

void  get_voxel_values_4d(
    VIO_Data_types  data_type,
    void        *void_ptr,
    int         steps[],
    int         counts[],
    VIO_Real        values[] );

void  get_voxel_values_3d(
    VIO_Data_types  data_type,
    void        *void_ptr,
    int         steps[],
    int         counts[],
    VIO_Real        values[] );

void  get_voxel_values_2d(
    VIO_Data_types  data_type,
    void        *void_ptr,
    int         steps[],
    int         counts[],
    VIO_Real        values[] );

void  get_voxel_values_1d(
    VIO_Data_types  data_type,
    void        *void_ptr,
    int         step0,
    int         n0,
    VIO_Real        values[] );

void  get_volume_voxel_hyperslab_5d(
    VIO_Volume   volume,
    int      v0,
    int      v1,
    int      v2,
    int      v3,
    int      v4,
    int      n0,
    int      n1,
    int      n2,
    int      n3,
    int      n4,
    VIO_Real     values[] );

void  get_volume_voxel_hyperslab_4d(
    VIO_Volume   volume,
    int      v0,
    int      v1,
    int      v2,
    int      v3,
    int      n0,
    int      n1,
    int      n2,
    int      n3,
    VIO_Real     values[] );

void  get_volume_voxel_hyperslab_3d(
    VIO_Volume   volume,
    int      v0,
    int      v1,
    int      v2,
    int      n0,
    int      n1,
    int      n2,
    VIO_Real     values[] );

void  get_volume_voxel_hyperslab_2d(
    VIO_Volume   volume,
    int      v0,
    int      v1,
    int      n0,
    int      n1,
    VIO_Real     values[] );

void  get_volume_voxel_hyperslab_1d(
    VIO_Volume   volume,
    int      v0,
    int      n0,
    VIO_Real     values[] );

void  get_volume_voxel_hyperslab(
    VIO_Volume   volume,
    int      v0,
    int      v1,
    int      v2,
    int      v3,
    int      v4,
    int      n0,
    int      n1,
    int      n2,
    int      n3,
    int      n4,
    VIO_Real     voxels[] );

VIO_Status  initialize_free_format_input(
    VIO_STR               filename,
    VIO_Volume               volume,
    volume_input_struct  *volume_input );

void  delete_free_format_input(
    volume_input_struct   *volume_input );

VIO_BOOL  input_more_free_format_file(
    VIO_Volume                volume,
    volume_input_struct   *volume_input,
    VIO_Real                  *fraction_done );

int   get_minc_file_n_dimensions(
    VIO_STR   filename );

Minc_file  initialize_minc_input_from_minc_id(
    int                  minc_id,
    VIO_Volume               volume,
    minc_input_options   *options );

Minc_file  initialize_minc_input(
    VIO_STR               filename,
    VIO_Volume               volume,
    minc_input_options   *options );

int  get_n_input_volumes(
    Minc_file  file );

VIO_Status  close_minc_input(
    Minc_file   file );

VIO_Status  input_minc_hyperslab(
    Minc_file        file,
    VIO_Data_types       data_type,
    int              n_array_dims,
    int              array_sizes[],
    void             *array_data_ptr,
    int              to_array[],
    int              start[],
    int              count[] );

VIO_BOOL  input_more_minc_file(
    Minc_file   file,
    VIO_Real        *fraction_done );

VIO_BOOL  advance_input_volume(
    Minc_file   file );

void  reset_input_volume(
    Minc_file   file );

int  get_minc_file_id(
    Minc_file  file );

void  set_default_minc_input_options(
    minc_input_options  *options );

void  set_minc_input_promote_invalid_to_zero_flag(
    minc_input_options  *options,
    VIO_BOOL             flag );

void  set_minc_input_promote_invalid_to_min_flag(
    minc_input_options  *options,
    VIO_BOOL             flag );

void  set_minc_input_vector_to_scalar_flag(
    minc_input_options  *options,
    VIO_BOOL             flag );

void  set_minc_input_vector_to_colour_flag(
    minc_input_options  *options,
    VIO_BOOL             flag );

void  set_minc_input_colour_dimension_size(
    minc_input_options  *options,
    int                 size );

void  set_minc_input_colour_max_dimension_size(
    minc_input_options  *options,
    int                 size );

void  set_minc_input_colour_indices(
    minc_input_options  *options,
    int                 indices[4] );

VIO_Status  start_volume_input(
    VIO_STR               filename,
    int                  n_dimensions,
    VIO_STR               dim_names[],
    nc_type              volume_nc_data_type,
    VIO_BOOL              volume_signed_flag,
    VIO_Real                 volume_voxel_min,
    VIO_Real                 volume_voxel_max,
    VIO_BOOL              create_volume_flag,
    VIO_Volume               *volume,
    minc_input_options   *options,
    volume_input_struct  *input_info );

void  delete_volume_input(
    volume_input_struct   *input_info );

VIO_BOOL  input_more_of_volume(
    VIO_Volume                volume,
    volume_input_struct   *input_info,
    VIO_Real                  *fraction_done );

void  cancel_volume_input(
    VIO_Volume                volume,
    volume_input_struct   *input_info );

VIO_Status  input_volume(
    VIO_STR               filename,
    int                  n_dimensions,
    VIO_STR               dim_names[],
    nc_type              volume_nc_data_type,
    VIO_BOOL              volume_signed_flag,
    VIO_Real                 volume_voxel_min,
    VIO_Real                 volume_voxel_max,
    VIO_BOOL              create_volume_flag,
    VIO_Volume               *volume,
    minc_input_options   *options );

Minc_file   get_volume_input_minc_file(
    volume_input_struct   *volume_input );

 void   create_empty_multidim_array(
    VIO_multidim_array  *array,
    int             n_dimensions,
    VIO_Data_types      data_type );

VIO_Data_types  get_multidim_data_type(
    VIO_multidim_array       *array );

void  set_multidim_data_type(
    VIO_multidim_array       *array,
    VIO_Data_types           data_type );

int  get_type_size(
    VIO_Data_types   type );

void  get_type_range(
    VIO_Data_types   type,
    VIO_Real         *min_value,
    VIO_Real         *max_value );

void  set_multidim_sizes(
    VIO_multidim_array   *array,
    int              sizes[] );

void  get_multidim_sizes(
    VIO_multidim_array   *array,
    int              sizes[] );

VIO_BOOL  multidim_array_is_alloced(
    VIO_multidim_array   *array );

void  alloc_multidim_array(
    VIO_multidim_array   *array );

 void   create_multidim_array(
    VIO_multidim_array  *array,
    int             n_dimensions,
    int             sizes[],
    VIO_Data_types      data_type );

void  delete_multidim_array(
    VIO_multidim_array   *array );

int  get_multidim_n_dimensions(
    VIO_multidim_array   *array );

void  copy_multidim_data_reordered(
    int                 type_size,
    void                *void_dest_ptr,
    int                 n_dest_dims,
    int                 dest_sizes[],
    void                *void_src_ptr,
    int                 n_src_dims,
    int                 src_sizes[],
    int                 counts[],
    int                 to_dest_index[],
    VIO_BOOL             use_src_order );

void  copy_multidim_reordered(
    VIO_multidim_array      *dest,
    int                 dest_ind[],
    VIO_multidim_array      *src,
    int                 src_ind[],
    int                 counts[],
    int                 to_dest_index[] );

Minc_file  initialize_minc_output(
    VIO_STR                 filename,
    int                    n_dimensions,
    VIO_STR                 dim_names[],
    int                    sizes[],
    nc_type                file_nc_data_type,
    VIO_BOOL                file_signed_flag,
    VIO_Real                   file_voxel_min,
    VIO_Real                   file_voxel_max,
    VIO_General_transform      *voxel_to_world_transform,
    VIO_Volume                 volume_to_attach,
    minc_output_options    *options );

VIO_Status  copy_auxiliary_data_from_minc_file(
    Minc_file   file,
    VIO_STR      filename,
    VIO_STR      history_string );

VIO_Status  copy_auxiliary_data_from_open_minc_file(
    Minc_file   file,
    int         src_cdfid,
    VIO_STR      history_string );

VIO_Status  add_minc_history(
    Minc_file   file,
    VIO_STR      history_string );

VIO_Status  set_minc_output_random_order(
    Minc_file   file );

VIO_Status  output_minc_hyperslab(
    Minc_file           file,
    VIO_Data_types          data_type,
    int                 n_array_dims,
    int                 array_sizes[],
    void                *array_data_ptr,
    int                 to_array[],
    int                 file_start[],
    int                 file_count[] );

VIO_Status  output_volume_to_minc_file_position(
    Minc_file   file,
    VIO_Volume      volume,
    int         volume_count[],
    long        file_start[] );

VIO_Status  output_minc_volume(
    Minc_file   file );

VIO_Status  close_minc_output(
    Minc_file   file );

void  set_default_minc_output_options(
    minc_output_options  *options           );

void  copy_minc_output_options(
    minc_output_options  *src,
    minc_output_options  *dest );

void  delete_minc_output_options(
    minc_output_options  *options           );

void  set_minc_output_dimensions_order(
    minc_output_options  *options,
    int                  n_dimensions,
    VIO_STR               dimension_names[] );

void  set_minc_output_real_range(
    minc_output_options  *options,
    VIO_Real                 real_min,
    VIO_Real                 real_max );

void  set_minc_output_use_volume_starts_and_steps_flag(
    minc_output_options  *options,
    VIO_BOOL              flag );

VIO_Status   get_file_dimension_names(
    VIO_STR   filename,
    int      *n_dims,
    VIO_STR   *dim_names[] );

VIO_STR  *create_output_dim_names(
    VIO_Volume                volume,
    VIO_STR                original_filename,
    minc_output_options   *options,
    int                   file_sizes[] );

VIO_Status   copy_volume_auxiliary_and_history(
    Minc_file   minc_file,
    VIO_STR      filename,
    VIO_STR      original_filename,
    VIO_STR      history );

VIO_Status  output_modified_volume(
    VIO_STR                filename,
    nc_type               file_nc_data_type,
    VIO_BOOL               file_signed_flag,
    VIO_Real                  file_voxel_min,
    VIO_Real                  file_voxel_max,
    VIO_Volume                volume,
    VIO_STR                original_filename,
    VIO_STR                history,
    minc_output_options   *options );

VIO_Status  output_volume(
    VIO_STR                filename,
    nc_type               file_nc_data_type,
    VIO_BOOL               file_signed_flag,
    VIO_Real                  file_voxel_min,
    VIO_Real                  file_voxel_max,
    VIO_Volume                volume,
    VIO_STR                history,
    minc_output_options   *options );

void  convert_values_to_voxels(
    VIO_Volume   volume,
    int      n_voxels,
    VIO_Real     values[],
    VIO_Real     voxels[] );

void  set_volume_value_hyperslab(
    VIO_Volume   volume,
    int      v0,
    int      v1,
    int      v2,
    int      v3,
    int      v4,
    int      n0,
    int      n1,
    int      n2,
    int      n3,
    int      n4,
    VIO_Real     values[] );

void  set_volume_value_hyperslab_5d(
    VIO_Volume   volume,
    int      v0,
    int      v1,
    int      v2,
    int      v3,
    int      v4,
    int      n0,
    int      n1,
    int      n2,
    int      n3,
    int      n4,
    VIO_Real     values[] );

void  set_volume_value_hyperslab_4d(
    VIO_Volume   volume,
    int      v0,
    int      v1,
    int      v2,
    int      v3,
    int      n0,
    int      n1,
    int      n2,
    int      n3,
    VIO_Real     values[] );

void  set_volume_value_hyperslab_3d(
    VIO_Volume   volume,
    int      v0,
    int      v1,
    int      v2,
    int      n0,
    int      n1,
    int      n2,
    VIO_Real     values[] );

void  set_volume_value_hyperslab_2d(
    VIO_Volume   volume,
    int      v0,
    int      v1,
    int      n0,
    int      n1,
    VIO_Real     values[] );

void  set_volume_value_hyperslab_1d(
    VIO_Volume   volume,
    int      v0,
    int      n0,
    VIO_Real     values[] );

void  set_volume_voxel_hyperslab_5d(
    VIO_Volume   volume,
    int      v0,
    int      v1,
    int      v2,
    int      v3,
    int      v4,
    int      n0,
    int      n1,
    int      n2,
    int      n3,
    int      n4,
    VIO_Real     values[] );

void  set_volume_voxel_hyperslab_4d(
    VIO_Volume   volume,
    int      v0,
    int      v1,
    int      v2,
    int      v3,
    int      n0,
    int      n1,
    int      n2,
    int      n3,
    VIO_Real     values[] );

void  set_volume_voxel_hyperslab_3d(
    VIO_Volume   volume,
    int      v0,
    int      v1,
    int      v2,
    int      n0,
    int      n1,
    int      n2,
    VIO_Real     values[] );

void  set_volume_voxel_hyperslab_2d(
    VIO_Volume   volume,
    int      v0,
    int      v1,
    int      n0,
    int      n1,
    VIO_Real     values[] );

void  set_volume_voxel_hyperslab_1d(
    VIO_Volume   volume,
    int      v0,
    int      n0,
    VIO_Real     values[] );

void  set_volume_voxel_hyperslab(
    VIO_Volume   volume,
    int      v0,
    int      v1,
    int      v2,
    int      v3,
    int      v4,
    int      n0,
    int      n1,
    int      n2,
    int      n3,
    int      n4,
    VIO_Real     voxels[] );

void  initialize_volume_cache(
    VIO_volume_cache_struct   *cache,
    VIO_Volume                volume );

VIO_BOOL  volume_cache_is_alloced(
    VIO_volume_cache_struct   *cache );

void  flush_volume_cache(
    VIO_Volume                volume );

void  delete_volume_cache(
    VIO_volume_cache_struct   *cache,
    VIO_Volume                volume );

void  set_cache_output_volume_parameters(
    VIO_Volume                      volume,
    VIO_STR                      filename,
    nc_type                     file_nc_data_type,
    VIO_BOOL                     file_signed_flag,
    VIO_Real                        file_voxel_min,
    VIO_Real                        file_voxel_max,
    VIO_STR                      original_filename,
    VIO_STR                      history,
    minc_output_options         *options )
;

void  open_cache_volume_input_file(
    VIO_volume_cache_struct   *cache,
    VIO_Volume                volume,
    VIO_STR                filename,
    minc_input_options    *options );

void  cache_volume_range_has_changed(
    VIO_Volume   volume );

void  set_cache_volume_file_offset(
    VIO_volume_cache_struct   *cache,
    VIO_Volume                volume,
    long                  file_offset[] );

VIO_Real  get_cached_volume_voxel(
    VIO_Volume   volume,
    int      x,
    int      y,
    int      z,
    int      t,
    int      v );

void  set_cached_volume_voxel(
    VIO_Volume   volume,
    int      x,
    int      y,
    int      z,
    int      t,
    int      v,
    VIO_Real     value );

VIO_BOOL  cached_volume_has_been_modified(
    VIO_volume_cache_struct  *cache );

void   set_volume_cache_debugging(
    VIO_Volume   volume,
    int      output_every );

VIO_STR  *get_default_dim_names(
    int    n_dimensions );

VIO_BOOL  convert_dim_name_to_spatial_axis(
    VIO_STR  name,
    int     *axis );

VIO_Data_types  get_volume_data_type(
    VIO_Volume       volume );

void  set_rgb_volume_flag(
    VIO_Volume   volume,
    VIO_BOOL  flag );

VIO_BOOL  is_an_rgb_volume(
    VIO_Volume   volume );

VIO_BOOL  volume_is_alloced(
    VIO_Volume   volume );

void  free_volume_data(
    VIO_Volume   volume );

void  delete_volume(
    VIO_Volume   volume );

void  compute_world_transform(
    int                 spatial_axes[VIO_N_DIMENSIONS],
    VIO_Real                separations[],
    VIO_Real                direction_cosines[][VIO_N_DIMENSIONS],
    VIO_Real                starts[],
    VIO_General_transform   *world_transform );

void  convert_transform_to_starts_and_steps(
    VIO_General_transform  *transform,
    int                n_volume_dimensions,
    VIO_Real               step_signs[],
    int                spatial_axes[],
    VIO_Real               starts[],
    VIO_Real               steps[],
    VIO_Real               dir_cosines[][VIO_N_DIMENSIONS] );

VIO_STR  *get_volume_dimension_names(
    VIO_Volume   volume );

void  delete_dimension_names(
    VIO_Volume   volume,
    VIO_STR   dimension_names[] );

void  reorder_voxel_to_xyz(
    VIO_Volume   volume,
    VIO_Real     voxel[],
    VIO_Real     xyz[] );

void  reorder_xyz_to_voxel(
    VIO_Volume   volume,
    VIO_Real     xyz[],
    VIO_Real      voxel[] );

void  convert_3D_voxel_to_world(
    VIO_Volume   volume,
    VIO_Real     voxel1,
    VIO_Real     voxel2,
    VIO_Real     voxel3,
    VIO_Real     *x_world,
    VIO_Real     *y_world,
    VIO_Real     *z_world );

void  convert_voxel_normal_vector_to_world(
    VIO_Volume          volume,
    VIO_Real            voxel_vector[],
    VIO_Real            *x_world,
    VIO_Real            *y_world,
    VIO_Real            *z_world );

void  convert_voxel_vector_to_world(
    VIO_Volume          volume,
    VIO_Real            voxel_vector[],
    VIO_Real            *x_world,
    VIO_Real            *y_world,
    VIO_Real            *z_world );

void  convert_world_vector_to_voxel(
    VIO_Volume          volume,
    VIO_Real            x_world,
    VIO_Real            y_world,
    VIO_Real            z_world,
    VIO_Real            voxel_vector[]  );

void  convert_3D_world_to_voxel(
    VIO_Volume   volume,
    VIO_Real     x_world,
    VIO_Real     y_world,
    VIO_Real     z_world,
    VIO_Real     *voxel1,
    VIO_Real     *voxel2,
    VIO_Real     *voxel3 );

VIO_Volume   copy_volume_definition_no_alloc(
    VIO_Volume   volume,
    nc_type  nc_data_type,
    VIO_BOOL  signed_flag,
    VIO_Real     voxel_min,
    VIO_Real     voxel_max );

VIO_Volume   copy_volume_definition(
    VIO_Volume   volume,
    nc_type  nc_data_type,
    VIO_BOOL  signed_flag,
    VIO_Real     voxel_min,
    VIO_Real     voxel_max );

void  grid_transform_point(
    VIO_General_transform   *transform,
    VIO_Real                x,
    VIO_Real                y,
    VIO_Real                z,
    VIO_Real                *x_transformed,
    VIO_Real                *y_transformed,
    VIO_Real                *z_transformed );

void  grid_inverse_transform_point(
    VIO_General_transform   *transform,
    VIO_Real                x,
    VIO_Real                y,
    VIO_Real                z,
    VIO_Real                *x_transformed,
    VIO_Real                *y_transformed,
    VIO_Real                *z_transformed );

VIO_Status  mni_get_nonwhite_character(
    FILE   *file,
    char   *ch );

VIO_Status  mni_skip_expected_character(
    FILE   *file,
    char   expected_ch );

VIO_Status  mni_input_line(
    FILE     *file,
    VIO_STR   *string );

VIO_Status  mni_input_string(
    FILE     *file,
    VIO_STR   *string,
    char     termination_char1,
    char     termination_char2 );

VIO_Status  mni_input_keyword_and_equal_sign(
    FILE         *file,
    const char   keyword[],
    VIO_BOOL      print_error_message );

VIO_Status  mni_input_real(
    FILE    *file,
    VIO_Real    *d );

VIO_Status  mni_input_reals(
    FILE    *file,
    int     *n,
    VIO_Real    *reals[] );

VIO_Status  mni_input_int(
    FILE    *file,
    int     *i );

void  output_comments(
    FILE     *file,
    VIO_STR   comments );

VIO_STR  get_default_tag_file_suffix( void );

VIO_Status  initialize_tag_file_output(
    FILE      *file,
    VIO_STR    comments,
    int       n_volumes );

VIO_Status  output_one_tag(
    FILE      *file,
    int       n_volumes,
    VIO_Real      tag_volume1[],
    VIO_Real      tag_volume2[],
    VIO_Real      *weight,
    int       *structure_id,
    int       *patient_id,
    VIO_STR    label );

void  terminate_tag_file_output(
    FILE    *file );

VIO_Status  output_tag_points(
    FILE      *file,
    VIO_STR    comments,
    int       n_volumes,
    int       n_tag_points,
    VIO_Real      **tags_volume1,
    VIO_Real      **tags_volume2,
    VIO_Real      weights[],
    int       structure_ids[],
    int       patient_ids[],
    VIO_STR    *labels );

void  free_tag_points(
    int       n_volumes,
    int       n_tag_points,
    VIO_Real      **tags_volume1,
    VIO_Real      **tags_volume2,
    VIO_Real      weights[],
    int       structure_ids[],
    int       patient_ids[],
    char      **labels );

VIO_Status  initialize_tag_file_input(
    FILE      *file,
    int       *n_volumes_ptr );

VIO_Status  output_tag_file(
    VIO_STR    filename,
    VIO_STR    comments,
    int       n_volumes,
    int       n_tag_points,
    VIO_Real      **tags_volume1,
    VIO_Real      **tags_volume2,
    VIO_Real      weights[],
    int       structure_ids[],
    int       patient_ids[],
    VIO_STR    labels[] );

VIO_Status  input_tag_file(
    VIO_STR    filename,
    int       *n_volumes,
    int       *n_tag_points,
    VIO_Real      ***tags_volume1,
    VIO_Real      ***tags_volume2,
    VIO_Real      **weights,
    int       **structure_ids,
    int       **patient_ids,
    VIO_STR    *labels[] );

VIO_BOOL  input_one_tag(
    FILE      *file,
    int       n_volumes,
    VIO_Real      tag_volume1[],
    VIO_Real      tag_volume2[],
    VIO_Real      *weight,
    int       *structure_id,
    int       *patient_id,
    VIO_STR    *label,
    VIO_Status    *status );

VIO_Status  input_tag_points(
    FILE      *file,
    int       *n_volumes_ptr,
    int       *n_tag_points,
    VIO_Real      ***tags_volume1,
    VIO_Real      ***tags_volume2,
    VIO_Real      **weights,
    int       **structure_ids,
    int       **patient_ids,
    VIO_STR    *labels[] );

void  evaluate_thin_plate_spline(
    int     n_dims,
    int     n_values,
    int     n_points,
    VIO_Real    **points,
    VIO_Real    **weights,
    VIO_Real    pos[],
    VIO_Real    values[],
    VIO_Real    **derivs );

void  thin_plate_spline_transform(
    int     n_dims,
    int     n_points,
    VIO_Real    **points,
    VIO_Real    **weights,
    VIO_Real    x,
    VIO_Real    y,
    VIO_Real    z,
    VIO_Real    *x_transformed,
    VIO_Real    *y_transformed,
    VIO_Real    *z_transformed );

void  thin_plate_spline_inverse_transform(
    int     n_dims,
    int     n_points,
    VIO_Real    **points,
    VIO_Real    **weights,
    VIO_Real    x,
    VIO_Real    y,
    VIO_Real    z,
    VIO_Real    *x_transformed,
    VIO_Real    *y_transformed,
    VIO_Real    *z_transformed );

VIO_Real  thin_plate_spline_U(
    VIO_Real   pos[],
    VIO_Real   landmark[],
    int    n_dims );

VIO_STR  get_default_transform_file_suffix( void );

VIO_Status  output_transform(
    FILE                *file,
    VIO_STR              filename,
    int                 *volume_count_ptr,
    VIO_STR              comments,
    VIO_General_transform   *transform );

VIO_Status  input_transform(
    FILE                *file,
    VIO_STR              filename,
    VIO_General_transform   *transform );

VIO_Status  output_transform_file(
    VIO_STR              filename,
    VIO_STR              comments,
    VIO_General_transform   *transform );

VIO_Status  input_transform_file(
    VIO_STR              filename,
    VIO_General_transform   *transform );

void  create_linear_transform(
    VIO_General_transform   *transform,
    VIO_Transform           *linear_transform );

void  create_thin_plate_transform_real(
    VIO_General_transform    *transform,
    int                  n_dimensions,
    int                  n_points,
    VIO_Real                 **points,
    VIO_Real                 **displacements );

void  create_thin_plate_transform(
    VIO_General_transform    *transform,
    int                  n_dimensions,
    int                  n_points,
    float                **points,
    float                **displacements );

void create_grid_transform(
    VIO_General_transform 	*transform,
    VIO_Volume 			displacement_volume,
    VIO_STR 			displacement_volume_file );

void create_grid_transform_no_copy(
    VIO_General_transform 	*transform,
    VIO_Volume 			displacement_volume,
    VIO_STR 			displacement_volume_file );

void  create_user_transform(
    VIO_General_transform         *transform,
    void                      *user_data,
    size_t                    size_user_data,
    VIO_User_transform_function   transform_function,
    VIO_User_transform_function   inverse_transform_function );

VIO_Transform_types  get_transform_type(
    VIO_General_transform   *transform );

int  get_n_concated_transforms(
    VIO_General_transform   *transform );

VIO_General_transform  *get_nth_general_transform(
    VIO_General_transform   *transform,
    int                 n );

VIO_Transform  *get_linear_transform_ptr(
    VIO_General_transform   *transform );

VIO_Transform  *get_inverse_linear_transform_ptr(
    VIO_General_transform   *transform );

void  general_transform_point(
    VIO_General_transform   *transform,
    VIO_Real                x,
    VIO_Real                y,
    VIO_Real                z,
    VIO_Real                *x_transformed,
    VIO_Real                *y_transformed,
    VIO_Real                *z_transformed );

void  general_inverse_transform_point(
    VIO_General_transform   *transform,
    VIO_Real                x,
    VIO_Real                y,
    VIO_Real                z,
    VIO_Real                *x_transformed,
    VIO_Real                *y_transformed,
    VIO_Real                *z_transformed );

void  copy_general_transform(
    VIO_General_transform   *transform,
    VIO_General_transform   *copy );

void  invert_general_transform(
    VIO_General_transform   *transform );

void  create_inverse_general_transform(
    VIO_General_transform   *transform,
    VIO_General_transform   *inverse );

void  concat_general_transforms(
    VIO_General_transform   *first,
    VIO_General_transform   *second,
    VIO_General_transform   *result );

void  delete_general_transform(
    VIO_General_transform   *transform );

VIO_Colour  make_rgba_Colour(
    int    r,
    int    g,
    int    b,
    int    a );

int  get_Colour_r(
    VIO_Colour   colour );

int  get_Colour_g(
    VIO_Colour   colour );

int  get_Colour_b(
    VIO_Colour   colour );

int  get_Colour_a(
    VIO_Colour   colour );

VIO_Colour  make_Colour(
    int   r,
    int   g,
    int   b );

VIO_Real  get_Colour_r_0_1(
    VIO_Colour   colour );

VIO_Real  get_Colour_g_0_1(
    VIO_Colour   colour );

VIO_Real  get_Colour_b_0_1(
    VIO_Colour   colour );

VIO_Real  get_Colour_a_0_1(
    VIO_Colour   colour );

VIO_Colour  make_Colour_0_1(
    VIO_Real   r,
    VIO_Real   g,
    VIO_Real   b );

VIO_Colour  make_rgba_Colour_0_1(
    VIO_Real   r,
    VIO_Real   g,
    VIO_Real   b,
    VIO_Real   a );

VIO_BOOL  scaled_maximal_pivoting_gaussian_elimination(
    int   n,
    int   row[],
    VIO_Real  **a,
    int   n_values,
    VIO_Real  **solution );

VIO_BOOL  solve_linear_system(
    int   n,
    VIO_Real  **coefs,
    VIO_Real  values[],
    VIO_Real  solution[] );

VIO_BOOL  invert_square_matrix(
    int   n,
    VIO_Real  **matrix,
    VIO_Real  **inverse );

/* VIO_BOOL  newton_root_find(
    int    n_dimensions,
    void   (*function) ( void *, VIO_Real [],  VIO_Real [], VIO_Real ** ),
    void   *function_data,
    VIO_Real   initial_guess[],
    VIO_Real   desired_values[],
    VIO_Real   solution[],
    VIO_Real   function_tolerance,
    VIO_Real   delta_tolerance,
    int    max_iterations );
*/

void  create_orthogonal_vector(
    VIO_Vector  *v,
    VIO_Vector  *ortho );

void  create_two_orthogonal_vectors(
    VIO_Vector   *v,
    VIO_Vector   *v1,
    VIO_Vector   *v2 );

VIO_BOOL   compute_transform_inverse(
    VIO_Transform  *transform,
    VIO_Transform  *inverse );

void  get_linear_spline_coefs(
    VIO_Real  **coefs );

void  get_quadratic_spline_coefs(
    VIO_Real  **coefs );

void  get_cubic_spline_coefs(
    VIO_Real  **coefs );

VIO_Real  cubic_interpolate(
    VIO_Real   u,
    VIO_Real   v0,
    VIO_Real   v1,
    VIO_Real   v2,
    VIO_Real   v3 );

void  evaluate_univariate_interpolating_spline(
    VIO_Real    u,
    int     degree,
    VIO_Real    coefs[],
    int     n_derivs,
    VIO_Real    derivs[] );

void  evaluate_bivariate_interpolating_spline(
    VIO_Real    u,
    VIO_Real    v,
    int     degree,
    VIO_Real    coefs[],
    int     n_derivs,
    VIO_Real    derivs[] );

void  evaluate_trivariate_interpolating_spline(
    VIO_Real    u,
    VIO_Real    v,
    VIO_Real    w,
    int     degree,
    VIO_Real    coefs[],
    int     n_derivs,
    VIO_Real    derivs[] );

void  evaluate_interpolating_spline(
    int     n_dims,
    VIO_Real    parameters[],
    int     degree,
    int     n_values,
    VIO_Real    coefs[],
    int     n_derivs,
    VIO_Real    derivs[] );

void  spline_tensor_product(
    int     n_dims,
    VIO_Real    positions[],
    int     degrees[],
    VIO_Real    *bases[],
    int     n_values,
    VIO_Real    coefs[],
    int     n_derivs[],
    VIO_Real    results[] );

void  make_identity_transform( VIO_Transform   *transform );

VIO_BOOL  close_to_identity(
    VIO_Transform   *transform );

void  get_transform_origin(
    VIO_Transform   *transform,
    VIO_Point       *origin );

void  set_transform_origin(
    VIO_Transform   *transform,
    VIO_Point       *origin );

void  get_transform_origin_real(
    VIO_Transform   *transform,
    VIO_Real        origin[] );

void  get_transform_x_axis(
    VIO_Transform   *transform,
    VIO_Vector      *x_axis );

void  get_transform_x_axis_real(
    VIO_Transform   *transform,
    VIO_Real        x_axis[] );

void  set_transform_x_axis(
    VIO_Transform   *transform,
    VIO_Vector      *x_axis );

void  set_transform_x_axis_real(
    VIO_Transform   *transform,
    VIO_Real        x_axis[] );

void  get_transform_y_axis(
    VIO_Transform   *transform,
    VIO_Vector      *y_axis );

void  get_transform_y_axis_real(
    VIO_Transform   *transform,
    VIO_Real        y_axis[] );

void  set_transform_y_axis(
    VIO_Transform   *transform,
    VIO_Vector      *y_axis );

void  set_transform_y_axis_real(
    VIO_Transform   *transform,
    VIO_Real        y_axis[] );

void  get_transform_z_axis(
    VIO_Transform   *transform,
    VIO_Vector      *z_axis );

void  get_transform_z_axis_real(
    VIO_Transform   *transform,
    VIO_Real        z_axis[] );

void  set_transform_z_axis(
    VIO_Transform   *transform,
    VIO_Vector      *z_axis );

void  set_transform_z_axis_real(
    VIO_Transform   *transform,
    VIO_Real        z_axis[] );

void   make_change_to_bases_transform(
    VIO_Point      *origin,
    VIO_Vector     *x_axis,
    VIO_Vector     *y_axis,
    VIO_Vector     *z_axis,
    VIO_Transform  *transform );

void   make_change_from_bases_transform(
    VIO_Point      *origin,
    VIO_Vector     *x_axis,
    VIO_Vector     *y_axis,
    VIO_Vector     *z_axis,
    VIO_Transform  *transform );

void   concat_transforms(
    VIO_Transform   *result,
    VIO_Transform   *t1,
    VIO_Transform   *t2 );

void  transform_point(
    VIO_Transform  *transform,
    VIO_Real       x,
    VIO_Real       y,
    VIO_Real       z,
    VIO_Real       *x_trans,
    VIO_Real       *y_trans,
    VIO_Real       *z_trans );

void  transform_vector(
    VIO_Transform  *transform,
    VIO_Real       x,
    VIO_Real       y,
    VIO_Real       z,
    VIO_Real       *x_trans,
    VIO_Real       *y_trans,
    VIO_Real       *z_trans );

/*
void  *alloc_memory_in_bytes(
    size_t       n_bytes
    _ALLOC_SOURCE_LINE_ARG_DEF );

void  *alloc_memory_1d(
    size_t       n_elements,
    size_t       type_size
    _ALLOC_SOURCE_LINE_ARG_DEF );

void  *alloc_memory_2d(
    size_t       n1,
    size_t       n2,
    size_t       type_size
    _ALLOC_SOURCE_LINE_ARG_DEF );

void  *alloc_memory_3d(
    size_t       n1,
    size_t       n2,
    size_t       n3,
    size_t       type_size
    _ALLOC_SOURCE_LINE_ARG_DEF );

void  *alloc_memory_4d(
    size_t       n1,
    size_t       n2,
    size_t       n3,
    size_t       n4,
    size_t       type_size
    _ALLOC_SOURCE_LINE_ARG_DEF );

void  *alloc_memory_5d(
    size_t       n1,
    size_t       n2,
    size_t       n3,
    size_t       n4,
    size_t       n5,
    size_t       type_size
    _ALLOC_SOURCE_LINE_ARG_DEF );

void  realloc_memory(
    void      **ptr,
    size_t    n_elements,
    size_t    type_size
    _ALLOC_SOURCE_LINE_ARG_DEF );

void  free_memory_1d(
    void   **ptr
    _ALLOC_SOURCE_LINE_ARG_DEF );

void  free_memory_2d(
    void   ***ptr
    _ALLOC_SOURCE_LINE_ARG_DEF );

void  free_memory_3d(
    void   ****ptr
    _ALLOC_SOURCE_LINE_ARG_DEF );

void  free_memory_4d(
    void   *****ptr
    _ALLOC_SOURCE_LINE_ARG_DEF );

void  free_memory_5d(
    void   ******ptr
    _ALLOC_SOURCE_LINE_ARG_DEF );

size_t  get_total_memory_alloced( void );

VIO_BOOL  alloc_checking_enabled( void );

void  set_alloc_checking( VIO_BOOL state );

void  record_ptr_alloc_check(
    void      *ptr,
    size_t    n_bytes,
    VIO_STR    source_file,
    int       line_number );

void  change_ptr_alloc_check(
    void      *old_ptr,
    void      *new_ptr,
    size_t    n_bytes,
    VIO_STR    source_file,
    int       line_number );

VIO_BOOL  unrecord_ptr_alloc_check(
    void     *ptr,
    VIO_STR   source_file,
    int      line_number );

void  output_alloc_to_file(
    VIO_STR   filename );

void  print_alloc_source_line(
    VIO_STR  filename,
    int     line_number );

void  set_array_size(
    void      **array,
    size_t    type_size,
    size_t    previous_n_elems,
    size_t    new_n_elems,
    size_t    chunk_size
    _ALLOC_SOURCE_LINE_ARG_DEF );

VIO_BOOL  real_is_double( void );
*/

VIO_BOOL  file_exists(
    VIO_STR        filename );

VIO_BOOL  file_directory_exists(
    VIO_STR        filename );

VIO_BOOL  check_clobber_file(
    VIO_STR   filename );

VIO_BOOL  check_clobber_file_default_suffix(
    VIO_STR   filename,
    VIO_STR   default_suffix );

VIO_Status  make_backup_file(
    VIO_STR   filename,
    VIO_STR   *backup_filename );

void  cleanup_backup_file(
    VIO_STR   filename,
    VIO_STR   backup_filename,
    VIO_Status   status_of_write );

void  remove_file(
    VIO_STR  filename );

VIO_Status  copy_file(
    VIO_STR  src,
    VIO_STR  dest );

VIO_Status  move_file(
    VIO_STR  src,
    VIO_STR  dest );

VIO_STR  expand_filename(
    VIO_STR  filename );

VIO_BOOL  filename_extension_matches(
    VIO_STR   filename,
    VIO_STR   extension );

VIO_STR  remove_directories_from_filename(
    VIO_STR  filename );

VIO_BOOL  file_exists_as_compressed(
    VIO_STR       filename,
    VIO_STR       *compressed_filename );

VIO_STR  get_temporary_filename( void );

VIO_Status  open_file(
    VIO_STR             filename,
    VIO_IO_types           io_type,
    VIO_File_formats       file_format,
    FILE               **file );

VIO_Status  open_file_with_default_suffix(
    VIO_STR             filename,
    VIO_STR             default_suffix,
    VIO_IO_types           io_type,
    VIO_File_formats       file_format,
    FILE               **file );

VIO_Status  set_file_position(
    FILE     *file,
    long     byte_position );

VIO_Status  close_file(
    FILE     *file );

VIO_STR  extract_directory(
    VIO_STR    filename );

VIO_STR  get_absolute_filename(
    VIO_STR    filename,
    VIO_STR    directory );

VIO_Status  flush_file(
    FILE     *file );

VIO_Status  input_character(
    FILE  *file,
    char   *ch );

VIO_Status  unget_character(
    FILE  *file,
    char  ch );

VIO_Status  input_nonwhite_character(
    FILE   *file,
    char   *ch );

VIO_Status  output_character(
    FILE   *file,
    char   ch );

VIO_Status   skip_input_until(
    FILE   *file,
    char   search_char );

VIO_Status  output_string(
    FILE    *file,
    VIO_STR  str );

VIO_Status  input_string(
    FILE    *file,
    VIO_STR  *str,
    char    termination_char );

VIO_Status  input_quoted_string(
    FILE            *file,
    VIO_STR          *str );

VIO_Status  input_possibly_quoted_string(
    FILE            *file,
    VIO_STR          *str );

VIO_Status  output_quoted_string(
    FILE            *file,
    VIO_STR          str );

VIO_Status  input_binary_data(
    FILE            *file,
    void            *data,
    size_t          element_size,
    int             n );

VIO_Status  output_binary_data(
    FILE            *file,
    void            *data,
    size_t          element_size,
    int             n );

VIO_Status  input_newline(
    FILE            *file );

VIO_Status  output_newline(
    FILE            *file );

VIO_Status  input_line(
    FILE    *file,
    VIO_STR  *line );

VIO_Status  input_boolean(
    FILE            *file,
    VIO_BOOL         *b );

VIO_Status  output_boolean(
    FILE            *file,
    VIO_BOOL         b );

VIO_Status  input_short(
    FILE            *file,
    short           *s );

VIO_Status  output_short(
    FILE            *file,
    short           s );

VIO_Status  input_unsigned_short(
    FILE            *file,
    unsigned short  *s );

VIO_Status  output_unsigned_short(
    FILE            *file,
    unsigned short  s );

VIO_Status  input_int(
    FILE  *file,
    int   *i );

VIO_Status  output_int(
    FILE            *file,
    int             i );

VIO_Status  input_real(
    FILE            *file,
    VIO_Real            *r );

VIO_Status  output_real(
    FILE            *file,
    VIO_Real            r );

VIO_Status  input_float(
    FILE            *file,
    float           *f );

VIO_Status  output_float(
    FILE            *file,
    float           f );

VIO_Status  input_double(
    FILE            *file,
    double          *d );

VIO_Status  output_double(
    FILE            *file,
    double          d );

VIO_Status  io_binary_data(
    FILE            *file,
    VIO_IO_types        io_flag,
    void            *data,
    size_t          element_size,
    int             n );

VIO_Status  io_newline(
    FILE            *file,
    VIO_IO_types        io_flag,
    VIO_File_formats    format );

VIO_Status  io_quoted_string(
    FILE            *file,
    VIO_IO_types        io_flag,
    VIO_File_formats    format,
    VIO_STR          *str );

VIO_Status  io_boolean(
    FILE            *file,
    VIO_IO_types        io_flag,
    VIO_File_formats    format,
    VIO_BOOL         *b );

VIO_Status  io_short(
    FILE            *file,
    VIO_IO_types        io_flag,
    VIO_File_formats    format,
    short           *short_int );

VIO_Status  io_unsigned_short(
    FILE            *file,
    VIO_IO_types        io_flag,
    VIO_File_formats    format,
    unsigned short  *unsigned_short );

VIO_Status  io_unsigned_char(
    FILE            *file,
    VIO_IO_types        io_flag,
    VIO_File_formats    format,
    unsigned  char  *c );

VIO_Status  io_int(
    FILE            *file,
    VIO_IO_types        io_flag,
    VIO_File_formats    format,
    int             *i );

VIO_Status  io_real(
    FILE            *file,
    VIO_IO_types        io_flag,
    VIO_File_formats    format,
    VIO_Real            *r );

VIO_Status  io_float(
    FILE            *file,
    VIO_IO_types        io_flag,
    VIO_File_formats    format,
    float           *f );

VIO_Status  io_double(
    FILE            *file,
    VIO_IO_types        io_flag,
    VIO_File_formats    format,
    double          *d );

VIO_Status  io_ints(
    FILE            *file,
    VIO_IO_types        io_flag,
    VIO_File_formats    format,
    int             n,
    int             *ints[] );

VIO_Status  io_unsigned_chars(
    FILE            *file,
    VIO_IO_types        io_flag,
    VIO_File_formats    format,
    int             n,
    unsigned char   *unsigned_chars[] );
/*
void  set_print_function( void  (*function) ( VIO_STR ) );

void  push_print_function( void );

void  pop_print_function( void );

void  print( VIO_STR format, ... );

void  set_print_error_function( void  (*function) ( char [] ) );

void  push_print_error_function( void );

void  pop_print_error_function( void );

void  print_error( char format[], ... );

void   handle_internal_error( char  str[] );

void  abort_if_allowed( void );
*/
void  initialize_progress_report(
    VIO_progress_struct   *progress,
    VIO_BOOL           one_line_only,
    int               n_steps,
    VIO_STR            title );

void  update_progress_report(
    VIO_progress_struct   *progress,
    int               current_step );

void  terminate_progress_report(
    VIO_progress_struct   *progress );

VIO_STR  alloc_string(
    int   length );

VIO_STR  create_string(
    VIO_STR    initial );

void  delete_string(
    VIO_STR   string );

VIO_STR  concat_strings(
    VIO_STR   str1,
    VIO_STR   str2 );

void  replace_string(
    VIO_STR   *string,
    VIO_STR   new_string );

void  concat_char_to_string(
    VIO_STR   *string,
    char     ch );

void  concat_to_string(
    VIO_STR   *string,
    VIO_STR   str2 );

int  string_length(
    VIO_STR   string );

VIO_BOOL  equal_strings(
    VIO_STR   str1,
    VIO_STR   str2 );

VIO_BOOL  is_lower_case(
    char  ch );

VIO_BOOL  is_upper_case(
    char  ch );

char  get_lower_case(
    char   ch );

char  get_upper_case(
    char   ch );

VIO_BOOL  string_ends_in(
    VIO_STR   string,
    VIO_STR   ending );

  VIO_STR   strip_outer_blanks(
    VIO_STR  str );

int  find_character(
    VIO_STR    string,
    char      ch );

void  make_string_upper_case(
    VIO_STR    string );

VIO_BOOL  blank_string(
    VIO_STR   string );

VIO_Real  current_cpu_seconds( void );

VIO_Real  current_realtime_seconds( void );

VIO_STR  format_time(
    VIO_STR   format,
    VIO_Real     seconds );

void  print_time(
    VIO_STR   format,
    VIO_Real     seconds );

VIO_STR  get_clock_time( void );

void  sleep_program( VIO_Real seconds );

VIO_STR  get_date( void );
