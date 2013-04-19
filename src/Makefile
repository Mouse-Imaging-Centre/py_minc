# Makefile for Python module: VolumeIO


#INCLUDES = -I/usr/include/python2.1 -I/usr/local/mni/include
INCLUDES = -I/usr/include/python1.5 -I/usr/local/mni/include
LIBPATH = -L/usr/local/lib -L/usr/local/mni/lib
LIBRARIES = -lvolume_io -lminc -lnetcdf

CFLAGS = $(INCLUDES) -shared -g
LDFLAGS = -shared $(LIBPATH) $(LIBRARIES)
CC = gcc

#INSTALLPATH = ${HOME}/lib/py_minc


all: VolumeIO.so VolumeIO_a.so VolumeIO_constants.so 


volume_io_wrapper.o: volume_io_wrapper.c Makefile


VolumeIO.so: volume_io_wrapper.o 
	$(CC) $(LDFLAGS) -o ../lib/py_minc/VolumeIO.so volume_io_wrapper.o 

VolumeIO_a.so: VolumeIO_a.o 
	$(CC) $(LDFLAGS) -o ../lib/py_minc/VolumeIO_a.so VolumeIO_a.o

VolumeIO_a.o: VolumeIO_a.c Makefile

VolumeIO_a.c: VolumeIO_a.i Makefile
	swig -python VolumeIO_a.i -o VolumeIO_a.c

VolumeIO_constants.o: VolumeIO_constants.c Makefile

VolumeIO_constants.so: VolumeIO_constants.o 
	$(CC) $(LDFLAGS) -o ../lib/py_minc/VolumeIO_constants.so VolumeIO_constants.o

py_minc_netcdf.o: py_minc_netcdf.c

py_minc_netcdf.so: py_minc_netcdf.o
	$(CC) $(LDFLAGS) -o ../lib/py_minc/py_minc_netcdf.so py_minc_netcdf.o

clean:
	rm -f *.so *.o

