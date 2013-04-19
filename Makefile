
INSTALLPATH=${HOME}/lib/python/py_minc
#INSTALLPATH=${HOME}/lib/python2.1/py_minc



build: FORCE
	(cd src; make all)

install: build
	install -m 644 lib/py_minc/{__init__.py,py_minc.py} ${INSTALLPATH}
	install -m 755 lib/py_minc/{VolumeIO_constants.so,VolumeIO_a.so,VolumeIO.so} ${INSTALLPATH}

clean:
	(cd src; make clean)

FORCE: