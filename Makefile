INCLUDES=
LIBNAME=libepr.so

OFILES=src/epr_quad.o \
	   src/epr_abs.o \
	   src/epr_im.o \
	   src/epr_pot.o

ifndef CC
	CC=gcc
endif

ifndef PREFIX
	PREFIX=${HOME}
endif

CFLAGS=-g3 -Wall -Wextra -O2 -Iinclude ${INCLUDES} -fPIC -std=c99 -fopenmp

${LIBNAME}: ${OFILES}
	${CC} ${CFLAGS} $^ -shared -fopenmp -lm -o $@ 

install: ${LIBNAME}
	mkdir -p ${PREFIX}/lib
	cp ${LIBNAME} ${PREFIX}/lib
	mkdir -p ${PREFIX}/include/epr
	cp include/*.h ${PREFIX}/include/epr

uninstall:
	rm ${PREFIX}/lib/${LIBNAME}
	rm -rf ${PREFIX}/include/lixel

clean: 
	${RM} ${LIBNAME} ${OFILES}

