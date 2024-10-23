# .c and .cu makefile

SU_HOME=/../../su/
SU_HOME1=/../../su/src/Complex
MPI_HOME=/../../intel/oneapi/mpi/
CUDA_HOME=/../../cuda/


So_HOME=$(shell pwd)
 
NVCC=$(CUDA_HOME)/bin/nvcc

CFLAGS = -I$(SU_HOME)/include -I$(SU_HOME1)/include \
         -I$(CUDA_HOME)/include -I/usr/local/cuda-11.6/Common \
         -I$(MPI_HOME)/include -I$(So_HOME)

# Lib path
LFLAGS = -L$(SU_HOME)/lib -L$(SU_HOME1)/lib \
         -L$(MPI_HOME)/lib/intel64 -L/../../intel/oneapi/mkl/latest/lib/intel64 \
		 -L$(CUDA_HOME)/lib64 -L$(So_HOME)

LDMKLlp64= -lmkl_rt -lsu -lpar -lcwp -lmpi -fopenmp 
LDMKLilp64=-lsu -lpar -lcwp -DMKL_ILP64 -liomp5 -lpthread -lm -lmkl_intel_ilp64 -lmkl_intel_thread \
-lmkl_core -lmkl_blacs_intelmpi_ilp64 -ldl


CFILES1=omp_string_subfunction
CFILES2=fre_solver_mkl
CFILES3=fre_gmg_solver_mkl
CFILES4=fre_compute_sparse_matrix3d_mkl
CFILES5=fre_forward_cpu_subfunction

CUFILES=fre_forward_gmres_cpu
EXECNAME=fre_forward_gmres_cpu

all:
	mpiicc -fopenmp -c -w -fPIC  $(CFILES1).c $(CFLAGS) $(LFLAGS) $(LDMKLilp64)
#	mpiicc -shared -fPIC -o lib$(CFILES1).so $(CFILES1).o
	ar -cr lib$(CFILES1).a $(CFILES1).o

	mpiicc -fopenmp -c -w -fPIC  $(CFILES2).c $(CFLAGS) $(LFLAGS) $(LDMKLilp64) $(So_HOME)/$(CFILES1).o
#	mpiicc -shared -fPIC -o lib$(CFILES2).so $(CFILES2).o
	ar -cr lib$(CFILES2).a $(CFILES2).o

	mpiicc -fopenmp -c -w -fPIC  $(CFILES3).c $(CFLAGS) $(LFLAGS) $(LDMKLilp64) $(So_HOME)/$(CFILES1).o $(So_HOME)/$(CFILES2).o
#	mpiicc -shared -fPIC -o lib$(CFILES3).so $(CFILES3).o
	ar -cr lib$(CFILES3).a $(CFILES3).o

	mpiicc -fopenmp -c -w -fPIC  $(CFILES4).c $(CFLAGS) $(LFLAGS) $(LDMKLilp64) $(So_HOME)/$(CFILES1).o
#	mpiicc -shared -fPIC -o lib$(CFILES4).so $(CFILES4).o
	ar -cr lib$(CFILES4).a $(CFILES4).o

	mpiicc -fopenmp -c -w -fPIC  $(CFILES5).c \
	$(So_HOME)/$(CFILES1).o \
	$(So_HOME)/$(CFILES2).o \
	$(So_HOME)/$(CFILES3).o \
	$(So_HOME)/$(CFILES4).o  \
	$(CFLAGS) $(LFLAGS) $(LDMKLilp64)

	ar -crv libmylib.a $(CFILES1).o $(CFILES2).o $(CFILES3).o $(CFILES4).o $(CFILES5).o
	

#	mpiicc -shared -o lib$(CFILES5).so $(CFILES5).o 



	mpiicc -O3 -w -o $(EXECNAME) $(CUFILES).c \
	 $(CFLAGS) $(LFLAGS)  -lmylib $(LDMKLilp64)

	 rm -f *.o


clean:
	rm -f $(EXECNAME)

#	mpiicc -O3 -w -o $(EXECNAME) $(CUFILES).c \
#	  $(CFLAGS) $(LFLAGS) $(LDMKLilp64)
#-$(MPICC) -O3 -arch=sm_60 -c $(CUFILES).cu $(CFLAGS) $(LFLAGS) $(LD)
# -$(MPICC) -fopenmp -c  $(CFILES).c $(CFLAGS) $(LFLAGS) $(LDC) -std=c99

#-$(MPICC) -fopenmp -c  $(CFILES).h $(CFLAGS) $(LFLAGS) $(LDC) -std=c99
#-$(NVCC) -O3 -arch=sm_60 -c $(CUFILES).cu $(CFLAGS) $(LFLAGS) $(LD)
#-$(NVCC) -O3 -arch=sm_60 -o $(EXECNAME) $(CFILES).o $(CUFILES).o $(CFLAGS) $(LFLAGS) $(LD)
