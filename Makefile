DEBUG=-g --gpu-architecture compute_37 -Wno-deprecated-gpu-targets
OPT=-O2
OBJ=scan.o
SIZE=400000000

all: fill-debug fill-opt

fill-debug: fill.cu scan.cu
	nvcc $(DEBUG) -o $@ fill.cu scan.cu

fill-opt: fill.cu scan.cu
	nvcc $(DEBUG) $(OPT) -o $@ fill.cu scan.cu

clean:
	rm -rf fill-*

clean-hpc:
	rm -rf hpctoolkit-fill*

run:
	./fill-opt -n $(SIZE) 

run-d:
	./fill-debug -n 20 -d

run-v:
	./fill-debug -n $(SIZE) -v

run-dv:
	./fill-debug -n 20 -d -v

run-hpc:
	hpcrun -e REALTIME -e gpu=nvidia ./fill-debug -n $(SIZE)
	hpcstruct hpctoolkit-fill-debug-measurements*
	hpcprof hpctoolkit-fill-debug-measurements*
	
view:
	hpcviewer hpctoolkit-fill-debug-database*
