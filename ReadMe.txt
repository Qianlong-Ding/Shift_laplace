What is this repository for?
This is a program for solving the Helmholtz equation using the shifted Laplacian method.

How do I get set up?

Before running the program, you need to install the following open-source libraries on the Ubuntu system: Intel® HPC Toolkit, Intel® oneAPI Base Toolkit, and Seismic Unix.
Software required: Ubuntu, Intel® HPC Toolkit, Intel® oneAPI Base Toolkit, Seismic Unix

Modify the paths of the different linked libraries in the Makefile located in the src_forward0916/ folder to match the paths on your computer.

Usage
1 From a terminal, cd to src_forward0916/ and enter make.
2 cd ../ and enter ./002_FDFD_timeslice.job
3 Use the file complex_r_172.bin to plot the single frequency component graph. Use norm2*.bin files to plot the convergence curve graph.
