# Structure Light System

The project implements structure light(SL) algorithm based on sinusoidal signal generation with 4-step phase shifting.
For phase unwrap following algorithms are used:
 - preconditioned conjugate gradient (PCG) - iterative method that use 4-step phase shifting(36-42HZ sinusoidal pattern);
 - Temporal phase unwrapping(TPU) - direct phase unwrap that use 4-step phase shifting with and 1HZ 4-step phase shifting as referecnce;

Both algorithms have CPU(scalar) and GPU(CUDA) implementation.

# Usage via CLI with precaptured test images

3d reconstruction algorithm based on structured light algotirhm using 2 different algorithm for phase unwraping:

Preconditioned conjugate gradient (PCG) phase unwrap algorithm:
./3dr -w 1024 -h 1024 -t 1 --hf_ref test_images/hf/ref/  --hf_obj test_images/hf/phase -c 1

Temporary phase unwraping algotirm
./3dr -w 1024 -h 1024 -t 2 --lf_ref test_images/lf/ref/ --lf_obj test_images/lf/phase/   --hf_ref test_images/hf/ref/  --hf_obj test_images/hf/phase -c 1
