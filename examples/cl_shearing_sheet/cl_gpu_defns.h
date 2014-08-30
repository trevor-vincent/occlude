#ifndef CL_GPU_DEFNS_H
#define CL_GPU_DEFNS_H

#define WAVEFRONT_SIZE 32
#define MAX_DEPTH 26
#define KAHAN_SUMMATION
#define NO_GBZ
// #define ERROR_CHECK

//ghost boxes in the z direction
#ifdef NO_GBZ
#define GBZ_COL 0
#else
#define GBZ_COL 1
#endif

#endif
