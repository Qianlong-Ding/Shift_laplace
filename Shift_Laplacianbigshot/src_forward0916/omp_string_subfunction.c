#include "omp_string_subfunction.h"
#include "par.h"
#include "su.h"
#include <omp.h>
#include "mkl.h"
#define ALIGN 64

void omp_memset_MKL_INT(int mkl_parallel,MKL_INT *p1,MKL_INT n)
{
    MKL_INT seize_multible;

    MKL_INT seize_remainder;

    seize_multible=(long long int)floor(n/mkl_parallel);
    seize_remainder=n-seize_multible*(mkl_parallel);

    #pragma omp parallel for num_threads(mkl_parallel)
    for (int i = 0; i < mkl_parallel; ++i)
    {
        switch(i)
        {
            case 0:memset(p1+mkl_parallel*seize_multible, 0, sizeof(MKL_INT)*seize_remainder);break;
        }
        memset(p1+i*seize_multible, 0, sizeof(MKL_INT)*seize_multible);
    }

}

void omp_memset_float(int mkl_parallel,float *p1,MKL_INT n)
{
    MKL_INT seize_multible;

    MKL_INT seize_remainder;

    seize_multible=(long long int)floor(n/mkl_parallel);
    seize_remainder=n-seize_multible*(mkl_parallel);

    #pragma omp parallel for num_threads(mkl_parallel)
    for (int i = 0; i < mkl_parallel; ++i)
    {
        switch(i)
        {
            case 0:memset(p1+mkl_parallel*seize_multible, 0, sizeof(float)*seize_remainder);break;
        }
        memset(p1+i*seize_multible, 0, sizeof(float)*seize_multible);
    }
}

void omp_memset_complex(int mkl_parallel,complex *p1,MKL_INT n)
{
    MKL_INT seize_multible;

    MKL_INT seize_remainder;

    seize_multible=(long long int)floor(n/mkl_parallel);
    seize_remainder=n-seize_multible*(mkl_parallel);


    #pragma omp parallel for num_threads(mkl_parallel)
    for (int i = 0; i < mkl_parallel; ++i)
    {
        switch(i)
        {
            case 0:memset(p1+mkl_parallel*seize_multible, 0, sizeof(complex)*seize_remainder);break;
        }
        memset(p1+i*seize_multible, 0, sizeof(complex)*seize_multible);
    }
}

void omp_memset_dcomplex(int mkl_parallel,dcomplex *p1,MKL_INT n)
{
    MKL_INT seize_multible;

    MKL_INT seize_remainder;

    seize_multible=(long long int)floor(n/mkl_parallel);
    seize_remainder=n-seize_multible*(mkl_parallel);

    #pragma omp parallel for num_threads(mkl_parallel)
    for (int i = 0; i < mkl_parallel; ++i)
    {
        switch(i)
        {
            case 0:memset(p1+mkl_parallel*seize_multible, 0, sizeof(dcomplex)*seize_remainder);break;
        }
        memset(p1+i*seize_multible, 0, sizeof(dcomplex)*seize_multible);
    }
}

void omp_memcpy_MKL_INT(int mkl_parallel,MKL_INT *p1,MKL_INT *p2,MKL_INT n)
{
    MKL_INT seize_multible;

    MKL_INT seize_remainder;

    seize_multible=(long long int)floor(n/mkl_parallel);
    seize_remainder=n-seize_multible*(mkl_parallel);

    #pragma omp parallel for num_threads(mkl_parallel)
    for (int i = 0; i < mkl_parallel; ++i)
    {
        switch(i)
        {
            case 0:memcpy(p1+mkl_parallel*seize_multible, p2+mkl_parallel*seize_multible, sizeof(MKL_INT)*seize_remainder);break;
        }
        memcpy(p1+i*seize_multible, p2+i*seize_multible, sizeof(MKL_INT)*seize_multible);
    }
}

void omp_memcpy_dcomplex(int mkl_parallel,dcomplex *p1,dcomplex *p2,MKL_INT n)
{
    MKL_INT seize_multible;

    MKL_INT seize_remainder;

    seize_multible=(long long int)floor(n/mkl_parallel);
    seize_remainder=n-seize_multible*(mkl_parallel);

    #pragma omp parallel for num_threads(mkl_parallel)
    for (int i = 0; i < mkl_parallel; ++i)
    {
        switch(i)
        {
            case 0:memcpy(p1+mkl_parallel*seize_multible, p2+mkl_parallel*seize_multible, sizeof(dcomplex)*seize_remainder);break;
        }
        memcpy(p1+i*seize_multible, p2+i*seize_multible,  sizeof(dcomplex)*seize_multible);
    }
}
void omp_mklgemv(int mkl_parallel,dcomplex *p1,dcomplex *p2,MKL_INT n)
{

}