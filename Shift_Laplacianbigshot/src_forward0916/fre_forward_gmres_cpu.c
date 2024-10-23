#include "fre_forward_cpu_subfunction.h"
#include "fre_compute_sparse_matrix3d_mkl.h"
#include "fre_gmg_solver_mkl.h"
#include "fre_solver_mkl.h"
#include "omp_string_subfunction.h"

#include "stdlib.h"
#include "time.h"
#include "par.h"
#include "su.h"
#include "segy.h"
#include <math.h>
#include "mkl.h"
#include <mpi.h>

#include "stdio.h"


#ifdef MKL_ILP64
#define INT_PRINT_FORMAT "%lld"
#else
#define INT_PRINT_FORMAT "%d"
#endif
#define LOOKFAC    2


#define ALIGN 64
/****************************** self documentation ****************************/
const char *sdoc[] = {
" 								                                              ",
" 								                                              ",
" 																			  ",
" Required Parameters:						       						      ",
" 															                  ",
" 															                  ",
" Optional Parameters:                                                        ",
" 									                                          ",
" Notes:								                                      ",
" 								                                              ",
NULL};
/*

 */
/****************************** end self doc **********************************/

int main(int argc, char **argv)
{
	/************** MPI parameters **************/
    int myid,npros;
    int dblock;
    char proname[20];
    int resultlen;
    MPI_Comm comm;
    int s_beg,s_end;
    int f_beg,f_end;

    /* GMRES  */

    int mkl_parallel;
    int LevelMax=15;

    /* fgmres norm2 */
    float *norm2_res;
    

    /***************** do model ******************/
    int op_fdmodel;
    int op_shot;
    int op_slice;
    int op_precond;
    int shift_laplace_option;

    int nfft;
    int ifreq;

    float constant_vel;

    float velmin;
    float velmax;

    int pml_thick;

    /************* model parameters ************/
    float fx;
    float fy;
    float fz;
    int nx;
    int ny;
    int nz;
    float dx;
    float dy;
    float dz;
    int nxz;

    int nxA;
    int nzA;
    int ix;
    int iy;
    int iz;
    int ixA;
    int izA;
    int ixz;

    int Nx;
    int Ny;
    int Nz;
    int no_zero_number;

    dcomplex *h_Val[LevelMax];
    MKL_INT *h_Col[LevelMax];
    MKL_INT *h_Row[LevelMax];

    dcomplex *h_ValA;
    MKL_INT *h_ColA;
    MKL_INT *h_RowA;

    dcomplex *h_ValR[LevelMax];
    MKL_INT *h_ColR[LevelMax];
    MKL_INT *h_RowR[LevelMax];

    dcomplex *h_ValP[LevelMax];
    MKL_INT *h_ColP[LevelMax];
    MKL_INT *h_RowP[LevelMax];

    dcomplex *h_ValSR[LevelMax];
    MKL_INT *h_ColSR[LevelMax];
    MKL_INT *h_RowSR[LevelMax];

    double iStart, iElaps;

    /****************************************/
    int nxR;
    int srmaxnum;
    int srnum;
    float *SR;
    int *OffsetSRx;
    int *OffsetSRy;
    int *OffsetSRz;

    int *NxIlevel;
    int *NyIlevel;
    int *NzIlevel;
    MKL_INT *RankIlevel;

    int ilevel;
    int Level;
    
    dcomplex *h_x;
    dcomplex *h_b;

    float ***vel;
    float ***nppw;
    complex ***A;


    /*********************PML*********************/

    float R;
    float alpha_max;

    float ***dampx;
    float ***dampy;
    float ***dampz;
    float ***dampxdiff;
    float ***dampydiff;
    float ***dampzdiff;

    float ***alphax;
    float ***alphay;
    float ***alphaz;

    float ***alphax_diff;
    float ***alphay_diff;
    float ***alphaz_diff;

    complex ***deno_ex2;
    complex ***deno_ey2;
    complex ***deno_ez2;

    complex ***deno_ex3;
    complex ***deno_ey3;
    complex ***deno_ez3;

    float *dux;
    float *duy;
    float *duz;
    float *duxx;
    float *duyy;
    float *duzz;

    float **slice2D;
    float ***slice3D;

    float *omegav_cofficient;
    complex *coefficient;

    /****************shot*******************/
    int ns;
    int is;
    float fxs;
    float fys;
    float fzs;
    float dxs;
    float dys;
    float dzs;

    float *xs;
    float *ys;
    float *zs;
    int *ixs;
    int *iys;
    int *izs;

    int nr;
    int ir;
    float fxr;
    float fyr;
    float fzr;
    float dxr;
    float dyr;
    float dzr;

    float *xr;
    float *yr;
    float *zr;
    int *ixr;
    int *iyr;
    int *izr;

    float ***shot;

    /**************************************/
    int nrec;
    float dr;
    float fr;

    /****************Viscosity**************/
    float Q_constant;
    float  ***h_Q;
    float alpha;
    float omega0;

    /**************source*******************/
    float fpeak;
    float multiple;
    int nt;
    float tdelay;
    float dt;
    
    float *source;
    complex *source_fre;
    float t;
    int it;




    /********************Freq*************************/
    int nt_fft;
    int nf;
    int nfreq;
    float df;

    float freq_refrence;

    /**************** frequent slice *****************/
    complex ***freq_slice;

    int *iifreq;
    int iiifreq;


    /**************************************************/
    float omega;


    /********************GMRES*************************/
    int nter_max;
    float tolerant;
    int m_fg;
    int m_g;
    int gmres_smoother;
    int nter_precond;
    int fgmres_out;
    int nter_real;

    /*************************************************/
    int solve_option;

    /*******************File name*********************/
    char optname[BUFSIZ];

    char *frslicepath="";
    char *shotpath="";
    char *slicepath="";

    char *qualfile="";
    char *velfile="";
    char *logfile="";

    /************* file pointer ***************/
    FILE *optfp=NULL;
    FILE *ipt=NULL;
    FILE *logfp=NULL;

    int verbose;
    int vel_plural;
    int Viscosity;
    int gmres_out;
    int slice_is;
    float slice_t; 

	initargs(argc,argv);
    requestdoc(0);

	/* MPI initializing */
    MPI_Init(&argc, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD,&comm);
    MPI_Comm_size(comm,&npros);
    MPI_Comm_rank(comm,&myid);
    MPI_Get_processor_name(proname,&resultlen);

    if(!getparint("verbose",&verbose))                        verbose=0;
    if(!getparint("op_shot", &op_shot))                       op_shot=1;
    if(!getparint("op_slice", &op_slice))                     op_slice=0;
    if(!getparint("op_fdmodel", &op_fdmodel))                 op_fdmodel=1;
    if(!getparint("op_precond", &op_precond))                 op_precond=1;
    if(!getparint("vel_plural",&vel_plural))                  vel_plural = 0;
    if(!getparint("Viscosity",&Viscosity))                    Viscosity = 1;
    

    if(!getparint("mkl_parallel",&mkl_parallel))              mkl_parallel=1;
    if(!getparint("srmaxnum",&srmaxnum))                      srmaxnum=4;

    //srmaxnum

    if(op_slice==1)
    {
        if (!getparint("slice_is",&slice_is))                 err("Please input 'slice_is'");
        if (!getparfloat("slice_t",&slice_t))                 err("Please input 'slice_t'");       
    }

    if(!getparfloat("fx",&fx))                                fx = 0.0;
    if(!getparfloat("fy",&fy))                                fy = 0.0;
    if(!getparfloat("fz",&fz))                                fz = 0.0;
    if(!getparint("nx",&nx))                                  err("Please input 'nx'");
    if(!getparint("ny",&ny))                                  err("Please input 'ny'");
    if(!getparint("nz",&nz))                                  err("Please input 'nz'");
    if(!getparfloat("dx",&dx))                                err("Please input 'dx'");
    if(!getparfloat("dy",&dy))                                err("Please input 'dy'");
    if(!getparfloat("dz",&dz))                                err("Please input 'dz'");

    if(!getparint("ns",&ns))                                  err("Please input 'ns'");
    if(!getparint("nr",&nr))                                  err("Please input 'nr'");

    if(!getparfloat("fxs",&fxs))                              fxs = 0.0;
    if(!getparfloat("fys",&fys))                              fys = 0.0;
    if(!getparfloat("fzs",&fzs))                              fzs = 0.0;
    if(!getparfloat("dxs",&dxs))                              dxs = 1.0;
    if(!getparfloat("dys",&dys))                              dys = 1.0;
    if(!getparfloat("dzs",&dzs))                              dzs = 1.0;

    if(!getparfloat("fxr",&fxr))                              fxr = 0.0;
    if(!getparfloat("fyr",&fyr))                              fyr = 0.0;
    if(!getparfloat("fzr",&fzr))                              fzr = 0.0;
    if(!getparfloat("dxr",&dxr))                              dxr = 1.0;
    if(!getparfloat("dzr",&dzr))                              dzr = 1.0;

    if(!getparfloat("multiple",&multiple))                    multiple = 1000.0;
    if(!getparfloat("fpeak",&fpeak))                          fpeak = 30.0;
    if(!getparfloat("omega0",&omega0))                        err("Please input 'omega0'");

    if(!getparint("pml_thick",&pml_thick))                    pml_thick= 10;
    if(!getparfloat("alpha_max",&alpha_max))                  alpha_max=0.0;
    if(!getparfloat("R",&R))                                  R=0.0001;
    
    if(!getparint("nter_max",&nter_max))                      nter_max=1000;

    if(!getparint("m_g",&m_g))                                m_g=5;
    if(!getparint("m_fg",&m_fg))                              m_fg=5;
    
    if(!getparint("solve_option",&solve_option))              solve_option=1;
    
    if(!getparint("fgmres_out",&fgmres_out))                  fgmres_out=1;
    if(!getparint("gmres_out",&gmres_out))                    gmres_out=0;
    if(!getparint("gmres_smoother",&gmres_smoother))          gmres_smoother=m_g;
    if(!getparint("nter_precond",&nter_precond))              nter_precond=1;
    if(!getparfloat("tolerant",&tolerant))                    tolerant=0.0001;


    if(!getparint("shift_laplace_option",&shift_laplace_option)) shift_laplace_option=1;
    if(!getparfloat("freq_refrence",&freq_refrence))             freq_refrence=30;

    if (!getparstring("logfile",&logfile)) logfile="log.txt";
    

    if(!getparint("nt",&nt))                                  err("Please input 'nt'");
    if(!getparfloat("dt",&dt))                                err("Please input 'dt'");

    if(!getparstring("frslicepath",&frslicepath)) 
        err("you must enter a frslicepath");
    if(!getparstring("shotpath",&shotpath)) 
        err("you must enter a shotpath");
    if(!getparstring("slicepath",&slicepath)) 
        err("you must enter a slicepath");

    if(!getparstring("velfile", &velfile))             err("Please input 'velfile' name");

    nf= npfaro(nt, LOOKFAC * nt);

    nfreq = nf/2+1;
    df=1.0/(nf*dt);
    warn("nf=%d",nf);


    warn("nt=%d dt=%f", nt, dt);
    warn("nfreq=%d df=%.8f", nfreq, df); 

    
    if(!getparfloat("Q_constant",&Q_constant)) Q_constant = 600.0;


    tdelay=1.0/fpeak;

    nt_fft= npfaro(nt, LOOKFAC * nt);

    warn("nt_fft=%d",nt_fft);

    nfreq = nt_fft/2+1;
    df=1.0/(nt_fft*dt);

    nfft=NINT(4.0*fpeak/df);

    warn("nfft=%d npros=%d",nfft,npros);

    nxz=nx*ny*nz;

    srmaxnum=3;

    //nxA=13;
    nxA=19;
    nxR=27;
    nzA=nxz;

    norm2_res=alloc1float(nter_max+1);


    OffsetSRx=alloc1int((srmaxnum*2+1)*(srmaxnum*2+1)*(srmaxnum*2+1));
    OffsetSRy=alloc1int((srmaxnum*2+1)*(srmaxnum*2+1)*(srmaxnum*2+1));
    OffsetSRz=alloc1int((srmaxnum*2+1)*(srmaxnum*2+1)*(srmaxnum*2+1));

    SR=alloc1float((srmaxnum*2+1)*(srmaxnum*2+1)*(srmaxnum*2+1));


    srnum=ComputeSR1d( OffsetSRx, OffsetSRy, OffsetSRz, SR, srmaxnum);


    source=alloc1float(nt_fft);
    source_fre=alloc1complex(nfreq);

    xs=alloc1float(ns);
    ys=alloc1float(ns);
    zs=alloc1float(ns);
    ixs=alloc1int(ns);
    iys=alloc1int(ns);
    izs=alloc1int(ns);

    xr=alloc1float(nr);
    yr=alloc1float(nr);
    zr=alloc1float(nr);
    ixr=alloc1int(nr);
    iyr=alloc1int(nr);
    izr=alloc1int(nr);

    vel=alloc3float(nz,ny,nx);
    nppw=alloc3float(nz,ny,nx);
    A=alloc3complex(nz,ny,nx);

    dampx=alloc3float(nz,ny,nx);
    dampy=alloc3float(nz,ny,nx);
    dampz=alloc3float(nz,ny,nx);

    dampxdiff=alloc3float(nz,ny,nx);
    dampydiff=alloc3float(nz,ny,nx);
    dampzdiff=alloc3float(nz,ny,nx);

    alphax=alloc3float(nz,ny,nx);
    alphay=alloc3float(nz,ny,nx);
    alphaz=alloc3float(nz,ny,nx);

    alphax_diff=alloc3float(nz,ny,nx);
    alphay_diff=alloc3float(nz,ny,nx);
    alphaz_diff=alloc3float(nz,ny,nx);

    deno_ex2=alloc3complex(nz,ny,nx);
    deno_ey2=alloc3complex(nz,ny,nx);
    deno_ez2=alloc3complex(nz,ny,nx);

    deno_ex3=alloc3complex(nz,ny,nx);
    deno_ey3=alloc3complex(nz,ny,nx);
    deno_ez3=alloc3complex(nz,ny,nx);


    omegav_cofficient=alloc1float(nxA);
    coefficient=alloc1complex(nxA);

    dux=alloc1float(nxA);
    duy=alloc1float(nxA);
    duz=alloc1float(nxA);
    duxx=alloc1float(nxA);
    duyy=alloc1float(nxA);
    duzz=alloc1float(nxA);



    freq_slice=alloc3complex(nz,ny,nx);

    iifreq=alloc1int(nfft);

    shot=alloc3float(nt_fft,ny,nx);

    NxIlevel = alloc1int(LevelMax);
    NyIlevel = alloc1int(LevelMax);
    NzIlevel = alloc1int(LevelMax);
    // RankIlevel = alloc1int(LevelMax);
    warn("mklint=%d int=%d",sizeof(MKL_INT),sizeof(int));

    RankIlevel = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (LevelMax), ALIGN);

    slice2D=alloc2float(nz,ny);

    slice3D=alloc3float(nz,ny,nx);

    memset(NxIlevel,   0, sizeof(int)*LevelMax);
    memset(NyIlevel,   0, sizeof(int)*LevelMax);
    memset(NzIlevel,   0, sizeof(int)*LevelMax);
    memset(RankIlevel, 0, sizeof(MKL_INT)*LevelMax);

    
    NxIlevel[0]=nx;
    NyIlevel[0]=ny;
    NzIlevel[0]=nz;
    for(ilevel = 1;ilevel < LevelMax;ilevel++)
    {
        NxIlevel[ilevel] = (int)((NxIlevel[ilevel-1] + 1)/2);
        NyIlevel[ilevel] = (int)((NyIlevel[ilevel-1] + 1)/2);
        NzIlevel[ilevel] = (int)((NzIlevel[ilevel-1] + 1)/2);

        if((NxIlevel[ilevel] < 7 ) || ( NyIlevel[ilevel] < 7) || (NzIlevel[ilevel] < 7))
        {
            Level = ilevel;
            break;
        }
    }
    
    warn("Level=%d",Level);

    // h_b=alloc1dcomplex(nzA);
    // h_x=alloc1dcomplex(nzA);

    

    for (ilevel = 0; ilevel < Level; ilevel++)
    {
        RankIlevel[ilevel]=NxIlevel[ilevel]*NyIlevel[ilevel]*NzIlevel[ilevel];
        warn("ilevel=%d\tNxIlevel[ilevel]=%d\tNyIlevel[ilevel]=%d\tNzIlevel[ilevel]=%d\tRankIlevel[ilevel]="INT_PRINT_FORMAT"",
            ilevel,NxIlevel[ilevel],NyIlevel[ilevel],NzIlevel[ilevel],RankIlevel[ilevel]);
    }

    // iStart = dsecnd();

    h_b = (dcomplex *)mkl_malloc(sizeof(dcomplex)*RankIlevel[0], ALIGN);
    h_x = (dcomplex *)mkl_malloc(sizeof(dcomplex)*RankIlevel[0], ALIGN);

    omp_memset_dcomplex(mkl_parallel,h_b,RankIlevel[0]);
    omp_memset_dcomplex(mkl_parallel,h_x,RankIlevel[0]);

    for(ilevel=0;ilevel<Level-1;ilevel++)
    {
        h_ValP[ilevel]=(dcomplex *)mkl_malloc(sizeof(dcomplex) * RankIlevel[ilevel+1]*nxR, ALIGN);
        h_ColP[ilevel]=(MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (RankIlevel[ilevel+1]*nxR), ALIGN);
        h_RowP[ilevel]=(MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (RankIlevel[ilevel]+1), ALIGN);

        h_ValR[ilevel]=(dcomplex *)mkl_malloc(sizeof(dcomplex) * RankIlevel[ilevel+1]*nxR, ALIGN);
        h_ColR[ilevel]=(MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (RankIlevel[ilevel+1]*nxR), ALIGN);
        h_RowR[ilevel]=(MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (RankIlevel[ilevel+1]+1), ALIGN);

        h_ValSR[ilevel]=(dcomplex *)mkl_malloc(sizeof(dcomplex) * RankIlevel[ilevel+1]*srnum, ALIGN);
        h_ColSR[ilevel]=(MKL_INT *)mkl_malloc(sizeof(MKL_INT) * RankIlevel[ilevel+1]*srnum, ALIGN);
        h_RowSR[ilevel]=(MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (RankIlevel[ilevel+1]+1), ALIGN);

        omp_memset_dcomplex(mkl_parallel,h_ValP[ilevel],RankIlevel[ilevel+1]*nxR);
        omp_memset_dcomplex(mkl_parallel,h_ValR[ilevel],RankIlevel[ilevel+1]*nxR);
        omp_memset_dcomplex(mkl_parallel,h_ValSR[ilevel],RankIlevel[ilevel+1]*srnum);

        omp_memset_MKL_INT(mkl_parallel,h_ColP[ilevel],RankIlevel[ilevel+1]*nxR);
        omp_memset_MKL_INT(mkl_parallel,h_RowP[ilevel],(RankIlevel[ilevel]+1));

        omp_memset_MKL_INT(mkl_parallel,h_ColR[ilevel],RankIlevel[ilevel+1]*nxR);
        omp_memset_MKL_INT(mkl_parallel,h_RowR[ilevel],(RankIlevel[ilevel+1]+1));

        omp_memset_MKL_INT(mkl_parallel,h_ColSR[ilevel],RankIlevel[ilevel+1]*srnum);
        omp_memset_MKL_INT(mkl_parallel,h_RowSR[ilevel],(RankIlevel[ilevel+1]+1));
    }

    h_Val[0]=(dcomplex *)mkl_malloc(sizeof(dcomplex) * RankIlevel[0]*nxA, ALIGN);
    h_Col[0]=(MKL_INT *)mkl_malloc(sizeof(MKL_INT) * RankIlevel[0]*nxA, ALIGN);
    h_Row[0]=(MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (RankIlevel[0]+1), ALIGN);

    h_ValA=(dcomplex *)mkl_malloc(sizeof(dcomplex) * RankIlevel[0]*nxA, ALIGN);
    h_ColA=(MKL_INT *)mkl_malloc(sizeof(MKL_INT) * RankIlevel[0]*nxA, ALIGN);
    h_RowA=(MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (RankIlevel[0]+1), ALIGN);

    omp_memset_dcomplex(mkl_parallel,h_Val[0],RankIlevel[0]*nxA);
    omp_memset_MKL_INT(mkl_parallel,h_Row[0],RankIlevel[0]*nxA);
    omp_memset_MKL_INT(mkl_parallel,h_Row[0],(RankIlevel[0]+1));

    omp_memset_dcomplex(mkl_parallel,h_ValA,RankIlevel[0]*nxA);
    omp_memset_MKL_INT(mkl_parallel,h_RowA,RankIlevel[0]*nxA);
    omp_memset_MKL_INT(mkl_parallel,h_RowA,(RankIlevel[0]+1));


    memset(source,     0, FSIZE*nt_fft);
    memset(source_fre, 0, sizeof(complex)*nfreq);

    omp_memset_float(mkl_parallel, dampx[0][0], RankIlevel[0]);
    omp_memset_float(mkl_parallel, dampy[0][0], RankIlevel[0]);
    omp_memset_float(mkl_parallel, dampz[0][0], RankIlevel[0]);

    omp_memset_float(mkl_parallel, dampxdiff[0][0], RankIlevel[0]);
    omp_memset_float(mkl_parallel, dampydiff[0][0], RankIlevel[0]);
    omp_memset_float(mkl_parallel, dampzdiff[0][0], RankIlevel[0]);

    omp_memset_float(mkl_parallel, alphax[0][0], RankIlevel[0]);
    omp_memset_float(mkl_parallel, alphay[0][0], RankIlevel[0]);
    omp_memset_float(mkl_parallel, alphaz[0][0], RankIlevel[0]);

    omp_memset_float(mkl_parallel, alphax_diff[0][0], RankIlevel[0]);
    omp_memset_float(mkl_parallel, alphay_diff[0][0], RankIlevel[0]);
    omp_memset_float(mkl_parallel, alphaz_diff[0][0], RankIlevel[0]);

    omp_memset_complex( mkl_parallel, deno_ex2[0][0],RankIlevel[0]);
    omp_memset_complex( mkl_parallel, deno_ey2[0][0],RankIlevel[0]);
    omp_memset_complex( mkl_parallel, deno_ez2[0][0],RankIlevel[0]);
    omp_memset_complex( mkl_parallel, deno_ex3[0][0],RankIlevel[0]);
    omp_memset_complex( mkl_parallel, deno_ey3[0][0],RankIlevel[0]);
    omp_memset_complex( mkl_parallel, deno_ez3[0][0],RankIlevel[0]);
    omp_memset_complex( mkl_parallel, freq_slice[0][0],RankIlevel[0]);


    memset(omegav_cofficient, 0, sizeof(float)*nxA);
    memset(coefficient,       0, sizeof(complex)*nxA);

    memset(dux,   0, sizeof(float)*nxA);
    memset(duy,   0, sizeof(float)*nxA);
    memset(duz,   0, sizeof(float)*nxA);
    memset(duxx,  0, sizeof(float)*nxA);
    memset(duyy,  0, sizeof(float)*nxA);
    memset(duzz,  0, sizeof(float)*nxA);

    memset(iifreq,0, sizeof(int)*nfft);


    // memset(h_b,              0, sizeof(dcomplex)*RankIlevel[0]);
    // memset(h_x,              0, sizeof(dcomplex)*RankIlevel[0]);

    // memset(freq_slice[0][0], 0, sizeof(complex)*RankIlevel[0]);
    memset(shot[0][0],       0, sizeof(float)*nx*ny*nt_fft);

    warn("ns=%d nr=%d",ns,nr);

    for(is=0;is<ns;is++)
    {
        xs[is] = fxs+dxs*is;
        ys[is] = fys+dys*is;
        zs[is] = fzs+dzs*is;

        ixs[is] = NINT((xs[is]-fx)/dx);
        iys[is] = NINT((ys[is]-fy)/dy);
        izs[is] = NINT((zs[is]-fz)/dz);
    }
    for(ir=0;ir<nr;ir++)
    {
        xr[ir] = fxr+dxr*ir;
        yr[ir] = fyr+dyr*ir;
        zr[ir] = fzr+dzr*ir;

        ixr[ir] = NINT((xr[ir]-fx)/dx);
        iyr[ir] = NINT((yr[ir]-fy)/dy);
        izr[ir] = NINT((zr[ir]-fz)/dz);
    }




    for (it=0,t=0;it<nt_fft;t=t+dt,it++)
    {
        source[it]=multiple*ricker(t-tdelay,fpeak);
    }
    FFT_tf(nt_fft,nfreq,source,source_fre);



    if (myid==0)
    {
        if(!getparfloat("constant_vel", &constant_vel) )
        {
            warn("Read in velocity model");
            if(!getparstring("velfile", &velfile))
                err("Please input 'velfile' name");

            if((ipt=efopen(velfile, "r"))==NULL)
                err("Velocity file is empty!!");
            else
            {
                fread(vel[0][0], sizeof(float), nx*ny*nz, ipt);
                fclose(ipt);
            }
        }
        else
        {
            warn("Constant velocity model");
            #pragma omp parallel for num_threads(mkl_parallel)
            for(int ix=0;ix<nx;++ix)
            {
                for(int iy=0;iy<ny;++iy)
                {
                    for(int iz=0;iz<nz;++iz)
                    {
                        vel[ix][iy][iz]=constant_vel;
                    }
                }
            }
        }

        // {
        //     warn("Constant velocity model");
        //     #pragma omp parallel for num_threads(mkl_parallel)
        //     for(int ix=0;ix<nx;++ix)
        //     {
        //         for(int iy=0;iy<ny;++iy)
        //         {
        //             for(int iz=0;iz<nz;++iz)
        //             {
        //                 vel[ix][iy][iz]=3000.0;
        //             }
        //         }
        //     }
        // }


    }
    MPI_Bcast(vel[0][0],RankIlevel[0],MPI_FLOAT,0,comm);
    MPI_Barrier(comm);

    velmin=vel[0][0][0];
    velmax=vel[0][0][0];

    #pragma omp parallel for num_threads(mkl_parallel)
    for(int ix=0;ix<nx;ix++)
    {
        for(int iy=0;iy<ny;iy++)
        {
            for(int iz=0;iz<nz;iz++)
            {
                A[ix][iy][iz].r=1.0;
                A[ix][iy][iz].i=0.0;

                if(velmin>vel[ix][iy][iz])
                {
                    velmin=vel[ix][iy][iz];
                }

                if(velmax<vel[ix][iy][iz])
                {
                    velmax=vel[ix][iy][iz];
                }
            }
        }
    }


    warn("velmin=%f velmax=%f",velmin,velmax);

    // for(ix=0;ix<nx;ix++)
    // {
    //     for(iy=0; iy<ny;iy++)
    //     {
    //         for(iz=70;iz<nz;iz++)
    //         {
    //             vel[ix][iy][iz]=3000.0;

    //             A[ix][iy][iz].r=1.0;
    //             A[ix][iy][iz].i=0.0;
    //         }
    //     }
    // }

   dblock=(int)(nfft/(npros));

    if ( myid<(nfft%npros) )
    {
        {
            f_beg=NINT(myid*(dblock+1));
            f_end=NINT((myid+1)*(dblock+1));
        }
    }
    else
    {
        f_beg=NINT(myid*dblock+nfft%npros);
        f_end=NINT((myid+1)*dblock + nfft%npros);
    }
    MPI_Barrier(comm);
    warn("rank: %d, on processor %s. Begin frequency:%d, end frequency:%d",
        myid,proname,f_beg,f_end-1);


    // if(myid==0)
    // {
    //     nfft=11;
    //     dblock=(int)(nfft/(npros));
    //     ifreq=0;
    // for(int imyid=0;imyid<npros;imyid++)
    // {
    //     int if_beg;
    //     int if_end;
    //     if ( imyid<(nfft%npros) )
    //     {
    //         if_beg=NINT(imyid*(dblock+1));
    //         if_end=NINT((imyid+1)*(dblock+1));
    //     }
    //     else
    //     {
    //         if_beg=NINT(imyid*dblock+nfft%npros);
    //         if_end=NINT((imyid+1)*dblock + nfft%npros);
    //     }

    //     for(int i=if_beg,itemp=0;i<if_end;i++,itemp++)
    //     {
    //         iifreq[ifreq]=itemp*npros+imyid;

    //         warn("iifreq[%d]=%d",ifreq,iifreq[ifreq]);
    //         ifreq++;

    //     }

    // }

    // }




    ifreq=0;
    for(int imyid=0;imyid<npros;imyid++)
    {
        int if_beg;
        int if_end;
        if ( imyid<(nfft%npros) )
        {
            if_beg=NINT(imyid*(dblock+1));
            if_end=NINT((imyid+1)*(dblock+1));
        }
        else
        {
            if_beg=NINT(imyid*dblock+nfft%npros);
            if_end=NINT((imyid+1)*dblock + nfft%npros);
        }

        for(int i=if_beg,itemp=0;i<if_end;i++,itemp++)
        {
            iifreq[ifreq]=itemp*npros+imyid;
            ifreq++;
        }
    }

    // for(int ifreq=0;ifreq<nfft;ifreq++)
    // {
    //     int iblock;
    //     ifreq%npros;
    //     iblock=(int)(ifreq/npros)
    //     iifreq[(ifreq%npros)*npros+iblock]=ifreq;
    // }


    // for(int ifreq=0;ifreq<nfft;ifreq++)
    // {
    //     iifreq[ifreq]=ifreq;
    // }

    // for(int ifreq=0;ifreq<12;ifreq++)
    // {
    //     int iblock;
    //     iblock=(int)(ifreq/npros);
        
    //     iifreq[(ifreq%npros)*npros+iblock]=ifreq;
    // }

    

    #pragma omp parallel for num_threads(mkl_parallel)
    for(int ix=0;ix<nx;ix++)
    {
        for(int iy=0;iy<ny;iy++)
        {
            for(int iz=0;iz<nz;iz++)
            {
                deno_ex2[ix][iy][iz].r=1.0;
                deno_ey2[ix][iy][iz].r=1.0;
                deno_ez2[ix][iy][iz].r=1.0;

                deno_ex3[ix][iy][iz].r=1.0;
                deno_ey3[ix][iy][iz].r=1.0;
                deno_ez3[ix][iy][iz].r=1.0;

            }
        }
    }

    if(myid==0)
    {
        logfp=fopen(logfile,"a");
        fseek(logfp,0,SEEK_END);
        fprintf(logfp,"\n");
        fprintf(logfp,"=========== Frequent_FDFD_forward 1.0 ===========\n");
        fprintf(logfp,"Copyright @CUPB Ding Qianlong\n");
        fprintf(logfp,"inputmodel=%s\n",velfile);
        fprintf(logfp,"velmin=%.2f,velmax=%.2f \n",velmin,velmax);
        fprintf(logfp,"fpeak=%.1f,multiple=%.1f,nt=%d,dt=%.6f\n",fpeak,multiple,nt,dt);
        fprintf(logfp,"nx=%d ny=%d nz=%d \n",nx,ny,nz);
        fprintf(logfp,"dx=%.1f dy=%.1f dz=%.1f \n",dx,dy,dz);
        fprintf(logfp,"pml_thick=%d \n",pml_thick);
        fprintf(logfp,"freq_refrence=%.2fHz \n",freq_refrence);
        fprintf(logfp,"**************************************************\n");

        fflush(logfp);
    }



    nxz=nx*ny*nz;

    switch(op_fdmodel)
    {
        case 1:
        iStart = dsecnd();
        //for(ifreq=f_beg;ifreq<f_end;++ifreq)
        {

            memset(norm2_res,0, sizeof(float)*(nter_max+1));

            for(int iter=0;iter<nter_max+1;iter++)
            {
                norm2_res[iter]=1e-10;
            }
            
            // if(ifreq>5)
            // {
            //      continue;
            // }
            // if(iifreq[ifreq]<2)
            // {
            //      continue;
            // }
            ifreq=10;
            iifreq[ifreq]=(int)(freq_refrence/df);
            iiifreq=iifreq[ifreq];
            warn("ifreq=%d",iifreq[ifreq]);

            #pragma omp parallel for num_threads(mkl_parallel)
            for(int ix=0;ix<nx;ix++)
            {
                for(int iy=0; iy<ny;iy++)
                {
                    for(int iz=0;iz<nz;iz++)
                    {
                        nppw[ix][iy][iz]=(velmax+velmin)/(2.0*iifreq[ifreq]*df*dx);
                        // nppw[ix][iy][iz]=vel[ix][iy][iz]/(ifreq*df*dx);
                        // if(nppw[ix][iy][iz]<pow(2,Level-1))
                        // {
                        //     nppw[ix][iy][iz]=pow(2,Level-1);
                        // }

                        // if(nppw[ix][iy][iz]>=128.0)
                        // {
                        //     //ppw[ix][iy][iz]=pow(2,14-NINT(log(nppw[ix][iy][iz])/log(2)));
                        // }
                    }
                }
            }

            iter_solve_fun_gmres(solve_option, shift_laplace_option, nter_max, &nter_real, norm2_res, tolerant,
              nter_precond,op_precond, gmres_smoother, fgmres_out, gmres_out, m_fg, m_g,
                                    myid, npros, comm, mkl_parallel, Level,
                                    h_ValA, h_ColA, h_RowA,
                                    nppw,
                                    h_Val, h_Col, h_Row, h_b, h_x,
                                    h_ValR, h_ColR, h_RowR,
                                    h_ValP, h_ColP, h_RowP,
                                    h_ValSR, h_ColSR, h_RowSR,
                                    OffsetSRx, OffsetSRy, OffsetSRz, SR, srnum,
                                    RankIlevel, NxIlevel, NyIlevel, NzIlevel,
                                    dx, dy, dz, nxA,
                                    nx, ny, nz, RankIlevel[0], &no_zero_number,
                                    R, fpeak, alpha_max,
                                    pml_thick,
                                    dampx, dampy, dampz,
                                    dampxdiff, dampydiff, dampzdiff,
                                    alphax, alphay, alphaz,
                                    alphax_diff, alphay_diff, alphaz_diff,
                                    deno_ex2, deno_ey2, deno_ez2,
                                    deno_ex3, deno_ey3, deno_ez3,
                                    dux, duy, duz,
                                    duxx, duyy, duzz,
                                    A, vel, omega, omegav_cofficient, coefficient,
                                    freq_slice,
                                    ns, xs, ys, zs, ixs, iys, izs,
                                    source_fre,iifreq[ifreq], df,
                                    frslicepath,logfp);

            // #pragma omp parallel for num_threads(mkl_parallel)
            // for(int ix=0; ix<nx; ix++)
            //     for(int iy=0; iy<ny; iy++)
            //         for(int iz=0; iz<nz; iz++)
            //         {
            //             freq_slice[ix][iy][iz].r=(float)h_x[ix*ny*nz+iy*nz+iz].r;
            //             freq_slice[ix][iy][iz].i=(float)h_x[ix*ny*nz+iy*nz+iz].i;
            //         }

            // sprintf(optname,"%s/frequency_slice_%d.bin",frslicepath,iifreq[ifreq]);
            // optfp=fopen(optname,"w");
            // fwrite(freq_slice[0][0],sizeof(complex),nx*ny*nz,optfp);
            // fclose(optfp);

            // #pragma omp parallel for num_threads(mkl_parallel)
            // for(int ix=0; ix<nx; ix++)
            // {
            //     for(int iy=0;iy<ny;iy++)
            //     {
            //         for(int iz=0;iz<nz;iz++)
            //         {
            //             slice3D[ix][iy][iz]=h_x[ix*ny*nz+iy*nz+iz].r;
            //         }
            //     }
            // }
            // sprintf(optname,"%s/complex_r_23.bin",frslicepath);
            // optfp=fopen(optname,"w");
            // fwrite(slice3D[0][0],sizeof(float),nx*ny*nz,optfp);
            // fclose(optfp);

            // ix=70;
            // {
            //     for(iy=0;iy<ny;iy++)
            //     {
            //         for(iz=0;iz<nz;iz++)
            //         {
            //             slice2D[iy][iz]=h_x[ix*ny*nz+iy*nz+iz].r;
            //         }
            //     }
            // }

            // sprintf(optname,"%s/complex_r2D_23.bin",frslicepath);
            // optfp=fopen(optname,"w");
            // fwrite(slice2D[0],sizeof(float),ny*nz,optfp);
            // fclose(optfp);

            
            // if(shift_laplace_option==1)
            // {
            //     sprintf(optname,"%s/norm2shiftlaplace_res23.bin",frslicepath);
            //     optfp=fopen(optname,"w");
            //     fwrite(norm2_res,sizeof(float),(nter_max+1),optfp);
            //     fclose(optfp);
            // }
            // else
            // {
            //     sprintf(optname,"%s/norm2_res23.bin",frslicepath);
            //     optfp=fopen(optname,"w");
            //     fwrite(norm2_res,sizeof(float),(nter_max+1),optfp);
            //     fclose(optfp);
            // }


            #pragma omp parallel for num_threads(mkl_parallel)
            for(int ix=0; ix<nx; ix++)
            {
                for(int iy=0;iy<ny;iy++)
                {
                    for(int iz=0;iz<nz;iz++)
                    {
                        slice3D[ix][iy][iz]=h_x[ix*ny*nz+iy*nz+iz].r;
                    }
                }
            }

            sprintf(optname,"%s/frequency_slice_%d.bin",frslicepath,iifreq[ifreq]);
            optfp=fopen(optname,"w");
            fwrite(freq_slice[0][0],sizeof(complex),nx*ny*nz,optfp);
            fclose(optfp);
            
            sprintf(optname,"%s/complex_r_%d.bin",frslicepath,iiifreq);
            optfp=fopen(optname,"w");
            fwrite(slice3D[0][0],sizeof(float),nx*ny*nz,optfp);
            fclose(optfp);

            ix=70;
            {
                for(iy=0;iy<ny;iy++)
                {
                    for(iz=0;iz<nz;iz++)
                    {
                        slice2D[iy][iz]=h_x[ix*ny*nz+iy*nz+iz].r;
                    }
                }
            }

            sprintf(optname,"%s/complex_r2D_%d.bin",frslicepath,iiifreq);
            optfp=fopen(optname,"w");
            fwrite(slice2D[0],sizeof(float),ny*nz,optfp);
            fclose(optfp);

            
            if(shift_laplace_option==1)
            {
                sprintf(optname,"%s/norm2shiftlaplace_res%d.bin",frslicepath,iiifreq);
                optfp=fopen(optname,"w");
                fwrite(norm2_res,sizeof(float),(nter_max+1),optfp);
                fclose(optfp);
            }
            else
            {
                sprintf(optname,"%s/norm2_res%d.bin",frslicepath,iiifreq);
                optfp=fopen(optname,"w");
                fwrite(norm2_res,sizeof(float),(nter_max+1),optfp);
                fclose(optfp);
            }


            


            for(ilevel = 1; ilevel < Level; ilevel++)
            {
                mkl_free(h_Val[ilevel]);
                mkl_free(h_Col[ilevel]);
                mkl_free(h_Row[ilevel]);
            }
        
        }

        iElaps =dsecnd() - iStart;
        break;
    }

    MPI_Barrier(comm);


    MPI_Barrier(comm);

    dblock=(int)(ns/(npros));
    if (myid==0)
        warn("ns = %d, number of processor = %d, dblock = %d",ns,npros,dblock);
    if ( myid<(ns%npros) )
    {
        {
            s_beg=myid*(dblock+1);
            s_end=(myid+1)*(dblock+1);
        }
    }
    else
    {
        s_beg=myid*dblock+ns%npros;
        s_end=(myid+1)*dblock + ns%npros;
    }
    MPI_Barrier(comm);
    warn("rank: %d, on processor %s. Begin source:%d, end source:%d",
            myid,proname,s_beg,s_end-1);
    if(op_shot==1)
    {
        // for (is=s_beg;is<s_end;++is)
        // {
        //     memset(hs[0], 0, sizeof(float)*nf*nrec);
        //     warn("Out put shotgather");

        // }
        is=0;
        Output_shotgather3d( mkl_parallel, nx, ny, nz, nt_fft, nfreq, dt, nfft, shot,
     is, xs, ys, zs, ixs, iys, izs,
     fx, fy, dx, dy, dz, fzr,
     tdelay,
     frslicepath, shotpath);
    }
    MPI_Barrier(comm);

    if(op_slice==1)
    {
        it=NINT(slice_t/dt);

        warn("it=%d,dt=%f",it,dt);

        iStart = dsecnd();
        Output_snapshot3d( mkl_parallel, nx, ny, nz, nt_fft, nfreq, it, dt, nfft, slice_is,
    frslicepath,slicepath);

        iElaps =dsecnd() - iStart;


        warn("iElaps=%f",iElaps);

    }

    free1float(norm2_res);


    free1int(NxIlevel);
    free1int(NyIlevel);
    free1int(NzIlevel);
    mkl_free(RankIlevel);

    for(ilevel=0;ilevel<Level-1;ilevel++)
    {
        mkl_free(h_ValP[ilevel]);
        mkl_free(h_ColP[ilevel]);
        mkl_free(h_RowP[ilevel]);

        mkl_free(h_ValR[ilevel]);
        mkl_free(h_ColR[ilevel]);
        mkl_free(h_RowR[ilevel]);

        mkl_free(h_ValSR[ilevel]);
        mkl_free(h_ColSR[ilevel]);
        mkl_free(h_RowSR[ilevel]);

    }

    mkl_free(h_Val[0]);
    mkl_free(h_Col[0]);
    mkl_free(h_Row[0]);

    free1int(OffsetSRx);
    free1int(OffsetSRy);
    free1int(OffsetSRz);

    free1float(SR);

    free1float(xs);
    free1float(ys);
    free1float(zs);
    free1int(ixs);
    free1int(iys);
    free1int(izs);

    free1float(xr);
    free1float(yr);
    free1float(zr);
    free1int(ixr);
    free1int(iyr);
    free1int(izr);

    free1float(source);
    free1complex(source_fre);

    mkl_free(h_b);
    mkl_free(h_x);

    free3float(vel);
    free3float(nppw);
    free3complex(A);


    free3float(dampx);
    free3float(dampy);
    free3float(dampz);
    free3float(dampxdiff);
    free3float(dampydiff);
    free3float(dampzdiff);

    free3float(alphax);
    free3float(alphay);
    free3float(alphaz);
    free3float(alphax_diff);
    free3float(alphay_diff);
    free3float(alphaz_diff);


    free3complex(deno_ex2);
    free3complex(deno_ey2);
    free3complex(deno_ez2);
    free3complex(deno_ex3);
    free3complex(deno_ey3);
    free3complex(deno_ez3);


    free1float(omegav_cofficient);
    free1complex(coefficient);

    free1float(dux);
    free1float(duy);
    free1float(duz);
    free1float(duxx);
    free1float(duyy);
    free1float(duzz);

    free3float(slice3D);
    free2float(slice2D);


    free3complex(freq_slice);
    free3float(shot);

    free1int(iifreq);

    if (myid==0)
    {
        //fprintf(logfp,"=========== Frequent_FDFD_forward 1.0 ===========\n");
        fprintf(logfp,"================ Compute Complete ===============\n");
        fclose(logfp);
    }

    MPI_Finalize();

    return(CWP_Exit());
}
