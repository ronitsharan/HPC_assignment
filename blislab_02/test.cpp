// using openmp for around micro-kernel loop 
//loop unrolling of matrix C
//storing initial values of matrix C in registers

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits.h>
#include <immintrin.h>
#include <omp.h>
#include <time.h>


//Broadwell (Tsubame architecture) 16x6
#define GEMM_SIMD_ALIGN_SIZE 32   //step5/single/include/bl_config.h
#define SGEMM_MC 128  
#define SGEMM_NC 4080
#define SGEMM_KC 256
#define SGEMM_MR 16
#define SGEMM_NR 6

typedef unsigned long long dim_t;  //include/bl_sgemm_kernel.h

struct aux_t {
  float *b_next;
  float *b_next_s;
  char  *flag;
  int   pc;
  int   m;
  int   n;
};

//    sgemm_( "N", "N", &m, &n, &k, &alpha, XA, &lda, XB, &ldb, &beta, XC, &ldc );

extern "C" void sgemm_(char*, char*, int*, int*, int*, float*, float*,     
                       int*, float*, int*, float*, float*, int*);  //sgemm/bl_sgemm_ref.c for referring to BLAS


// sgemm/bl_sgemm_util.c **
float *bl_malloc_aligned(int m, int n, int size) {
  float *ptr;
  int err = posix_memalign((void**)&ptr,(size_t)GEMM_SIMD_ALIGN_SIZE, size*m*n); //**
  if (err) {
    printf( "bl_malloc_aligned(): posix_memalign() failures" );
    exit(1);
  }
  return ptr;
}

//--------------------------------------------------------------------------------------
//sgemm/my_dgemm.c
inline void packA_mcxkc_d(int m, int k, float *XA, int ldXA, int offseta, float *packA) {
  float *a_pntr[SGEMM_MR];
  for (int i=0; i<m; i++) {
    a_pntr[i] = XA + offseta + i;
  }
  for (int i=m; i<SGEMM_MR; i++) {
    a_pntr[i] = XA + offseta + 0;
  }
  for (int p=0; p<k; p++) {
    for (int i=0; i<SGEMM_MR; i++) {
      *packA = *a_pntr[i];
      packA++;
      a_pntr[i] = a_pntr[i] + ldXA;
    }
  }
}

//---------------------------------------------------------------------------------------

inline void packB_kcxnc_d(int n, int k, float *XB, int ldXB, int offsetb, float *packB) {
  float *b_pntr[SGEMM_NR];
  for (int j=0; j<n; j++) {
    b_pntr[j] = XB + ldXB * (offsetb + j);
  }
  for (int j=n; j<SGEMM_NR; j++) {
    b_pntr[j] = XB + ldXB * (offsetb + 0);
  }

  for (int p=0; p<k; p++) {
    for (int j=0; j<SGEMM_NR; j++) {
      *packB++ = *b_pntr[j]++;
    }
  }
}

//--------------------------------------------------------------------------------------

#if 1
//#include "micro_kernel.h"
#include "bl_sgemm_asm_16x6.h"
#else
void micro_kernel(int kc, float* A, float* B, float* C, dim_t ldc, aux_t* aux) {
  int nr = aux->n;
  int mr = aux->m;
  for (int kr=0; kr<kc; kr++) {
    for (int j=0; j<nr; j++) {
      for (int i=0; i<mr; i++) {
        C[j*ldc+i] += A[i*kc+kr] * B[j*kc+kr];
      }
    }
  }
}
#endif

//----------------------------------------------------------------------------------------
void macro_kernel(int mc, int nc, int kc, float *packA, float *packB, float *C, int ldc) {
  aux_t aux;
  aux.b_next = packB;
  int bl_ic_nt = 16;
  char *str = getenv( "BLISLAB_IC_NT" );
  if ( str != NULL ) {
    bl_ic_nt = (int)strtol(str, NULL, 10);
  }

#pragma omp parallel for num_threads( bl_ic_nt )
  for (int jr=0; jr<nc; jr+=SGEMM_NR) {            //2nd loop around micro-kernel
    aux.n = std::min(nc-jr, SGEMM_NR);
    for (int ir=0; ir<mc; ir+=SGEMM_MR) {          //1st loop around micro-kernel
      aux.m = std::min(mc-ir, SGEMM_MR);
      if (ir+SGEMM_MR >= mc) {
        aux.b_next += SGEMM_NR * kc;
      }
      micro_kernel(kc, &packA[ir*kc], &packB[jr*kc], &C[jr*ldc+ir], ldc, &aux);
    }                                              //1st loop around micro-kernel
  }                                                //2nd loop around micro-kernel 
}

//---------------------------------------------------------------------------------------------

void bl_sgemm(int m, int n, int k, float *XA, int lda, float *XB, int ldb, float *C, int ldc) {
  int bl_ic_nt = 16;    // ****multi_threads, 1 set in blislab

  char *str = getenv( "BLISLAB_IC_NT" );
  if ( str != NULL ) {
    bl_ic_nt = (int)strtol(str, NULL, 10);
  }
//Allocate packing buffers
  float * packA = bl_malloc_aligned(SGEMM_KC, (SGEMM_MC+1)*bl_ic_nt, sizeof(float));
  float * packB = bl_malloc_aligned(SGEMM_KC, (SGEMM_NC+1)         , sizeof(float)); //****bl_ic_nt insert

  for (int jc=0; jc<n; jc+=SGEMM_NC) {           //5-th loop around micro-kernel  
    int nc = std::min(n-jc, SGEMM_NC);
    for (int pc=0; pc<k; pc+=SGEMM_KC) {         //4-th loop around micro-kernel 
      int kc = std::min(k-pc, SGEMM_KC);

#pragma omp parallel for num_threads(bl_ic_nt)
      for (int j=0; j<nc; j+=SGEMM_NR) {
      
        packB_kcxnc_d(std::min(nc-j, SGEMM_NR), kc, &XB[pc], k, jc+j, &packB[j*kc]); 
      }
#pragma omp parallel for num_threads(bl_ic_nt)   //3-rd loop around micro-kernel
      for (int ic=0; ic<m; ic+=SGEMM_MC) {
        int tid = omp_get_thread_num();           //**
        int mc = std::min(m-ic, SGEMM_MC);
        for (int i=0; i<mc; i+=SGEMM_MR) {
          packA_mcxkc_d(std::min(mc-i, SGEMM_MR), kc, &XA[pc*lda], m, ic+i, &packA[tid*SGEMM_MC*kc+i*kc]);
        }
        macro_kernel(mc, nc, kc, packA+tid*SGEMM_MC*kc, packB, &C[jc*ldc+ic], ldc); 
      }       //End 3.rd loop around micro-kernel
    }         //End 4.th loop around micro-kernel
  }           //End 5.th loop around micro-kernel
  free(packA);
  free(packB);
}

#define A( i, j ) A[(j)*lda+(i)]
#define B( i, j ) B[(j)*ldb+(i)]
#define C( i, j ) C[(j)*ldc+(i)]
#define C_ref( i, j ) C_ref[(j)*ldc_ref+(i)]

int main(int argc, char *argv[]) {
  int m, n, k;
  sscanf(argv[1], "%d", &m);
  sscanf(argv[2], "%d", &n);
  sscanf(argv[3], "%d", &k);
  float alpha = 1.0, beta = 1.0;
  double ref_rectime, bl_sgemm_rectime;  //sgemm/test_bl_sgemm.c
  float * A = (float*)malloc(sizeof(float)*m*k);
  float * B = (float*)malloc(sizeof(float)*k*n);
  int lda = m;
  int ldb = k;
  int ldc = ((m-1) / SGEMM_MR + 1) * SGEMM_MR;
  int ldc_ref = m;
  float * C = bl_malloc_aligned(ldc, n+4, sizeof(float));
  float * C_ref = (float*)malloc(sizeof(float)*m*n);
  int nrepeats = 3;
  srand(time(NULL));
  for (int p=0; p<k; p++) {
    for (int i=0; i<m; i++) {
      A(i,p) = drand48();
    }
  }
  for (int j=0; j<n; j++) {
    for (int p=0; p<k; p++) {
      B(p,j) = drand48();
    }
  }

 
  for (int j=0; j<n; j++) {
    for (int i=0; i<m; i++) {
      C_ref(i,j) = 0.0;
      C(i,j) = 0.0;
      
    }
  }

//minor change using pointers 
/* float *cp;
  for (int j=0; j<n; j++) {
    cp = &C[j*ldc];             // point cp to top of the jth column
    for (int i=0; i<m; i++) {
      //C_ref(i,j) = 0.0;
      //C(i,j) = 0.0;
      *cp++ = 0.0;             // 
    }
  }
*/
//Loop unrolling with unrolling factor of 4
/*  float *cp;
  for (int j=0; j<n; j++) {
    cp = &C[j*ldc];             // point cp to top of the jth column
    for (int i=0; i<m; i+=4) {
      //C_ref(i,j) = 0.0;
      //C(i,j) = 0.0;
      *(cp+0) = 0.0;
      *(cp+1) = 0.0;
      *(cp+2) = 0.0;
      *(cp+3) = 0.0;
      cp += 4;
    }
  }*/

// storing matrix C in registers
/*  float *cp;
  for (int j=0; j<n; j++) {
    cp = &C[j*ldc];             // point cp to top of the jth column
    for (int i=0; i<m; i+=4) {
      register double c0=0.0, c1=0.0, c2=0.0, c3=0.0, c4=0.0, c5=0.0; 
      //C_ref(i,j) = 0.0;
      //C(i,j) = 0.0;
      *(cp+0) = c0;
      *(cp+1) = c1;
      *(cp+2) = c2;
      *(cp+3) = c3;
      *(cp+4) = c4;
      *(cp+5) = c5;
      cp += 6;
    }
  }
*/

  for (int i=0; i<nrepeats; i++) {
    double bl_sgemm_beg = omp_get_wtime();
    bl_sgemm(m, n, k, A, lda, B, ldb, C, ldc);      //bl_sgemm call
    double bl_sgemm_time = omp_get_wtime() - bl_sgemm_beg;
    if (i == 0) {
      bl_sgemm_rectime = bl_sgemm_time;
    } else {
      bl_sgemm_rectime = bl_sgemm_time < bl_sgemm_rectime ? bl_sgemm_time : bl_sgemm_rectime;
    }
  }
  for (int i=0; i<nrepeats; i++) {
    double ref_beg = omp_get_wtime();
    sgemm_( "N", "N", &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C_ref, &ldc_ref );
    double ref_time = omp_get_wtime() - ref_beg;
    if (i == 0) {
      ref_rectime = ref_time;
    } else {
      ref_rectime = ref_time < ref_rectime ? ref_time : ref_rectime;
    }
  }
  for (int i=0; i<m; i++) {
    for (int j=0; j<n; j++) {
      if (fabs(C(i,j) - C_ref(i,j)) > 1) {
        printf( "C[ %d ][ %d ] != C_ref, %E, %E\n", i, j, C(i,j), C_ref(i,j));
        break;
      }
    }
  }
  float flops = (m * n / (1000.0 * 1000.0 * 1000.0)) * (2 * k);
  printf("%5d\t %5d\t %5d\t %5.2lf\t\t %5.2lf\n",
         m, n, k, flops / bl_sgemm_rectime, flops / ref_rectime);
  free(A);
  free(B);
  free(C);
  free(C_ref);
  return 0;
}
