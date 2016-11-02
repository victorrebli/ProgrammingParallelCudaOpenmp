#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

#define N 1024
#define TILE 16
#define MIN(a,b) (((a)<(b))?(a):(b))
//float A[N][N], B[N][N], C[N][N]; // declaring matrices of NxN size

void matriz_seq(double **A,double **B,double **C)
{
int i,j,k,y,x,z,l,l1,p,m;
srand ( time(NULL) );
for(l=0;l<N;l++) {
for(l1=0;l1<N;l1++) {
C[l][l1] = 0.0;
}
}
for ( i=0; i<N; i+=TILE )
        //printf("valor: %d", i);
        for ( j=0; j<N; j+=TILE )
            for ( k=0; k<N; k+=TILE )
                for ( y=i; y<i+TILE; y++ )
                    for ( x=j; x<j+TILE; x++ )
                        for ( z=k; z<k+TILE; z++ )
                            C[y][x] += A[y][z]*B[z][x];



}


void matriz_openmp(double **A,double **B,double **C)
{
int i,j,k,y,x,z,l,l1,p,m;
double sum;
srand ( time(NULL) );
for(l=0;l<N;l++) {
for(l1=0;l1<N;l1++) {
C[l][l1] = 0.0;
}
}
//#pragma omp paralled shared(A,B,C) private(i,j,k,y,x,z)
//{
//#pragma omp for
#pragma omp parallel for private(j,k,y,x,z,sum)
for ( i=0; i<N; i+=TILE ){
        for ( j=0; j<N; j+=TILE ){
       //#pragma parallel for reduction(+: C[y,i,MIN(y+TILE,N)][x,j,MIN(j+TILE,N)])
            for ( k=0; k<N; k+=TILE ){
                for ( y=i; y<MIN(i+TILE,N); y++ ){
                    for ( x=j; x<MIN(j+TILE,N); x++ ){
                        for ( z=k; z<MIN(k+TILE,N); z++ ){
                            sum += A[y][z]*B[z][x];
                        }
                            C[y][x] += sum;
  }
}
}
}
}
//}


/* for (i = 0; i < N; i += TILE)
                {
                        for (j = 0; j < N; j += TILE)
                        {
                                #pragma omp parallel for collapse(2)
                                for (x = 0; x < TILE; ++x)
                                {
                                        for (y = 0; y < TILE; ++y)
                                        {
                                                for (k = 0; k < N; ++k)
                                                {
                                                        #pragma omp critical
                                                        C[i + x][j + y] += A[i + x][k] * B[k][j + y];
                                                }
                                        }
                                }
                        }
                } */
}

int main ()
{
int i,j,k,y,x,z,l,l1,p,m,b;
int tamanho = N * N;
int iter = 1000;

FILE *arq;

/* DECLARING VARIABLES */
//int i, j, m; // indices for matrix multiplication
float t_1; // Execution time measures
clock_t c_1, c_2;
double **A, **B, **C;
   int  h;
   A = malloc (N * sizeof (double *));
   for (h = 0; h < N; ++h)
      A[h] = malloc (N * sizeof (double));

    B = malloc (N * sizeof (double *));
   for (h = 0; h < N; ++h)
      B[h] = malloc (N * sizeof (double));

    C = malloc (N * sizeof (double *));
   for (h = 0; h < N; ++h)
      C[h] = malloc (N * sizeof (double));

/* FILLING MATRICES WITH RANDOM NUMBERS */
srand ( time(NULL) );
for(l=0;l<N;l++) {
for(l1=0;l1<N;l1++) {
A[l][l1]= ((double)rand()/(double)(RAND_MAX)) * 3.0;
B[l][l1]= ((double)rand()/(double)(RAND_MAX)) * 3.0;
C[l][l1] = 0.0;
}
}
c_1=time(NULL); // time measure: start mm

printf("Entrando na matriz seq\n");
for ( b= 0; b<iter; b++)
{
printf("imprimindo valor de b: %d \n", b);
matriz_seq(A,B,C);
}

//#define TILE 64
/* for ( i=0; i<N; i+=TILE )
        for ( j=0; j<N; j+=TILE )
            for ( k=0; k<N; k+=TILE )

                for ( y=i; y<i+TILE; y++ )
                    for ( x=j; x<j+TILE; x++ )
                        for ( z=k; z<k+TILE; z++ )
                            C[y][x] += A[y][z]*B[z][x]; */




c_2=time(NULL); // time measure: end mm
t_1 = (float)(c_2-c_1); // time elapsed for job row-wise
arq = fopen("sequencial.txt","w");
fprintf(arq, "Execution time: %f \n", t_1);
fprintf(arq,"Effective Bandwidth (GB/s): %f \n", 2*tamanho/t_1/1e6);
float msecPerMatrixMul = t_1;
double flopsPerMatrixMul = 2.0 * (double)tamanho;
double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
fprintf(arq, "Performance= %.2f GFlop/s\n",
        gigaFlops);

fprintf(arq,"Time = %.3f msec \n",
        msecPerMatrixMul);

fclose(arq);
/*printf("Execution time: %f \n",t_1);
printf("Effective Bandwidth (GB/s): %f \n", 2*tamanho/t_1/1e6);
float msecPerMatrixMul = t_1;
double flopsPerMatrixMul = 2.0 * (double)tamanho;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul);*/
/* TERMINATE PROGRAM */

/*for(l=0;l<N;l++) {
for(l1=0;l1<N;l1++) {
//A[l][l1]= (rand()%10);
//B[l][l1]= (rand()%10);
C[l][l1] = 0.0;
}
}*/


//double start_time = omp_get_wtime();
c_1=time(NULL); // time measure: start mm

for( b = 0; b<iter; b++)
{
matriz_openmp(A,B,C);
}
/*
for ( i=0; i<N; i+=TILE )
        for ( j=0; j<N; j+=TILE )
            for ( k=0; k<N; k+=TILE )

                #pragma omp parallel
                #pragma omp parallel for private(x,z) shared(TILE)
                for ( y=i; y<i+TILE; y++ )
                    for ( x=j; x<j+TILE; x++ )
                        //C[y][x] = 0.0;
                        for ( z=k; z<k+TILE; z++ )
                            C[y][x] += A[y][z]*B[z][x]; */


//double time = omp_get_wtime() - start_time;
c_2=time(NULL); // time measure: end mm
t_1 = (float)(c_2-c_1); // time elapsed for job row-wise

arq = fopen("openmp.txt","w");
fprintf(arq, "Execution time: %f \n", t_1);
fprintf(arq,"Effective Bandwidth (GB/s): %f \n", 2*tamanho/t_1/1e6);
msecPerMatrixMul = t_1;
flopsPerMatrixMul = 2.0 * (double)tamanho;
gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
fprintf(arq, "Performance= %.2f GFlop/s\n",
        gigaFlops);

fprintf(arq,"Time = %.3f msec \n",
        msecPerMatrixMul);

fclose(arq);

/*printf("Execution time: %f \n",t_1);
printf("Effective Bandwidth (GB/s): %f \n", 2*tamanho/t_1/1e6);
msecPerMatrixMul = t_1;
flopsPerMatrixMul = 2.0 * (double)tamanho;
    gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul);*/

/* for (i=0; i<N; i++){
for(j=0; j<N; j++){

printf("%f \t", A[i][j]);

}
printf("\n");

}
printf("\n"); */
//printf("valor: %d", meu);


//int meu = 0;
/*for (i=0; i<N; i++){
for(j=0; j<N; j++){

printf("%f \t", B[i][j]);

}
printf("\n");

}
printf("\n"); */

//printf("valor: %d", meu);

//int meu = 0;
/* for (i=0; i<N; i++){
for(j=0; j<N; j++){

printf("%f \t", C[i][j]);

}
printf("\n");

}
printf("\n"); */
//printf("valor: %d", meu);

}
