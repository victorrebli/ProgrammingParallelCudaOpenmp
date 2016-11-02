#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#define Tile_size 64

int numARows = 1024;   // number of rows in the matrix A
int numAColumns = 1024;  // number of columns in the matrix A
int numBRows = 1024;   // number of rows in the matrix B
int numBColumns = 1024;  // number of columns in the matrix B
int numCRows = 1024;  // number of rows in the matrix C (you have to set this)
int numCColumns = 1024; // number of columns in the matrix C (you have to set this)


__global__ void matrixMultiplyShared(float * A, float * B, float * C,
                                    int numARows, int numAColumns,
                                    int numBRows, int numBColumns,
                                    int numCRows, int numCColumns)
{
    __shared__ float sA[Tile_size][Tile_size];   // Tile size to store elements in shared memory
    __shared__ float sB[Tile_size][Tile_size];

    int Row = blockDim.y*blockIdx.y + threadIdx.y; //To generate ids of threads.
    int Col = blockDim.x*blockIdx.x + threadIdx.x;
    float Cvalue = 0.0;
    sA[threadIdx.y][threadIdx.x] = 0.0;
    sB[threadIdx.y][threadIdx.x] = 0.0;

    for (int k = 0; k < (((numAColumns - 1)/ Tile_size) + 1); k++)
    {
        if ( (Row < numARows) && (threadIdx.x + (k*Tile_size)) < numAColumns)//Copy Data to Tile from Matrix (Global Memory to Shared Memory)
        {
            sA[threadIdx.y][threadIdx.x] = A[(Row*numAColumns) + threadIdx.x + (k*Tile_size)];
        }
        else
        {
            sA[threadIdx.y][threadIdx.x] = 0.0;
        }
        if ( Col < numBColumns && (threadIdx.y + k*Tile_size) < numBRows)//Copy Data to Tile from Matrix (Global Memory to Shared Memory)
        {
            sB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + k*Tile_size)*numBColumns + Col];
        }
        else
        {
            sB[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();

        for (int j = 0; j < Tile_size; ++j)//Multiplying Elements present in tile
        {
            Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }
    }
    if (Row < numCRows && Col < numCColumns)//Saving Final result into Matrix C
    {
        C[Row*numCColumns + Col] = Cvalue;
    }
}
//*************************************************************
void Print_Mat(int Row,int Col,float * Mat)
{
 for(int i=0;i<Row*Col;i++)
   {
   printf("%f  ",*(Mat+i));

   if((i%Col)==0 )
    {
     printf("\n");
    }
   }
}//Function close

int main(int argc, char ** argv) {
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * hostComputedC;
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int tamanho = numARows * numAColumns;
    float time_1;
    cudaEvent_t start, stop;

    hostA = (float *) malloc(sizeof(float)*numARows*numAColumns);
    hostB = (float *) malloc(sizeof(float)*numBRows*numBColumns);

    for (int i = 0; i < numARows*numAColumns; i++)//Matrix Initialization
    {
        hostA[i]=((float)rand()/(float)(RAND_MAX)) * 3.0;
    }
    for (int i = 0; i < numBRows*numBColumns; i++)
    {
        hostB[i]=((float)rand()/(float)(RAND_MAX)) * 3.0;
    }

   
    numCRows = numARows;
    numCColumns = numBColumns;

    hostC = (float *) malloc(sizeof(float)*numCRows*numCColumns);
    hostComputedC = (float *) malloc(sizeof(float)*numCRows*numCColumns);

    
    cudaMalloc((void **)&deviceA, sizeof(float)*numARows*numAColumns);
    cudaMalloc((void **)&deviceB, sizeof(float)*numBRows*numBColumns);
    cudaMalloc((void **)&deviceC, sizeof(float)*numCRows*numCColumns);

    
    cudaMemcpy(deviceA, hostA, sizeof(float)*numARows*numAColumns, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, sizeof(float)*numBRows*numBColumns, cudaMemcpyHostToDevice);

   

    dim3 dimGrid((numCColumns/Tile_size) + 1, (numCRows/Tile_size) + 1, 1);//Number of Blocks required
    dim3 dimBlock(Tile_size, Tile_size, 1);//Number of threads in each block

    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start, NULL);
    int iter = 1000;
    for (int j =0; j<iter; j++)
    {
    //printf("chamada: %d \n", j);
    matrixMultiplyShared<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
    }

    cudaEventRecord( stop, NULL);
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time_1, start, stop );

    cudaError_t err1 = cudaPeekAtLastError();

    cudaDeviceSynchronize();

    
    cudaMemcpy(hostC, deviceC, sizeof(float)*numCRows*numCColumns, cudaMemcpyDeviceToHost);

    printf("Effective Bandwidth (GB/s): %f \n", 2*tamanho/time_1/1e6);
    float msecPerMatrixMul = time_1;
    double flopsPerMatrixMul = 2.0 * (double)tamanho;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul);

    // Free the GPU memory
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    //Free the Pointer Memory
    free(hostA);
    free(hostB);
    free(hostC);
    free(hostComputedC);

    return 0;
}
