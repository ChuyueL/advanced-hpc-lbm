/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <mm_malloc.h>
#include <omp.h>
#include <string.h>
#include "/mnt/storage/software/languages/intel/intel-parallel_studio_xe_2020_u4/compilers_and_libraries_2020.4.304/linux/mpi/intel64/include/mpi.h"

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

typedef struct {
  float* speed0;
  float* speed1;
  float* speed2;
  float* speed3;
  float* speed4;
  float* speed5;
  float* speed6;
  float* speed7;
  float* speed8;

  //array of pointers
  //float* speeds[9];
} t_speed_soa;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed_soa* cells_ptr, t_speed_soa* tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, int rank, int nprocs);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int timestep(const t_param params, t_speed_soa* cells, t_speed_soa* tmp_cells, int* obstacles);
float timestep_vectorised(const t_param params, t_speed_soa* restrict cells, t_speed_soa* restrict tmp_cells, int* restrict obstacles);
float timestep_parallel(const t_param params, t_speed_soa* restrict cells, t_speed_soa* restrict tmp_cells, int* restrict obstacles);
float timestep_mpi(const t_param params, t_speed_soa* restrict cells, t_speed_soa* restrict tmp_cells, int* restrict obstacles, int rank, int nprocs, int work, int start, int end);
float timestep_mpi_2(const t_param params, t_speed_soa* restrict cells, t_speed_soa* restrict tmp_cells, int* restrict obstacles, int rank, int nprocs, int work, int start, int end);
float timestep_mpi_3(const t_param params, t_speed_soa* restrict cells, t_speed_soa* restrict tmp_cells, int* restrict obstacles, int rank, int nprocs, int work, int start, int end);


void initialise_buffers(const t_param params, t_speed_soa* global_cells, t_speed_soa* global_tmp_cells);

int accelerate_flow(const t_param params, t_speed_soa* restrict cells, int* obstacles);
int propagate(const t_param params, t_speed_soa* restrict cells, t_speed_soa* restrict tmp_cells);
int propagate_fast(const t_param params, t_speed_soa* restrict cells, t_speed_soa* restrict tmp_cells);


int rebound(const t_param params, t_speed_soa* restrict cells, t_speed_soa* restrict tmp_cells, int* obstacles);
int rebound_old(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);

int collision(const t_param params, t_speed_soa* restrict cells, t_speed_soa* restrict tmp_cells, int* obstacles);
int collision_old(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);

int write_values(const t_param params, t_speed_soa* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed_soa* cells_ptr, t_speed_soa* tmp_cells_ptr, t_speed_soa* global_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed_soa* restrict cells, int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed_soa* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

//int r;
/* int work;
int start;
int end; */
//int normal_work;
int top_neighbour_rank;
int bottom_neighbour_rank;
int accelerate_rank;
int accelerate_jj;

inline void swap(t_speed_soa** cells_ptr, t_speed_soa **tmp_cells_ptr) {
    t_speed_soa* temp = *cells_ptr;
    *cells_ptr = *tmp_cells_ptr;
    *tmp_cells_ptr = temp;
}

inline void swap_buffer(float **a, float **b) {
  float* temp = *a;
  *a = *b; 
  *b = temp;
}

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  int nprocs, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  //t_speed* cells     = NULL;    /* grid containing fluid densities */
  //t_speed* tmp_cells = NULL;    /* scratch space */
  t_speed_soa cells;
  t_speed_soa tmp_cells;
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;                                                             /* structure to hold elapsed time */
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* Total/init time starts here: initialise our data structures and load values from file */
  gettimeofday(&timstr, NULL);
  tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  init_tic=tot_tic;
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels, rank, nprocs);

  t_speed_soa* cells_ptr = &cells;
  t_speed_soa* tmp_cells_ptr = &tmp_cells;

  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;

  int r = params.ny % nprocs; //remainder
  int work;
  int start;
  int end;
  if (rank < r) {
    work = params.ny / nprocs + 1;
    start = rank * work;
    end = start + work;
  }
  else {
    work = params.ny / nprocs;
    start = rank * work + r;
    end = start + work;
  }
  int normal_work = params.ny / nprocs;

  if (rank != 0) {
    top_neighbour_rank = rank - 1;
  }
  else {
    top_neighbour_rank = nprocs - 1;
  }

  if (rank != nprocs - 1) {
    bottom_neighbour_rank = rank + 1;
  }
  else {
    bottom_neighbour_rank = 0;
  }

  if (normal_work >= 2) {
    accelerate_rank = nprocs - 1;
    accelerate_jj = work - 1;
  }
  else {
    accelerate_rank = nprocs - 2;
    accelerate_jj = work;
  }

  /* if (rank == 0) {
    printf("Ranks: %d\n", nprocs);
  } */


  for (int tt = 0; tt < params.maxIters; tt++)
  {
    
    av_vels[tt] = timestep_mpi_3(params, cells_ptr, tmp_cells_ptr, obstacles, rank, nprocs, work, start, end);
    swap(&cells_ptr, &tmp_cells_ptr);

    
#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }
  
  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic=comp_toc;

  // Collate data from ranks here 
  t_speed_soa global_cells;

  t_speed_soa* global_cells_ptr = &global_cells;
  //initialise_buffers(params, global_cells_ptr, global_tmp_cells_ptr);

  int sendwork = work * params.nx;
  int* recvcounts = (int*)malloc(sizeof(int) * nprocs);

  for (size_t i = 0; i < nprocs; i++)
  {
    if (i < r) {
      recvcounts[i] = (normal_work + 1) * params.nx;
    }
    else {
      recvcounts[i] = normal_work * params.nx;
    }
  }

  int* displacements = (int*)malloc(sizeof(int) * nprocs);
  displacements[0] = 0;

  for (size_t i = 1; i < nprocs; i++)
  {
    displacements[i] = displacements[i-1] + recvcounts[i-1];
  }
  
  /* printf("Displacements: ");
  for (size_t i = 0; i < nprocs; i++)
  {
    printf("%d ", displacements[i]);
  }
  printf("\n");

  printf("Rank %d\n", rank); */

 /*  if (rank == 0) {
    for (size_t jj = 1; jj < work + 1; jj++)
    {
      for (size_t ii = 0; ii < params.nx; ii++)
      {
        if(obstacles[(start - 1) * params.nx + ii + jj*params.nx]) printf("0 ");
        else printf("1 ");
      }
      printf("\n");
    }
  } */
  
  
  float* sendbuf = &cells_ptr->speed0[params.nx];
  float* recvbuf0 = (float*)_mm_malloc(sizeof(float) * params.nx * params.ny, 64);

  MPI_Gatherv(sendbuf, sendwork, MPI_FLOAT, recvbuf0, recvcounts, displacements, MPI_FLOAT, 0, MPI_COMM_WORLD);
  
  global_cells_ptr->speed0 = recvbuf0;

  sendbuf = &cells_ptr->speed1[params.nx];
  float* recvbuf1 = (float*)_mm_malloc(sizeof(float) * params.nx * params.ny, 64);
  MPI_Gatherv(sendbuf, sendwork, MPI_FLOAT, recvbuf1, recvcounts, displacements, MPI_FLOAT, 0, MPI_COMM_WORLD);

  global_cells_ptr->speed1 = recvbuf1;

  sendbuf = &cells_ptr->speed2[params.nx];
  float* recvbuf2 = (float*)_mm_malloc(sizeof(float) * params.nx * params.ny, 64);
  MPI_Gatherv(sendbuf, sendwork, MPI_FLOAT, recvbuf2, recvcounts, displacements, MPI_FLOAT, 0, MPI_COMM_WORLD);

  global_cells_ptr->speed2 = recvbuf2;

  sendbuf = &cells_ptr->speed3[params.nx];
  float* recvbuf3 = (float*)_mm_malloc(sizeof(float) * params.nx * params.ny, 64);
  MPI_Gatherv(sendbuf, sendwork, MPI_FLOAT, recvbuf3, recvcounts, displacements, MPI_FLOAT, 0, MPI_COMM_WORLD);

  global_cells_ptr->speed3 = recvbuf3;

  sendbuf = &cells_ptr->speed4[params.nx];
  float* recvbuf4 = (float*)_mm_malloc(sizeof(float) * params.nx * params.ny, 64);
  MPI_Gatherv(sendbuf, sendwork, MPI_FLOAT, recvbuf4, recvcounts, displacements, MPI_FLOAT, 0, MPI_COMM_WORLD);

  global_cells_ptr->speed4 = recvbuf4;

  sendbuf = &cells_ptr->speed5[params.nx];
  float* recvbuf5 = (float*)_mm_malloc(sizeof(float) * params.nx * params.ny, 64);
  MPI_Gatherv(sendbuf, sendwork, MPI_FLOAT, recvbuf5, recvcounts, displacements, MPI_FLOAT, 0, MPI_COMM_WORLD);

  global_cells_ptr->speed5 = recvbuf5;

  sendbuf = &cells_ptr->speed6[params.nx];
  float* recvbuf6 = (float*)_mm_malloc(sizeof(float) * params.nx * params.ny, 64);
  MPI_Gatherv(sendbuf, sendwork, MPI_FLOAT, recvbuf6, recvcounts, displacements, MPI_FLOAT, 0, MPI_COMM_WORLD);
  global_cells_ptr->speed6 = recvbuf6;

  sendbuf = &cells_ptr->speed7[params.nx];
  float* recvbuf7 = (float*)_mm_malloc(sizeof(float) * params.nx * params.ny, 64);
  MPI_Gatherv(sendbuf, sendwork, MPI_FLOAT, recvbuf7, recvcounts, displacements, MPI_FLOAT, 0, MPI_COMM_WORLD);

  global_cells_ptr->speed7 = recvbuf7;

  sendbuf = &cells_ptr->speed8[params.nx];
  float* recvbuf8 = (float*)_mm_malloc(sizeof(float) * params.nx * params.ny, 64);
  MPI_Gatherv(sendbuf, sendwork, MPI_FLOAT, recvbuf8, recvcounts, displacements, MPI_FLOAT, 0, MPI_COMM_WORLD);

  global_cells_ptr->speed8 = recvbuf8;

  /* Total/collate time stops here.*/
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;
  
  /* if (rank == 0) {
    for (size_t jj = 0; jj < params.ny; jj++)
    {
      for (size_t ii = 0; ii < params.nx; ii++)
      {
        printf("%f ", global_cells_ptr->speed3[ii + jj * params.nx]);
      }
      printf("\n");
    }
  } */
  
  

  if (rank == 0) {
    /* write final values and free memory */
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, global_cells_ptr, obstacles));
    printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
    printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
    printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
    printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
    write_values(params, global_cells_ptr, obstacles, av_vels);
    
  }
  
  finalise(&params, cells_ptr, tmp_cells_ptr, global_cells_ptr, &obstacles, &av_vels);
  MPI_Finalize();
  return EXIT_SUCCESS;
}

int timestep(const t_param params, t_speed_soa* cells, t_speed_soa* tmp_cells, int* obstacles)
{
  accelerate_flow(params, cells, obstacles);
  propagate_fast(params, cells, tmp_cells);
  rebound(params, cells, tmp_cells, obstacles);
  collision(params, cells, tmp_cells, obstacles);
  return EXIT_SUCCESS;
}

void initialise_buffers(const t_param params, t_speed_soa* global_cells, t_speed_soa* global_tmp_cells) {
  global_cells->speed0 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  global_cells->speed1 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  global_cells->speed2 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  global_cells->speed3 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  global_cells->speed4 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  global_cells->speed5 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  global_cells->speed6 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  global_cells->speed7 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  global_cells->speed8 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);

  global_tmp_cells->speed0 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  global_tmp_cells->speed1 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  global_tmp_cells->speed2 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  global_tmp_cells->speed3 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  global_tmp_cells->speed4 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  global_tmp_cells->speed5 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  global_tmp_cells->speed6 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  global_tmp_cells->speed7 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  global_tmp_cells->speed8 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);

}

inline float timestep_mpi(const t_param params, t_speed_soa* restrict cells, t_speed_soa* restrict tmp_cells, int* obstacles, int rank, int nprocs, int work, int start, int end) {
  //Exchange halo regions
  int tag = 0;
  MPI_Status status;

  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

    /* modify the 2nd row of the grid */

  int jj = accelerate_jj;
  
  if (rank == accelerate_rank) {
    __assume_aligned(cells->speed0, 64);
    __assume_aligned(cells->speed1, 64);
    __assume_aligned(cells->speed2, 64);
    __assume_aligned(cells->speed3, 64);
    __assume_aligned(cells->speed4, 64);
    __assume_aligned(cells->speed5, 64);
    __assume_aligned(cells->speed6, 64);
    __assume_aligned(cells->speed7, 64);
    __assume_aligned(cells->speed8, 64);

    __assume_aligned(tmp_cells->speed0, 64);
    __assume_aligned(tmp_cells->speed1, 64);
    __assume_aligned(tmp_cells->speed2, 64);
    __assume_aligned(tmp_cells->speed3, 64);
    __assume_aligned(tmp_cells->speed4, 64);
    __assume_aligned(tmp_cells->speed5, 64);
    __assume_aligned(tmp_cells->speed6, 64);
    __assume_aligned(tmp_cells->speed7, 64);
    __assume_aligned(tmp_cells->speed8, 64);

    __assume((params.nx) % 2 == 0);
    __assume((params.nx) % 4 == 0);
    __assume((params.nx) % 8 == 0);
    __assume((params.nx) % 16 == 0);
    __assume((params.nx) % 32 == 0);
    __assume((params.nx) % 64 == 0);
    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* if the cell is not occupied and
      ** we don't send a negative density */
      if (!obstacles[(start - 1) * params.nx + ii + jj*params.nx]
          && (cells->speed3[ii + jj*params.nx] - w1) > 0.f
          && (cells->speed6[ii + jj*params.nx] - w2) > 0.f
          && (cells->speed7[ii + jj*params.nx] - w2) > 0.f)
      {
        /* increase 'east-side' densities */
        cells->speed1[ii + jj*params.nx] += w1;
        cells->speed5[ii + jj*params.nx] += w2;
        cells->speed8[ii + jj*params.nx] += w2;
        /* decrease 'west-side' densities */
        cells->speed3[ii + jj*params.nx] -= w1;
        cells->speed6[ii + jj*params.nx] -= w2;
        cells->speed7[ii + jj*params.nx] -= w2;
      }
    }
  }

  //Send to top and receive from bottom

  /* if (rank != 0) {
    float* sendbuf = &cells->speed0[start * params.nx]; //top row
    float* recvbuf;
    MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, rank - 1, tag, recvbuf, params.nx, MPI_FLOAT, rank - 1, tag, MPI_COMM_WORLD, &status);
  }
  else {
    float* sendbuf = &cells->speed0[0];
    float* recvbuf;
    MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, nprocs - 1, tag, recvbuf, params.nx, MPI_FLOAT, nprocs - 1, tag, MPI_COMM_WORLD, &status);

  } */

  float* sendbuf = &cells->speed0[params.nx]; //top row
  float* recvbuf = (float*)_mm_malloc(sizeof(float) * params.nx, 64);
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  
  __assume_aligned(cells->speed0, 64);
  __assume_aligned(recvbuf, 64);

  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);

  /* for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed0[i + (work + 1) * params.nx] = recvbuf[i];
  } */

  memcpy((void*)&cells->speed0[(work + 1) * params.nx], (void*)&recvbuf[0], sizeof(float)*params.nx);

  sendbuf = &cells->speed1[params.nx]; //top row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed1, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);

  /* for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed1[i + (work + 1) * params.nx] = recvbuf[i];
  } */

  memcpy((void*)&cells->speed1[(work + 1) * params.nx], (void*)&recvbuf[0], sizeof(float)*params.nx);

  sendbuf = &cells->speed2[params.nx]; //top row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed2, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);

  /* for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed2[i + (work + 1) * params.nx] = recvbuf[i];
  } */
  memcpy((void*)&cells->speed2[(work + 1) * params.nx], (void*)&recvbuf[0], sizeof(float)*params.nx);

  sendbuf = &cells->speed3[params.nx]; //top row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed3, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);

  /* #pragma omp simd
  for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed3[i + (work + 1) * params.nx] = recvbuf[i];
  } */
  memcpy((void*)&cells->speed3[(work + 1) * params.nx], (void*)&recvbuf[0], sizeof(float)*params.nx);

  sendbuf = &cells->speed4[params.nx]; //top row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed4, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);

  /* #pragma omp simd
  for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed4[i + (work + 1) * params.nx] = recvbuf[i];
  } */
  memcpy((void*)&cells->speed4[(work + 1) * params.nx], (void*)&recvbuf[0], sizeof(float)*params.nx);

  sendbuf = &cells->speed5[params.nx]; //top row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed5, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);

  /* #pragma omp simd
  for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed5[i + (work + 1) * params.nx] = recvbuf[i];
  } */
  memcpy((void*)&cells->speed5[(work + 1) * params.nx], (void*)&recvbuf[0], sizeof(float)*params.nx);

  sendbuf = &cells->speed6[params.nx]; //top row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed6, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);

  /* #pragma omp simd
  for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed6[i + (work + 1) * params.nx] = recvbuf[i];
  } */
  memcpy((void*)&cells->speed6[(work + 1) * params.nx], (void*)&recvbuf[0], sizeof(float)*params.nx);

  sendbuf = &cells->speed7[params.nx]; //top row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed7, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);

  /* #pragma omp simd
  for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed7[i + (work + 1) * params.nx] = recvbuf[i];
  } */

  memcpy((void*)&cells->speed7[(work + 1) * params.nx], (void*)&recvbuf[0], sizeof(float)*params.nx);

  sendbuf = &cells->speed8[params.nx]; //top row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed8, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);

  /* #pragma omp simd
  for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed8[i + (work + 1) * params.nx] = recvbuf[i];
  } */
  memcpy((void*)&cells->speed8[(work + 1) * params.nx], (void*)&recvbuf[0], sizeof(float)*params.nx);

  //Send to bottom and receive from top
  sendbuf = &cells->speed0[work * params.nx]; //bottom row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed0, 64);
  __assume_aligned(recvbuf, 64);

  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);
  
  /* #pragma omp simd
  for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed0[i] = recvbuf[i];
  } */
  memcpy((void*)&cells->speed0[0], (void*)&recvbuf[0], sizeof(float) * params.nx);

  sendbuf = &cells->speed1[work * params.nx]; //bottom row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed1, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);
  
  /* #pragma omp simd
  for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed1[i] = recvbuf[i];
  } */
  memcpy((void*)&cells->speed1[0], (void*)&recvbuf[0], sizeof(float) * params.nx);


  sendbuf = &cells->speed2[work * params.nx]; //bottom row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed2, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);
  
  /* #pragma omp simd
  for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed2[i] = recvbuf[i];
  } */
  memcpy((void*)&cells->speed2[0], (void*)&recvbuf[0], sizeof(float) * params.nx);

  sendbuf = &cells->speed3[work * params.nx]; //bottom row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed3, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);
  
  /* #pragma omp simd
  for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed3[i] = recvbuf[i];
  } */
  memcpy((void*)&cells->speed3[0], (void*)&recvbuf[0], sizeof(float) * params.nx);

  sendbuf = &cells->speed4[work * params.nx]; //bottom row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed4, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);
  
  /* #pragma omp simd
  for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed4[i] = recvbuf[i];
  } */
  memcpy((void*)&cells->speed4[0], (void*)&recvbuf[0], sizeof(float) * params.nx);

  sendbuf = &cells->speed5[work * params.nx]; //bottom row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed5, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);
  
  /* #pragma omp simd
  for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed5[i] = recvbuf[i];
  } */
  memcpy((void*)&cells->speed5[0], (void*)&recvbuf[0], sizeof(float) * params.nx);

  sendbuf = &cells->speed6[work * params.nx]; //bottom row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed6, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);
  
  /* #pragma omp simd
  for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed6[i] = recvbuf[i];
  } */
  memcpy((void*)&cells->speed6[0], (void*)&recvbuf[0], sizeof(float) * params.nx);

  sendbuf = &cells->speed7[work * params.nx]; //bottom row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed7, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);
  
  /* #pragma omp simd
  for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed7[i] = recvbuf[i];
  } */
  memcpy((void*)&cells->speed7[0], (void*)&recvbuf[0], sizeof(float) * params.nx);

  sendbuf = &cells->speed8[work * params.nx]; //bottom row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed8, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);
  
  /* #pragma omp simd
  for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed8[i] = recvbuf[i];
  } */
  memcpy((void*)&cells->speed8[0], (void*)&recvbuf[0], sizeof(float) * params.nx);

  //Compute
  int   tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;
  

  /* loop over _all_ cells */
  for (int jj = 1; jj < work + 1; jj++)
  {
    //printf("%d \n", omp_get_num_threads());
    __assume_aligned(cells->speed0, 64);
    __assume_aligned(cells->speed1, 64);
    __assume_aligned(cells->speed2, 64);
    __assume_aligned(cells->speed3, 64);
    __assume_aligned(cells->speed4, 64);
    __assume_aligned(cells->speed5, 64);
    __assume_aligned(cells->speed6, 64);
    __assume_aligned(cells->speed7, 64);
    __assume_aligned(cells->speed8, 64);

    __assume_aligned(tmp_cells->speed0, 64);
    __assume_aligned(tmp_cells->speed1, 64);
    __assume_aligned(tmp_cells->speed2, 64);
    __assume_aligned(tmp_cells->speed3, 64);
    __assume_aligned(tmp_cells->speed4, 64);
    __assume_aligned(tmp_cells->speed5, 64);
    __assume_aligned(tmp_cells->speed6, 64);
    __assume_aligned(tmp_cells->speed7, 64);
    __assume_aligned(tmp_cells->speed8, 64);

    __assume((params.nx) % 2 == 0);
    __assume((params.nx) % 4 == 0);
    __assume((params.nx) % 8 == 0);
    __assume((params.nx) % 16 == 0);
    __assume((params.nx) % 32 == 0);
    __assume((params.nx) % 64 == 0);

    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */

      const int y_n = jj + 1;
      const int x_e = (ii == params.nx - 1) ? 0 : ii + 1;
      const int y_s = jj - 1;
      const int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);

      if (obstacles[(start - 1) * params.nx + jj*params.nx + ii]) {
        tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx];
        tmp_cells->speed1[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx];
        tmp_cells->speed2[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx];
        tmp_cells->speed3[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx];
        tmp_cells->speed4[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx];
        tmp_cells->speed5[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx];
        tmp_cells->speed6[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx];
        tmp_cells->speed7[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx];
        tmp_cells->speed8[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx];
      }
      else {
        const float c_sq = 1.f / 3.f; /* square of speed of sound */
        const float w0 = 4.f / 9.f;  /* weighting factor */
        const float w1 = 1.f / 9.f;  /* weighting factor */
        const float w2 = 1.f / 36.f; /* weighting factor */
        /* compute local density total */
        float local_density = 0.f;

        local_density += cells->speed0[ii + jj*params.nx];
        local_density += cells->speed1[x_w + jj*params.nx];
        local_density += cells->speed2[ii + y_s*params.nx];
        local_density += cells->speed3[x_e + jj*params.nx];
        local_density += cells->speed4[ii + y_n*params.nx];
        local_density += cells->speed5[x_w + y_s*params.nx];
        local_density += cells->speed6[x_e + y_s*params.nx];
        local_density += cells->speed7[x_e + y_n*params.nx];
        local_density += cells->speed8[x_w + y_n*params.nx];
      
        /* compute x velocity component */
        float u_x = (cells->speed1[x_w + jj*params.nx]
                      + cells->speed5[x_w + y_s*params.nx]
                      + cells->speed8[x_w + y_n*params.nx]
                      - (cells->speed3[x_e + jj*params.nx]
                          + cells->speed6[x_e + y_s*params.nx]
                          + cells->speed7[x_e + y_n*params.nx]))
                      / local_density;
        /* compute y velocity component */
        float u_y = (cells->speed2[ii + y_s*params.nx]
                      + cells->speed5[x_w + y_s*params.nx]
                      + cells->speed6[x_e + y_s*params.nx]
                      - (cells->speed4[ii + y_n*params.nx]
                          + cells->speed7[x_e + y_n*params.nx]
                          + cells->speed8[x_w + y_n*params.nx]))
                      / local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u[NSPEEDS];
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */

        /* equilibrium densities */
        float d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */
        d_equ[0] = w0 * local_density
                    * (1.f - u_sq / (2.f * c_sq));
        /* axis speeds: weight w1 */
        d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                          + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                          + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                          + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                          + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        /* diagonal speeds: weight w2 */
        d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                          + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                          + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                          + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                          + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));

        /* relaxation step */

        tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx]
                                                  + params.omega
                                                  * (d_equ[0] - cells->speed0[ii + jj*params.nx]);

        tmp_cells->speed1[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx]
                                                  + params.omega
                                                  * (d_equ[1] - cells->speed1[x_w + jj*params.nx]);

        tmp_cells->speed2[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx]
                                                  + params.omega
                                                  * (d_equ[2] - cells->speed2[ii + y_s*params.nx]);                                          
        
        tmp_cells->speed3[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx]
                                                  + params.omega
                                                  * (d_equ[3] - cells->speed3[x_e + jj*params.nx]);

        tmp_cells->speed4[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx]
                                                  + params.omega
                                                  * (d_equ[4] - cells->speed4[ii + y_n*params.nx]);
        
        tmp_cells->speed5[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx]
                                                  + params.omega
                                                  * (d_equ[5] - cells->speed5[x_w + y_s*params.nx]);
        
        tmp_cells->speed6[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx]
                                                  + params.omega
                                                  * (d_equ[6] - cells->speed6[x_e + y_s*params.nx]);
        
        tmp_cells->speed7[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx]
                                                  + params.omega
                                                  * (d_equ[7] - cells->speed7[x_e + y_n*params.nx]);
        
        tmp_cells->speed8[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx]
                                                  + params.omega
                                                  * (d_equ[8] - cells->speed8[x_w + y_n*params.nx]);


        /* local density total */
        local_density = 0.f;

        local_density += tmp_cells->speed0[ii + jj*params.nx];
        local_density += tmp_cells->speed1[ii + jj*params.nx];
        local_density += tmp_cells->speed2[ii + jj*params.nx];
        local_density += tmp_cells->speed3[ii + jj*params.nx];
        local_density += tmp_cells->speed4[ii + jj*params.nx];
        local_density += tmp_cells->speed5[ii + jj*params.nx];
        local_density += tmp_cells->speed6[ii + jj*params.nx];
        local_density += tmp_cells->speed7[ii + jj*params.nx];
        local_density += tmp_cells->speed8[ii + jj*params.nx];

        /* x-component of velocity */
        u_x = (tmp_cells->speed1[ii + jj*params.nx]
                      + tmp_cells->speed5[ii + jj*params.nx]
                      + tmp_cells->speed8[ii + jj*params.nx]
                      - (tmp_cells->speed3[ii + jj*params.nx]
                          + tmp_cells->speed6[ii + jj*params.nx]
                          + tmp_cells->speed7[ii + jj*params.nx]))
                      / local_density;
        /* compute y velocity component */
        u_y = (tmp_cells->speed2[ii + jj*params.nx]
                      + tmp_cells->speed5[ii + jj*params.nx]
                      + tmp_cells->speed6[ii + jj*params.nx]
                      - (tmp_cells->speed4[ii + jj*params.nx]
                          + tmp_cells->speed7[ii + jj*params.nx]
                          + tmp_cells->speed8[ii + jj*params.nx]))
                      / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
      
    }
  }

  int local_tot_cells = tot_cells;
  int global_tot_cells = 0;
  float local_tot_u = tot_u;
  float global_tot_u = 0.0;

  MPI_Reduce(&local_tot_cells, &global_tot_cells, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_tot_u, &global_tot_u, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

  /* if (rank == 0) {
    printf("%f\n", global_tot_u/(float)global_tot_cells);
    printf("global tot_u: %f\n", global_tot_u);
    printf("global tot_cells: %d\n", global_tot_cells);
  } */
  return global_tot_u/(float)global_tot_cells;

}

inline float timestep_mpi_2(const t_param params, t_speed_soa* restrict cells, t_speed_soa* restrict tmp_cells, int* obstacles, int rank, int nprocs, int work, int start, int end) {
  //Exchange halo regions
  int tag = 0;
  MPI_Status status;

  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

    /* modify the 2nd row of the grid */
  //int jj = params.ny - 2;  
  int jj = accelerate_jj;

  
  if (rank == accelerate_rank) {
    __assume_aligned(cells->speed0, 64);
    __assume_aligned(cells->speed1, 64);
    __assume_aligned(cells->speed2, 64);
    __assume_aligned(cells->speed3, 64);
    __assume_aligned(cells->speed4, 64);
    __assume_aligned(cells->speed5, 64);
    __assume_aligned(cells->speed6, 64);
    __assume_aligned(cells->speed7, 64);
    __assume_aligned(cells->speed8, 64);

    __assume_aligned(tmp_cells->speed0, 64);
    __assume_aligned(tmp_cells->speed1, 64);
    __assume_aligned(tmp_cells->speed2, 64);
    __assume_aligned(tmp_cells->speed3, 64);
    __assume_aligned(tmp_cells->speed4, 64);
    __assume_aligned(tmp_cells->speed5, 64);
    __assume_aligned(tmp_cells->speed6, 64);
    __assume_aligned(tmp_cells->speed7, 64);
    __assume_aligned(tmp_cells->speed8, 64);

    __assume((params.nx) % 2 == 0);
    __assume((params.nx) % 4 == 0);
    __assume((params.nx) % 8 == 0);
    __assume((params.nx) % 16 == 0);
    __assume((params.nx) % 32 == 0);
    __assume((params.nx) % 64 == 0);
    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* if the cell is not occupied and
      ** we don't send a negative density */
      if (!obstacles[(start - 1) * params.nx + ii + jj*params.nx]
          && (cells->speed3[ii + jj*params.nx] - w1) > 0.f
          && (cells->speed6[ii + jj*params.nx] - w2) > 0.f
          && (cells->speed7[ii + jj*params.nx] - w2) > 0.f)
      {
        /* increase 'east-side' densities */
        cells->speed1[ii + jj*params.nx] += w1;
        cells->speed5[ii + jj*params.nx] += w2;
        cells->speed8[ii + jj*params.nx] += w2;
        /* decrease 'west-side' densities */
        cells->speed3[ii + jj*params.nx] -= w1;
        cells->speed6[ii + jj*params.nx] -= w2;
        cells->speed7[ii + jj*params.nx] -= w2;
      }
    }
  }

  //Send to top and receive from bottom

  float* sendbuf = &cells->speed0[params.nx]; //top row
  float* recvbuf = (float*)_mm_malloc(sizeof(float) * params.nx, 64);
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  
  __assume_aligned(cells->speed0, 64);
  __assume_aligned(recvbuf, 64);

  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);

  /* #pragma omp simd
  for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed0[i + (work + 1) * params.nx] = recvbuf[i];
  } */
  memcpy((void*)&cells->speed0[(work + 1) * params.nx], (void*)&recvbuf[0], sizeof(float)*params.nx);

  sendbuf = &cells->speed1[params.nx]; //top row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed1, 64);
  __assume_aligned(recvbuf, 64);

  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);

/*   #pragma omp simd
  for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed1[i + (work + 1) * params.nx] = recvbuf[i];
  }
 */
  memcpy((void*)&cells->speed1[(work + 1) * params.nx], (void*)&recvbuf[0], sizeof(float)*params.nx);

  sendbuf = &cells->speed2[params.nx]; //top row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed2, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);

  /* #pragma omp simd
  for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed2[i + (work + 1) * params.nx] = recvbuf[i];
  } */
  memcpy((void*)&cells->speed2[(work + 1) * params.nx], (void*)&recvbuf[0], sizeof(float)*params.nx);

  sendbuf = &cells->speed3[params.nx]; //top row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed3, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);

  /* #pragma omp simd
  for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed3[i + (work + 1) * params.nx] = recvbuf[i];
  } */
  memcpy((void*)&cells->speed3[(work + 1) * params.nx], (void*)&recvbuf[0], sizeof(float)*params.nx);

  sendbuf = &cells->speed4[params.nx]; //top row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed4, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);

  /* #pragma omp simd
  for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed4[i + (work + 1) * params.nx] = recvbuf[i];
  } */
  memcpy((void*)&cells->speed4[(work + 1) * params.nx], (void*)&recvbuf[0], sizeof(float)*params.nx);

  sendbuf = &cells->speed5[params.nx]; //top row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed5, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);

  /* #pragma omp simd
  for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed5[i + (work + 1) * params.nx] = recvbuf[i];
  } */
  memcpy((void*)&cells->speed5[(work + 1) * params.nx], (void*)&recvbuf[0], sizeof(float)*params.nx);

  sendbuf = &cells->speed6[params.nx]; //top row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed6, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);

  /* #pragma omp simd
  for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed6[i + (work + 1) * params.nx] = recvbuf[i];
  } */
  memcpy((void*)&cells->speed6[(work + 1) * params.nx], (void*)&recvbuf[0], sizeof(float)*params.nx);

  sendbuf = &cells->speed7[params.nx]; //top row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed7, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);

  /* #pragma omp simd
  for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed7[i + (work + 1) * params.nx] = recvbuf[i];
  } */

  memcpy((void*)&cells->speed7[(work + 1) * params.nx], (void*)&recvbuf[0], sizeof(float)*params.nx);

  sendbuf = &cells->speed8[params.nx]; //top row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed8, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);

  /* #pragma omp simd
  for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed8[i + (work + 1) * params.nx] = recvbuf[i];
  } */
  memcpy((void*)&cells->speed8[(work + 1) * params.nx], (void*)&recvbuf[0], sizeof(float)*params.nx);

  //Send to bottom and receive from top
  sendbuf = &cells->speed0[work * params.nx]; //bottom row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed0, 64);
  __assume_aligned(recvbuf, 64);

  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);
  
  /* #pragma omp simd
  for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed0[i] = recvbuf[i];
  } */
  memcpy((void*)&cells->speed0[0], (void*)&recvbuf[0], sizeof(float) * params.nx);

  sendbuf = &cells->speed1[work * params.nx]; //bottom row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed1, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);
  
  /* #pragma omp simd
  for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed1[i] = recvbuf[i];
  } */
  memcpy((void*)&cells->speed1[0], (void*)&recvbuf[0], sizeof(float) * params.nx);


  sendbuf = &cells->speed2[work * params.nx]; //bottom row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed2, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);
  
  /* #pragma omp simd
  for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed2[i] = recvbuf[i];
  } */
  memcpy((void*)&cells->speed2[0], (void*)&recvbuf[0], sizeof(float) * params.nx);

  sendbuf = &cells->speed3[work * params.nx]; //bottom row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed3, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);
  
  /* #pragma omp simd
  for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed3[i] = recvbuf[i];
  } */
  memcpy((void*)&cells->speed3[0], (void*)&recvbuf[0], sizeof(float) * params.nx);

  sendbuf = &cells->speed4[work * params.nx]; //bottom row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed4, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);
  
  /* #pragma omp simd
  for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed4[i] = recvbuf[i];
  } */
  memcpy((void*)&cells->speed4[0], (void*)&recvbuf[0], sizeof(float) * params.nx);

  sendbuf = &cells->speed5[work * params.nx]; //bottom row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed5, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);
  
  /* #pragma omp simd
  for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed5[i] = recvbuf[i];
  } */
  memcpy((void*)&cells->speed5[0], (void*)&recvbuf[0], sizeof(float) * params.nx);

  sendbuf = &cells->speed6[work * params.nx]; //bottom row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed6, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);
  
  /* #pragma omp simd
  for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed6[i] = recvbuf[i];
  } */
  memcpy((void*)&cells->speed6[0], (void*)&recvbuf[0], sizeof(float) * params.nx);

  sendbuf = &cells->speed7[work * params.nx]; //bottom row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed7, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);
  
  /* #pragma omp simd
  for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed7[i] = recvbuf[i];
  } */
  memcpy((void*)&cells->speed7[0], (void*)&recvbuf[0], sizeof(float) * params.nx);

  sendbuf = &cells->speed8[work * params.nx]; //bottom row
  MPI_Sendrecv(sendbuf, params.nx, MPI_FLOAT, bottom_neighbour_rank, tag, recvbuf, params.nx, MPI_FLOAT, top_neighbour_rank, tag, MPI_COMM_WORLD, &status);
  __assume_aligned(cells->speed8, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.nx) % 64 == 0);
  
  /* #pragma omp simd
  for (size_t i = 0; i < params.nx; i++)
  {
    cells->speed8[i] = recvbuf[i];
  } */
  memcpy((void*)&cells->speed8[0], (void*)&recvbuf[0], sizeof(float) * params.nx);

  //Compute
  int   tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;
  

  /* loop over _all_ cells */
  for (int jj = 1; jj < work + 1; jj++)
  {
    const int y_n = jj + 1;
    const int y_s = jj - 1;

    int ii = 0;
    int x_e = 1;
    int x_w = params.nx - 1;

    if (obstacles[(start - 1) * params.nx + jj*params.nx + ii]) {
      tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx];
      tmp_cells->speed1[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx];
      tmp_cells->speed2[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx];
      tmp_cells->speed3[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx];
      tmp_cells->speed4[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx];
      tmp_cells->speed5[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx];
      tmp_cells->speed6[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx];
      tmp_cells->speed7[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx];
      tmp_cells->speed8[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx];
    }
    else {
      const float c_sq = 1.f / 3.f; /* square of speed of sound */
      const float w0 = 4.f / 9.f;  /* weighting factor */
      const float w1 = 1.f / 9.f;  /* weighting factor */
      const float w2 = 1.f / 36.f; /* weighting factor */
      /* compute local density total */
      float local_density = 0.f;

      local_density += cells->speed0[ii + jj*params.nx];
      local_density += cells->speed1[x_w + jj*params.nx];
      local_density += cells->speed2[ii + y_s*params.nx];
      local_density += cells->speed3[x_e + jj*params.nx];
      local_density += cells->speed4[ii + y_n*params.nx];
      local_density += cells->speed5[x_w + y_s*params.nx];
      local_density += cells->speed6[x_e + y_s*params.nx];
      local_density += cells->speed7[x_e + y_n*params.nx];
      local_density += cells->speed8[x_w + y_n*params.nx];
    
      /* compute x velocity component */
      float u_x = (cells->speed1[x_w + jj*params.nx]
                    + cells->speed5[x_w + y_s*params.nx]
                    + cells->speed8[x_w + y_n*params.nx]
                    - (cells->speed3[x_e + jj*params.nx]
                        + cells->speed6[x_e + y_s*params.nx]
                        + cells->speed7[x_e + y_n*params.nx]))
                    / local_density;
      /* compute y velocity component */
      float u_y = (cells->speed2[ii + y_s*params.nx]
                    + cells->speed5[x_w + y_s*params.nx]
                    + cells->speed6[x_e + y_s*params.nx]
                    - (cells->speed4[ii + y_n*params.nx]
                        + cells->speed7[x_e + y_n*params.nx]
                        + cells->speed8[x_w + y_n*params.nx]))
                    / local_density;

      /* velocity squared */
      float u_sq = u_x * u_x + u_y * u_y;

      /* directional velocity components */
      float u[NSPEEDS];
      u[1] =   u_x;        /* east */
      u[2] =         u_y;  /* north */
      u[3] = - u_x;        /* west */
      u[4] =       - u_y;  /* south */
      u[5] =   u_x + u_y;  /* north-east */
      u[6] = - u_x + u_y;  /* north-west */
      u[7] = - u_x - u_y;  /* south-west */
      u[8] =   u_x - u_y;  /* south-east */

      /* equilibrium densities */
      float d_equ[NSPEEDS];
      /* zero velocity density: weight w0 */
      d_equ[0] = w0 * local_density
                  * (1.f - u_sq / (2.f * c_sq));
      /* axis speeds: weight w1 */
      d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                        + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                        + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                        + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                        + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      /* diagonal speeds: weight w2 */
      d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                        + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                        + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                        + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                        + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));

      /* relaxation step */

      tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx]
                                                + params.omega
                                                * (d_equ[0] - cells->speed0[ii + jj*params.nx]);

      tmp_cells->speed1[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx]
                                                + params.omega
                                                * (d_equ[1] - cells->speed1[x_w + jj*params.nx]);

      tmp_cells->speed2[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx]
                                                + params.omega
                                                * (d_equ[2] - cells->speed2[ii + y_s*params.nx]);                                          
      
      tmp_cells->speed3[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx]
                                                + params.omega
                                                * (d_equ[3] - cells->speed3[x_e + jj*params.nx]);

      tmp_cells->speed4[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx]
                                                + params.omega
                                                * (d_equ[4] - cells->speed4[ii + y_n*params.nx]);
      
      tmp_cells->speed5[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx]
                                                + params.omega
                                                * (d_equ[5] - cells->speed5[x_w + y_s*params.nx]);
      
      tmp_cells->speed6[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx]
                                                + params.omega
                                                * (d_equ[6] - cells->speed6[x_e + y_s*params.nx]);
      
      tmp_cells->speed7[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx]
                                                + params.omega
                                                * (d_equ[7] - cells->speed7[x_e + y_n*params.nx]);
      
      tmp_cells->speed8[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx]
                                                + params.omega
                                                * (d_equ[8] - cells->speed8[x_w + y_n*params.nx]);


      /* local density total */
      /* local_density = 0.f;

      local_density += tmp_cells->speed0[ii + jj*params.nx];
      local_density += tmp_cells->speed1[ii + jj*params.nx];
      local_density += tmp_cells->speed2[ii + jj*params.nx];
      local_density += tmp_cells->speed3[ii + jj*params.nx];
      local_density += tmp_cells->speed4[ii + jj*params.nx];
      local_density += tmp_cells->speed5[ii + jj*params.nx];
      local_density += tmp_cells->speed6[ii + jj*params.nx];
      local_density += tmp_cells->speed7[ii + jj*params.nx];
      local_density += tmp_cells->speed8[ii + jj*params.nx]; */

      /* x-component of velocity */
      /* u_x = (tmp_cells->speed1[ii + jj*params.nx]
                    + tmp_cells->speed5[ii + jj*params.nx]
                    + tmp_cells->speed8[ii + jj*params.nx]
                    - (tmp_cells->speed3[ii + jj*params.nx]
                        + tmp_cells->speed6[ii + jj*params.nx]
                        + tmp_cells->speed7[ii + jj*params.nx]))
                    / local_density; */
      /* compute y velocity component */
      /* u_y = (tmp_cells->speed2[ii + jj*params.nx]
                    + tmp_cells->speed5[ii + jj*params.nx]
                    + tmp_cells->speed6[ii + jj*params.nx]
                    - (tmp_cells->speed4[ii + jj*params.nx]
                        + tmp_cells->speed7[ii + jj*params.nx]
                        + tmp_cells->speed8[ii + jj*params.nx]))
                    / local_density; */
      /* accumulate the norm of x- and y- velocity components */
      tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
      /* increase counter of inspected cells */
      ++tot_cells;
    }

    //printf("%d \n", omp_get_num_threads());
    __assume_aligned(cells->speed0, 64);
    __assume_aligned(cells->speed1, 64);
    __assume_aligned(cells->speed2, 64);
    __assume_aligned(cells->speed3, 64);
    __assume_aligned(cells->speed4, 64);
    __assume_aligned(cells->speed5, 64);
    __assume_aligned(cells->speed6, 64);
    __assume_aligned(cells->speed7, 64);
    __assume_aligned(cells->speed8, 64);

    __assume_aligned(tmp_cells->speed0, 64);
    __assume_aligned(tmp_cells->speed1, 64);
    __assume_aligned(tmp_cells->speed2, 64);
    __assume_aligned(tmp_cells->speed3, 64);
    __assume_aligned(tmp_cells->speed4, 64);
    __assume_aligned(tmp_cells->speed5, 64);
    __assume_aligned(tmp_cells->speed6, 64);
    __assume_aligned(tmp_cells->speed7, 64);
    __assume_aligned(tmp_cells->speed8, 64);

    __assume((params.nx) % 2 == 0);
    __assume((params.nx) % 4 == 0);
    __assume((params.nx) % 8 == 0);
    __assume((params.nx) % 16 == 0);
    __assume((params.nx) % 32 == 0);
    __assume((params.nx) % 64 == 0);

    __assume_aligned(cells->speed0, 64);
    __assume_aligned(cells->speed1, 64);
    __assume_aligned(cells->speed2, 64);
    __assume_aligned(cells->speed3, 64);
    __assume_aligned(cells->speed4, 64);
    __assume_aligned(cells->speed5, 64);
    __assume_aligned(cells->speed6, 64);
    __assume_aligned(cells->speed7, 64);
    __assume_aligned(cells->speed8, 64);

    __assume_aligned(tmp_cells->speed0, 64);
    __assume_aligned(tmp_cells->speed1, 64);
    __assume_aligned(tmp_cells->speed2, 64);
    __assume_aligned(tmp_cells->speed3, 64);
    __assume_aligned(tmp_cells->speed4, 64);
    __assume_aligned(tmp_cells->speed5, 64);
    __assume_aligned(tmp_cells->speed6, 64);
    __assume_aligned(tmp_cells->speed7, 64);
    __assume_aligned(tmp_cells->speed8, 64);
    #pragma omp simd
    for (int ii = 1; ii < params.nx - 1; ii++)
    {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */

      //const int x_e = (ii == params.nx - 1) ? 0 : ii + 1;
      //const int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
      const int x_e = ii + 1;
      const int x_w = ii - 1;

      if (obstacles[(start - 1) * params.nx + jj*params.nx + ii]) {
        tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx];
        tmp_cells->speed1[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx];
        tmp_cells->speed2[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx];
        tmp_cells->speed3[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx];
        tmp_cells->speed4[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx];
        tmp_cells->speed5[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx];
        tmp_cells->speed6[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx];
        tmp_cells->speed7[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx];
        tmp_cells->speed8[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx];
      }
      else {
        const float c_sq = 1.f / 3.f; /* square of speed of sound */
        const float w0 = 4.f / 9.f;  /* weighting factor */
        const float w1 = 1.f / 9.f;  /* weighting factor */
        const float w2 = 1.f / 36.f; /* weighting factor */
        /* compute local density total */
        float local_density = 0.f;

        local_density += cells->speed0[ii + jj*params.nx];
        local_density += cells->speed1[x_w + jj*params.nx];
        local_density += cells->speed2[ii + y_s*params.nx];
        local_density += cells->speed3[x_e + jj*params.nx];
        local_density += cells->speed4[ii + y_n*params.nx];
        local_density += cells->speed5[x_w + y_s*params.nx];
        local_density += cells->speed6[x_e + y_s*params.nx];
        local_density += cells->speed7[x_e + y_n*params.nx];
        local_density += cells->speed8[x_w + y_n*params.nx];
      
        /* compute x velocity component */
        float u_x = (cells->speed1[x_w + jj*params.nx]
                      + cells->speed5[x_w + y_s*params.nx]
                      + cells->speed8[x_w + y_n*params.nx]
                      - (cells->speed3[x_e + jj*params.nx]
                          + cells->speed6[x_e + y_s*params.nx]
                          + cells->speed7[x_e + y_n*params.nx]))
                      / local_density;
        /* compute y velocity component */
        float u_y = (cells->speed2[ii + y_s*params.nx]
                      + cells->speed5[x_w + y_s*params.nx]
                      + cells->speed6[x_e + y_s*params.nx]
                      - (cells->speed4[ii + y_n*params.nx]
                          + cells->speed7[x_e + y_n*params.nx]
                          + cells->speed8[x_w + y_n*params.nx]))
                      / local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u[NSPEEDS];
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */

        /* equilibrium densities */
        float d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */
        d_equ[0] = w0 * local_density
                    * (1.f - u_sq / (2.f * c_sq));
        /* axis speeds: weight w1 */
        d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                          + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                          + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                          + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                          + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        /* diagonal speeds: weight w2 */
        d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                          + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                          + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                          + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                          + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));

        /* relaxation step */

        tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx]
                                                  + params.omega
                                                  * (d_equ[0] - cells->speed0[ii + jj*params.nx]);

        tmp_cells->speed1[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx]
                                                  + params.omega
                                                  * (d_equ[1] - cells->speed1[x_w + jj*params.nx]);

        tmp_cells->speed2[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx]
                                                  + params.omega
                                                  * (d_equ[2] - cells->speed2[ii + y_s*params.nx]);                                          
        
        tmp_cells->speed3[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx]
                                                  + params.omega
                                                  * (d_equ[3] - cells->speed3[x_e + jj*params.nx]);

        tmp_cells->speed4[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx]
                                                  + params.omega
                                                  * (d_equ[4] - cells->speed4[ii + y_n*params.nx]);
        
        tmp_cells->speed5[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx]
                                                  + params.omega
                                                  * (d_equ[5] - cells->speed5[x_w + y_s*params.nx]);
        
        tmp_cells->speed6[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx]
                                                  + params.omega
                                                  * (d_equ[6] - cells->speed6[x_e + y_s*params.nx]);
        
        tmp_cells->speed7[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx]
                                                  + params.omega
                                                  * (d_equ[7] - cells->speed7[x_e + y_n*params.nx]);
        
        tmp_cells->speed8[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx]
                                                  + params.omega
                                                  * (d_equ[8] - cells->speed8[x_w + y_n*params.nx]);


        /* local density total */
        /* local_density = 0.f;

        local_density += tmp_cells->speed0[ii + jj*params.nx];
        local_density += tmp_cells->speed1[ii + jj*params.nx];
        local_density += tmp_cells->speed2[ii + jj*params.nx];
        local_density += tmp_cells->speed3[ii + jj*params.nx];
        local_density += tmp_cells->speed4[ii + jj*params.nx];
        local_density += tmp_cells->speed5[ii + jj*params.nx];
        local_density += tmp_cells->speed6[ii + jj*params.nx];
        local_density += tmp_cells->speed7[ii + jj*params.nx];
        local_density += tmp_cells->speed8[ii + jj*params.nx]; */

        /* x-component of velocity */
        /* u_x = (tmp_cells->speed1[ii + jj*params.nx]
                      + tmp_cells->speed5[ii + jj*params.nx]
                      + tmp_cells->speed8[ii + jj*params.nx]
                      - (tmp_cells->speed3[ii + jj*params.nx]
                          + tmp_cells->speed6[ii + jj*params.nx]
                          + tmp_cells->speed7[ii + jj*params.nx]))
                      / local_density; */
        /* compute y velocity component */
        /* u_y = (tmp_cells->speed2[ii + jj*params.nx]
                      + tmp_cells->speed5[ii + jj*params.nx]
                      + tmp_cells->speed6[ii + jj*params.nx]
                      - (tmp_cells->speed4[ii + jj*params.nx]
                          + tmp_cells->speed7[ii + jj*params.nx]
                          + tmp_cells->speed8[ii + jj*params.nx]))
                      / local_density; */
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
      
    }

    ii = params.nx - 1; 
    x_e = 0;
    x_w = ii - 1;

    if (obstacles[(start - 1) * params.nx + jj*params.nx + ii]) {
      tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx];
      tmp_cells->speed1[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx];
      tmp_cells->speed2[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx];
      tmp_cells->speed3[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx];
      tmp_cells->speed4[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx];
      tmp_cells->speed5[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx];
      tmp_cells->speed6[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx];
      tmp_cells->speed7[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx];
      tmp_cells->speed8[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx];
    }
    else {
      const float c_sq = 1.f / 3.f; /* square of speed of sound */
      const float w0 = 4.f / 9.f;  /* weighting factor */
      const float w1 = 1.f / 9.f;  /* weighting factor */
      const float w2 = 1.f / 36.f; /* weighting factor */
      /* compute local density total */
      float local_density = 0.f;

      local_density += cells->speed0[ii + jj*params.nx];
      local_density += cells->speed1[x_w + jj*params.nx];
      local_density += cells->speed2[ii + y_s*params.nx];
      local_density += cells->speed3[x_e + jj*params.nx];
      local_density += cells->speed4[ii + y_n*params.nx];
      local_density += cells->speed5[x_w + y_s*params.nx];
      local_density += cells->speed6[x_e + y_s*params.nx];
      local_density += cells->speed7[x_e + y_n*params.nx];
      local_density += cells->speed8[x_w + y_n*params.nx];
    
      /* compute x velocity component */
      float u_x = (cells->speed1[x_w + jj*params.nx]
                    + cells->speed5[x_w + y_s*params.nx]
                    + cells->speed8[x_w + y_n*params.nx]
                    - (cells->speed3[x_e + jj*params.nx]
                        + cells->speed6[x_e + y_s*params.nx]
                        + cells->speed7[x_e + y_n*params.nx]))
                    / local_density;
      /* compute y velocity component */
      float u_y = (cells->speed2[ii + y_s*params.nx]
                    + cells->speed5[x_w + y_s*params.nx]
                    + cells->speed6[x_e + y_s*params.nx]
                    - (cells->speed4[ii + y_n*params.nx]
                        + cells->speed7[x_e + y_n*params.nx]
                        + cells->speed8[x_w + y_n*params.nx]))
                    / local_density;

      /* velocity squared */
      float u_sq = u_x * u_x + u_y * u_y;

      /* directional velocity components */
      float u[NSPEEDS];
      u[1] =   u_x;        /* east */
      u[2] =         u_y;  /* north */
      u[3] = - u_x;        /* west */
      u[4] =       - u_y;  /* south */
      u[5] =   u_x + u_y;  /* north-east */
      u[6] = - u_x + u_y;  /* north-west */
      u[7] = - u_x - u_y;  /* south-west */
      u[8] =   u_x - u_y;  /* south-east */

      /* equilibrium densities */
      float d_equ[NSPEEDS];
      /* zero velocity density: weight w0 */
      d_equ[0] = w0 * local_density
                  * (1.f - u_sq / (2.f * c_sq));
      /* axis speeds: weight w1 */
      d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                        + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                        + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                        + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                        + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      /* diagonal speeds: weight w2 */
      d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                        + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                        + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                        + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                        + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));

      /* relaxation step */

      tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx]
                                                + params.omega
                                                * (d_equ[0] - cells->speed0[ii + jj*params.nx]);

      tmp_cells->speed1[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx]
                                                + params.omega
                                                * (d_equ[1] - cells->speed1[x_w + jj*params.nx]);

      tmp_cells->speed2[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx]
                                                + params.omega
                                                * (d_equ[2] - cells->speed2[ii + y_s*params.nx]);                                          
      
      tmp_cells->speed3[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx]
                                                + params.omega
                                                * (d_equ[3] - cells->speed3[x_e + jj*params.nx]);

      tmp_cells->speed4[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx]
                                                + params.omega
                                                * (d_equ[4] - cells->speed4[ii + y_n*params.nx]);
      
      tmp_cells->speed5[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx]
                                                + params.omega
                                                * (d_equ[5] - cells->speed5[x_w + y_s*params.nx]);
      
      tmp_cells->speed6[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx]
                                                + params.omega
                                                * (d_equ[6] - cells->speed6[x_e + y_s*params.nx]);
      
      tmp_cells->speed7[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx]
                                                + params.omega
                                                * (d_equ[7] - cells->speed7[x_e + y_n*params.nx]);
      
      tmp_cells->speed8[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx]
                                                + params.omega
                                                * (d_equ[8] - cells->speed8[x_w + y_n*params.nx]);


      /* local density total */
      /* local_density = 0.f;

      local_density += tmp_cells->speed0[ii + jj*params.nx];
      local_density += tmp_cells->speed1[ii + jj*params.nx];
      local_density += tmp_cells->speed2[ii + jj*params.nx];
      local_density += tmp_cells->speed3[ii + jj*params.nx];
      local_density += tmp_cells->speed4[ii + jj*params.nx];
      local_density += tmp_cells->speed5[ii + jj*params.nx];
      local_density += tmp_cells->speed6[ii + jj*params.nx];
      local_density += tmp_cells->speed7[ii + jj*params.nx];
      local_density += tmp_cells->speed8[ii + jj*params.nx];
 */
      /* x-component of velocity */
      /* u_x = (tmp_cells->speed1[ii + jj*params.nx]
                    + tmp_cells->speed5[ii + jj*params.nx]
                    + tmp_cells->speed8[ii + jj*params.nx]
                    - (tmp_cells->speed3[ii + jj*params.nx]
                        + tmp_cells->speed6[ii + jj*params.nx]
                        + tmp_cells->speed7[ii + jj*params.nx]))
                    / local_density; */
      /* compute y velocity component */
      /* u_y = (tmp_cells->speed2[ii + jj*params.nx]
                    + tmp_cells->speed5[ii + jj*params.nx]
                    + tmp_cells->speed6[ii + jj*params.nx]
                    - (tmp_cells->speed4[ii + jj*params.nx]
                        + tmp_cells->speed7[ii + jj*params.nx]
                        + tmp_cells->speed8[ii + jj*params.nx]))
                    / local_density; */
      /* accumulate the norm of x- and y- velocity components */
      tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
      /* increase counter of inspected cells */
      ++tot_cells;
    }
  }

  int local_tot_cells = tot_cells;
  int global_tot_cells = 1;
  float local_tot_u = tot_u;
  float global_tot_u = 0.0;

  MPI_Reduce(&local_tot_cells, &global_tot_cells, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_tot_u, &global_tot_u, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

  /* if (rank == 0) {
    printf("%f\n", global_tot_u/(float)global_tot_cells);
    printf("global tot_u: %f\n", global_tot_u);
    printf("global tot_cells: %d\n", global_tot_cells);
  } */
  return global_tot_u/(float)global_tot_cells;
}

inline float timestep_mpi_3(const t_param params, t_speed_soa* restrict cells, t_speed_soa* restrict tmp_cells, int* obstacles, int rank, int nprocs, int work, int start, int end) {
  //Exchange halo regions
  int tag = 0;
  MPI_Status status;

  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

    /* modify the 2nd row of the grid */
  //int jj = params.ny - 2;  
  int jj = accelerate_jj;

  
  if (rank == accelerate_rank) {
    __assume_aligned(cells->speed0, 64);
    __assume_aligned(cells->speed1, 64);
    __assume_aligned(cells->speed2, 64);
    __assume_aligned(cells->speed3, 64);
    __assume_aligned(cells->speed4, 64);
    __assume_aligned(cells->speed5, 64);
    __assume_aligned(cells->speed6, 64);
    __assume_aligned(cells->speed7, 64);
    __assume_aligned(cells->speed8, 64);

    __assume_aligned(tmp_cells->speed0, 64);
    __assume_aligned(tmp_cells->speed1, 64);
    __assume_aligned(tmp_cells->speed2, 64);
    __assume_aligned(tmp_cells->speed3, 64);
    __assume_aligned(tmp_cells->speed4, 64);
    __assume_aligned(tmp_cells->speed5, 64);
    __assume_aligned(tmp_cells->speed6, 64);
    __assume_aligned(tmp_cells->speed7, 64);
    __assume_aligned(tmp_cells->speed8, 64);

    __assume((params.nx) % 2 == 0);
    __assume((params.nx) % 4 == 0);
    __assume((params.nx) % 8 == 0);
    __assume((params.nx) % 16 == 0);
    __assume((params.nx) % 32 == 0);
    __assume((params.nx) % 64 == 0);
    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* if the cell is not occupied and
      ** we don't send a negative density */
      if (!obstacles[(start - 1) * params.nx + ii + jj*params.nx]
          && (cells->speed3[ii + jj*params.nx] - w1) > 0.f
          && (cells->speed6[ii + jj*params.nx] - w2) > 0.f
          && (cells->speed7[ii + jj*params.nx] - w2) > 0.f)
      {
        /* increase 'east-side' densities */
        cells->speed1[ii + jj*params.nx] += w1;
        cells->speed5[ii + jj*params.nx] += w2;
        cells->speed8[ii + jj*params.nx] += w2;
        /* decrease 'west-side' densities */
        cells->speed3[ii + jj*params.nx] -= w1;
        cells->speed6[ii + jj*params.nx] -= w2;
        cells->speed7[ii + jj*params.nx] -= w2;
      }
    }
  }

  //Send to top and receive from bottom
  float* sendbuf = (float*)_mm_malloc(sizeof(float) * params.nx * 9, 64);

  __assume_aligned(sendbuf, 64);
  __assume_aligned(cells->speed0, 64);
  __assume_aligned(cells->speed1, 64);
  __assume_aligned(cells->speed2, 64);
  __assume_aligned(cells->speed3, 64);
  __assume_aligned(cells->speed4, 64);
  __assume_aligned(cells->speed5, 64);
  __assume_aligned(cells->speed6, 64);
  __assume_aligned(cells->speed7, 64);
  __assume_aligned(cells->speed8, 64);
  memcpy((void*)&sendbuf[0], (void*)&cells->speed0[params.nx], sizeof(float) * params.nx);
  memcpy((void*)&sendbuf[params.nx], (void*)&cells->speed1[params.nx], sizeof(float) * params.nx);
  memcpy((void*)&sendbuf[2*params.nx], (void*)&cells->speed2[params.nx], sizeof(float) * params.nx);
  memcpy((void*)&sendbuf[3*params.nx], (void*)&cells->speed3[params.nx], sizeof(float) * params.nx);
  memcpy((void*)&sendbuf[4*params.nx], (void*)&cells->speed4[params.nx], sizeof(float) * params.nx);
  memcpy((void*)&sendbuf[5*params.nx], (void*)&cells->speed5[params.nx], sizeof(float) * params.nx);
  memcpy((void*)&sendbuf[6*params.nx], (void*)&cells->speed6[params.nx], sizeof(float) * params.nx);
  memcpy((void*)&sendbuf[7*params.nx], (void*)&cells->speed7[params.nx], sizeof(float) * params.nx);
  memcpy((void*)&sendbuf[8*params.nx], (void*)&cells->speed8[params.nx], sizeof(float) * params.nx);

  float* recvbuf = (float*)_mm_malloc(sizeof(float) * params.nx * 9, 64);
  
  MPI_Sendrecv(sendbuf, params.nx * 9, MPI_FLOAT, top_neighbour_rank, tag, recvbuf, params.nx * 9, MPI_FLOAT, bottom_neighbour_rank, tag, MPI_COMM_WORLD, &status);

  __assume_aligned(recvbuf, 64);
  __assume_aligned(cells->speed0, 64);
  __assume_aligned(cells->speed1, 64);
  __assume_aligned(cells->speed2, 64);
  __assume_aligned(cells->speed3, 64);
  __assume_aligned(cells->speed4, 64);
  __assume_aligned(cells->speed5, 64);
  __assume_aligned(cells->speed6, 64);
  __assume_aligned(cells->speed7, 64);
  __assume_aligned(cells->speed8, 64);
  memcpy((void*)&cells->speed0[(work + 1) * params.nx], (void*)&recvbuf[0], sizeof(float)*params.nx);
  memcpy((void*)&cells->speed1[(work + 1) * params.nx], (void*)&recvbuf[params.nx], sizeof(float)*params.nx);
  memcpy((void*)&cells->speed2[(work + 1) * params.nx], (void*)&recvbuf[2*params.nx], sizeof(float)*params.nx);
  memcpy((void*)&cells->speed3[(work + 1) * params.nx], (void*)&recvbuf[3*params.nx], sizeof(float)*params.nx);
  memcpy((void*)&cells->speed4[(work + 1) * params.nx], (void*)&recvbuf[4*params.nx], sizeof(float)*params.nx);
  memcpy((void*)&cells->speed5[(work + 1) * params.nx], (void*)&recvbuf[5*params.nx], sizeof(float)*params.nx);
  memcpy((void*)&cells->speed6[(work + 1) * params.nx], (void*)&recvbuf[6*params.nx], sizeof(float)*params.nx);
  memcpy((void*)&cells->speed7[(work + 1) * params.nx], (void*)&recvbuf[7*params.nx], sizeof(float)*params.nx);
  memcpy((void*)&cells->speed8[(work + 1) * params.nx], (void*)&recvbuf[8*params.nx], sizeof(float)*params.nx);

  //Send to bottom and receive from top

  __assume_aligned(sendbuf, 64);

  __assume_aligned(cells->speed0, 64);
  memcpy((void*)&sendbuf[0], (void*)&cells->speed0[work * params.nx], sizeof(float) * params.nx);

  __assume_aligned(cells->speed1, 64);
  memcpy((void*)&sendbuf[params.nx], (void*)&cells->speed1[work * params.nx], sizeof(float) * params.nx);

  __assume_aligned(cells->speed2, 64);
  memcpy((void*)&sendbuf[2*params.nx], (void*)&cells->speed2[work * params.nx], sizeof(float) * params.nx);
  
  __assume_aligned(cells->speed3, 64);
  memcpy((void*)&sendbuf[3*params.nx], (void*)&cells->speed3[work * params.nx], sizeof(float) * params.nx);
  __assume_aligned(cells->speed4, 64);
  memcpy((void*)&sendbuf[4*params.nx], (void*)&cells->speed4[work * params.nx], sizeof(float) * params.nx);
  __assume_aligned(cells->speed5, 64);
  memcpy((void*)&sendbuf[5*params.nx], (void*)&cells->speed5[work * params.nx], sizeof(float) * params.nx);

  __assume_aligned(cells->speed6, 64);
  memcpy((void*)&sendbuf[6*params.nx], (void*)&cells->speed6[work * params.nx], sizeof(float) * params.nx);

  __assume_aligned(cells->speed7, 64);
  memcpy((void*)&sendbuf[7*params.nx], (void*)&cells->speed7[work * params.nx], sizeof(float) * params.nx);

  __assume_aligned(cells->speed8, 64);
  memcpy((void*)&sendbuf[8*params.nx], (void*)&cells->speed8[work * params.nx], sizeof(float) * params.nx);

  MPI_Sendrecv(sendbuf, params.nx * 9, MPI_FLOAT, bottom_neighbour_rank, tag, recvbuf, params.nx * 9, MPI_FLOAT, top_neighbour_rank, tag, MPI_COMM_WORLD, &status);

  __assume_aligned(recvbuf, 64);
  __assume_aligned(cells->speed0, 64);
  __assume_aligned(cells->speed1, 64);
  __assume_aligned(cells->speed2, 64);
  __assume_aligned(cells->speed3, 64);
  __assume_aligned(cells->speed4, 64);
  __assume_aligned(cells->speed5, 64);
  __assume_aligned(cells->speed6, 64);
  __assume_aligned(cells->speed7, 64);
  __assume_aligned(cells->speed8, 64);
  memcpy((void*)&cells->speed0[0], (void*)&recvbuf[0], sizeof(float)*params.nx);
  memcpy((void*)&cells->speed1[0], (void*)&recvbuf[params.nx], sizeof(float)*params.nx);
  memcpy((void*)&cells->speed2[0], (void*)&recvbuf[2*params.nx], sizeof(float)*params.nx);
  memcpy((void*)&cells->speed3[0], (void*)&recvbuf[3*params.nx], sizeof(float)*params.nx);
  memcpy((void*)&cells->speed4[0], (void*)&recvbuf[4*params.nx], sizeof(float)*params.nx);
  memcpy((void*)&cells->speed5[0], (void*)&recvbuf[5*params.nx], sizeof(float)*params.nx);
  memcpy((void*)&cells->speed6[0], (void*)&recvbuf[6*params.nx], sizeof(float)*params.nx);
  memcpy((void*)&cells->speed7[0], (void*)&recvbuf[7*params.nx], sizeof(float)*params.nx);
  memcpy((void*)&cells->speed8[0], (void*)&recvbuf[8*params.nx], sizeof(float)*params.nx);

  //Compute
  int   tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;
  
/* loop over _all_ cells */
  for (int jj = 1; jj < work + 1; jj++)
  {
    __assume_aligned(cells->speed0, 64);
    __assume_aligned(cells->speed1, 64);
    __assume_aligned(cells->speed2, 64);
    __assume_aligned(cells->speed3, 64);
    __assume_aligned(cells->speed4, 64);
    __assume_aligned(cells->speed5, 64);
    __assume_aligned(cells->speed6, 64);
    __assume_aligned(cells->speed7, 64);
    __assume_aligned(cells->speed8, 64);

    __assume_aligned(tmp_cells->speed0, 64);
    __assume_aligned(tmp_cells->speed1, 64);
    __assume_aligned(tmp_cells->speed2, 64);
    __assume_aligned(tmp_cells->speed3, 64);
    __assume_aligned(tmp_cells->speed4, 64);
    __assume_aligned(tmp_cells->speed5, 64);
    __assume_aligned(tmp_cells->speed6, 64);
    __assume_aligned(tmp_cells->speed7, 64);
    __assume_aligned(tmp_cells->speed8, 64);

    __assume((params.nx) % 2 == 0);
    __assume((params.nx) % 4 == 0);
    __assume((params.nx) % 8 == 0);
    __assume((params.nx) % 16 == 0);
    __assume((params.nx) % 32 == 0);
    __assume((params.nx) % 64 == 0);

    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */

      const int y_n = jj + 1;
      const int x_e = (ii == params.nx - 1) ? 0 : ii + 1;
      const int y_s = jj - 1;
      const int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);

      if (obstacles[(start - 1) * params.nx + jj*params.nx + ii]) {
        tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx];
        tmp_cells->speed1[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx];
        tmp_cells->speed2[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx];
        tmp_cells->speed3[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx];
        tmp_cells->speed4[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx];
        tmp_cells->speed5[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx];
        tmp_cells->speed6[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx];
        tmp_cells->speed7[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx];
        tmp_cells->speed8[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx];
      }
      else {
        const float c_sq = 1.f / 3.f; /* square of speed of sound */
        const float w0 = 4.f / 9.f;  /* weighting factor */
        const float w1 = 1.f / 9.f;  /* weighting factor */
        const float w2 = 1.f / 36.f; /* weighting factor */
        /* compute local density total */
        float local_density = 0.f;

        local_density += cells->speed0[ii + jj*params.nx];
        local_density += cells->speed1[x_w + jj*params.nx];
        local_density += cells->speed2[ii + y_s*params.nx];
        local_density += cells->speed3[x_e + jj*params.nx];
        local_density += cells->speed4[ii + y_n*params.nx];
        local_density += cells->speed5[x_w + y_s*params.nx];
        local_density += cells->speed6[x_e + y_s*params.nx];
        local_density += cells->speed7[x_e + y_n*params.nx];
        local_density += cells->speed8[x_w + y_n*params.nx];
      
        /* compute x velocity component */
        float u_x = (cells->speed1[x_w + jj*params.nx]
                      + cells->speed5[x_w + y_s*params.nx]
                      + cells->speed8[x_w + y_n*params.nx]
                      - (cells->speed3[x_e + jj*params.nx]
                          + cells->speed6[x_e + y_s*params.nx]
                          + cells->speed7[x_e + y_n*params.nx]))
                      / local_density;
        /* compute y velocity component */
        float u_y = (cells->speed2[ii + y_s*params.nx]
                      + cells->speed5[x_w + y_s*params.nx]
                      + cells->speed6[x_e + y_s*params.nx]
                      - (cells->speed4[ii + y_n*params.nx]
                          + cells->speed7[x_e + y_n*params.nx]
                          + cells->speed8[x_w + y_n*params.nx]))
                      / local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u[NSPEEDS];
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */

        /* equilibrium densities */
        float d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */
        d_equ[0] = w0 * local_density
                    * (1.f - u_sq / (2.f * c_sq));
        /* axis speeds: weight w1 */
        d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                          + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                          + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                          + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                          + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        /* diagonal speeds: weight w2 */
        d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                          + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                          + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                          + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                          + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));

        /* relaxation step */

        tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx]
                                                  + params.omega
                                                  * (d_equ[0] - cells->speed0[ii + jj*params.nx]);

        tmp_cells->speed1[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx]
                                                  + params.omega
                                                  * (d_equ[1] - cells->speed1[x_w + jj*params.nx]);

        tmp_cells->speed2[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx]
                                                  + params.omega
                                                  * (d_equ[2] - cells->speed2[ii + y_s*params.nx]);                                          
        
        tmp_cells->speed3[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx]
                                                  + params.omega
                                                  * (d_equ[3] - cells->speed3[x_e + jj*params.nx]);

        tmp_cells->speed4[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx]
                                                  + params.omega
                                                  * (d_equ[4] - cells->speed4[ii + y_n*params.nx]);
        
        tmp_cells->speed5[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx]
                                                  + params.omega
                                                  * (d_equ[5] - cells->speed5[x_w + y_s*params.nx]);
        
        tmp_cells->speed6[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx]
                                                  + params.omega
                                                  * (d_equ[6] - cells->speed6[x_e + y_s*params.nx]);
        
        tmp_cells->speed7[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx]
                                                  + params.omega
                                                  * (d_equ[7] - cells->speed7[x_e + y_n*params.nx]);
        
        tmp_cells->speed8[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx]
                                                  + params.omega
                                                  * (d_equ[8] - cells->speed8[x_w + y_n*params.nx]);


        /* local density total */
        /* local_density = 0.f;

        local_density += tmp_cells->speed0[ii + jj*params.nx];
        local_density += tmp_cells->speed1[ii + jj*params.nx];
        local_density += tmp_cells->speed2[ii + jj*params.nx];
        local_density += tmp_cells->speed3[ii + jj*params.nx];
        local_density += tmp_cells->speed4[ii + jj*params.nx];
        local_density += tmp_cells->speed5[ii + jj*params.nx];
        local_density += tmp_cells->speed6[ii + jj*params.nx];
        local_density += tmp_cells->speed7[ii + jj*params.nx];
        local_density += tmp_cells->speed8[ii + jj*params.nx]; */

        /* x-component of velocity */
        /* u_x = (tmp_cells->speed1[ii + jj*params.nx]
                      + tmp_cells->speed5[ii + jj*params.nx]
                      + tmp_cells->speed8[ii + jj*params.nx]
                      - (tmp_cells->speed3[ii + jj*params.nx]
                          + tmp_cells->speed6[ii + jj*params.nx]
                          + tmp_cells->speed7[ii + jj*params.nx]))
                      / local_density; */
        /* compute y velocity component */
        /* u_y = (tmp_cells->speed2[ii + jj*params.nx]
                      + tmp_cells->speed5[ii + jj*params.nx]
                      + tmp_cells->speed6[ii + jj*params.nx]
                      - (tmp_cells->speed4[ii + jj*params.nx]
                          + tmp_cells->speed7[ii + jj*params.nx]
                          + tmp_cells->speed8[ii + jj*params.nx]))
                      / local_density; */
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
      
    }
  }
  

  int local_tot_cells = tot_cells;
  int global_tot_cells = 0;
  float local_tot_u = tot_u;
  float global_tot_u = 0.0;

  MPI_Reduce(&local_tot_cells, &global_tot_cells, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_tot_u, &global_tot_u, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

  /* if (rank == 0) {
    printf("%f\n", global_tot_u/(float)global_tot_cells);
    printf("global tot_u: %f\n", global_tot_u);
    printf("global tot_cells: %d\n", global_tot_cells);
  } */
  return global_tot_u/(float)global_tot_cells;

}

float timestep_parallel(const t_param params, t_speed_soa* restrict cells, t_speed_soa* restrict tmp_cells, int* restrict obstacles) {
  __assume_aligned(cells->speed0, 64);
  __assume_aligned(cells->speed1, 64);
  __assume_aligned(cells->speed2, 64);
  __assume_aligned(cells->speed3, 64);
  __assume_aligned(cells->speed4, 64);
  __assume_aligned(cells->speed5, 64);
  __assume_aligned(cells->speed6, 64);
  __assume_aligned(cells->speed7, 64);
  __assume_aligned(cells->speed8, 64);

  __assume_aligned(tmp_cells->speed0, 64);
  __assume_aligned(tmp_cells->speed1, 64);
  __assume_aligned(tmp_cells->speed2, 64);
  __assume_aligned(tmp_cells->speed3, 64);
  __assume_aligned(tmp_cells->speed4, 64);
  __assume_aligned(tmp_cells->speed5, 64);
  __assume_aligned(tmp_cells->speed6, 64);
  __assume_aligned(tmp_cells->speed7, 64);
  __assume_aligned(tmp_cells->speed8, 64);

  __assume((params.nx) % 2 == 0);
  __assume((params.ny) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.ny) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.ny) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.ny) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.ny) % 32 == 0);
  __assume((params.nx) % 64 == 0);
  __assume((params.ny) % 64 == 0);

  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  int   tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* modify the 2nd row of the grid */
  int jj = params.ny - 2;

  #pragma omp simd
  for (int ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + jj*params.nx]
        && (cells->speed3[ii + jj*params.nx] - w1) > 0.f
        && (cells->speed6[ii + jj*params.nx] - w2) > 0.f
        && (cells->speed7[ii + jj*params.nx] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells->speed1[ii + jj*params.nx] += w1;
      cells->speed5[ii + jj*params.nx] += w2;
      cells->speed8[ii + jj*params.nx] += w2;
      /* decrease 'west-side' densities */
      cells->speed3[ii + jj*params.nx] -= w1;
      cells->speed6[ii + jj*params.nx] -= w2;
      cells->speed7[ii + jj*params.nx] -= w2;
    }
  }

  #pragma omp parallel for reduction(+:tot_u, tot_cells)
  /* loop over _all_ cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    //printf("%d \n", omp_get_num_threads());
    __assume_aligned(cells->speed0, 64);
    __assume_aligned(cells->speed1, 64);
    __assume_aligned(cells->speed2, 64);
    __assume_aligned(cells->speed3, 64);
    __assume_aligned(cells->speed4, 64);
    __assume_aligned(cells->speed5, 64);
    __assume_aligned(cells->speed6, 64);
    __assume_aligned(cells->speed7, 64);
    __assume_aligned(cells->speed8, 64);

    __assume_aligned(tmp_cells->speed0, 64);
    __assume_aligned(tmp_cells->speed1, 64);
    __assume_aligned(tmp_cells->speed2, 64);
    __assume_aligned(tmp_cells->speed3, 64);
    __assume_aligned(tmp_cells->speed4, 64);
    __assume_aligned(tmp_cells->speed5, 64);
    __assume_aligned(tmp_cells->speed6, 64);
    __assume_aligned(tmp_cells->speed7, 64);
    __assume_aligned(tmp_cells->speed8, 64);

    #pragma omp simd reduction(+:tot_u, tot_cells)
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      //int y_n = (jj + 1) % params.ny;
      //int x_e = (ii + 1) % params.nx;
      int y_n = (jj == params.ny - 1) ? 0 : jj + 1;
      int x_e = (ii == params.nx - 1) ? 0 : ii + 1;
      int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
      int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);

      if (obstacles[jj*params.nx + ii]) {
        tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx];
        tmp_cells->speed1[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx];
        tmp_cells->speed2[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx];
        tmp_cells->speed3[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx];
        tmp_cells->speed4[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx];
        tmp_cells->speed5[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx];
        tmp_cells->speed6[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx];
        tmp_cells->speed7[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx];
        tmp_cells->speed8[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx];
      }
      else {
        const float c_sq = 1.f / 3.f; /* square of speed of sound */
        const float w0 = 4.f / 9.f;  /* weighting factor */
        const float w1 = 1.f / 9.f;  /* weighting factor */
        const float w2 = 1.f / 36.f; /* weighting factor */
        /* compute local density total */
        float local_density = 0.f;

        local_density += cells->speed0[ii + jj*params.nx];
        local_density += cells->speed1[x_w + jj*params.nx];
        local_density += cells->speed2[ii + y_s*params.nx];
        local_density += cells->speed3[x_e + jj*params.nx];
        local_density += cells->speed4[ii + y_n*params.nx];
        local_density += cells->speed5[x_w + y_s*params.nx];
        local_density += cells->speed6[x_e + y_s*params.nx];
        local_density += cells->speed7[x_e + y_n*params.nx];
        local_density += cells->speed8[x_w + y_n*params.nx];
        
      
        /* compute x velocity component */
        float u_x = (cells->speed1[x_w + jj*params.nx]
                      + cells->speed5[x_w + y_s*params.nx]
                      + cells->speed8[x_w + y_n*params.nx]
                      - (cells->speed3[x_e + jj*params.nx]
                          + cells->speed6[x_e + y_s*params.nx]
                          + cells->speed7[x_e + y_n*params.nx]))
                      / local_density;
        /* compute y velocity component */
        float u_y = (cells->speed2[ii + y_s*params.nx]
                      + cells->speed5[x_w + y_s*params.nx]
                      + cells->speed6[x_e + y_s*params.nx]
                      - (cells->speed4[ii + y_n*params.nx]
                          + cells->speed7[x_e + y_n*params.nx]
                          + cells->speed8[x_w + y_n*params.nx]))
                      / local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u[NSPEEDS];
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */

        /* equilibrium densities */
        float d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */
        d_equ[0] = w0 * local_density
                    * (1.f - u_sq / (2.f * c_sq));
        /* axis speeds: weight w1 */
        d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                          + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                          + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                          + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                          + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        /* diagonal speeds: weight w2 */
        d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                          + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                          + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                          + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                          + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));

        /* relaxation step */

        tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx]
                                                  + params.omega
                                                  * (d_equ[0] - cells->speed0[ii + jj*params.nx]);

        tmp_cells->speed1[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx]
                                                  + params.omega
                                                  * (d_equ[1] - cells->speed1[x_w + jj*params.nx]);

        tmp_cells->speed2[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx]
                                                  + params.omega
                                                  * (d_equ[2] - cells->speed2[ii + y_s*params.nx]);                                          
        
        tmp_cells->speed3[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx]
                                                  + params.omega
                                                  * (d_equ[3] - cells->speed3[x_e + jj*params.nx]);

        tmp_cells->speed4[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx]
                                                  + params.omega
                                                  * (d_equ[4] - cells->speed4[ii + y_n*params.nx]);
        
        tmp_cells->speed5[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx]
                                                  + params.omega
                                                  * (d_equ[5] - cells->speed5[x_w + y_s*params.nx]);
        
        tmp_cells->speed6[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx]
                                                  + params.omega
                                                  * (d_equ[6] - cells->speed6[x_e + y_s*params.nx]);
        
        tmp_cells->speed7[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx]
                                                  + params.omega
                                                  * (d_equ[7] - cells->speed7[x_e + y_n*params.nx]);
        
        tmp_cells->speed8[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx]
                                                  + params.omega
                                                  * (d_equ[8] - cells->speed8[x_w + y_n*params.nx]);


        /* local density total */
        local_density = 0.f;

        local_density += tmp_cells->speed0[ii + jj*params.nx];
        local_density += tmp_cells->speed1[ii + jj*params.nx];
        local_density += tmp_cells->speed2[ii + jj*params.nx];
        local_density += tmp_cells->speed3[ii + jj*params.nx];
        local_density += tmp_cells->speed4[ii + jj*params.nx];
        local_density += tmp_cells->speed5[ii + jj*params.nx];
        local_density += tmp_cells->speed6[ii + jj*params.nx];
        local_density += tmp_cells->speed7[ii + jj*params.nx];
        local_density += tmp_cells->speed8[ii + jj*params.nx];

        /* x-component of velocity */
        u_x = (tmp_cells->speed1[ii + jj*params.nx]
                      + tmp_cells->speed5[ii + jj*params.nx]
                      + tmp_cells->speed8[ii + jj*params.nx]
                      - (tmp_cells->speed3[ii + jj*params.nx]
                          + tmp_cells->speed6[ii + jj*params.nx]
                          + tmp_cells->speed7[ii + jj*params.nx]))
                      / local_density;
        /* compute y velocity component */
        u_y = (tmp_cells->speed2[ii + jj*params.nx]
                      + tmp_cells->speed5[ii + jj*params.nx]
                      + tmp_cells->speed6[ii + jj*params.nx]
                      - (tmp_cells->speed4[ii + jj*params.nx]
                          + tmp_cells->speed7[ii + jj*params.nx]
                          + tmp_cells->speed8[ii + jj*params.nx]))
                      / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
      
    }
  }
  return tot_u / (float)tot_cells;

}

float timestep_vectorised(const t_param params, t_speed_soa* restrict cells, t_speed_soa* restrict tmp_cells, int* restrict obstacles) {
  __assume_aligned(cells->speed0, 64);
  __assume_aligned(cells->speed1, 64);
  __assume_aligned(cells->speed2, 64);
  __assume_aligned(cells->speed3, 64);
  __assume_aligned(cells->speed4, 64);
  __assume_aligned(cells->speed5, 64);
  __assume_aligned(cells->speed6, 64);
  __assume_aligned(cells->speed7, 64);
  __assume_aligned(cells->speed8, 64);

  __assume_aligned(tmp_cells->speed0, 64);
  __assume_aligned(tmp_cells->speed1, 64);
  __assume_aligned(tmp_cells->speed2, 64);
  __assume_aligned(tmp_cells->speed3, 64);
  __assume_aligned(tmp_cells->speed4, 64);
  __assume_aligned(tmp_cells->speed5, 64);
  __assume_aligned(tmp_cells->speed6, 64);
  __assume_aligned(tmp_cells->speed7, 64);
  __assume_aligned(tmp_cells->speed8, 64);

  __assume((params.nx) % 2 == 0);
  __assume((params.ny) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.ny) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.ny) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.ny) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.ny) % 32 == 0);
  __assume((params.nx) % 64 == 0);
  __assume((params.ny) % 64 == 0);

  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* modify the 2nd row of the grid */
  int jj = params.ny - 2;

  #pragma omp simd
  for (int ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + jj*params.nx]
        && (cells->speed3[ii + jj*params.nx] - w1) > 0.f
        && (cells->speed6[ii + jj*params.nx] - w2) > 0.f
        && (cells->speed7[ii + jj*params.nx] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells->speed1[ii + jj*params.nx] += w1;
      cells->speed5[ii + jj*params.nx] += w2;
      cells->speed8[ii + jj*params.nx] += w2;
      /* decrease 'west-side' densities */
      cells->speed3[ii + jj*params.nx] -= w1;
      cells->speed6[ii + jj*params.nx] -= w2;
      cells->speed7[ii + jj*params.nx] -= w2;
    }
  }

  //Case where jj = 0
  jj = 0;
  int y_s = jj + params.ny - 1;
  int y_n = jj + 1;


  //Case where ii = 0
  int ii = 0;
  int x_e = ii + 1;
  int x_w = ii + params.nx - 1;

  if (obstacles[jj*params.nx + ii]) {
    tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx];
    tmp_cells->speed1[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx];
    tmp_cells->speed2[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx];
    tmp_cells->speed3[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx];
    tmp_cells->speed4[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx];
    tmp_cells->speed5[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx];
    tmp_cells->speed6[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx];
    tmp_cells->speed7[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx];
    tmp_cells->speed8[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx];
  }
  else {
    const float c_sq = 1.f / 3.f; /* square of speed of sound */
    const float w0 = 4.f / 9.f;  /* weighting factor */
    const float w1 = 1.f / 9.f;  /* weighting factor */
    const float w2 = 1.f / 36.f; /* weighting factor */
    /* compute local density total */
    float local_density = 0.f;

    local_density += cells->speed0[ii + jj*params.nx];
    local_density += cells->speed1[x_w + jj*params.nx];
    local_density += cells->speed2[ii + y_s*params.nx];
    local_density += cells->speed3[x_e + jj*params.nx];
    local_density += cells->speed4[ii + y_n*params.nx];
    local_density += cells->speed5[x_w + y_s*params.nx];
    local_density += cells->speed6[x_e + y_s*params.nx];
    local_density += cells->speed7[x_e + y_n*params.nx];
    local_density += cells->speed8[x_w + y_n*params.nx];
    
  
    /* compute x velocity component */
    float u_x = (cells->speed1[x_w + jj*params.nx]
                  + cells->speed5[x_w + y_s*params.nx]
                  + cells->speed8[x_w + y_n*params.nx]
                  - (cells->speed3[x_e + jj*params.nx]
                      + cells->speed6[x_e + y_s*params.nx]
                      + cells->speed7[x_e + y_n*params.nx]))
                  / local_density;
    /* compute y velocity component */
    float u_y = (cells->speed2[ii + y_s*params.nx]
                  + cells->speed5[x_w + y_s*params.nx]
                  + cells->speed6[x_e + y_s*params.nx]
                  - (cells->speed4[ii + y_n*params.nx]
                      + cells->speed7[x_e + y_n*params.nx]
                      + cells->speed8[x_w + y_n*params.nx]))
                  / local_density;

    /* velocity squared */
    float u_sq = u_x * u_x + u_y * u_y;

    /* directional velocity components */
    float u[NSPEEDS];
    u[1] =   u_x;        /* east */
    u[2] =         u_y;  /* north */
    u[3] = - u_x;        /* west */
    u[4] =       - u_y;  /* south */
    u[5] =   u_x + u_y;  /* north-east */
    u[6] = - u_x + u_y;  /* north-west */
    u[7] = - u_x - u_y;  /* south-west */
    u[8] =   u_x - u_y;  /* south-east */

    /* equilibrium densities */
    float d_equ[NSPEEDS];
    /* zero velocity density: weight w0 */
    d_equ[0] = w0 * local_density
                * (1.f - u_sq / (2.f * c_sq));
    /* axis speeds: weight w1 */
    d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                      + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                      + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                      + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                      + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    /* diagonal speeds: weight w2 */
    d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                      + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                      + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                      + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                      + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));

    /* relaxation step */

    tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx]
                                              + params.omega
                                              * (d_equ[0] - cells->speed0[ii + jj*params.nx]);

    tmp_cells->speed1[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx]
                                              + params.omega
                                              * (d_equ[1] - cells->speed1[x_w + jj*params.nx]);

    tmp_cells->speed2[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx]
                                              + params.omega
                                              * (d_equ[2] - cells->speed2[ii + y_s*params.nx]);                                          
    
    tmp_cells->speed3[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx]
                                              + params.omega
                                              * (d_equ[3] - cells->speed3[x_e + jj*params.nx]);

    tmp_cells->speed4[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx]
                                              + params.omega
                                              * (d_equ[4] - cells->speed4[ii + y_n*params.nx]);
    
    tmp_cells->speed5[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx]
                                              + params.omega
                                              * (d_equ[5] - cells->speed5[x_w + y_s*params.nx]);
    
    tmp_cells->speed6[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx]
                                              + params.omega
                                              * (d_equ[6] - cells->speed6[x_e + y_s*params.nx]);
    
    tmp_cells->speed7[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx]
                                              + params.omega
                                              * (d_equ[7] - cells->speed7[x_e + y_n*params.nx]);
    
    tmp_cells->speed8[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx]
                                              + params.omega
                                              * (d_equ[8] - cells->speed8[x_w + y_n*params.nx]);


    /* local density total */
    local_density = 0.f;

    local_density += tmp_cells->speed0[ii + jj*params.nx];
    local_density += tmp_cells->speed1[ii + jj*params.nx];
    local_density += tmp_cells->speed2[ii + jj*params.nx];
    local_density += tmp_cells->speed3[ii + jj*params.nx];
    local_density += tmp_cells->speed4[ii + jj*params.nx];
    local_density += tmp_cells->speed5[ii + jj*params.nx];
    local_density += tmp_cells->speed6[ii + jj*params.nx];
    local_density += tmp_cells->speed7[ii + jj*params.nx];
    local_density += tmp_cells->speed8[ii + jj*params.nx];

    /* x-component of velocity */
    u_x = (tmp_cells->speed1[ii + jj*params.nx]
                  + tmp_cells->speed5[ii + jj*params.nx]
                  + tmp_cells->speed8[ii + jj*params.nx]
                  - (tmp_cells->speed3[ii + jj*params.nx]
                      + tmp_cells->speed6[ii + jj*params.nx]
                      + tmp_cells->speed7[ii + jj*params.nx]))
                  / local_density;
    /* compute y velocity component */
    u_y = (tmp_cells->speed2[ii + jj*params.nx]
                  + tmp_cells->speed5[ii + jj*params.nx]
                  + tmp_cells->speed6[ii + jj*params.nx]
                  - (tmp_cells->speed4[ii + jj*params.nx]
                      + tmp_cells->speed7[ii + jj*params.nx]
                      + tmp_cells->speed8[ii + jj*params.nx]))
                  / local_density;
    /* accumulate the norm of x- and y- velocity components */
    tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
    /* increase counter of inspected cells */
    ++tot_cells;
  }

  #pragma omp simd
  for (int ii = 1; ii < params.nx - 1; ii++)
  {
    /* determine indices of axis-direction neighbours
    ** respecting periodic boundary conditions (wrap around) */
    int x_e = ii + 1;
    int x_w = ii - 1;
    /* propagate densities from neighbouring cells, following
    ** appropriate directions of travel and writing into
    ** scratch space grid */
    if (obstacles[jj*params.nx + ii]) {
    tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx];
    tmp_cells->speed1[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx];
    tmp_cells->speed2[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx];
    tmp_cells->speed3[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx];
    tmp_cells->speed4[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx];
    tmp_cells->speed5[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx];
    tmp_cells->speed6[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx];
    tmp_cells->speed7[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx];
    tmp_cells->speed8[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx];
  }
  else {
    const float c_sq = 1.f / 3.f; /* square of speed of sound */
    const float w0 = 4.f / 9.f;  /* weighting factor */
    const float w1 = 1.f / 9.f;  /* weighting factor */
    const float w2 = 1.f / 36.f; /* weighting factor */
    /* compute local density total */
    float local_density = 0.f;

    local_density += cells->speed0[ii + jj*params.nx];
    local_density += cells->speed1[x_w + jj*params.nx];
    local_density += cells->speed2[ii + y_s*params.nx];
    local_density += cells->speed3[x_e + jj*params.nx];
    local_density += cells->speed4[ii + y_n*params.nx];
    local_density += cells->speed5[x_w + y_s*params.nx];
    local_density += cells->speed6[x_e + y_s*params.nx];
    local_density += cells->speed7[x_e + y_n*params.nx];
    local_density += cells->speed8[x_w + y_n*params.nx];
    
  
    /* compute x velocity component */
    float u_x = (cells->speed1[x_w + jj*params.nx]
                  + cells->speed5[x_w + y_s*params.nx]
                  + cells->speed8[x_w + y_n*params.nx]
                  - (cells->speed3[x_e + jj*params.nx]
                      + cells->speed6[x_e + y_s*params.nx]
                      + cells->speed7[x_e + y_n*params.nx]))
                  / local_density;
    /* compute y velocity component */
    float u_y = (cells->speed2[ii + y_s*params.nx]
                  + cells->speed5[x_w + y_s*params.nx]
                  + cells->speed6[x_e + y_s*params.nx]
                  - (cells->speed4[ii + y_n*params.nx]
                      + cells->speed7[x_e + y_n*params.nx]
                      + cells->speed8[x_w + y_n*params.nx]))
                  / local_density;

    /* velocity squared */
    float u_sq = u_x * u_x + u_y * u_y;

    /* directional velocity components */
    float u[NSPEEDS];
    u[1] =   u_x;        /* east */
    u[2] =         u_y;  /* north */
    u[3] = - u_x;        /* west */
    u[4] =       - u_y;  /* south */
    u[5] =   u_x + u_y;  /* north-east */
    u[6] = - u_x + u_y;  /* north-west */
    u[7] = - u_x - u_y;  /* south-west */
    u[8] =   u_x - u_y;  /* south-east */

    /* equilibrium densities */
    float d_equ[NSPEEDS];
    /* zero velocity density: weight w0 */
    d_equ[0] = w0 * local_density
                * (1.f - u_sq / (2.f * c_sq));
    /* axis speeds: weight w1 */
    d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                      + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                      + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                      + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                      + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    /* diagonal speeds: weight w2 */
    d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                      + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                      + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                      + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                      + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));

    /* relaxation step */

    tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx]
                                              + params.omega
                                              * (d_equ[0] - cells->speed0[ii + jj*params.nx]);

    tmp_cells->speed1[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx]
                                              + params.omega
                                              * (d_equ[1] - cells->speed1[x_w + jj*params.nx]);

    tmp_cells->speed2[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx]
                                              + params.omega
                                              * (d_equ[2] - cells->speed2[ii + y_s*params.nx]);                                          
    
    tmp_cells->speed3[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx]
                                              + params.omega
                                              * (d_equ[3] - cells->speed3[x_e + jj*params.nx]);

    tmp_cells->speed4[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx]
                                              + params.omega
                                              * (d_equ[4] - cells->speed4[ii + y_n*params.nx]);
    
    tmp_cells->speed5[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx]
                                              + params.omega
                                              * (d_equ[5] - cells->speed5[x_w + y_s*params.nx]);
    
    tmp_cells->speed6[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx]
                                              + params.omega
                                              * (d_equ[6] - cells->speed6[x_e + y_s*params.nx]);
    
    tmp_cells->speed7[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx]
                                              + params.omega
                                              * (d_equ[7] - cells->speed7[x_e + y_n*params.nx]);
    
    tmp_cells->speed8[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx]
                                              + params.omega
                                              * (d_equ[8] - cells->speed8[x_w + y_n*params.nx]);


    /* local density total */
    local_density = 0.f;

    local_density += tmp_cells->speed0[ii + jj*params.nx];
    local_density += tmp_cells->speed1[ii + jj*params.nx];
    local_density += tmp_cells->speed2[ii + jj*params.nx];
    local_density += tmp_cells->speed3[ii + jj*params.nx];
    local_density += tmp_cells->speed4[ii + jj*params.nx];
    local_density += tmp_cells->speed5[ii + jj*params.nx];
    local_density += tmp_cells->speed6[ii + jj*params.nx];
    local_density += tmp_cells->speed7[ii + jj*params.nx];
    local_density += tmp_cells->speed8[ii + jj*params.nx];

    /* x-component of velocity */
    u_x = (tmp_cells->speed1[ii + jj*params.nx]
                  + tmp_cells->speed5[ii + jj*params.nx]
                  + tmp_cells->speed8[ii + jj*params.nx]
                  - (tmp_cells->speed3[ii + jj*params.nx]
                      + tmp_cells->speed6[ii + jj*params.nx]
                      + tmp_cells->speed7[ii + jj*params.nx]))
                  / local_density;
    /* compute y velocity component */
    u_y = (tmp_cells->speed2[ii + jj*params.nx]
                  + tmp_cells->speed5[ii + jj*params.nx]
                  + tmp_cells->speed6[ii + jj*params.nx]
                  - (tmp_cells->speed4[ii + jj*params.nx]
                      + tmp_cells->speed7[ii + jj*params.nx]
                      + tmp_cells->speed8[ii + jj*params.nx]))
                  / local_density;
    /* accumulate the norm of x- and y- velocity components */
    tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
    /* increase counter of inspected cells */
    ++tot_cells;
  }
  }

  ii = params.nx - 1;
  x_e = 0;
  x_w = (ii - 1);
  if (obstacles[jj*params.nx + ii]) {
    tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx];
    tmp_cells->speed1[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx];
    tmp_cells->speed2[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx];
    tmp_cells->speed3[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx];
    tmp_cells->speed4[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx];
    tmp_cells->speed5[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx];
    tmp_cells->speed6[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx];
    tmp_cells->speed7[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx];
    tmp_cells->speed8[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx];
  }
  else {
    const float c_sq = 1.f / 3.f; /* square of speed of sound */
    const float w0 = 4.f / 9.f;  /* weighting factor */
    const float w1 = 1.f / 9.f;  /* weighting factor */
    const float w2 = 1.f / 36.f; /* weighting factor */
    /* compute local density total */
    float local_density = 0.f;

    local_density += cells->speed0[ii + jj*params.nx];
    local_density += cells->speed1[x_w + jj*params.nx];
    local_density += cells->speed2[ii + y_s*params.nx];
    local_density += cells->speed3[x_e + jj*params.nx];
    local_density += cells->speed4[ii + y_n*params.nx];
    local_density += cells->speed5[x_w + y_s*params.nx];
    local_density += cells->speed6[x_e + y_s*params.nx];
    local_density += cells->speed7[x_e + y_n*params.nx];
    local_density += cells->speed8[x_w + y_n*params.nx];
    
  
    /* compute x velocity component */
    float u_x = (cells->speed1[x_w + jj*params.nx]
                  + cells->speed5[x_w + y_s*params.nx]
                  + cells->speed8[x_w + y_n*params.nx]
                  - (cells->speed3[x_e + jj*params.nx]
                      + cells->speed6[x_e + y_s*params.nx]
                      + cells->speed7[x_e + y_n*params.nx]))
                  / local_density;
    /* compute y velocity component */
    float u_y = (cells->speed2[ii + y_s*params.nx]
                  + cells->speed5[x_w + y_s*params.nx]
                  + cells->speed6[x_e + y_s*params.nx]
                  - (cells->speed4[ii + y_n*params.nx]
                      + cells->speed7[x_e + y_n*params.nx]
                      + cells->speed8[x_w + y_n*params.nx]))
                  / local_density;

    /* velocity squared */
    float u_sq = u_x * u_x + u_y * u_y;

    /* directional velocity components */
    float u[NSPEEDS];
    u[1] =   u_x;        /* east */
    u[2] =         u_y;  /* north */
    u[3] = - u_x;        /* west */
    u[4] =       - u_y;  /* south */
    u[5] =   u_x + u_y;  /* north-east */
    u[6] = - u_x + u_y;  /* north-west */
    u[7] = - u_x - u_y;  /* south-west */
    u[8] =   u_x - u_y;  /* south-east */

    /* equilibrium densities */
    float d_equ[NSPEEDS];
    /* zero velocity density: weight w0 */
    d_equ[0] = w0 * local_density
                * (1.f - u_sq / (2.f * c_sq));
    /* axis speeds: weight w1 */
    d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                      + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                      + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                      + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                      + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    /* diagonal speeds: weight w2 */
    d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                      + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                      + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                      + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                      + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));

    /* relaxation step */

    tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx]
                                              + params.omega
                                              * (d_equ[0] - cells->speed0[ii + jj*params.nx]);

    tmp_cells->speed1[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx]
                                              + params.omega
                                              * (d_equ[1] - cells->speed1[x_w + jj*params.nx]);

    tmp_cells->speed2[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx]
                                              + params.omega
                                              * (d_equ[2] - cells->speed2[ii + y_s*params.nx]);                                          
    
    tmp_cells->speed3[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx]
                                              + params.omega
                                              * (d_equ[3] - cells->speed3[x_e + jj*params.nx]);

    tmp_cells->speed4[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx]
                                              + params.omega
                                              * (d_equ[4] - cells->speed4[ii + y_n*params.nx]);
    
    tmp_cells->speed5[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx]
                                              + params.omega
                                              * (d_equ[5] - cells->speed5[x_w + y_s*params.nx]);
    
    tmp_cells->speed6[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx]
                                              + params.omega
                                              * (d_equ[6] - cells->speed6[x_e + y_s*params.nx]);
    
    tmp_cells->speed7[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx]
                                              + params.omega
                                              * (d_equ[7] - cells->speed7[x_e + y_n*params.nx]);
    
    tmp_cells->speed8[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx]
                                              + params.omega
                                              * (d_equ[8] - cells->speed8[x_w + y_n*params.nx]);


    /* local density total */
    local_density = 0.f;

    local_density += tmp_cells->speed0[ii + jj*params.nx];
    local_density += tmp_cells->speed1[ii + jj*params.nx];
    local_density += tmp_cells->speed2[ii + jj*params.nx];
    local_density += tmp_cells->speed3[ii + jj*params.nx];
    local_density += tmp_cells->speed4[ii + jj*params.nx];
    local_density += tmp_cells->speed5[ii + jj*params.nx];
    local_density += tmp_cells->speed6[ii + jj*params.nx];
    local_density += tmp_cells->speed7[ii + jj*params.nx];
    local_density += tmp_cells->speed8[ii + jj*params.nx];

    /* x-component of velocity */
    u_x = (tmp_cells->speed1[ii + jj*params.nx]
                  + tmp_cells->speed5[ii + jj*params.nx]
                  + tmp_cells->speed8[ii + jj*params.nx]
                  - (tmp_cells->speed3[ii + jj*params.nx]
                      + tmp_cells->speed6[ii + jj*params.nx]
                      + tmp_cells->speed7[ii + jj*params.nx]))
                  / local_density;
    /* compute y velocity component */
    u_y = (tmp_cells->speed2[ii + jj*params.nx]
                  + tmp_cells->speed5[ii + jj*params.nx]
                  + tmp_cells->speed6[ii + jj*params.nx]
                  - (tmp_cells->speed4[ii + jj*params.nx]
                      + tmp_cells->speed7[ii + jj*params.nx]
                      + tmp_cells->speed8[ii + jj*params.nx]))
                  / local_density;
    /* accumulate the norm of x- and y- velocity components */
    tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
    /* increase counter of inspected cells */
    ++tot_cells;
  }

  for (int jj = 1; jj < params.ny - 1; jj++)
  {
    y_s = (jj - 1);

    //Case where ii = 0
    int ii = 0;
    int y_n = jj + 1;
    int x_e = ii + 1;
    int x_w = ii + params.nx - 1;

    if (obstacles[jj*params.nx + ii]) {
      tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx];
      tmp_cells->speed1[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx];
      tmp_cells->speed2[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx];
      tmp_cells->speed3[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx];
      tmp_cells->speed4[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx];
      tmp_cells->speed5[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx];
      tmp_cells->speed6[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx];
      tmp_cells->speed7[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx];
      tmp_cells->speed8[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx];
    }
    else {
      const float c_sq = 1.f / 3.f; /* square of speed of sound */
      const float w0 = 4.f / 9.f;  /* weighting factor */
      const float w1 = 1.f / 9.f;  /* weighting factor */
      const float w2 = 1.f / 36.f; /* weighting factor */
      /* compute local density total */
      float local_density = 0.f;

      local_density += cells->speed0[ii + jj*params.nx];
      local_density += cells->speed1[x_w + jj*params.nx];
      local_density += cells->speed2[ii + y_s*params.nx];
      local_density += cells->speed3[x_e + jj*params.nx];
      local_density += cells->speed4[ii + y_n*params.nx];
      local_density += cells->speed5[x_w + y_s*params.nx];
      local_density += cells->speed6[x_e + y_s*params.nx];
      local_density += cells->speed7[x_e + y_n*params.nx];
      local_density += cells->speed8[x_w + y_n*params.nx];
      
    
      /* compute x velocity component */
      float u_x = (cells->speed1[x_w + jj*params.nx]
                    + cells->speed5[x_w + y_s*params.nx]
                    + cells->speed8[x_w + y_n*params.nx]
                    - (cells->speed3[x_e + jj*params.nx]
                        + cells->speed6[x_e + y_s*params.nx]
                        + cells->speed7[x_e + y_n*params.nx]))
                    / local_density;
      /* compute y velocity component */
      float u_y = (cells->speed2[ii + y_s*params.nx]
                    + cells->speed5[x_w + y_s*params.nx]
                    + cells->speed6[x_e + y_s*params.nx]
                    - (cells->speed4[ii + y_n*params.nx]
                        + cells->speed7[x_e + y_n*params.nx]
                        + cells->speed8[x_w + y_n*params.nx]))
                    / local_density;

      /* velocity squared */
      float u_sq = u_x * u_x + u_y * u_y;

      /* directional velocity components */
      float u[NSPEEDS];
      u[1] =   u_x;        /* east */
      u[2] =         u_y;  /* north */
      u[3] = - u_x;        /* west */
      u[4] =       - u_y;  /* south */
      u[5] =   u_x + u_y;  /* north-east */
      u[6] = - u_x + u_y;  /* north-west */
      u[7] = - u_x - u_y;  /* south-west */
      u[8] =   u_x - u_y;  /* south-east */

      /* equilibrium densities */
      float d_equ[NSPEEDS];
      /* zero velocity density: weight w0 */
      d_equ[0] = w0 * local_density
                  * (1.f - u_sq / (2.f * c_sq));
      /* axis speeds: weight w1 */
      d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                        + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                        + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                        + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                        + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      /* diagonal speeds: weight w2 */
      d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                        + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                        + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                        + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                        + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));

      /* relaxation step */

      tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx]
                                                + params.omega
                                                * (d_equ[0] - cells->speed0[ii + jj*params.nx]);

      tmp_cells->speed1[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx]
                                                + params.omega
                                                * (d_equ[1] - cells->speed1[x_w + jj*params.nx]);

      tmp_cells->speed2[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx]
                                                + params.omega
                                                * (d_equ[2] - cells->speed2[ii + y_s*params.nx]);                                                                                                                                                    
                                                                                       
      tmp_cells->speed3[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx]
                                                + params.omega
                                                * (d_equ[3] - cells->speed3[x_e + jj*params.nx]);

      tmp_cells->speed4[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx]
                                                + params.omega
                                                * (d_equ[4] - cells->speed4[ii + y_n*params.nx]);
      
      tmp_cells->speed5[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx]
                                                + params.omega
                                                * (d_equ[5] - cells->speed5[x_w + y_s*params.nx]);
      
      tmp_cells->speed6[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx]
                                                + params.omega
                                                * (d_equ[6] - cells->speed6[x_e + y_s*params.nx]);
      
      tmp_cells->speed7[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx]
                                                + params.omega
                                                * (d_equ[7] - cells->speed7[x_e + y_n*params.nx]);
      
      tmp_cells->speed8[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx]
                                                + params.omega
                                                * (d_equ[8] - cells->speed8[x_w + y_n*params.nx]);

      /* local density total */
      local_density = 0.f;

      local_density += tmp_cells->speed0[ii + jj*params.nx];
      local_density += tmp_cells->speed1[ii + jj*params.nx];
      local_density += tmp_cells->speed2[ii + jj*params.nx];
      local_density += tmp_cells->speed3[ii + jj*params.nx];
      local_density += tmp_cells->speed4[ii + jj*params.nx];
      local_density += tmp_cells->speed5[ii + jj*params.nx];
      local_density += tmp_cells->speed6[ii + jj*params.nx];
      local_density += tmp_cells->speed7[ii + jj*params.nx];
      local_density += tmp_cells->speed8[ii + jj*params.nx];

      /* x-component of velocity */
      u_x = (tmp_cells->speed1[ii + jj*params.nx]
                    + tmp_cells->speed5[ii + jj*params.nx]
                    + tmp_cells->speed8[ii + jj*params.nx]
                    - (tmp_cells->speed3[ii + jj*params.nx]
                        + tmp_cells->speed6[ii + jj*params.nx]
                        + tmp_cells->speed7[ii + jj*params.nx]))
                    / local_density;
      /* compute y velocity component */
      u_y = (tmp_cells->speed2[ii + jj*params.nx]
                    + tmp_cells->speed5[ii + jj*params.nx]
                    + tmp_cells->speed6[ii + jj*params.nx]
                    - (tmp_cells->speed4[ii + jj*params.nx]
                        + tmp_cells->speed7[ii + jj*params.nx]
                        + tmp_cells->speed8[ii + jj*params.nx]))
                    / local_density;
      /* accumulate the norm of x- and y- velocity components */
      tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
      /* increase counter of inspected cells */
      ++tot_cells;

    }

    #pragma omp simd
    //Case where i > 0 and i < params.nx - 1
    for (int ii = 1; ii < params.nx - 1; ii++)
    {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int x_e = ii + 1;
      int x_w = ii - 1;
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      if (obstacles[jj*params.nx + ii]) {
        tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx];
        tmp_cells->speed1[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx];
        tmp_cells->speed2[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx];
        tmp_cells->speed3[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx];
        tmp_cells->speed4[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx];
        tmp_cells->speed5[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx];
        tmp_cells->speed6[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx];
        tmp_cells->speed7[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx];
        tmp_cells->speed8[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx];
      }
      else {
        const float c_sq = 1.f / 3.f; /* square of speed of sound */
        const float w0 = 4.f / 9.f;  /* weighting factor */
        const float w1 = 1.f / 9.f;  /* weighting factor */
        const float w2 = 1.f / 36.f; /* weighting factor */
        /* compute local density total */
        float local_density = 0.f;

        local_density += cells->speed0[ii + jj*params.nx];
        local_density += cells->speed1[x_w + jj*params.nx];
        local_density += cells->speed2[ii + y_s*params.nx];
        local_density += cells->speed3[x_e + jj*params.nx];
        local_density += cells->speed4[ii + y_n*params.nx];
        local_density += cells->speed5[x_w + y_s*params.nx];
        local_density += cells->speed6[x_e + y_s*params.nx];
        local_density += cells->speed7[x_e + y_n*params.nx];
        local_density += cells->speed8[x_w + y_n*params.nx];
        
      
        /* compute x velocity component */
        float u_x = (cells->speed1[x_w + jj*params.nx]
                      + cells->speed5[x_w + y_s*params.nx]
                      + cells->speed8[x_w + y_n*params.nx]
                      - (cells->speed3[x_e + jj*params.nx]
                          + cells->speed6[x_e + y_s*params.nx]
                          + cells->speed7[x_e + y_n*params.nx]))
                      / local_density;
        /* compute y velocity component */
        float u_y = (cells->speed2[ii + y_s*params.nx]
                      + cells->speed5[x_w + y_s*params.nx]
                      + cells->speed6[x_e + y_s*params.nx]
                      - (cells->speed4[ii + y_n*params.nx]
                          + cells->speed7[x_e + y_n*params.nx]
                          + cells->speed8[x_w + y_n*params.nx]))
                      / local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u[NSPEEDS];
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */

        /* equilibrium densities */
        float d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */
        d_equ[0] = w0 * local_density
                    * (1.f - u_sq / (2.f * c_sq));
        /* axis speeds: weight w1 */
        d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                          + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                          + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                          + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                          + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        /* diagonal speeds: weight w2 */
        d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                          + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                          + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                          + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                          + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));

        /* relaxation step */

        tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx]
                                                  + params.omega
                                                  * (d_equ[0] - cells->speed0[ii + jj*params.nx]);

        tmp_cells->speed1[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx]
                                                  + params.omega
                                                  * (d_equ[1] - cells->speed1[x_w + jj*params.nx]);

        tmp_cells->speed2[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx]
                                                  + params.omega
                                                  * (d_equ[2] - cells->speed2[ii + y_s*params.nx]);                                          
        
        tmp_cells->speed3[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx]
                                                  + params.omega
                                                  * (d_equ[3] - cells->speed3[x_e + jj*params.nx]);

        tmp_cells->speed4[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx]
                                                  + params.omega
                                                  * (d_equ[4] - cells->speed4[ii + y_n*params.nx]);
        
        tmp_cells->speed5[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx]
                                                  + params.omega
                                                  * (d_equ[5] - cells->speed5[x_w + y_s*params.nx]);
        
        tmp_cells->speed6[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx]
                                                  + params.omega
                                                  * (d_equ[6] - cells->speed6[x_e + y_s*params.nx]);
        
        tmp_cells->speed7[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx]
                                                  + params.omega
                                                  * (d_equ[7] - cells->speed7[x_e + y_n*params.nx]);
        
        tmp_cells->speed8[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx]
                                                  + params.omega
                                                  * (d_equ[8] - cells->speed8[x_w + y_n*params.nx]);


        /* local density total */
        local_density = 0.f;

        local_density += tmp_cells->speed0[ii + jj*params.nx];
        local_density += tmp_cells->speed1[ii + jj*params.nx];
        local_density += tmp_cells->speed2[ii + jj*params.nx];
        local_density += tmp_cells->speed3[ii + jj*params.nx];
        local_density += tmp_cells->speed4[ii + jj*params.nx];
        local_density += tmp_cells->speed5[ii + jj*params.nx];
        local_density += tmp_cells->speed6[ii + jj*params.nx];
        local_density += tmp_cells->speed7[ii + jj*params.nx];
        local_density += tmp_cells->speed8[ii + jj*params.nx];

        /* x-component of velocity */
        u_x = (tmp_cells->speed1[ii + jj*params.nx]
                      + tmp_cells->speed5[ii + jj*params.nx]
                      + tmp_cells->speed8[ii + jj*params.nx]
                      - (tmp_cells->speed3[ii + jj*params.nx]
                          + tmp_cells->speed6[ii + jj*params.nx]
                          + tmp_cells->speed7[ii + jj*params.nx]))
                      / local_density;
        /* compute y velocity component */
        u_y = (tmp_cells->speed2[ii + jj*params.nx]
                      + tmp_cells->speed5[ii + jj*params.nx]
                      + tmp_cells->speed6[ii + jj*params.nx]
                      - (tmp_cells->speed4[ii + jj*params.nx]
                          + tmp_cells->speed7[ii + jj*params.nx]
                          + tmp_cells->speed8[ii + jj*params.nx]))
                      / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;

      }
    }

    ii = params.nx - 1;
    x_e = 0;
    x_w = ii - 1;
    if (obstacles[jj*params.nx + ii]) {
    tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx];
    tmp_cells->speed1[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx];
    tmp_cells->speed2[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx];
    tmp_cells->speed3[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx];
    tmp_cells->speed4[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx];
    tmp_cells->speed5[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx];
    tmp_cells->speed6[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx];
    tmp_cells->speed7[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx];
    tmp_cells->speed8[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx];
  }
  else {
    const float c_sq = 1.f / 3.f; /* square of speed of sound */
    const float w0 = 4.f / 9.f;  /* weighting factor */
    const float w1 = 1.f / 9.f;  /* weighting factor */
    const float w2 = 1.f / 36.f; /* weighting factor */
    /* compute local density total */
    float local_density = 0.f;

    local_density += cells->speed0[ii + jj*params.nx];
    local_density += cells->speed1[x_w + jj*params.nx];
    local_density += cells->speed2[ii + y_s*params.nx];
    local_density += cells->speed3[x_e + jj*params.nx];
    local_density += cells->speed4[ii + y_n*params.nx];
    local_density += cells->speed5[x_w + y_s*params.nx];
    local_density += cells->speed6[x_e + y_s*params.nx];
    local_density += cells->speed7[x_e + y_n*params.nx];
    local_density += cells->speed8[x_w + y_n*params.nx];
    
  
    /* compute x velocity component */
    float u_x = (cells->speed1[x_w + jj*params.nx]
                  + cells->speed5[x_w + y_s*params.nx]
                  + cells->speed8[x_w + y_n*params.nx]
                  - (cells->speed3[x_e + jj*params.nx]
                      + cells->speed6[x_e + y_s*params.nx]
                      + cells->speed7[x_e + y_n*params.nx]))
                  / local_density;
    /* compute y velocity component */
    float u_y = (cells->speed2[ii + y_s*params.nx]
                  + cells->speed5[x_w + y_s*params.nx]
                  + cells->speed6[x_e + y_s*params.nx]
                  - (cells->speed4[ii + y_n*params.nx]
                      + cells->speed7[x_e + y_n*params.nx]
                      + cells->speed8[x_w + y_n*params.nx]))
                  / local_density;

    /* velocity squared */
    float u_sq = u_x * u_x + u_y * u_y;

    /* directional velocity components */
    float u[NSPEEDS];
    u[1] =   u_x;        /* east */
    u[2] =         u_y;  /* north */
    u[3] = - u_x;        /* west */
    u[4] =       - u_y;  /* south */
    u[5] =   u_x + u_y;  /* north-east */
    u[6] = - u_x + u_y;  /* north-west */
    u[7] = - u_x - u_y;  /* south-west */
    u[8] =   u_x - u_y;  /* south-east */

    /* equilibrium densities */
    float d_equ[NSPEEDS];
    /* zero velocity density: weight w0 */
    d_equ[0] = w0 * local_density
                * (1.f - u_sq / (2.f * c_sq));
    /* axis speeds: weight w1 */
    d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                      + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                      + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                      + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                      + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    /* diagonal speeds: weight w2 */
    d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                      + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                      + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                      + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                      + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));

    /* relaxation step */

    tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx]
                                              + params.omega
                                              * (d_equ[0] - cells->speed0[ii + jj*params.nx]);

    tmp_cells->speed1[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx]
                                              + params.omega
                                              * (d_equ[1] - cells->speed1[x_w + jj*params.nx]);

    tmp_cells->speed2[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx]
                                              + params.omega
                                              * (d_equ[2] - cells->speed2[ii + y_s*params.nx]);                                          
    
    tmp_cells->speed3[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx]
                                              + params.omega
                                              * (d_equ[3] - cells->speed3[x_e + jj*params.nx]);

    tmp_cells->speed4[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx]
                                              + params.omega
                                              * (d_equ[4] - cells->speed4[ii + y_n*params.nx]);
    
    tmp_cells->speed5[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx]
                                              + params.omega
                                              * (d_equ[5] - cells->speed5[x_w + y_s*params.nx]);
    
    tmp_cells->speed6[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx]
                                              + params.omega
                                              * (d_equ[6] - cells->speed6[x_e + y_s*params.nx]);
    
    tmp_cells->speed7[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx]
                                              + params.omega
                                              * (d_equ[7] - cells->speed7[x_e + y_n*params.nx]);
    
    tmp_cells->speed8[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx]
                                              + params.omega
                                              * (d_equ[8] - cells->speed8[x_w + y_n*params.nx]);


    /* local density total */
    local_density = 0.f;

    local_density += tmp_cells->speed0[ii + jj*params.nx];
    local_density += tmp_cells->speed1[ii + jj*params.nx];
    local_density += tmp_cells->speed2[ii + jj*params.nx];
    local_density += tmp_cells->speed3[ii + jj*params.nx];
    local_density += tmp_cells->speed4[ii + jj*params.nx];
    local_density += tmp_cells->speed5[ii + jj*params.nx];
    local_density += tmp_cells->speed6[ii + jj*params.nx];
    local_density += tmp_cells->speed7[ii + jj*params.nx];
    local_density += tmp_cells->speed8[ii + jj*params.nx];

    /* x-component of velocity */
    u_x = (tmp_cells->speed1[ii + jj*params.nx]
                  + tmp_cells->speed5[ii + jj*params.nx]
                  + tmp_cells->speed8[ii + jj*params.nx]
                  - (tmp_cells->speed3[ii + jj*params.nx]
                      + tmp_cells->speed6[ii + jj*params.nx]
                      + tmp_cells->speed7[ii + jj*params.nx]))
                  / local_density;
    /* compute y velocity component */
    u_y = (tmp_cells->speed2[ii + jj*params.nx]
                  + tmp_cells->speed5[ii + jj*params.nx]
                  + tmp_cells->speed6[ii + jj*params.nx]
                  - (tmp_cells->speed4[ii + jj*params.nx]
                      + tmp_cells->speed7[ii + jj*params.nx]
                      + tmp_cells->speed8[ii + jj*params.nx]))
                  / local_density;
    /* accumulate the norm of x- and y- velocity components */
    tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
    /* increase counter of inspected cells */
    ++tot_cells;
  }
  
  }


  //Case where jj = params.ny - 1
  jj = params.ny - 1;
  y_s = jj - 1;
  y_n = 0;

  //Case where ii = 0
  ii = 0;
  x_e = ii + 1;
  x_w = ii + params.nx - 1;

  if (obstacles[jj*params.nx + ii]) {
    tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx];
    tmp_cells->speed1[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx];
    tmp_cells->speed2[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx];
    tmp_cells->speed3[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx];
    tmp_cells->speed4[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx];
    tmp_cells->speed5[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx];
    tmp_cells->speed6[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx];
    tmp_cells->speed7[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx];
    tmp_cells->speed8[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx];
  }
  else {
    const float c_sq = 1.f / 3.f; /* square of speed of sound */
    const float w0 = 4.f / 9.f;  /* weighting factor */
    const float w1 = 1.f / 9.f;  /* weighting factor */
    const float w2 = 1.f / 36.f; /* weighting factor */
    /* compute local density total */
    float local_density = 0.f;

    local_density += cells->speed0[ii + jj*params.nx];
    local_density += cells->speed1[x_w + jj*params.nx];
    local_density += cells->speed2[ii + y_s*params.nx];
    local_density += cells->speed3[x_e + jj*params.nx];
    local_density += cells->speed4[ii + y_n*params.nx];
    local_density += cells->speed5[x_w + y_s*params.nx];
    local_density += cells->speed6[x_e + y_s*params.nx];
    local_density += cells->speed7[x_e + y_n*params.nx];
    local_density += cells->speed8[x_w + y_n*params.nx];
    
  
    /* compute x velocity component */
    float u_x = (cells->speed1[x_w + jj*params.nx]
                  + cells->speed5[x_w + y_s*params.nx]
                  + cells->speed8[x_w + y_n*params.nx]
                  - (cells->speed3[x_e + jj*params.nx]
                      + cells->speed6[x_e + y_s*params.nx]
                      + cells->speed7[x_e + y_n*params.nx]))
                  / local_density;
    /* compute y velocity component */
    float u_y = (cells->speed2[ii + y_s*params.nx]
                  + cells->speed5[x_w + y_s*params.nx]
                  + cells->speed6[x_e + y_s*params.nx]
                  - (cells->speed4[ii + y_n*params.nx]
                      + cells->speed7[x_e + y_n*params.nx]
                      + cells->speed8[x_w + y_n*params.nx]))
                  / local_density;

    /* velocity squared */
    float u_sq = u_x * u_x + u_y * u_y;

    /* directional velocity components */
    float u[NSPEEDS];
    u[1] =   u_x;        /* east */
    u[2] =         u_y;  /* north */
    u[3] = - u_x;        /* west */
    u[4] =       - u_y;  /* south */
    u[5] =   u_x + u_y;  /* north-east */
    u[6] = - u_x + u_y;  /* north-west */
    u[7] = - u_x - u_y;  /* south-west */
    u[8] =   u_x - u_y;  /* south-east */

    /* equilibrium densities */
    float d_equ[NSPEEDS];
    /* zero velocity density: weight w0 */
    d_equ[0] = w0 * local_density
                * (1.f - u_sq / (2.f * c_sq));
    /* axis speeds: weight w1 */
    d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                      + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                      + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                      + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                      + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    /* diagonal speeds: weight w2 */
    d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                      + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                      + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                      + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                      + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));

    /* relaxation step */

    tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx]
                                              + params.omega
                                              * (d_equ[0] - cells->speed0[ii + jj*params.nx]);

    tmp_cells->speed1[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx]
                                              + params.omega
                                              * (d_equ[1] - cells->speed1[x_w + jj*params.nx]);

    tmp_cells->speed2[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx]
                                              + params.omega
                                              * (d_equ[2] - cells->speed2[ii + y_s*params.nx]);                                          
    
    tmp_cells->speed3[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx]
                                              + params.omega
                                              * (d_equ[3] - cells->speed3[x_e + jj*params.nx]);

    tmp_cells->speed4[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx]
                                              + params.omega
                                              * (d_equ[4] - cells->speed4[ii + y_n*params.nx]);
    
    tmp_cells->speed5[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx]
                                              + params.omega
                                              * (d_equ[5] - cells->speed5[x_w + y_s*params.nx]);
    
    tmp_cells->speed6[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx]
                                              + params.omega
                                              * (d_equ[6] - cells->speed6[x_e + y_s*params.nx]);
    
    tmp_cells->speed7[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx]
                                              + params.omega
                                              * (d_equ[7] - cells->speed7[x_e + y_n*params.nx]);
    
    tmp_cells->speed8[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx]
                                              + params.omega
                                              * (d_equ[8] - cells->speed8[x_w + y_n*params.nx]);

    /* local density total */
    local_density = 0.f;

    local_density += tmp_cells->speed0[ii + jj*params.nx];
    local_density += tmp_cells->speed1[ii + jj*params.nx];
    local_density += tmp_cells->speed2[ii + jj*params.nx];
    local_density += tmp_cells->speed3[ii + jj*params.nx];
    local_density += tmp_cells->speed4[ii + jj*params.nx];
    local_density += tmp_cells->speed5[ii + jj*params.nx];
    local_density += tmp_cells->speed6[ii + jj*params.nx];
    local_density += tmp_cells->speed7[ii + jj*params.nx];
    local_density += tmp_cells->speed8[ii + jj*params.nx];

    /* x-component of velocity */
    u_x = (tmp_cells->speed1[ii + jj*params.nx]
                  + tmp_cells->speed5[ii + jj*params.nx]
                  + tmp_cells->speed8[ii + jj*params.nx]
                  - (tmp_cells->speed3[ii + jj*params.nx]
                      + tmp_cells->speed6[ii + jj*params.nx]
                      + tmp_cells->speed7[ii + jj*params.nx]))
                  / local_density;
    /* compute y velocity component */
    u_y = (tmp_cells->speed2[ii + jj*params.nx]
                  + tmp_cells->speed5[ii + jj*params.nx]
                  + tmp_cells->speed6[ii + jj*params.nx]
                  - (tmp_cells->speed4[ii + jj*params.nx]
                      + tmp_cells->speed7[ii + jj*params.nx]
                      + tmp_cells->speed8[ii + jj*params.nx]))
                  / local_density;
    /* accumulate the norm of x- and y- velocity components */
    tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
    /* increase counter of inspected cells */
    ++tot_cells;

  }

  #pragma omp simd
  for (int ii = 1; ii < params.nx - 1; ii++)
    {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int x_e = ii + 1;
      int x_w = ii - 1;
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      if (obstacles[jj*params.nx + ii]) {
    tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx];
    tmp_cells->speed1[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx];
    tmp_cells->speed2[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx];
    tmp_cells->speed3[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx];
    tmp_cells->speed4[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx];
    tmp_cells->speed5[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx];
    tmp_cells->speed6[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx];
    tmp_cells->speed7[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx];
    tmp_cells->speed8[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx];
  }
  else {
    const float c_sq = 1.f / 3.f; /* square of speed of sound */
    const float w0 = 4.f / 9.f;  /* weighting factor */
    const float w1 = 1.f / 9.f;  /* weighting factor */
    const float w2 = 1.f / 36.f; /* weighting factor */
    /* compute local density total */
    float local_density = 0.f;

    local_density += cells->speed0[ii + jj*params.nx];
    local_density += cells->speed1[x_w + jj*params.nx];
    local_density += cells->speed2[ii + y_s*params.nx];
    local_density += cells->speed3[x_e + jj*params.nx];
    local_density += cells->speed4[ii + y_n*params.nx];
    local_density += cells->speed5[x_w + y_s*params.nx];
    local_density += cells->speed6[x_e + y_s*params.nx];
    local_density += cells->speed7[x_e + y_n*params.nx];
    local_density += cells->speed8[x_w + y_n*params.nx];
    
  
    /* compute x velocity component */
    float u_x = (cells->speed1[x_w + jj*params.nx]
                  + cells->speed5[x_w + y_s*params.nx]
                  + cells->speed8[x_w + y_n*params.nx]
                  - (cells->speed3[x_e + jj*params.nx]
                      + cells->speed6[x_e + y_s*params.nx]
                      + cells->speed7[x_e + y_n*params.nx]))
                  / local_density;
    /* compute y velocity component */
    float u_y = (cells->speed2[ii + y_s*params.nx]
                  + cells->speed5[x_w + y_s*params.nx]
                  + cells->speed6[x_e + y_s*params.nx]
                  - (cells->speed4[ii + y_n*params.nx]
                      + cells->speed7[x_e + y_n*params.nx]
                      + cells->speed8[x_w + y_n*params.nx]))
                  / local_density;

    /* velocity squared */
    float u_sq = u_x * u_x + u_y * u_y;

    /* directional velocity components */
    float u[NSPEEDS];
    u[1] =   u_x;        /* east */
    u[2] =         u_y;  /* north */
    u[3] = - u_x;        /* west */
    u[4] =       - u_y;  /* south */
    u[5] =   u_x + u_y;  /* north-east */
    u[6] = - u_x + u_y;  /* north-west */
    u[7] = - u_x - u_y;  /* south-west */
    u[8] =   u_x - u_y;  /* south-east */

    /* equilibrium densities */
    float d_equ[NSPEEDS];
    /* zero velocity density: weight w0 */
    d_equ[0] = w0 * local_density
                * (1.f - u_sq / (2.f * c_sq));
    /* axis speeds: weight w1 */
    d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                      + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                      + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                      + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                      + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    /* diagonal speeds: weight w2 */
    d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                      + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                      + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                      + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                      + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));

    /* relaxation step */

    tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx]
                                              + params.omega
                                              * (d_equ[0] - cells->speed0[ii + jj*params.nx]);

    tmp_cells->speed1[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx]
                                              + params.omega
                                              * (d_equ[1] - cells->speed1[x_w + jj*params.nx]);

    tmp_cells->speed2[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx]
                                              + params.omega
                                              * (d_equ[2] - cells->speed2[ii + y_s*params.nx]);                                          
    
    tmp_cells->speed3[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx]
                                              + params.omega
                                              * (d_equ[3] - cells->speed3[x_e + jj*params.nx]);

    tmp_cells->speed4[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx]
                                              + params.omega
                                              * (d_equ[4] - cells->speed4[ii + y_n*params.nx]);
    
    tmp_cells->speed5[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx]
                                              + params.omega
                                              * (d_equ[5] - cells->speed5[x_w + y_s*params.nx]);
    
    tmp_cells->speed6[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx]
                                              + params.omega
                                              * (d_equ[6] - cells->speed6[x_e + y_s*params.nx]);
    
    tmp_cells->speed7[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx]
                                              + params.omega
                                              * (d_equ[7] - cells->speed7[x_e + y_n*params.nx]);
    
    tmp_cells->speed8[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx]
                                              + params.omega
                                              * (d_equ[8] - cells->speed8[x_w + y_n*params.nx]);


    /* local density total */
    local_density = 0.f;

    local_density += tmp_cells->speed0[ii + jj*params.nx];
    local_density += tmp_cells->speed1[ii + jj*params.nx];
    local_density += tmp_cells->speed2[ii + jj*params.nx];
    local_density += tmp_cells->speed3[ii + jj*params.nx];
    local_density += tmp_cells->speed4[ii + jj*params.nx];
    local_density += tmp_cells->speed5[ii + jj*params.nx];
    local_density += tmp_cells->speed6[ii + jj*params.nx];
    local_density += tmp_cells->speed7[ii + jj*params.nx];
    local_density += tmp_cells->speed8[ii + jj*params.nx];

    /* x-component of velocity */
    u_x = (tmp_cells->speed1[ii + jj*params.nx]
                  + tmp_cells->speed5[ii + jj*params.nx]
                  + tmp_cells->speed8[ii + jj*params.nx]
                  - (tmp_cells->speed3[ii + jj*params.nx]
                      + tmp_cells->speed6[ii + jj*params.nx]
                      + tmp_cells->speed7[ii + jj*params.nx]))
                  / local_density;
    /* compute y velocity component */
    u_y = (tmp_cells->speed2[ii + jj*params.nx]
                  + tmp_cells->speed5[ii + jj*params.nx]
                  + tmp_cells->speed6[ii + jj*params.nx]
                  - (tmp_cells->speed4[ii + jj*params.nx]
                      + tmp_cells->speed7[ii + jj*params.nx]
                      + tmp_cells->speed8[ii + jj*params.nx]))
                  / local_density;
    /* accumulate the norm of x- and y- velocity components */
    tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
    /* increase counter of inspected cells */
    ++tot_cells;
  }
    }

    ii = params.nx - 1;
    x_e = 0;
    x_w = ii - 1;
    if (obstacles[jj*params.nx + ii]) {
    tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx];
    tmp_cells->speed1[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx];
    tmp_cells->speed2[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx];
    tmp_cells->speed3[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx];
    tmp_cells->speed4[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx];
    tmp_cells->speed5[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx];
    tmp_cells->speed6[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx];
    tmp_cells->speed7[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx];
    tmp_cells->speed8[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx];
  }
  else {
    const float c_sq = 1.f / 3.f; /* square of speed of sound */
    const float w0 = 4.f / 9.f;  /* weighting factor */
    const float w1 = 1.f / 9.f;  /* weighting factor */
    const float w2 = 1.f / 36.f; /* weighting factor */
    /* compute local density total */
    float local_density = 0.f;

    local_density += cells->speed0[ii + jj*params.nx];
    local_density += cells->speed1[x_w + jj*params.nx];
    local_density += cells->speed2[ii + y_s*params.nx];
    local_density += cells->speed3[x_e + jj*params.nx];
    local_density += cells->speed4[ii + y_n*params.nx];
    local_density += cells->speed5[x_w + y_s*params.nx];
    local_density += cells->speed6[x_e + y_s*params.nx];
    local_density += cells->speed7[x_e + y_n*params.nx];
    local_density += cells->speed8[x_w + y_n*params.nx];
    
  
    /* compute x velocity component */
    float u_x = (cells->speed1[x_w + jj*params.nx]
                  + cells->speed5[x_w + y_s*params.nx]
                  + cells->speed8[x_w + y_n*params.nx]
                  - (cells->speed3[x_e + jj*params.nx]
                      + cells->speed6[x_e + y_s*params.nx]
                      + cells->speed7[x_e + y_n*params.nx]))
                  / local_density;
    /* compute y velocity component */
    float u_y = (cells->speed2[ii + y_s*params.nx]
                  + cells->speed5[x_w + y_s*params.nx]
                  + cells->speed6[x_e + y_s*params.nx]
                  - (cells->speed4[ii + y_n*params.nx]
                      + cells->speed7[x_e + y_n*params.nx]
                      + cells->speed8[x_w + y_n*params.nx]))
                  / local_density;

    /* velocity squared */
    float u_sq = u_x * u_x + u_y * u_y;

    /* directional velocity components */
    float u[NSPEEDS];
    u[1] =   u_x;        /* east */
    u[2] =         u_y;  /* north */
    u[3] = - u_x;        /* west */
    u[4] =       - u_y;  /* south */
    u[5] =   u_x + u_y;  /* north-east */
    u[6] = - u_x + u_y;  /* north-west */
    u[7] = - u_x - u_y;  /* south-west */
    u[8] =   u_x - u_y;  /* south-east */

    /* equilibrium densities */
    float d_equ[NSPEEDS];
    /* zero velocity density: weight w0 */
    d_equ[0] = w0 * local_density
                * (1.f - u_sq / (2.f * c_sq));
    /* axis speeds: weight w1 */
    d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                      + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                      + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                      + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                      + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    /* diagonal speeds: weight w2 */
    d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                      + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                      + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                      + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                      + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));

    /* relaxation step */

    tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx]
                                              + params.omega
                                              * (d_equ[0] - cells->speed0[ii + jj*params.nx]);

    tmp_cells->speed1[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx]
                                              + params.omega
                                              * (d_equ[1] - cells->speed1[x_w + jj*params.nx]);

    tmp_cells->speed2[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx]
                                              + params.omega
                                              * (d_equ[2] - cells->speed2[ii + y_s*params.nx]);                                          
    
    tmp_cells->speed3[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx]
                                              + params.omega
                                              * (d_equ[3] - cells->speed3[x_e + jj*params.nx]);

    tmp_cells->speed4[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx]
                                              + params.omega
                                              * (d_equ[4] - cells->speed4[ii + y_n*params.nx]);
    
    tmp_cells->speed5[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx]
                                              + params.omega
                                              * (d_equ[5] - cells->speed5[x_w + y_s*params.nx]);
    
    tmp_cells->speed6[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx]
                                              + params.omega
                                              * (d_equ[6] - cells->speed6[x_e + y_s*params.nx]);
    
    tmp_cells->speed7[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx]
                                              + params.omega
                                              * (d_equ[7] - cells->speed7[x_e + y_n*params.nx]);
    
    tmp_cells->speed8[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx]
                                              + params.omega
                                              * (d_equ[8] - cells->speed8[x_w + y_n*params.nx]);


  /* local density total */
    local_density = 0.f;

    local_density += tmp_cells->speed0[ii + jj*params.nx];
    local_density += tmp_cells->speed1[ii + jj*params.nx];
    local_density += tmp_cells->speed2[ii + jj*params.nx];
    local_density += tmp_cells->speed3[ii + jj*params.nx];
    local_density += tmp_cells->speed4[ii + jj*params.nx];
    local_density += tmp_cells->speed5[ii + jj*params.nx];
    local_density += tmp_cells->speed6[ii + jj*params.nx];
    local_density += tmp_cells->speed7[ii + jj*params.nx];
    local_density += tmp_cells->speed8[ii + jj*params.nx];

    /* x-component of velocity */
    u_x = (tmp_cells->speed1[ii + jj*params.nx]
                  + tmp_cells->speed5[ii + jj*params.nx]
                  + tmp_cells->speed8[ii + jj*params.nx]
                  - (tmp_cells->speed3[ii + jj*params.nx]
                      + tmp_cells->speed6[ii + jj*params.nx]
                      + tmp_cells->speed7[ii + jj*params.nx]))
                  / local_density;
    /* compute y velocity component */
    u_y = (tmp_cells->speed2[ii + jj*params.nx]
                  + tmp_cells->speed5[ii + jj*params.nx]
                  + tmp_cells->speed6[ii + jj*params.nx]
                  - (tmp_cells->speed4[ii + jj*params.nx]
                      + tmp_cells->speed7[ii + jj*params.nx]
                      + tmp_cells->speed8[ii + jj*params.nx]))
                  / local_density;
    /* accumulate the norm of x- and y- velocity components */
    tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
    /* increase counter of inspected cells */
    ++tot_cells;
  }
  return tot_u / (float)tot_cells;

}

int accelerate_flow(const t_param params, t_speed_soa* restrict cells, int* obstacles)
{


  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  int jj = params.ny - 2;

  __assume((params.nx) % 2 == 0);
  for (int ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + jj*params.nx]
        && (cells->speed3[ii + jj*params.nx] - w1) > 0.f
        && (cells->speed6[ii + jj*params.nx] - w2) > 0.f
        && (cells->speed7[ii + jj*params.nx] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells->speed1[ii + jj*params.nx] += w1;
      cells->speed5[ii + jj*params.nx] += w2;
      cells->speed8[ii + jj*params.nx] += w2;
      /* decrease 'west-side' densities */
      cells->speed3[ii + jj*params.nx] -= w1;
      cells->speed6[ii + jj*params.nx] -= w2;
      cells->speed7[ii + jj*params.nx] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

int propagate_fast(const t_param params, t_speed_soa* restrict cells, t_speed_soa* restrict tmp_cells) {

  __assume_aligned(cells->speed0, 64);
  __assume_aligned(cells->speed1, 64);
  __assume_aligned(cells->speed2, 64);
  __assume_aligned(cells->speed3, 64);
  __assume_aligned(cells->speed4, 64);
  __assume_aligned(cells->speed5, 64);
  __assume_aligned(cells->speed6, 64);
  __assume_aligned(cells->speed7, 64);
  __assume_aligned(cells->speed8, 64);

  __assume_aligned(tmp_cells->speed0, 64);
  __assume_aligned(tmp_cells->speed1, 64);
  __assume_aligned(tmp_cells->speed2, 64);
  __assume_aligned(tmp_cells->speed3, 64);
  __assume_aligned(tmp_cells->speed4, 64);
  __assume_aligned(tmp_cells->speed5, 64);
  __assume_aligned(tmp_cells->speed6, 64);
  __assume_aligned(tmp_cells->speed7, 64);
  __assume_aligned(tmp_cells->speed8, 64);
  //Case where jj = 0
  int jj = 0;
  int y_s = jj + params.ny - 1;

  //Case where ii = 0
  int ii = 0;
  int y_n = jj + 1;
  int x_e = ii + 1;
  int x_w = ii + params.nx - 1;

  tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx]; /* central cell, no movement */
  tmp_cells->speed1[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx]; /* east */
  tmp_cells->speed2[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx]; /* north */
  tmp_cells->speed3[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx]; /* west */
  tmp_cells->speed4[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx]; /* south */
  tmp_cells->speed5[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx]; /* north-east */
  tmp_cells->speed6[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx]; /* north-west */
  tmp_cells->speed7[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx]; /* south-west */
  tmp_cells->speed8[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx]; /* south-east */

  __assume((params.nx) % 2 == 0);
  __assume((params.ny) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.ny) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.ny) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.ny) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.ny) % 32 == 0);
  __assume((params.nx) % 64 == 0);
  __assume((params.ny) % 64 == 0);
    //Case where i > 0 and i < params.nx - 1

  #pragma omp simd
  for (int ii = 1; ii < params.nx - 1; ii++)
  {
    /* determine indices of axis-direction neighbours
    ** respecting periodic boundary conditions (wrap around) */
    int y_n = jj + 1;
    int x_e = ii + 1;
    int x_w = ii - 1;
    /* propagate densities from neighbouring cells, following
    ** appropriate directions of travel and writing into
    ** scratch space grid */
    tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx]; /* central cell, no movement */
    tmp_cells->speed1[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx]; /* east */
    tmp_cells->speed2[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx]; /* north */
    tmp_cells->speed3[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx]; /* west */
    tmp_cells->speed4[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx]; /* south */
    tmp_cells->speed5[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx]; /* north-east */
    tmp_cells->speed6[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx]; /* north-west */
    tmp_cells->speed7[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx]; /* south-west */
    tmp_cells->speed8[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx]; /* south-east */
  }

  ii = params.nx - 1;
  y_n = jj + 1;
  x_e = 0;
  x_w = (ii - 1);
  tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx]; /* central cell, no movement */
  tmp_cells->speed1[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx]; /* east */
  tmp_cells->speed2[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx]; /* north */
  tmp_cells->speed3[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx]; /* west */
  tmp_cells->speed4[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx]; /* south */
  tmp_cells->speed5[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx]; /* north-east */
  tmp_cells->speed6[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx]; /* north-west */
  tmp_cells->speed7[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx]; /* south-west */
  tmp_cells->speed8[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx]; /* south-east */

  
  for (int jj = 1; jj < params.ny - 1; jj++)
  {
    y_s = (jj - 1);

    //Case where ii = 0
    int ii = 0;
    int y_n = jj + 1;
    int x_e = ii + 1;
    int x_w = ii + params.nx - 1;

    tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx]; /* central cell, no movement */
    tmp_cells->speed1[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx]; /* east */
    tmp_cells->speed2[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx]; /* north */
    tmp_cells->speed3[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx]; /* west */
    tmp_cells->speed4[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx]; /* south */
    tmp_cells->speed5[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx]; /* north-east */
    tmp_cells->speed6[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx]; /* north-west */
    tmp_cells->speed7[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx]; /* south-west */
    tmp_cells->speed8[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx]; /* south-east */

    #pragma omp simd
    //Case where i > 0 and i < params.nx - 1
    for (int ii = 1; ii < params.nx - 1; ii++)
    {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int y_n = jj + 1;
      int x_e = ii + 1;
      int x_w = ii - 1;
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx]; /* central cell, no movement */
      tmp_cells->speed1[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx]; /* east */
      tmp_cells->speed2[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx]; /* north */
      tmp_cells->speed3[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx]; /* west */
      tmp_cells->speed4[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx]; /* south */
      tmp_cells->speed5[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx]; /* north-east */
      tmp_cells->speed6[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx]; /* north-west */
      tmp_cells->speed7[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx]; /* south-west */
      tmp_cells->speed8[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx]; /* south-east */
    }

    ii = params.nx - 1;
    y_n = jj + 1;
    x_e = 0;
    x_w = ii - 1;
    tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx]; /* central cell, no movement */
    tmp_cells->speed1[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx]; /* east */
    tmp_cells->speed2[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx]; /* north */
    tmp_cells->speed3[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx]; /* west */
    tmp_cells->speed4[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx]; /* south */
    tmp_cells->speed5[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx]; /* north-east */
    tmp_cells->speed6[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx]; /* north-west */
    tmp_cells->speed7[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx]; /* south-west */
    tmp_cells->speed8[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx]; /* south-east */
  
  }


  //Case where jj = params.ny - 1
  jj = params.ny - 1;
  y_s = jj - 1;

  //Case where ii = 0
  ii = 0;
  y_n = jj + 1;
  x_e = ii + 1;
  x_w = ii + params.nx - 1;

  tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx]; /* central cell, no movement */
  tmp_cells->speed1[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx]; /* east */
  tmp_cells->speed2[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx]; /* north */
  tmp_cells->speed3[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx]; /* west */
  tmp_cells->speed4[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx]; /* south */
  tmp_cells->speed5[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx]; /* north-east */
  tmp_cells->speed6[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx]; /* north-west */
  tmp_cells->speed7[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx]; /* south-west */
  tmp_cells->speed8[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx]; /* south-east */

  #pragma omp simd
  for (int ii = 1; ii < params.nx - 1; ii++)
    {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int y_n = jj + 1;
      int x_e = ii + 1;
      int x_w = ii - 1;
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx]; /* central cell, no movement */
      tmp_cells->speed1[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx]; /* east */
      tmp_cells->speed2[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx]; /* north */
      tmp_cells->speed3[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx]; /* west */
      tmp_cells->speed4[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx]; /* south */
      tmp_cells->speed5[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx]; /* north-east */
      tmp_cells->speed6[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx]; /* north-west */
      tmp_cells->speed7[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx]; /* south-west */
      tmp_cells->speed8[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx]; /* south-east */
    }

    ii = params.nx - 1;
    y_n = jj + 1;
    x_e = 0;
    x_w = ii - 1;
    tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx]; /* central cell, no movement */
    tmp_cells->speed1[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx]; /* east */
    tmp_cells->speed2[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx]; /* north */
    tmp_cells->speed3[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx]; /* west */
    tmp_cells->speed4[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx]; /* south */
    tmp_cells->speed5[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx]; /* north-east */
    tmp_cells->speed6[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx]; /* north-west */
    tmp_cells->speed7[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx]; /* south-west */
    tmp_cells->speed8[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx]; /* south-east */
   return EXIT_SUCCESS;

}

int propagate(const t_param params, t_speed_soa* restrict cells, t_speed_soa* restrict tmp_cells)
{
  __assume_aligned(cells->speed0, 64);
  __assume_aligned(cells->speed1, 64);
  __assume_aligned(cells->speed2, 64);
  __assume_aligned(cells->speed3, 64);
  __assume_aligned(cells->speed4, 64);
  __assume_aligned(cells->speed5, 64);
  __assume_aligned(cells->speed6, 64);
  __assume_aligned(cells->speed7, 64);
  __assume_aligned(cells->speed8, 64);

  __assume_aligned(tmp_cells->speed0, 64);
  __assume_aligned(tmp_cells->speed1, 64);
  __assume_aligned(tmp_cells->speed2, 64);
  __assume_aligned(tmp_cells->speed3, 64);
  __assume_aligned(tmp_cells->speed4, 64);
  __assume_aligned(tmp_cells->speed5, 64);
  __assume_aligned(tmp_cells->speed6, 64);
  __assume_aligned(tmp_cells->speed7, 64);
  __assume_aligned(tmp_cells->speed8, 64);
  /* loop over _all_ cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int y_n = (jj + 1) % params.ny;
      int x_e = (ii + 1) % params.nx;
      int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
      int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx]; /* central cell, no movement */
      tmp_cells->speed1[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx]; /* east */
      tmp_cells->speed2[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx]; /* north */
      tmp_cells->speed3[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx]; /* west */
      tmp_cells->speed4[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx]; /* south */
      tmp_cells->speed5[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx]; /* north-east */
      tmp_cells->speed6[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx]; /* north-west */
      tmp_cells->speed7[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx]; /* south-west */
      tmp_cells->speed8[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx]; /* south-east */
      
    }
  }

  return EXIT_SUCCESS;
}

int rebound_old(const t_param params, t_speed* restrict cells, t_speed* restrict tmp_cells, int* obstacles) {
  /* loop over the cells in the grid */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* if the cell contains an obstacle */
      if (obstacles[jj*params.nx + ii])
      {
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        cells[ii + jj*params.nx].speeds[1] = tmp_cells[ii + jj*params.nx].speeds[3];
        cells[ii + jj*params.nx].speeds[2] = tmp_cells[ii + jj*params.nx].speeds[4];
        cells[ii + jj*params.nx].speeds[3] = tmp_cells[ii + jj*params.nx].speeds[1];
        cells[ii + jj*params.nx].speeds[4] = tmp_cells[ii + jj*params.nx].speeds[2];
        cells[ii + jj*params.nx].speeds[5] = tmp_cells[ii + jj*params.nx].speeds[7];
        cells[ii + jj*params.nx].speeds[6] = tmp_cells[ii + jj*params.nx].speeds[8];
        cells[ii + jj*params.nx].speeds[7] = tmp_cells[ii + jj*params.nx].speeds[5];
        cells[ii + jj*params.nx].speeds[8] = tmp_cells[ii + jj*params.nx].speeds[6];
      }
    }
  }

  return EXIT_SUCCESS;
}

int rebound(const t_param params, t_speed_soa* restrict cells, t_speed_soa* restrict tmp_cells, int* obstacles)
{
  __assume_aligned(cells->speed0, 64);
  __assume_aligned(cells->speed1, 64);
  __assume_aligned(cells->speed2, 64);
  __assume_aligned(cells->speed3, 64);
  __assume_aligned(cells->speed4, 64);
  __assume_aligned(cells->speed5, 64);
  __assume_aligned(cells->speed6, 64);
  __assume_aligned(cells->speed7, 64);
  __assume_aligned(cells->speed8, 64);

  __assume_aligned(tmp_cells->speed0, 64);
  __assume_aligned(tmp_cells->speed1, 64);
  __assume_aligned(tmp_cells->speed2, 64);
  __assume_aligned(tmp_cells->speed3, 64);
  __assume_aligned(tmp_cells->speed4, 64);
  __assume_aligned(tmp_cells->speed5, 64);
  __assume_aligned(tmp_cells->speed6, 64);
  __assume_aligned(tmp_cells->speed7, 64);
  __assume_aligned(tmp_cells->speed8, 64);

  __assume((params.nx) % 2 == 0);
  __assume((params.ny) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.ny) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.ny) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.ny) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.ny) % 32 == 0);
  __assume((params.nx) % 64 == 0);
  __assume((params.ny) % 64 == 0);

  /* loop over the cells in the grid */
  for (int jj = 0; jj < params.ny; jj++)
  {
    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* if the cell contains an obstacle */
      if (obstacles[jj*params.nx + ii])
      {
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        float temp = tmp_cells->speed1[ii + jj*params.nx];
        tmp_cells->speed1[ii + jj*params.nx] = tmp_cells->speed3[ii + jj*params.nx];
        tmp_cells->speed3[ii + jj*params.nx] = temp;

        temp = tmp_cells->speed2[ii + jj*params.nx];
        tmp_cells->speed2[ii + jj*params.nx] = tmp_cells->speed4[ii + jj*params.nx];
        tmp_cells->speed4[ii + jj*params.nx] = temp;

        temp = tmp_cells->speed5[ii + jj*params.nx];
        tmp_cells->speed5[ii + jj*params.nx] = tmp_cells->speed7[ii + jj*params.nx];
        tmp_cells->speed7[ii + jj*params.nx] = temp;

        temp = tmp_cells->speed6[ii + jj*params.nx];
        tmp_cells->speed6[ii + jj*params.nx] = tmp_cells->speed8[ii + jj*params.nx];
        tmp_cells->speed8[ii + jj*params.nx] = temp;
      } 
    }
  }

  return EXIT_SUCCESS;
}


int collision_old(const t_param params, t_speed* restrict cells, t_speed* restrict tmp_cells, int* obstacles) {
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* don't consider occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* compute local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += tmp_cells[ii + jj*params.nx].speeds[kk];
        }

        /* compute x velocity component */
        float u_x = (tmp_cells[ii + jj*params.nx].speeds[1]
                      + tmp_cells[ii + jj*params.nx].speeds[5]
                      + tmp_cells[ii + jj*params.nx].speeds[8]
                      - (tmp_cells[ii + jj*params.nx].speeds[3]
                         + tmp_cells[ii + jj*params.nx].speeds[6]
                         + tmp_cells[ii + jj*params.nx].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (tmp_cells[ii + jj*params.nx].speeds[2]
                      + tmp_cells[ii + jj*params.nx].speeds[5]
                      + tmp_cells[ii + jj*params.nx].speeds[6]
                      - (tmp_cells[ii + jj*params.nx].speeds[4]
                         + tmp_cells[ii + jj*params.nx].speeds[7]
                         + tmp_cells[ii + jj*params.nx].speeds[8]))
                     / local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u[NSPEEDS];
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */

        /* equilibrium densities */
        float d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */
        d_equ[0] = w0 * local_density
                   * (1.f - u_sq / (2.f * c_sq));
        /* axis speeds: weight w1 */
        d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                         + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                         + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                         + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                         + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        /* diagonal speeds: weight w2 */
        d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                         + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                         + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                         + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                         + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));

        /* relaxation step */
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          cells[ii + jj*params.nx].speeds[kk] = tmp_cells[ii + jj*params.nx].speeds[kk]
                                                  + params.omega
                                                  * (d_equ[kk] - tmp_cells[ii + jj*params.nx].speeds[kk]);
        }

        //swap tmp_cells and cells
        //t_speed **temp = 


      }
    }
  }
  //swap(&cells, &tmp_cells);
  /* t_speed* temp = cells;
  *cells = *tmp_cells;
  *tmp_cells = *temp; */
  return EXIT_SUCCESS;
}

int collision(const t_param params, t_speed_soa* restrict cells, t_speed_soa* restrict tmp_cells, int* obstacles)
{
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  
  __assume_aligned(cells->speed0, 64);
  __assume_aligned(cells->speed1, 64);
  __assume_aligned(cells->speed2, 64);
  __assume_aligned(cells->speed3, 64);
  __assume_aligned(cells->speed4, 64);
  __assume_aligned(cells->speed5, 64);
  __assume_aligned(cells->speed6, 64);
  __assume_aligned(cells->speed7, 64);
  __assume_aligned(cells->speed8, 64);

  __assume_aligned(tmp_cells->speed0, 64);
  __assume_aligned(tmp_cells->speed1, 64);
  __assume_aligned(tmp_cells->speed2, 64);
  __assume_aligned(tmp_cells->speed3, 64);
  __assume_aligned(tmp_cells->speed4, 64);
  __assume_aligned(tmp_cells->speed5, 64);
  __assume_aligned(tmp_cells->speed6, 64);
  __assume_aligned(tmp_cells->speed7, 64);
  __assume_aligned(tmp_cells->speed8, 64);

  __assume((params.nx) % 2 == 0);
  __assume((params.ny) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.ny) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.ny) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.ny) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.ny) % 32 == 0);
  __assume((params.nx) % 64 == 0);
  __assume((params.ny) % 64 == 0);

  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
  
  for (int jj = 0; jj < params.ny; jj++)
  {
    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* don't consider occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* compute local density total */
        float local_density = 0.f;

        local_density += tmp_cells->speed0[ii + jj*params.nx];
        local_density += tmp_cells->speed1[ii + jj*params.nx];
        local_density += tmp_cells->speed2[ii + jj*params.nx];
        local_density += tmp_cells->speed3[ii + jj*params.nx];
        local_density += tmp_cells->speed4[ii + jj*params.nx];
        local_density += tmp_cells->speed5[ii + jj*params.nx];
        local_density += tmp_cells->speed6[ii + jj*params.nx];
        local_density += tmp_cells->speed7[ii + jj*params.nx];
        local_density += tmp_cells->speed8[ii + jj*params.nx];
        
       

        /* compute x velocity component */
        float u_x = (tmp_cells->speed1[ii + jj*params.nx]
                      + tmp_cells->speed5[ii + jj*params.nx]
                      + tmp_cells->speed8[ii + jj*params.nx]
                      - (tmp_cells->speed3[ii + jj*params.nx]
                         + tmp_cells->speed6[ii + jj*params.nx]
                         + tmp_cells->speed7[ii + jj*params.nx]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (tmp_cells->speed2[ii + jj*params.nx]
                      + tmp_cells->speed5[ii + jj*params.nx]
                      + tmp_cells->speed6[ii + jj*params.nx]
                      - (tmp_cells->speed4[ii + jj*params.nx]
                         + tmp_cells->speed7[ii + jj*params.nx]
                         + tmp_cells->speed8[ii + jj*params.nx]))
                     / local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u[NSPEEDS];
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */

        /* equilibrium densities */
        float d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */
        d_equ[0] = w0 * local_density
                   * (1.f - u_sq / (2.f * c_sq));
        /* axis speeds: weight w1 */
        d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                         + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                         + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                         + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                         + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        /* diagonal speeds: weight w2 */
        d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                         + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                         + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                         + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                         + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));

        /* relaxation step */

        tmp_cells->speed0[ii + jj*params.nx] = tmp_cells->speed0[ii + jj*params.nx]
                                                  + params.omega
                                                  * (d_equ[0] - tmp_cells->speed0[ii + jj*params.nx]);

        tmp_cells->speed1[ii + jj*params.nx] = tmp_cells->speed1[ii + jj*params.nx]
                                                  + params.omega
                                                  * (d_equ[1] - tmp_cells->speed1[ii + jj*params.nx]);

        tmp_cells->speed2[ii + jj*params.nx] = tmp_cells->speed2[ii + jj*params.nx]
                                                  + params.omega
                                                  * (d_equ[2] - tmp_cells->speed2[ii + jj*params.nx]);                                          
        
        tmp_cells->speed3[ii + jj*params.nx] = tmp_cells->speed3[ii + jj*params.nx]
                                                  + params.omega
                                                  * (d_equ[3] - tmp_cells->speed3[ii + jj*params.nx]);

        tmp_cells->speed4[ii + jj*params.nx] = tmp_cells->speed4[ii + jj*params.nx]
                                                  + params.omega
                                                  * (d_equ[4] - tmp_cells->speed4[ii + jj*params.nx]);
        
        tmp_cells->speed5[ii + jj*params.nx] = tmp_cells->speed5[ii + jj*params.nx]
                                                  + params.omega
                                                  * (d_equ[5] - tmp_cells->speed5[ii + jj*params.nx]);
        
        tmp_cells->speed6[ii + jj*params.nx] = tmp_cells->speed6[ii + jj*params.nx]
                                                  + params.omega
                                                  * (d_equ[6] - tmp_cells->speed6[ii + jj*params.nx]);
        
        tmp_cells->speed7[ii + jj*params.nx] = tmp_cells->speed7[ii + jj*params.nx]
                                                  + params.omega
                                                  * (d_equ[7] - tmp_cells->speed7[ii + jj*params.nx]);
        
        tmp_cells->speed8[ii + jj*params.nx] = tmp_cells->speed8[ii + jj*params.nx]
                                                  + params.omega
                                                  * (d_equ[8] - tmp_cells->speed8[ii + jj*params.nx]);
        //swap tmp_cells and cells
        //t_speed **temp = 


      }
    }
  }
  //swap(&cells, &tmp_cells);
  /* t_speed* temp = cells;
  *cells = *tmp_cells;
  *tmp_cells = *temp; */
  return EXIT_SUCCESS;
}

float av_velocity(const t_param params, t_speed_soa* restrict cells, int* obstacles)
{
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;
  
  __assume_aligned(cells->speed0, 64);
  __assume_aligned(cells->speed1, 64);
  __assume_aligned(cells->speed2, 64);
  __assume_aligned(cells->speed3, 64);
  __assume_aligned(cells->speed4, 64);
  __assume_aligned(cells->speed5, 64);
  __assume_aligned(cells->speed6, 64);
  __assume_aligned(cells->speed7, 64);
  __assume_aligned(cells->speed8, 64);

  __assume((params.nx) % 2 == 0);
  __assume((params.ny) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.ny) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.ny) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  __assume((params.ny) % 16 == 0);
  __assume((params.nx) % 32 == 0);
  __assume((params.ny) % 32 == 0);
  __assume((params.nx) % 64 == 0);
  __assume((params.ny) % 64 == 0);
  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* local density total */
        float local_density = 0.f;

        local_density += cells->speed0[ii + jj*params.nx];
        local_density += cells->speed1[ii + jj*params.nx];
        local_density += cells->speed2[ii + jj*params.nx];
        local_density += cells->speed3[ii + jj*params.nx];
        local_density += cells->speed4[ii + jj*params.nx];
        local_density += cells->speed5[ii + jj*params.nx];
        local_density += cells->speed6[ii + jj*params.nx];
        local_density += cells->speed7[ii + jj*params.nx];
        local_density += cells->speed8[ii + jj*params.nx];

        /* x-component of velocity */
        float u_x = (cells->speed1[ii + jj*params.nx]
                      + cells->speed5[ii + jj*params.nx]
                      + cells->speed8[ii + jj*params.nx]
                      - (cells->speed3[ii + jj*params.nx]
                         + cells->speed6[ii + jj*params.nx]
                         + cells->speed7[ii + jj*params.nx]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells->speed2[ii + jj*params.nx]
                      + cells->speed5[ii + jj*params.nx]
                      + cells->speed6[ii + jj*params.nx]
                      - (cells->speed4[ii + jj*params.nx]
                         + cells->speed7[ii + jj*params.nx]
                         + cells->speed8[ii + jj*params.nx]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed_soa* cells_ptr, t_speed_soa* tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, int rank, int nprocs)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  //*cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));
  //if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* for (size_t i = 0; i < 9; i++)
  {
    cells_ptr->speeds[i] = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
    if (cells_ptr->speeds[i] == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  } */

  int r = params->ny % nprocs; //remainder
  int work;
  if (rank < r) {
    work = params->ny / nprocs + 1;
  }
  else {
    work = params->ny / nprocs;
  }

  cells_ptr->speed0 = (float*)_mm_malloc(sizeof(float) * ((work + 2) * params->nx), 64);
  cells_ptr->speed1 = (float*)_mm_malloc(sizeof(float) * ((work + 2) * params->nx), 64);
  cells_ptr->speed2 = (float*)_mm_malloc(sizeof(float) * ((work + 2) * params->nx), 64);
  cells_ptr->speed3 = (float*)_mm_malloc(sizeof(float) * ((work + 2) * params->nx), 64);
  cells_ptr->speed4 = (float*)_mm_malloc(sizeof(float) * ((work + 2) * params->nx), 64);
  cells_ptr->speed5 = (float*)_mm_malloc(sizeof(float) * ((work + 2) * params->nx), 64);
  cells_ptr->speed6 = (float*)_mm_malloc(sizeof(float) * ((work + 2) * params->nx), 64);
  cells_ptr->speed7 = (float*)_mm_malloc(sizeof(float) * ((work + 2) * params->nx), 64);
  cells_ptr->speed8 = (float*)_mm_malloc(sizeof(float) * ((work + 2) * params->nx), 64);
  
  /* 'helper' grid, used as scratch space */
  //*tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  //if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* for (size_t i = 0; i < 9; i++)
  {
    tmp_cells_ptr->speeds[i] = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
    if (tmp_cells_ptr->speeds[i] == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  } */

  tmp_cells_ptr->speed0 = (float*)_mm_malloc(sizeof(float) * ((work + 2) * params->nx), 64);
  tmp_cells_ptr->speed1 = (float*)_mm_malloc(sizeof(float) * ((work + 2) * params->nx), 64);
  tmp_cells_ptr->speed2 = (float*)_mm_malloc(sizeof(float) * ((work + 2) * params->nx), 64);
  tmp_cells_ptr->speed3 = (float*)_mm_malloc(sizeof(float) * ((work + 2) * params->nx), 64);
  tmp_cells_ptr->speed4 = (float*)_mm_malloc(sizeof(float) * ((work + 2) * params->nx), 64);
  tmp_cells_ptr->speed5 = (float*)_mm_malloc(sizeof(float) * ((work + 2) * params->nx), 64);
  tmp_cells_ptr->speed6 = (float*)_mm_malloc(sizeof(float) * ((work + 2) * params->nx), 64);
  tmp_cells_ptr->speed7 = (float*)_mm_malloc(sizeof(float) * ((work + 2) * params->nx), 64);
  tmp_cells_ptr->speed8 = (float*)_mm_malloc(sizeof(float) * ((work + 2) * params->nx), 64);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 1; jj < work+1; jj++) //adjust for halo regions
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* centre */

      
      //(*cells_ptr)[ii + jj*params->nx].speeds[0] = w0;
      cells_ptr->speed0[ii + jj*params->nx] = w0;

      /* axis directions */
      
      //(*cells_ptr)[ii + jj*params->nx].speeds[1] = w1;
      //(*cells_ptr)[ii + jj*params->nx].speeds[2] = w1;
      //(*cells_ptr)[ii + jj*params->nx].speeds[3] = w1;
      //(*cells_ptr)[ii + jj*params->nx].speeds[4] = w1;
      cells_ptr->speed1[ii + jj*params->nx] = w1;
      cells_ptr->speed2[ii + jj*params->nx] = w1;
      cells_ptr->speed3[ii + jj*params->nx] = w1;
      cells_ptr->speed4[ii + jj*params->nx] = w1;

      /* diagonals */
      //(*cells_ptr)[ii + jj*params->nx].speeds[5] = w2;
      //(*cells_ptr)[ii + jj*params->nx].speeds[6] = w2;
      //(*cells_ptr)[ii + jj*params->nx].speeds[7] = w2;
      //(*cells_ptr)[ii + jj*params->nx].speeds[8] = w2;

      cells_ptr->speed5[ii + jj*params->nx] = w2;
      cells_ptr->speed6[ii + jj*params->nx] = w2;
      cells_ptr->speed7[ii + jj*params->nx] = w2;
      cells_ptr->speed8[ii + jj*params->nx] = w2;
    }
  }


  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed_soa* cells_ptr, t_speed_soa* tmp_cells_ptr, t_speed_soa* global_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  /* free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL; */

  //free cells
/*   for (size_t i = 0; i < 9; i++)
  {
    _mm_free(cells_ptr->speeds[i]);
    _mm_free(tmp_cells_ptr->speeds[i]);
  } */

  _mm_free(cells_ptr->speed0);
  _mm_free(cells_ptr->speed1);
  _mm_free(cells_ptr->speed2);
  _mm_free(cells_ptr->speed3);
  _mm_free(cells_ptr->speed4);
  _mm_free(cells_ptr->speed5);
  _mm_free(cells_ptr->speed6);
  _mm_free(cells_ptr->speed7);
  _mm_free(cells_ptr->speed8);

  _mm_free(tmp_cells_ptr->speed0);
  _mm_free(tmp_cells_ptr->speed1);
  _mm_free(tmp_cells_ptr->speed2);
  _mm_free(tmp_cells_ptr->speed3);
  _mm_free(tmp_cells_ptr->speed4);
  _mm_free(tmp_cells_ptr->speed5);
  _mm_free(tmp_cells_ptr->speed6);
  _mm_free(tmp_cells_ptr->speed7);
  _mm_free(tmp_cells_ptr->speed8);
  
  _mm_free(global_cells_ptr->speed0);
  _mm_free(global_cells_ptr->speed1);
  _mm_free(global_cells_ptr->speed2);
  _mm_free(global_cells_ptr->speed3);
  _mm_free(global_cells_ptr->speed4);
  _mm_free(global_cells_ptr->speed5);
  _mm_free(global_cells_ptr->speed6);
  _mm_free(global_cells_ptr->speed7);
  _mm_free(global_cells_ptr->speed8);

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed_soa* cells, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[ii + jj*params.nx].speeds[kk];
      }
    }
  }

  return total;
}

int write_values(const t_param params, t_speed_soa* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.f;

        local_density += cells->speed0[ii + jj*params.nx];
        local_density += cells->speed1[ii + jj*params.nx];
        local_density += cells->speed2[ii + jj*params.nx];
        local_density += cells->speed3[ii + jj*params.nx];
        local_density += cells->speed4[ii + jj*params.nx];
        local_density += cells->speed5[ii + jj*params.nx];
        local_density += cells->speed6[ii + jj*params.nx];
        local_density += cells->speed7[ii + jj*params.nx];
        local_density += cells->speed8[ii + jj*params.nx];

        /* x-component of velocity */
        float u_x = (cells->speed1[ii + jj*params.nx]
                      + cells->speed5[ii + jj*params.nx]
                      + cells->speed8[ii + jj*params.nx]
                      - (cells->speed3[ii + jj*params.nx]
                         + cells->speed6[ii + jj*params.nx]
                         + cells->speed7[ii + jj*params.nx]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells->speed2[ii + jj*params.nx]
                      + cells->speed5[ii + jj*params.nx]
                      + cells->speed6[ii + jj*params.nx]
                      - (cells->speed4[ii + jj*params.nx]
                         + cells->speed7[ii + jj*params.nx]
                         + cells->speed8[ii + jj*params.nx]))
                     / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii + params.nx * jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}
