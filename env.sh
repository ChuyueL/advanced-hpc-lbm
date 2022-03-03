# Add any `module load` or `export` commands that your code needs to
# compile and run to this file.
export OMP_NUM_THREADS=28
export OMP_PROC_BIND=true
export OMP_PLACES=cores

module load languages/anaconda2
module load languages/intel
module load languages/gcc