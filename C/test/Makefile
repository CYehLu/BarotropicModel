lib_path = ../lib/
dep_objs = $(lib_path)/utils.o $(lib_path)/helper_spatial.o $(lib_path)/helper_time.o $(lib_path)/model.o

lib_flags = -lfftw3 -lm
extern_lib_path = ../package/lib/
extern_inc_path = ../package/include/

other_flags = -std=c99 -fopenmp -Wall -Wextra

all: test_compute_rhs.out test_poisson.out test_model.out

test_poisson.out: test_poisson.c $(dep_objs)
	gcc $^ -o $@ $(other_flags) -I$(extern_inc_path) -L$(extern_lib_path) $(lib_flags)
    
test_compute_rhs.out: test_compute_rhs.c $(dep_objs)
	gcc $^ -o $@ $(other_flags) -I$(extern_inc_path) -L$(extern_lib_path) $(lib_flags)
    
test_model.out: test_model.c $(dep_objs)
	gcc $^ -o $@ $(other_flags) -I$(extern_inc_path) -L$(extern_lib_path) $(lib_flags)
    
clean:
	rm *.out
	rm -f *.dat
    