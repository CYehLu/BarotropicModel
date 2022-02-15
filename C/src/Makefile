obj_path = ../lib/

lib_flags = -lfftw3 -lm
extern_lib_path = ../package/lib/
extern_inc_path = ../package/include/

other_flags = -std=c99 -Wall -Wextra

.PHONY: all clean

all: $(obj_path)/model.o

$(obj_path)/model.o: model.c $(obj_path)/helper_spatial.o $(obj_path)/helper_time.o $(obj_path)/utils.o
	gcc -c $< $(other_flags) -o $@ -I$(extern_inc_path) -L$(extern_lib_path) $(lib_flags)

$(obj_path)/helper_spatial.o: helper_spatial.c $(obj_path)/utils.o
	gcc -c $< $(other_flags) -o $@ -I$(extern_inc_path) -L$(extern_lib_path) $(lib_flags)

$(obj_path)/%.o: %.c
	gcc -c $< $(other_flags) -o $@ 

clean:
	rm $(obj_path)/*.o
