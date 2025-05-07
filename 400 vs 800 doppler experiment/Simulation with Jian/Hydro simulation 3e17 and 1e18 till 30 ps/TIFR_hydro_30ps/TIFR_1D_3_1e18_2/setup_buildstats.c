/* !!! DO NOT EDIT, FILES WRITTEN BY SETUP SCRIPT !!!
   
!!****f* object/setup_buildstats
!!
!! NAME
!!
!!  setup_buildstats
!!
!!
!! SYNOPSIS
!!
!!  call setup_buildstats(build_date, build_dir, build_machine, setup_call)
!!
!!  call setup_buildstats(character, character, character, character)
!!
!! 
!! DESCRIPTION
!!
!!  Simple subroutine generated at build time that returns the build date
!!  build directory, build machine, c and f flags and the full setup 
!!  command used to assemble the FLASH executable.
!!
!!
!!
!!***
*/

#include "mangle_names.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

void FTOC(setup_buildstats)(char* build_date, 
		    char* build_dir, 
		    char* build_machine, 
		    char* setup_call, 
		    char* c_flags, 
		    char* f_flags){



     strncpy(build_date, "Thu 10 Apr 2025 10:07:32 AM EEST",80);
     strncpy(build_dir, "/media/ong/WORKDIR1/TIFR_1D_3_1e18_2", 80);
     strncpy(build_machine, "Linux ong-ThinkStation-P720 6.11.0-21-generic #21~24.04.1-Ubuntu SMP PREEMPT_DYN", 80);
     strncpy(setup_call, "/home/ong/FLASH4.8/bin/setup.py -auto -objdir=/media/ong/WORKDIR1/TIFR_1D_3_1e18_2 LaserTIFR -1d +cartesian -nxb=32 +pm4dev +hdf5typeio species=cham,targ +mtmmmt +uhd3t +laser ed_maxPulseSections=64 +mgd mgd_meshgroups=6 -parfile=tifr_1D_3.par ",400);
     strncpy(c_flags, "/home/ong/software/src/openmpi/4.1.6-gcc-13.3.0/bin/mpicc -I/home/ong/software/src/hypre/2.32.0-gcc-13.3.0/include -I/home/ong/software/src/hdf5/1.14.6-gcc-13.3.0/include -DH5_USE_16_API -ggdb -c -O2 -Wuninitialized -D_FORTIFY_SOURCE=2 -DFLASH_3T -DMAXBLOCKS=1000 -DNXB=32 -DNYB=1 -DNZB=1 -DN_DIM=1", 400);
     strncpy(f_flags, "/home/ong/software/src/openmpi/4.1.6-gcc-13.3.0/bin/mpif90 -I/home/ong/software/src/hypre/2.32.0-gcc-13.3.0/include -ggdb -c -O3 -fdefault-real-8 -fdefault-double-8 -Wuninitialized -fallow-argument-mismatch -DFLASH_3T -DMAXBLOCKS=1000 -DNXB=32 -DNYB=1 -DNZB=1 -DN_DIM=1", 400);


}

