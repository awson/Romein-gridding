ifeq		($(shell hostname), gpu01)
ARCH=		compute_30
else
ARCH=		compute_20
endif

PTXASFLAGS=	-Xptxas -v #,-dlcm=cg
CXX=		g++
#CXX=		icc
#CXXFLAGS=	-O3 -fopenmp -g -msse4.1 -vec-report1
#CXXFLAGS=	-O3 -fopenmp -g -march=core-avx-i -vec-report5
CXXFLAGS=	-O3 -fopenmp -g -I.

ifneq		"$(shell fgrep avx /proc/cpuinfo)" ""
CXXFLAGS:=	$(CXXFLAGS) -mavx
else
CXXFLAGS:=	$(CXXFLAGS) -msse3
endif


DEFINES=	$(if ${MODE}, -DMODE=${MODE})\
		$(if ${GRID_U}, -DGRID_U=${GRID_U})\
		$(if ${GRID_V}, -DGRID_V=${GRID_V})\
		$(if ${SUPPORT_U}, -DSUPPORT_U=${SUPPORT_U})\
		$(if ${SUPPORT_V}, -DSUPPORT_V=${SUPPORT_V})\
		$(if ${W_PLANES}, -DW_PLANES=${W_PLANES})\
		$(if ${X}, -DX=${X})\
		$(if ${TIMESTEPS}, -DTIMESTEPS=${TIMESTEPS})\
		$(if ${BLOCKS}, -DBLOCKS=${BLOCKS})\
		$(if ${USE_TEXTURE}, -DUSE_TEXTURE)

a.out-CPU:	Gridding-CPU.o UVW.o
		$(CXX) $(DEFINES) $(CXXFLAGS) $^ -o $@

a.out-Cuda:	Gridding-Cuda.o UVW.o
		nvcc $(DEFINES) $(PTXASFLAGS) -ccbin=${CXX} -Xcompiler "$(CXXFLAGS)" $^ -o $@

a.out-OpenCL:	Gridding-OpenCL.o UVW.o
		$(CXX) -g -L/opt/AMDAPP/lib/x86_64 $(DEFINES) $(CXXFLAGS) $^ -lOpenCL -o $@


Gridding-CPU.o:	Gridding.cc Common.h Defines.h
		$(CXX) -c $(DEFINES) $(CXXFLAGS) $< -o $@

Gridding-Cuda.ptx:Gridding.cc Common.h Defines.h
		nvcc $(PTXASFLAGS) -x cu --ptx -ccbin=${CXX} -g -D__CUDA__ $(DEFINES) -use_fast_math -arch=$(ARCH) -code=$(ARCH) -Xcompiler "$(CXXFLAGS)" $< -o $@

Gridding-Cuda.o:Gridding.cc Common.h Defines.h
		#nvcc $(PTXASFLAGS) -x cu --compile -ccbin=${CXX} -g -D__CUDA__ $(DEFINES) -use_fast_math -arch=$(ARCH) -code=$(ARCH) -Xcompiler "$(CXXFLAGS)" $< -o $@
		nvcc $(PTXASFLAGS) -x cu --compile -ccbin=${CXX} -g -D__CUDA__ $(DEFINES) -use_fast_math -arch=$(ARCH) -code=$(ARCH) -Xcompiler "-O3,-fopenmp" $< -o $@

Gridding-OpenCL.o:	Gridding.cc Common.h Defines.h
		$(CXX) -c -D__OPENCL__ -I/usr/local/cuda/include -I/opt/AMDAPP/include $(DEFINES) $(CXXFLAGS) $< -o $@
