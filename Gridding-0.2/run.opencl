#!/bin/bash

mkdir -p benchmarks

#DEVICE=2xE5-2680
#DEVICE=2xE5620
#DEVICE=GTX680
#DEVICE=GTX580
#DEVICE=HD6970
#DEVICE=HD7970

#PLATFORM="Intel(R) OpenCL"
#PLATFORM="NVIDIA CUDA"
#PLATFORM="AMD Accelerated Parallel Processing"

#VENDOR="Intel"
#VENDOR="Nvidia"
#VENDOR="AMD"

CXX=icc

export PLATFORM

for c in 16 24 32 48 64 96 128 192 256; do
  for w in 32; do
    if [ $w -le 32 -o -$c -le 128 ]; then
      if [ "$DEVICE" == "GTX580" ]; then
	TIMESTEPS=60
	BLOCKS=36
	USE_TEXTURE=1
      elif [ "$DEVICE" == "GTX680" ]; then
        if [ $c -le 16 ]; then
	  TIMESTEPS=10
	  BLOCKS=216
	elif [ $c -le 24 ]; then
	  TIMESTEPS=20
	  BLOCKS=108
	else
	  TIMESTEPS=30
	  BLOCKS=72
	fi
	#USE_TEXTURE=1
      elif [ "$DEVICE" == "HD6970" -o "$DEVICE" == "HD7970" ]; then
	TIMESTEPS=20
	BLOCKS=108
      else
	TIMESTEPS=40
	BLOCKS=54
      fi

      echo oversampling $DEVICE $VENDOR $c $w
      echo $CXX -D__OPENCL__ -I/cm/shared/package/amd-app-sdk/2.5/include -DMODE=1 -DGRID_U=2048 -DGRID_V=2048 -DSUPPORT_U=$c -DSUPPORT_V=$c -DW_PLANES=$w -DTIMESTEPS=$TIMESTEPS -DBLOCKS=$BLOCKS ${USE_TEXTURE:+-DUSE_TEXTURE} -O3 -L/cm/shared/package/intel-ocl-sdk/1.1/lib64 -lOpenCL -fopenmp -g -mavx Gridding.cc UVW.o -o a.out.$DEVICE.$VENDOR.OpenCL.Wprojection.$c.$w
      $CXX -D__OPENCL__ -I/cm/shared/package/amd-app-sdk/2.5/include -DMODE=1 -DGRID_U=2048 -DGRID_V=2048 -DSUPPORT_U=$c -DSUPPORT_V=$c -DW_PLANES=$w -DTIMESTEPS=$TIMESTEPS -DBLOCKS=$BLOCKS ${USE_TEXTURE:+-DUSE_TEXTURE} -O3 Gridding.cc UVW.o -L/cm/shared/package/intel-ocl-sdk/1.1/lib64 -lOpenCL -fopenmp -g -mavx -o a.out.$DEVICE.$VENDOR.OpenCL.Wprojection.$c.$w
      NR_GPUS=1 a.out.$DEVICE.$VENDOR.OpenCL.Wprojection.$c.$w 2>&1 | tee benchmarks/out.$DEVICE.$VENDOR.OpenCL.Wprojection.$c.$w
    fi
  done
done

#for c in 256 192 128 96 64 48 32 24 16; do
#  for i in 1024 512 256 128 64; do
#    if [ $c -le $i ]; then
#      echo interpolation $c $i
#      $CXX -g -DMODE=2 -DX=`expr $c - 1` -DSUPPORT_U=`expr $i - 2` -DSUPPORT_V=`expr $i - 2` -DW_PLANES=32 -DTIMESTEPS=54 -DBLOCKS=40 -O3 -fopenmp UVW.o -x c++ -mavx Gridding.cc -o opencl.out.2.$c.$i
#      prun -t 24:00:00 opencl.out.2.$c.$i 1 </dev/null >benchmarks/CPU.2.$c.$i 2>&1 &
#    fi
#  done
#done
