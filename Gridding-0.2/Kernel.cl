// (C) 2012  John Romein/ASTRON

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "Defines.h"

typedef __global float2 (*GridType)[GRID_V][GRID_U][POLARIZATIONS];
typedef __global struct { float x, y, z; } (*UVWtype)[BASELINES][TIMESTEPS][CHANNELS];
typedef __global float2 (*VisibilitiesType)[BASELINES][TIMESTEPS][CHANNELS][POLARIZATIONS];

#if MODE == MODE_SIMPLE
typedef __global float2 (*SupportType)[W_PLANES][SUPPORT_V][SUPPORT_U];
#elif MODE == MODE_OVERSAMPLE
#if ORDER == ORDER_W_OV_OU_V_U
typedef __global float2 (*SupportType)[W_PLANES][OVERSAMPLE_V][OVERSAMPLE_U][SUPPORT_V][SUPPORT_U];
#elif ORDER == ORDER_W_OV_V_OU_U
typedef __global float2 (*SupportType)[W_PLANES][OVERSAMPLE_V][SUPPORT_V][OVERSAMPLE_U][SUPPORT_U];
#endif
#endif

 

#if 0

inline void atomic_add_float(__global float *ptr, float value)
{
  *ptr += value; // NOT ATOMIC AT ALL!!!
}

#elif 0 && defined cl_khr_int64_base_atomics

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

inline void atomic_add_float2(__global float2 *ptr, float2 value)
{
  ulong old, new;

  do {
    old = * (__global ulong *) ptr;
    new = as_ulong(value + as_float2(old));
  } while (atom_cmpxchg((__global ulong *) ptr, old, new) != old);
}

#elif defined cl_khr_global_int32_base_atomics

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

inline void atomic_add_float(__global float *ptr, float value)
{
  __global int *addr = (__global int *) ptr;

  int old, new;

  do {
    old = addr[0];
    new = as_int(value + as_float(old));
  } while (atom_cmpxchg(addr, old, new) != old);
}


inline void atomic_add_float2(__global float2 *ptr, float2 value)
{
  __global int *addr = (__global int *) ptr;

  int old, new;

  do {
    old = addr[0];
    new = as_int(value.x + as_float(old));
  } while (atom_cmpxchg(addr, old, new) != old);

  do {
    old = addr[1];
    new = as_int(value.y + as_float(old));
  } while (atom_cmpxchg(addr + 1, old, new) != old);
}

#else
#error need atomic add
#endif


inline void atomic_add_to_grid(__global float *ptr, float4 sumR, float4 sumI)
{
#if 1
  atomic_add_float(ptr + 0, sumR.x);
  atomic_add_float(ptr + 1, sumI.x);
  atomic_add_float(ptr + 2, sumR.y);
  atomic_add_float(ptr + 3, sumI.y);
  atomic_add_float(ptr + 4, sumR.z);
  atomic_add_float(ptr + 5, sumI.z);
  atomic_add_float(ptr + 6, sumR.w);
  atomic_add_float(ptr + 7, sumI.w);
#else
  float dummy;
  asm volatile ("atom.global.add.f32 %0, [%1+0], %2;" : "=f" (dummy) : "l" (ptr), "f" (sumR.x) : "memory");
  asm volatile ("atom.global.add.f32 %0, [%1+4], %2;" : "=f" (dummy) : "l" (ptr), "f" (sumI.x) : "memory");
  asm volatile ("atom.global.add.f32 %0, [%1+8], %2;" : "=f" (dummy) : "l" (ptr), "f" (sumR.y) : "memory");
  asm volatile ("atom.global.add.f32 %0, [%1+12], %2;" : "=f" (dummy) : "l" (ptr), "f" (sumI.y) : "memory");
  asm volatile ("atom.global.add.f32 %0, [%1+16], %2;" : "=f" (dummy) : "l" (ptr), "f" (sumR.z) : "memory");
  asm volatile ("atom.global.add.f32 %0, [%1+20], %2;" : "=f" (dummy) : "l" (ptr), "f" (sumI.z) : "memory");
  asm volatile ("atom.global.add.f32 %0, [%1+24], %2;" : "=f" (dummy) : "l" (ptr), "f" (sumR.w) : "memory");
  asm volatile ("atom.global.add.f32 %0, [%1+28], %2;" : "=f" (dummy) : "l" (ptr), "f" (sumI.w) : "memory");
#endif
}


__kernel void clear_grid(__global float2 *grid_ptr)
{
  uint nrGridPoints	 = GRID_V * GRID_U * POLARIZATIONS;
  uint nrGridPointsPerSM = (nrGridPoints + get_global_size(1) - 1) / get_global_size(1);
  uint first		 = nrGridPointsPerSM * get_global_id(1) + get_local_id(0);
  uint last		 = min(first + nrGridPointsPerSM, nrGridPoints);
  uint step		 = get_local_size(0);

  for (uint i = first; i < last; i += step)
    grid_ptr[i] = (float2) (0, 0);
}


#if defined USE_TEXTURE
__constant sampler_t supportSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
#endif


__kernel
#if defined INTEL
__attribute__((vec_type_hint(float4))) // avoid auto-vectorization
#endif
void add_to_grid(__global void *grid_ptr,
			  __global void *visibilities_ptr,
			  __global void *uvw_ptr,
			  __global uint2 *supportPixelsUsed,
#if defined USE_TEXTURE
			  __read_only image3d_t supportImage
#else
			  __global void *support_ptr
#endif
)
{
  GridType	   grid		= (GridType) grid_ptr;
  VisibilitiesType visibilities = (VisibilitiesType) visibilities_ptr;
  UVWtype	   uvw		= (UVWtype) uvw_ptr;
#if !defined USE_TEXTURE
  SupportType	   support	= (SupportType) support_ptr;
#endif

  uint	bl	    = get_global_id(1);
  uint2	supportSize = supportPixelsUsed[bl];

  __local uint4  shared_info[TIMESTEPS][CHANNELS];
  __local float4 shared_visR[TIMESTEPS][CHANNELS], shared_visI[TIMESTEPS][CHANNELS];

  for (uint i = get_local_id(0); i < TIMESTEPS * CHANNELS; i += get_local_size(0)) {
    float2 visXX = (*visibilities)[bl][0][i][0];
    float2 visXY = (*visibilities)[bl][0][i][1];
    float2 visYX = (*visibilities)[bl][0][i][2];
    float2 visYY = (*visibilities)[bl][0][i][3];
    shared_visR[0][i] = (float4) (visXX.x, visXY.x, visYX.x, visYY.x);
    shared_visI[0][i] = (float4) (visXX.y, visXY.y, visYX.y, visYY.y);

    float u = (*uvw)[bl][0][i].x;
    float v = (*uvw)[bl][0][i].y;
    float w = (*uvw)[bl][0][i].z;

#if MODE == MODE_SIMPLE
    uint u_int = round(u);
    uint v_int = round(v);
    shared_info[0][i] = (uint4) { -u_int % supportSize.x, -v_int % supportSize.y, 0, u_int + GRID_U * v_int };
#elif MODE == MODE_OVERSAMPLE
    unsigned u_int  = (unsigned) (u);
    unsigned v_int  = (unsigned) (v);
    float    u_frac = u - u_int;
    float    v_frac = v - v_int;

#if defined USE_TEXTURE
    unsigned uv_frac_w_offset = (unsigned) w * OVERSAMPLE_V * OVERSAMPLE_U + (unsigned) (OVERSAMPLE_V * v_frac) * OVERSAMPLE_U + (unsigned) (OVERSAMPLE_U * u_frac);
#else
#if ORDER == ORDER_W_OV_OU_V_U
    unsigned uv_frac_w_offset = (unsigned) w * OVERSAMPLE_V * OVERSAMPLE_U * SUPPORT_V * SUPPORT_U + (unsigned) (OVERSAMPLE_V * v_frac) * OVERSAMPLE_U * SUPPORT_V * SUPPORT_U + (unsigned) (OVERSAMPLE_U * u_frac) * SUPPORT_V * SUPPORT_U;
#elif ORDER == ORDER_W_OV_V_OU_U
    unsigned uv_frac_w_offset = (unsigned) w * OVERSAMPLE_V * SUPPORT_V * OVERSAMPLE_U * SUPPORT_U + (unsigned) (OVERSAMPLE_V * v_frac) * SUPPORT_V * OVERSAMPLE_U * SUPPORT_U + (unsigned) (OVERSAMPLE_U * u_frac) * SUPPORT_U;
#endif
#endif

    shared_info[0][i] = (uint4) { -u_int % supportSize.x, -v_int % supportSize.y, uv_frac_w_offset, u_int + GRID_U * v_int };
#elif MODE == MODE_INTERPOLATE
#endif
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  //for (uint i = get_local_id(0); i < supportSize.x * supportSize.y; i += get_local_size(0)) {
  for (int i = supportSize.x * supportSize.y - get_local_id(0) - 1; i >= 0; i -= get_local_size(0)) {
    uint box_u = - (i % supportSize.x);
    uint box_v = - (i / supportSize.x);

    float4 sumR = (float4) (0, 0, 0, 0), sumI = (float4) (0, 0, 0, 0);

    uint grid_point = get_local_id(0);

    //for (uint time = 0; time < TIMESTEPS; time ++) {
      for (uint ch = 0; ch < TIMESTEPS * CHANNELS; ch ++) {
	uint4 info = shared_info[0][ch];
	int my_support_u = box_u + info.x;
	int my_support_v = box_v + info.y;

	if (my_support_u < 0) my_support_u += supportSize.x;
	if (my_support_v < 0) my_support_v += supportSize.y;

#if defined USE_TEXTURE
	float4 supportPixel = read_imagef(supportImage, supportSampler, (int4) (my_support_u, my_support_v, info.z, 0));
#else
#if ORDER == ORDER_W_OV_OU_V_U
	float2 supportPixel = (*support)[0][0][0][my_support_v][my_support_u + info.z];
#elif ORDER == ORDER_W_OV_V_OU_U
	float2 supportPixel = (*support)[0][0][my_support_v][0][my_support_u + info.z];
#endif
#endif

	uint new_grid_point = my_support_u + GRID_U * my_support_v + info.w;

	if (new_grid_point != grid_point) {
	  atomic_add_to_grid((__global float *) &(*grid)[0][grid_point][0], sumR, sumI);
	  grid_point = new_grid_point;
	  sumR = sumI = (float4) (0, 0, 0, 0);
	}

	float4 visR = shared_visR[0][ch];
	float4 visI = shared_visI[0][ch];

	sumR += supportPixel.xxxx * visR;
	sumI += supportPixel.yyyy * visR;
	sumR -= supportPixel.yyyy * visI;
	sumI += supportPixel.xxxx * visI;
      }
    //}

    atomic_add_to_grid((__global float *) &(*grid)[0][grid_point][0], sumR, sumI);
  }
}
