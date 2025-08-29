/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "yuv_to_rgb_kernel.h"

#define checkRuntime(call) check_runtime(call, #call, __LINE__, __FILE__)
#define half2short(h) (*(unsigned short*)&h)

typedef unsigned char uint8_t;

template <typename _T>
struct AsUnion4 {};
template <>
struct AsUnion4<uint8_t> {
  typedef uchar4 type;
};
template <>
struct AsUnion4<float> {
  typedef float4 type;
};
template <>
struct AsUnion4<__half> {
  typedef ushort4 type;
};
template <>
struct AsUnion4<int8_t> {
  typedef char4 type;
};

template <typename _T>
struct AsUnion3 {};
template <>
struct AsUnion3<uint8_t> {
  typedef uchar3 type;
};
template <>
struct AsUnion3<float> {
  typedef float3 type;
};
template <>
struct AsUnion3<__half> {
  typedef ushort3 type;
};
template <>
struct AsUnion3<int8_t> {
  typedef char3 type;
};

template <DataType _T>
struct AsPODType {};
template <>
struct AsPODType<DataType::Uint8> {
  typedef uint8_t type;
};
template <>
struct AsPODType<DataType::Float32> {
  typedef float type;
};
template <>
struct AsPODType<DataType::Float16> {
  typedef __half type;
};
template <>
struct AsPODType<DataType::Int8> {
  typedef int8_t type;
};

enum class Parallel : unsigned int { None = 0, SinglePixel = 1, FourPixel = 2 };

static __device__ __forceinline__ uchar4 make4(uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3) {
  return make_uchar4(v0, v1, v2, v3);
}
static __device__ __forceinline__ char4 make4(int8_t v0, int8_t v1, int8_t v2, int8_t v3) {
  return make_char4(v0, v1, v2, v3);
}
static __device__ __forceinline__ float4 make4(float v0, float v1, float v2, float v3) {
  return make_float4(v0, v1, v2, v3);
}
static __device__ __forceinline__ ushort4 make4(__half v0, __half v1, __half v2, __half v3) {
  return make_ushort4(half2short(v0), half2short(v1), half2short(v2), half2short(v3));
}

static __device__ __forceinline__ uchar3 make3(uint8_t v0, uint8_t v1, uint8_t v2) {
  return make_uchar3(v0, v1, v2);
}
static __device__ __forceinline__ char3 make3(int8_t v0, int8_t v1, int8_t v2) {
  return make_char3(v0, v1, v2);
}
static __device__ __forceinline__ float3 make3(float v0, float v1, float v2) {
  return make_float3(v0, v1, v2);
}
static __device__ __forceinline__ ushort3 make3(__half v0, __half v1, __half v2) {
  return make_ushort3(half2short(v0), half2short(v1), half2short(v2));
}

#define INTER_RESIZE_COEF_BITS 11
#define INTER_RESIZE_COEF_SCALE (1 << INTER_RESIZE_COEF_BITS)
#define CAST_BITS (INTER_RESIZE_COEF_BITS << 1)

template <typename _T>
static __forceinline__ __device__ _T limit(_T value, _T low, _T high) {
  return value < low ? low : (value > high ? high : value);
}

template <typename _T>
static __device__ __forceinline__ uint8_t u8cast(_T value) {
  return value < 0 ? 0 : (value >= 255 ? 255 : uint8_t(value));
}

template <typename _T>
static __device__ __forceinline__ int8_t s8cast(_T value) {
  return value <= -128 ? -128 : (value >= 127 ? 127 : int8_t(value));
}

template <typename _T>
struct Saturate {};
template <>
struct Saturate<int8_t> {
  __device__ __forceinline__ static int8_t cast(float x) { return s8cast(x); }
};
template <>
struct Saturate<uint8_t> {
  __device__ __forceinline__ static uint8_t cast(float x) { return u8cast(x); }
};
template <>
struct Saturate<float> {
  __device__ __forceinline__ static float cast(float x) { return x; }
};
template <>
struct Saturate<__half> {
  __device__ __forceinline__ static __half cast(float x) { return __half(x); }
};

static bool __inline__ check_runtime(cudaError_t e, const char* call, int line, const char* file) {
  if (e != cudaSuccess) {
    fprintf(
      stderr,
      "CUDA Runtime error %s # %s, code = %s [ %d ] in file %s:%d\n",
      call,
      cudaGetErrorString(e),
      cudaGetErrorName(e),
      e,
      file,
      line);
    return false;
  }
  return true;
}

template <typename _DataType, Parallel parallel, PixelLayout _Layout>
struct DataLayoutInvoker {};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// NHWC RGB
template <typename _DataType>
struct DataLayoutInvoker<_DataType, Parallel::SinglePixel, PixelLayout::NHWC_RGB> {
  static __device__ __forceinline__ void call(
    _DataType* pdst, _DataType r, _DataType g, _DataType b, int ib, int x, int y, int width, int stride, int height) {
    _DataType* p = pdst + (ib * height + y) * stride + x * 3;
    p[0] = r;
    p[1] = g;
    p[2] = b;
  }
};

template <typename _DataType>
struct DataLayoutInvoker<_DataType, Parallel::FourPixel, PixelLayout::NHWC_RGB> {
  static __device__ __forceinline__ void call(
    _DataType* pdst,
    _DataType r[4],
    _DataType g[4],
    _DataType b[4],
    int ib,
    int x,
    int y,
    int width,
    int stride,
    int height) {
    _DataType* p0 = pdst + (ib * height + y) * stride + (x + 0) * 3;
    _DataType* p1 = pdst + (ib * height + y) * stride + (x + 1) * 3;
    _DataType* p2 = pdst + (ib * height + y) * stride + (x + 2) * 3;
    _DataType* p3 = pdst + (ib * height + y) * stride + (x + 3) * 3;
    *(typename AsUnion3<_DataType>::type*)p0 = make3(r[0], g[0], b[0]);
    *(typename AsUnion3<_DataType>::type*)p1 = make3(r[1], g[1], b[1]);
    *(typename AsUnion3<_DataType>::type*)p2 = make3(r[2], g[2], b[2]);
    *(typename AsUnion3<_DataType>::type*)p3 = make3(r[3], g[3], b[3]);
  }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// NHWC BGR
template <typename _DataType>
struct DataLayoutInvoker<_DataType, Parallel::SinglePixel, PixelLayout::NHWC_BGR> {
  static __device__ __forceinline__ void call(
    _DataType* pdst, _DataType r, _DataType g, _DataType b, int ib, int x, int y, int width, int stride, int height) {
    _DataType* p = pdst + (ib * height + y) * stride + x * 3;
    p[0] = b;
    p[1] = g;
    p[2] = r;
  }
};

template <typename _DataType>
struct DataLayoutInvoker<_DataType, Parallel::FourPixel, PixelLayout::NHWC_BGR> {
  static __device__ __forceinline__ void call(
    _DataType* pdst,
    _DataType r[4],
    _DataType g[4],
    _DataType b[4],
    int ib,
    int x,
    int y,
    int width,
    int stride,
    int height) {
    _DataType* p0 = pdst + (ib * height + y) * stride + (x + 0) * 3;
    _DataType* p1 = pdst + (ib * height + y) * stride + (x + 1) * 3;
    _DataType* p2 = pdst + (ib * height + y) * stride + (x + 2) * 3;
    _DataType* p3 = pdst + (ib * height + y) * stride + (x + 3) * 3;
    *(typename AsUnion3<_DataType>::type*)p0 = make3(b[0], g[0], r[0]);
    *(typename AsUnion3<_DataType>::type*)p1 = make3(b[1], g[1], r[1]);
    *(typename AsUnion3<_DataType>::type*)p2 = make3(b[2], g[2], r[2]);
    *(typename AsUnion3<_DataType>::type*)p3 = make3(b[3], g[3], r[3]);
  }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// NCHW RGB
template <typename _DataType>
struct DataLayoutInvoker<_DataType, Parallel::SinglePixel, PixelLayout::NCHW_RGB> {
  static __device__ __forceinline__ void call(
    _DataType* pdst, _DataType r, _DataType g, _DataType b, int ib, int x, int y, int width, int stride, int height) {
    *(pdst + (((ib * 3 + 0) * height + y) * width + x)) = r;
    *(pdst + (((ib * 3 + 1) * height + y) * width + x)) = g;
    *(pdst + (((ib * 3 + 2) * height + y) * width + x)) = b;
  }
};

template <typename _DataType>
struct DataLayoutInvoker<_DataType, Parallel::FourPixel, PixelLayout::NCHW_RGB> {
  static __device__ __forceinline__ void call(
    _DataType* pdst,
    _DataType r[4],
    _DataType g[4],
    _DataType b[4],
    int ib,
    int x,
    int y,
    int width,
    int stride,
    int height) {
    *(typename AsUnion4<_DataType>::type*)(pdst + (((ib * 3 + 0) * height + y) * width + x)) =
      make4(r[0], r[1], r[2], r[3]);
    *(typename AsUnion4<_DataType>::type*)(pdst + (((ib * 3 + 1) * height + y) * width + x)) =
      make4(g[0], g[1], g[2], g[3]);
    *(typename AsUnion4<_DataType>::type*)(pdst + (((ib * 3 + 2) * height + y) * width + x)) =
      make4(b[0], b[1], b[2], b[3]);
  }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// NCHW BGR
template <typename _DataType>
struct DataLayoutInvoker<_DataType, Parallel::SinglePixel, PixelLayout::NCHW_BGR> {
  static __device__ __forceinline__ void call(
    _DataType* pdst, _DataType r, _DataType g, _DataType b, int ib, int x, int y, int width, int stride, int height) {
    *(pdst + (((ib * 3 + 0) * height + y) * width + x)) = b;
    *(pdst + (((ib * 3 + 1) * height + y) * width + x)) = g;
    *(pdst + (((ib * 3 + 2) * height + y) * width + x)) = r;
  }
};

template <typename _DataType>
struct DataLayoutInvoker<_DataType, Parallel::FourPixel, PixelLayout::NCHW_BGR> {
  static __device__ __forceinline__ void call(
    _DataType* pdst,
    _DataType r[4],
    _DataType g[4],
    _DataType b[4],
    int ib,
    int x,
    int y,
    int width,
    int stride,
    int height) {
    *(typename AsUnion4<_DataType>::type*)(pdst + (((ib * 3 + 0) * height + y) * width + x)) =
      make4(b[0], b[1], b[2], b[3]);
    *(typename AsUnion4<_DataType>::type*)(pdst + (((ib * 3 + 1) * height + y) * width + x)) =
      make4(g[0], g[1], g[2], g[3]);
    *(typename AsUnion4<_DataType>::type*)(pdst + (((ib * 3 + 2) * height + y) * width + x)) =
      make4(r[0], r[1], r[2], r[3]);
  }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// NCHWx RGB
template <typename _DataType, int _NSC, Parallel parallel>
struct NCHWxSaveRGBInvoker {};

template <typename _DataType, int _NSC>
struct NCHWxSaveRGBInvoker<_DataType, _NSC, Parallel::SinglePixel> {
  static __device__ __forceinline__ void call(
    _DataType* pdst, _DataType r, _DataType g, _DataType b, int ib, int x, int y, int width, int stride, int height) {
    // for NCHW(_NSC) tensor.view(N, C/_NSC, _NSC, H, W).transpose(0, 1, 3, 4, 2)
    _DataType* p = pdst + ((ib * height + y) * width + x) * _NSC;
    p[0] = r;
    p[1] = g;
    p[2] = b;
  }
};

template <typename _DataType, int _NSC>
struct NCHWxSaveRGBInvoker<_DataType, _NSC, Parallel::FourPixel> {
  static __device__ __forceinline__ void call(
    _DataType* pdst,
    _DataType r[4],
    _DataType g[4],
    _DataType b[4],
    int ib,
    int x,
    int y,
    int width,
    int stride,
    int height) {
    _DataType* p0 = pdst + ((ib * height + y) * width + x + 0) * _NSC;
    _DataType* p1 = pdst + ((ib * height + y) * width + x + 1) * _NSC;
    _DataType* p2 = pdst + ((ib * height + y) * width + x + 2) * _NSC;
    _DataType* p3 = pdst + ((ib * height + y) * width + x + 3) * _NSC;
    *(typename AsUnion3<_DataType>::type*)p0 = make3(r[0], g[0], b[0]);
    *(typename AsUnion3<_DataType>::type*)p1 = make3(r[1], g[1], b[1]);
    *(typename AsUnion3<_DataType>::type*)p2 = make3(r[2], g[2], b[2]);
    *(typename AsUnion3<_DataType>::type*)p3 = make3(r[3], g[3], b[3]);
  }
};

template <typename _DataType, int _NSC, Parallel parallel>
struct NCHWxSaveBGRInvoker {};

template <typename _DataType, int _NSC>
struct NCHWxSaveBGRInvoker<_DataType, _NSC, Parallel::SinglePixel> {
  static __device__ __forceinline__ void call(
    _DataType* pdst, _DataType r, _DataType g, _DataType b, int ib, int x, int y, int width, int stride, int height) {
    // for NCHW(_NSC) tensor.view(N, C/_NSC, _NSC, H, W).transpose(0, 1, 3, 4, 2)
    _DataType* p = pdst + ((ib * height + y) * width + x) * _NSC;
    p[0] = b;
    p[1] = g;
    p[2] = r;
  }
};

template <typename _DataType, int _NSC>
struct NCHWxSaveBGRInvoker<_DataType, _NSC, Parallel::FourPixel> {
  static __device__ __forceinline__ void call(
    _DataType* pdst,
    _DataType r[4],
    _DataType g[4],
    _DataType b[4],
    int ib,
    int x,
    int y,
    int width,
    int stride,
    int height) {
    _DataType* p0 = pdst + ((ib * height + y) * width + x + 0) * _NSC;
    _DataType* p1 = pdst + ((ib * height + y) * width + x + 1) * _NSC;
    _DataType* p2 = pdst + ((ib * height + y) * width + x + 2) * _NSC;
    _DataType* p3 = pdst + ((ib * height + y) * width + x + 3) * _NSC;
    *(typename AsUnion3<_DataType>::type*)p0 = make3(b[0], g[0], r[0]);
    *(typename AsUnion3<_DataType>::type*)p1 = make3(b[1], g[1], r[1]);
    *(typename AsUnion3<_DataType>::type*)p2 = make3(b[2], g[2], r[2]);
    *(typename AsUnion3<_DataType>::type*)p3 = make3(b[3], g[3], r[3]);
  }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// NCHWx RGB/BGR
template <typename _DataType>
struct DataLayoutInvoker<_DataType, Parallel::SinglePixel, PixelLayout::NCHW16_RGB> {
  static __device__ __forceinline__ void call(
    _DataType* pdst, _DataType r, _DataType g, _DataType b, int ib, int x, int y, int width, int stride, int height) {
    NCHWxSaveRGBInvoker<_DataType, 16, Parallel::SinglePixel>::call(pdst, r, g, b, ib, x, y, width, stride, height);
  }
};

template <typename _DataType>
struct DataLayoutInvoker<_DataType, Parallel::SinglePixel, PixelLayout::NCHW16_BGR> {
  static __device__ __forceinline__ void call(
    _DataType* pdst, _DataType r, _DataType g, _DataType b, int ib, int x, int y, int width, int stride, int height) {
    NCHWxSaveBGRInvoker<_DataType, 16, Parallel::SinglePixel>::call(pdst, r, g, b, ib, x, y, width, stride, height);
  }
};

template <typename _DataType>
struct DataLayoutInvoker<_DataType, Parallel::SinglePixel, PixelLayout::NCHW32_RGB> {
  static __device__ __forceinline__ void call(
    _DataType* pdst, _DataType r, _DataType g, _DataType b, int ib, int x, int y, int width, int stride, int height) {
    NCHWxSaveRGBInvoker<_DataType, 32, Parallel::SinglePixel>::call(pdst, r, g, b, ib, x, y, width, stride, height);
  }
};

template <typename _DataType>
struct DataLayoutInvoker<_DataType, Parallel::SinglePixel, PixelLayout::NCHW32_BGR> {
  static __device__ __forceinline__ void call(
    _DataType* pdst, _DataType r, _DataType g, _DataType b, int ib, int x, int y, int width, int stride, int height) {
    NCHWxSaveBGRInvoker<_DataType, 32, Parallel::SinglePixel>::call(pdst, r, g, b, ib, x, y, width, stride, height);
  }
};

template <typename _DataType>
struct DataLayoutInvoker<_DataType, Parallel::SinglePixel, PixelLayout::NCHW4_RGB> {
  static __device__ __forceinline__ void call(
    _DataType* pdst, _DataType r, _DataType g, _DataType b, int ib, int x, int y, int width, int stride, int height) {
    NCHWxSaveRGBInvoker<_DataType, 4, Parallel::SinglePixel>::call(pdst, r, g, b, ib, x, y, width, stride, height);
  }
};

template <typename _DataType>
struct DataLayoutInvoker<_DataType, Parallel::SinglePixel, PixelLayout::NCHW4_BGR> {
  static __device__ __forceinline__ void call(
    _DataType* pdst, _DataType r, _DataType g, _DataType b, int ib, int x, int y, int width, int stride, int height) {
    NCHWxSaveBGRInvoker<_DataType, 4, Parallel::SinglePixel>::call(pdst, r, g, b, ib, x, y, width, stride, height);
  }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename _DataType>
struct DataLayoutInvoker<_DataType, Parallel::FourPixel, PixelLayout::NCHW16_RGB> {
  static __device__ __forceinline__ void call(
    _DataType* pdst,
    _DataType r[4],
    _DataType g[4],
    _DataType b[4],
    int ib,
    int x,
    int y,
    int width,
    int stride,
    int height) {
    NCHWxSaveRGBInvoker<_DataType, 16, Parallel::FourPixel>::call(pdst, r, g, b, ib, x, y, width, stride, height);
  }
};

template <typename _DataType>
struct DataLayoutInvoker<_DataType, Parallel::FourPixel, PixelLayout::NCHW16_BGR> {
  static __device__ __forceinline__ void call(
    _DataType* pdst,
    _DataType r[4],
    _DataType g[4],
    _DataType b[4],
    int ib,
    int x,
    int y,
    int width,
    int stride,
    int height) {
    NCHWxSaveBGRInvoker<_DataType, 16, Parallel::FourPixel>::call(pdst, r, g, b, ib, x, y, width, stride, height);
  }
};

template <typename _DataType>
struct DataLayoutInvoker<_DataType, Parallel::FourPixel, PixelLayout::NCHW32_RGB> {
  static __device__ __forceinline__ void call(
    _DataType* pdst,
    _DataType r[4],
    _DataType g[4],
    _DataType b[4],
    int ib,
    int x,
    int y,
    int width,
    int stride,
    int height) {
    NCHWxSaveRGBInvoker<_DataType, 32, Parallel::FourPixel>::call(pdst, r, g, b, ib, x, y, width, stride, height);
  }
};

template <typename _DataType>
struct DataLayoutInvoker<_DataType, Parallel::FourPixel, PixelLayout::NCHW32_BGR> {
  static __device__ __forceinline__ void call(
    _DataType* pdst,
    _DataType r[4],
    _DataType g[4],
    _DataType b[4],
    int ib,
    int x,
    int y,
    int width,
    int stride,
    int height) {
    NCHWxSaveBGRInvoker<_DataType, 32, Parallel::FourPixel>::call(pdst, r, g, b, ib, x, y, width, stride, height);
  }
};

template <typename _DataType>
struct DataLayoutInvoker<_DataType, Parallel::FourPixel, PixelLayout::NCHW4_RGB> {
  static __device__ __forceinline__ void call(
    _DataType* pdst,
    _DataType r[4],
    _DataType g[4],
    _DataType b[4],
    int ib,
    int x,
    int y,
    int width,
    int stride,
    int height) {
    NCHWxSaveRGBInvoker<_DataType, 4, Parallel::FourPixel>::call(pdst, r, g, b, ib, x, y, width, stride, height);
  }
};

template <typename _DataType>
struct DataLayoutInvoker<_DataType, Parallel::FourPixel, PixelLayout::NCHW4_BGR> {
  static __device__ __forceinline__ void call(
    _DataType* pdst,
    _DataType r[4],
    _DataType g[4],
    _DataType b[4],
    int ib,
    int x,
    int y,
    int width,
    int stride,
    int height) {
    NCHWxSaveBGRInvoker<_DataType, 4, Parallel::FourPixel>::call(pdst, r, g, b, ib, x, y, width, stride, height);
  }
};
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static __device__ unsigned int __forceinline__ round_down2(unsigned int num) {
  return num & (~1);
}

template <typename _T>
static __device__ void __forceinline__ scale_rgb(
  uint8_t r0,
  uint8_t g0,
  uint8_t b0,
  _T& r,
  _T& g,
  _T& b,
  float mean0,
  float mean1,
  float mean2,
  float scale0,
  float scale1,
  float scale2) {
  r = Saturate<_T>::cast((r0 - mean0) * scale0);
  g = Saturate<_T>::cast((g0 - mean1) * scale1);
  b = Saturate<_T>::cast((b0 - mean2) * scale2);
}

// here align to opencv
static __device__ void __forceinline__ yuv2rgb(int y, int u, int v, uint8_t& r, uint8_t& g, uint8_t& b) {
  int iyval = 1220542 * max(0, y - 16);
  r = u8cast((iyval + 1673527 * (v - 128) + (1 << 19)) >> 20);
  g = u8cast((iyval - 852492 * (v - 128) - 409993 * (u - 128) + (1 << 19)) >> 20);
  b = u8cast((iyval + 2116026 * (u - 128) + (1 << 19)) >> 20);
}

template <YUVFormat yuv_format>
static __device__ void __forceinline__ load_yuv_pixel(
  const void* luma,
  const void* chroma,
  int x,
  int y,
  int down_x,
  int width,
  int stride,
  uint8_t& r,
  uint8_t& g,
  uint8_t& b);

// BL sample pixel implmentation
template <>
__device__ void __forceinline__ load_yuv_pixel<YUVFormat::NV12BlockLinear>(
  const void* luma,
  const void* chroma,
  int x,
  int y,
  int down_x,
  int width,
  int stride,
  uint8_t& r,
  uint8_t& g,
  uint8_t& b) {
  uint8_t yv = tex2D<uint8_t>((cudaTextureObject_t)luma, x, y);
  // If chroma bytes per pixel = 1.
  // uint8_t uv = tex2D<uint8_t>((cudaTextureObject_t)chroma, down_x + 0, y / 2);
  // uint8_t vv = tex2D<uint8_t>((cudaTextureObject_t)chroma, down_x + 1, y / 2);
  // yuv2rgb(yv, uv, vv, r, g, b);

  // If chroma bytes per pixel = 2.
  uchar2 uv = tex2D<uchar2>((cudaTextureObject_t)chroma, x / 2, y / 2);
  yuv2rgb(yv, uv.x, uv.y, r, g, b);
}

// PL sample pixel implmentation
template <>
__device__ void __forceinline__ load_yuv_pixel<YUVFormat::NV12PitchLinear>(
  const void* luma,
  const void* chroma,
  int x,
  int y,
  int down_x,
  int width,
  int stride,
  uint8_t& r,
  uint8_t& g,
  uint8_t& b) {
  uint8_t yv = *((const unsigned char*)luma + y * stride + x);
  uint8_t uv = *((const unsigned char*)chroma + (y / 2) * stride + down_x + 0);
  uint8_t vv = *((const unsigned char*)chroma + (y / 2) * stride + down_x + 1);
  yuv2rgb(yv, uv, vv, r, g, b);
}

//     Y U Y V Y U Y V      Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y
//     Y U Y V Y U Y V      Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y
//     Y U Y V Y U Y V      Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y
//     Y U Y V Y U Y V      Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y
//     Y U Y V Y U Y V      U U U U U U      V V V V V V      U V U V U V      V U V U V U
//     Y U Y V Y U Y V      V V V V V V      U U U U U U      U V U V U V      V U V U V U
//        - YUYV -           - I420 -          - YV12 -         - NV12 -         - NV21 -

// YUV422Packed_YUYV sample pixel implmentation
template <>
__device__ void __forceinline__ load_yuv_pixel<YUVFormat::YUV422Packed_YUYV>(
  const void* luma,
  const void* chroma,
  int x,
  int y,
  int down_x,
  int width,
  int stride,
  uint8_t& r,
  uint8_t& g,
  uint8_t& b) {
  // 0, 1, 2, 3, 4, 5, 6, 7
  // 0, 0, 2, 2, 4, 4, 6, 6
  // Y, U, Y, V, Y, U, Y, V
  // 0,    1,    2,    3
  uchar4 yuv = *(uchar4*)((const uint8_t*)luma + y * stride + (x / 2) * 4);
  if (x == down_x) {
    yuv2rgb(yuv.x, yuv.y, yuv.w, r, g, b);
  } else {
    yuv2rgb(yuv.z, yuv.y, yuv.w, r, g, b);
  }
}

template <typename DType, YUVFormat yuv_format, Interpolation interp, bool fully_coverage>
struct SamplePixel {};

// BL sample pixel implmentation
template <typename DType, YUVFormat format>
struct SamplePixel<DType, format, Interpolation::Nearest, false> {
  static __device__ void __forceinline__ call(
    const void* luma,
    const void* chroma,
    int x,
    int y,
    float sx,
    float sy,
    int output_xoffset,
    int output_yoffset,
    int ybatch_offset,
    int width,
    int stride,
    int height,
    uint8_t& r,
    uint8_t& g,
    uint8_t& b,
    FillColor fillcolor) {
    // In some cases, the floating point precision will lead to miscalculation of the value,
    // making the result not exactly match with opencv,
    // so here you need to add eps as precision compensation
    //
    // A special case is when the input is 3840 and the output is 446, x = 223:
    // const int src_x_double = 223.0  * (3840.0  / 446.0);            // -> 1920
    // const int src_x_float  = 223.0f * (3840.0f / 446.0f);           // -> 1919
    // const int src_x_float  = 223.0f * (3840.0f / 446.0f) + 1e-5;    // -> 1920
    //
    // !!! If you want to use the double for sx/sy, you'll get a 2x speed drop
    // const float eps = 1e-5;
    // int ix = x * sx + eps;
    // int iy = y * sy + eps + ybatch_offset;
    //
    int transed_dx = x - output_xoffset;
    int transed_dy = y - output_yoffset;
    const float eps = 1e-5;
    int ix = transed_dx * sx + eps;
    int iy = transed_dy * sy + eps + ybatch_offset;
    if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
      load_yuv_pixel<format>(luma, chroma, ix, iy, round_down2(ix), width, stride, r, g, b);
    } else {
      r = fillcolor.color[0];
      g = fillcolor.color[1];
      b = fillcolor.color[2];
    }
  }
};

template <typename DType, YUVFormat format>
struct SamplePixel<DType, format, Interpolation::Bilinear, false> {
  static __device__ void __forceinline__ call(
    const void* luma,
    const void* chroma,
    int x,
    int y,
    float sx,
    float sy,
    int output_xoffset,
    int output_yoffset,
    int ybatch_offset,
    int width,
    int stride,
    int height,
    uint8_t& r,
    uint8_t& g,
    uint8_t& b,
    FillColor fillcolor) {
    uint8_t r0[4], g0[4], b0[4];
    int transed_dx = x - output_xoffset;
    int transed_dy = y - output_yoffset;
    float src_x = (transed_dx + 0.5f) * sx - 0.5f;
    float src_y = (transed_dy + 0.5f) * sy - 0.5f;
    int y_low = floorf(src_y);
    int x_low = floorf(src_x);
    int y_high = y_low + 1;
    int x_high = x_low + 1;

    int ly = rint((src_y - y_low) * INTER_RESIZE_COEF_SCALE);
    int lx = rint((src_x - x_low) * INTER_RESIZE_COEF_SCALE);
    int hy = INTER_RESIZE_COEF_SCALE - ly;
    int hx = INTER_RESIZE_COEF_SCALE - lx;

    if (x_low >= 0 && x_low < width && y_low >= 0 && y_low < height)
      load_yuv_pixel<format>(
        luma, chroma, x_low, y_low + ybatch_offset, round_down2(x_low), width, stride, r0[0], g0[0], b0[0]);
    else {
      r0[0] = fillcolor.color[0];
      g0[0] = fillcolor.color[1];
      b0[0] = fillcolor.color[2];
    }

    if (x_high >= 0 && x_high < width && y_low >= 0 && y_low < height)
      load_yuv_pixel<format>(
        luma, chroma, x_high, y_low + ybatch_offset, round_down2(x_high), width, stride, r0[1], g0[1], b0[1]);
    else {
      r0[1] = fillcolor.color[0];
      g0[1] = fillcolor.color[1];
      b0[1] = fillcolor.color[2];
    }

    if (x_low >= 0 && x_low < width && y_high >= 0 && y_high < height)
      load_yuv_pixel<format>(
        luma, chroma, x_low, y_high + ybatch_offset, round_down2(x_low), width, stride, r0[2], g0[2], b0[2]);
    else {
      r0[2] = fillcolor.color[0];
      g0[2] = fillcolor.color[1];
      b0[2] = fillcolor.color[2];
    }

    if (x_high >= 0 && x_high < width && y_high >= 0 && y_high < height)
      load_yuv_pixel<format>(
        luma, chroma, x_high, y_high + ybatch_offset, round_down2(x_high), width, stride, r0[3], g0[3], b0[3]);
    else {
      r0[3] = fillcolor.color[0];
      g0[3] = fillcolor.color[1];
      b0[3] = fillcolor.color[2];
    }

    r = (((hy * ((hx * r0[0] + lx * r0[1]) >> 4)) >> 16) + ((ly * ((hx * r0[2] + lx * r0[3]) >> 4)) >> 16) + 2) >> 2;
    g = (((hy * ((hx * g0[0] + lx * g0[1]) >> 4)) >> 16) + ((ly * ((hx * g0[2] + lx * g0[3]) >> 4)) >> 16) + 2) >> 2;
    b = (((hy * ((hx * b0[0] + lx * b0[1]) >> 4)) >> 16) + ((ly * ((hx * b0[2] + lx * b0[3]) >> 4)) >> 16) + 2) >> 2;
  }
};

// BL sample pixel implmentation
template <typename DType, YUVFormat format>
struct SamplePixel<DType, format, Interpolation::Nearest, true> {
  static __device__ void __forceinline__ call(
    const void* luma,
    const void* chroma,
    int x,
    int y,
    float sx,
    float sy,
    int output_xoffset,
    int output_yoffset,
    int ybatch_offset,
    int width,
    int stride,
    int height,
    uint8_t& r,
    uint8_t& g,
    uint8_t& b,
    FillColor fillcolor) {
    // In some cases, the floating point precision will lead to miscalculation of the value,
    // making the result not exactly match with opencv,
    // so here you need to add eps as precision compensation
    //
    // A special case is when the input is 3840 and the output is 446, x = 223:
    // const int src_x_double = 223.0  * (3840.0  / 446.0);            // -> 1920
    // const int src_x_float  = 223.0f * (3840.0f / 446.0f);           // -> 1919
    // const int src_x_float  = 223.0f * (3840.0f / 446.0f) + 1e-5;    // -> 1920
    //
    // !!! If you want to use the double for sx/sy, you'll get a 2x speed drop
    //
    const float eps = 1e-5;
    int ix = x * sx + eps;
    int iy = y * sy + eps + ybatch_offset;
    load_yuv_pixel<format>(luma, chroma, ix, iy, round_down2(ix), width, stride, r, g, b);
  }
};

template <typename DType, YUVFormat format>
struct SamplePixel<DType, format, Interpolation::Bilinear, true> {
  static __device__ void __forceinline__ call(
    const void* luma,
    const void* chroma,
    int x,
    int y,
    float sx,
    float sy,
    int output_xoffset,
    int output_yoffset,
    int ybatch_offset,
    int width,
    int stride,
    int height,
    uint8_t& r,
    uint8_t& g,
    uint8_t& b,
    FillColor fillcolor) {
    uint8_t r0[4], g0[4], b0[4];
    float src_x = (x + 0.5f) * sx - 0.5f;
    float src_y = (y + 0.5f) * sy - 0.5f;
    int y_low = floorf(src_y);
    int x_low = floorf(src_x);
    int y_high = limit(y_low + 1, 0, height - 1);
    int x_high = limit(x_low + 1, 0, width - 1);
    y_low = limit(y_low, 0, height - 1);
    x_low = limit(x_low, 0, width - 1);

    int ly = rint((src_y - y_low) * INTER_RESIZE_COEF_SCALE);
    int lx = rint((src_x - x_low) * INTER_RESIZE_COEF_SCALE);
    int hy = INTER_RESIZE_COEF_SCALE - ly;
    int hx = INTER_RESIZE_COEF_SCALE - lx;

    load_yuv_pixel<format>(
      luma, chroma, x_low, y_low + ybatch_offset, round_down2(x_low), width, stride, r0[0], g0[0], b0[0]);
    load_yuv_pixel<format>(
      luma, chroma, x_high, y_low + ybatch_offset, round_down2(x_high), width, stride, r0[1], g0[1], b0[1]);
    load_yuv_pixel<format>(
      luma, chroma, x_low, y_high + ybatch_offset, round_down2(x_low), width, stride, r0[2], g0[2], b0[2]);
    load_yuv_pixel<format>(
      luma, chroma, x_high, y_high + ybatch_offset, round_down2(x_high), width, stride, r0[3], g0[3], b0[3]);

    r = (((hy * ((hx * r0[0] + lx * r0[1]) >> 4)) >> 16) + ((ly * ((hx * r0[2] + lx * r0[3]) >> 4)) >> 16) + 2) >> 2;
    g = (((hy * ((hx * g0[0] + lx * g0[1]) >> 4)) >> 16) + ((ly * ((hx * g0[2] + lx * g0[3]) >> 4)) >> 16) + 2) >> 2;
    b = (((hy * ((hx * b0[0] + lx * b0[1]) >> 4)) >> 16) + ((ly * ((hx * b0[2] + lx * b0[3]) >> 4)) >> 16) + 2) >> 2;
  }
};

template <YUVFormat yuv_format, typename OutDType, PixelLayout layout, Interpolation interp, bool fully_coverage>
static __global__ void convert_yuv_to_rgb_kernel_4x(
  const void* luma[],
  const void* chroma[],
  OutDType* pdst,
  float sx,
  float sy,
  int output_xoffset,
  int output_yoffset,
  FillColor fillcolor,
  int src_height,
  int src_width,
  int src_stride,
  float mean0,
  float mean1,
  float mean2,
  float scale0,
  float scale1,
  float scale2,
  int dst_width,
  int dst_stride,
  int dst_height,
  int nbatch) {
  int x = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dst_width - 3 || y >= dst_height)
    return;

  OutDType r[4], g[4], b[4];
  uint8_t r0, g0, b0;
  for (int ib = blockIdx.z; ib < nbatch; ib += gridDim.z) {
    int ybatch_offset = 0; // ib * src_height;
    for (int ip = 0; ip < 4; ++ip) {
      SamplePixel<OutDType, yuv_format, interp, fully_coverage>::call(
        luma[ib],
        chroma[ib],
        x + ip,
        y,
        sx,
        sy,
        output_xoffset,
        output_yoffset,
        ybatch_offset,
        src_width,
        src_stride,
        src_height,
        r0,
        g0,
        b0,
        fillcolor);
      scale_rgb(r0, g0, b0, r[ip], g[ip], b[ip], mean0, mean1, mean2, scale0, scale1, scale2);
    }

    DataLayoutInvoker<OutDType, Parallel::FourPixel, layout>::call(
      pdst, r, g, b, ib, x, y, dst_width, dst_stride, dst_height);
  }
}

template <YUVFormat yuv_format, typename OutDType, PixelLayout layout, Interpolation interp, bool fully_coverage>
static __global__ void convert_yuv_to_rgb_kernel_1x(
  const void* luma[],
  const void* chroma[],
  OutDType* pdst,
  float sx,
  float sy,
  int output_xoffset,
  int output_yoffset,
  FillColor fillcolor,
  int src_height,
  int src_width,
  int src_stride,
  float mean0,
  float mean1,
  float mean2,
  float scale0,
  float scale1,
  float scale2,
  int dst_width,
  int dst_stride,
  int dst_height,
  int nbatch) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dst_width || y >= dst_height)
    return;

  OutDType r, g, b;
  uint8_t r0, g0, b0;
  for (int ib = blockIdx.z; ib < nbatch; ib += gridDim.z) {
    int ybatch_offset = 0; // ib * src_height;
    SamplePixel<OutDType, yuv_format, interp, fully_coverage>::call(
      luma[ib],
      chroma[ib],
      x,
      y,
      sx,
      sy,
      output_xoffset,
      output_yoffset,
      ybatch_offset,
      src_width,
      src_stride,
      src_height,
      r0,
      g0,
      b0,
      fillcolor);

    scale_rgb(r0, g0, b0, r, g, b, mean0, mean1, mean2, scale0, scale1, scale2);
    DataLayoutInvoker<OutDType, Parallel::SinglePixel, layout>::call(
      pdst, r, g, b, ib, x, y, dst_width, dst_stride, dst_height);
  }
}

template <YUVFormat yuv_format, DataType out_dtype, PixelLayout layout, Interpolation interp, bool fully_coverage>
void batched_convert_yuv_to_rgb_impl(
  const void* luma[],
  const void* chroma[],
  int input_width,
  int input_stride,
  int input_height,
  int input_batch,
  int scaled_width,
  int scaled_height,
  int output_xoffset,
  int output_yoffset,
  FillColor fillcolor,
  void* out_ptr,
  int out_width,
  int out_stride,
  int out_height,
  float mean0,
  float mean1,
  float mean2,
  float scale0,
  float scale1,
  float scale2,
  cudaStream_t stream) {
  float sx = input_width / (float)scaled_width;
  float sy = input_height / (float)scaled_height;

  using OutDType = typename AsPODType<out_dtype>::type;

  if (
    layout == PixelLayout::NHWC_BGR || // Better performance
    layout == PixelLayout::NHWC_RGB || layout == PixelLayout::NCHW16_RGB || layout == PixelLayout::NCHW16_BGR ||
    layout == PixelLayout::NCHW32_RGB || layout == PixelLayout::NCHW32_BGR ||
    out_stride % 4 != 0 // Avoid misaligned addresses when writing
  ) {
    int grid_z = input_batch >= 32 ? 32 : input_batch;
    dim3 dim_block(32, 32);
    dim3 dim_grid((out_width + dim_block.x - 1) / dim_block.x, (out_height + dim_block.y - 1) / dim_block.y, grid_z);
    convert_yuv_to_rgb_kernel_1x<yuv_format, OutDType, layout, interp, fully_coverage>
      <<<dim_grid, dim_block, 0, stream>>>(
        luma,
        chroma,
        (OutDType*)out_ptr,
        sx,
        sy,
        output_xoffset,
        output_yoffset,
        fillcolor,
        input_height,
        input_width,
        input_stride,
        mean0,
        mean1,
        mean2,
        scale0,
        scale1,
        scale2,
        out_width,
        out_stride,
        out_height,
        input_batch);
  } else {
    int grid_z = input_batch >= 32 ? 32 : input_batch;
    dim3 dim_block(16, 32);
    dim3 dim_grid(
      ((out_width + 3) / 4 + dim_block.x - 1) / dim_block.x, (out_height + dim_block.y - 1) / dim_block.y, grid_z);
    convert_yuv_to_rgb_kernel_4x<yuv_format, OutDType, layout, interp, fully_coverage>
      <<<dim_grid, dim_block, 0, stream>>>(
        luma,
        chroma,
        (OutDType*)out_ptr,
        sx,
        sy,
        output_xoffset,
        output_yoffset,
        fillcolor,
        input_height,
        input_width,
        input_stride,
        mean0,
        mean1,
        mean2,
        scale0,
        scale1,
        scale2,
        out_width,
        out_stride,
        out_height,
        input_batch);
  }
  checkRuntime(cudaPeekAtLastError());
}

typedef void (*batched_convert_yuv_to_rgb_impl_function)(
  const void* luma[],
  const void* chroma[],
  int input_width,
  int input_stride,
  int input_height,
  int input_batch,
  int scaled_width,
  int scaled_height,
  int output_xoffset,
  int output_yoffset,
  FillColor fillcolor,
  void* out_ptr,
  int out_width,
  int out_stride,
  int out_height,
  float mean0,
  float mean1,
  float mean2,
  float scale0,
  float scale1,
  float scale2,
  cudaStream_t stream);

// If you want to modify this part of the code,
// please note that the order of the enumerated types must match the integer values of this type
// (note: that the order starts from 1)

#define DefineYUVFormat(...)                                                  \
  batched_convert_yuv_to_rgb_impl<YUVFormat::NV12BlockLinear, __VA_ARGS__>,   \
    batched_convert_yuv_to_rgb_impl<YUVFormat::NV12PitchLinear, __VA_ARGS__>, \
    batched_convert_yuv_to_rgb_impl<YUVFormat::YUV422Packed_YUYV, __VA_ARGS__>,

#define DefineDType(...)                                                                        \
  DefineYUVFormat(DataType::Uint8, __VA_ARGS__) DefineYUVFormat(DataType::Float32, __VA_ARGS__) \
    DefineYUVFormat(DataType::Float16, __VA_ARGS__) DefineYUVFormat(DataType::Int8, __VA_ARGS__)

#define DefineLayout(...)                                                                                   \
  DefineDType(PixelLayout::NCHW_RGB, __VA_ARGS__) DefineDType(PixelLayout::NCHW_BGR, __VA_ARGS__)           \
    DefineDType(PixelLayout::NHWC_RGB, __VA_ARGS__) DefineDType(PixelLayout::NHWC_BGR, __VA_ARGS__)         \
      DefineDType(PixelLayout::NCHW16_RGB, __VA_ARGS__) DefineDType(PixelLayout::NCHW16_BGR, __VA_ARGS__)   \
        DefineDType(PixelLayout::NCHW32_RGB, __VA_ARGS__) DefineDType(PixelLayout::NCHW32_BGR, __VA_ARGS__) \
          DefineDType(PixelLayout::NCHW4_RGB, __VA_ARGS__) DefineDType(PixelLayout::NCHW4_BGR, __VA_ARGS__)

#define DefineInterp(...) \
  DefineLayout(Interpolation::Nearest, __VA_ARGS__) DefineLayout(Interpolation::Bilinear, __VA_ARGS__)

#define DefinefullyCoverage DefineInterp(false) DefineInterp(true)

#define DefineAllFunction DefinefullyCoverage

template <typename T>
struct EnumCount {};
template <>
struct EnumCount<YUVFormat> {
  static const unsigned int value = 3;
};
template <>
struct EnumCount<DataType> {
  static const unsigned int value = 4;
};
template <>
struct EnumCount<PixelLayout> {
  static const unsigned int value = 10;
};
template <>
struct EnumCount<Interpolation> {
  static const unsigned int value = 2;
};

static const batched_convert_yuv_to_rgb_impl_function func_list[] = {DefineAllFunction nullptr};

void batched_convert_yuv_to_rgb_v2(
  const void* luma[],
  const void* chroma[],
  int input_width,
  int input_stride,
  int input_height,
  int input_batch,
  YUVFormat yuv_format,
  int scaled_width,
  int scaled_height,
  int output_xoffset,
  int output_yoffset,
  FillColor fillcolor,
  void* out_ptr,
  int out_width,
  int out_stride,
  int out_height,
  DataType out_dtype,
  PixelLayout out_layout,
  Interpolation interp,
  float mean0,
  float mean1,
  float mean2,
  float scale0,
  float scale1,
  float scale2,
  void* stream) {
  int iformat = (int)yuv_format - 1;
  int odtype = (int)out_dtype - 1;
  int olayout = (int)out_layout - 1;
  int iinterp = (int)interp - 1;
  int ifully_coverage = 0;
  //
  if (scaled_width == out_width && scaled_height == out_height && output_xoffset == 0 && output_yoffset == 0)
    ifully_coverage = 1;

  int index =
    (((ifully_coverage * EnumCount<Interpolation>::value + iinterp) * EnumCount<PixelLayout>::value + olayout) *
       EnumCount<DataType>::value +
     odtype) *
      EnumCount<YUVFormat>::value +
    iformat;

  // clang-format off
  if (
    iformat < 0 || iformat >= EnumCount<YUVFormat>::value || 
    odtype  < 0 || odtype >= EnumCount<DataType>::value ||
    olayout < 0 || olayout >= EnumCount<PixelLayout>::value || 
    iinterp < 0 || iinterp >= EnumCount<Interpolation>::value ||
    index < 0   || index >= sizeof(func_list) / sizeof(func_list[0]) - 1) {
    fprintf(stderr, "Unsupported configure %d.\n", index);
    return;
  }
  // clang-format on

  batched_convert_yuv_to_rgb_impl_function func = func_list[index];
  func(
    luma,
    chroma,
    input_width,
    input_stride,
    input_height,
    input_batch,
    scaled_width,
    scaled_height,
    output_xoffset,
    output_yoffset,
    fillcolor,
    out_ptr,
    out_width,
    out_stride,
    out_height,
    mean0,
    mean1,
    mean2,
    scale0,
    scale1,
    scale2,
    (cudaStream_t)stream);
}

#if __DEFINE_INDEPEND_TEST__
#include "bin.h"
#include "opencv2/opencv.hpp"

cv::Mat BGR2YUV_NV12(const cv::Mat& src) {
  auto src_h = src.rows;
  auto src_w = src.cols;
  cv::Mat dst(src_h * 1.5, src_w, CV_8UC1);
  cv::cvtColor(src, dst, cv::COLOR_BGR2YUV_I420); // I420: YYYY...UU...VV...

  auto n_y = src_h * src_w;
  auto n_uv = n_y / 2;
  auto n_u = n_y / 4;
  std::vector<uint8_t> uv(n_uv);
  std::copy(dst.data + n_y, dst.data + n_y + n_uv, uv.data());
  for (auto i = 0; i < n_u; i++) {
    dst.data[n_y + 2 * i] = uv[i]; // U
    dst.data[n_y + 2 * i + 1] = uv[n_u + i]; // V
  }
  return dst;
}

int t_nv12() {
  printf("@%d\n", __LINE__);

  auto img_bgr = cv::imread("front_mid.jpg"); //  3840*2160  960*540  PAD_H = 544

  {
    int w = img_bgr.cols;
    int h = img_bgr.rows;
    int c = img_bgr.channels();
    printf("------whc %d  %d  %d\n", w, h, c);
  }

  ////// Convert from BGR to NV12
  cv::Mat nv12 = BGR2YUV_NV12(img_bgr);
  cv::imwrite("nv12.png", nv12);

  const int h = 2160, w = 3840, new_h = 540, new_w = 960, batch = 3, pad_h = 544;
  const int w_pitch = 3840; // 2048;
  using T = uint8_t;
  using OT = uint8_t; // float;

  void* d_src;
  int nbytes = h * 1.5 * w;
  cudaMalloc(&d_src, nbytes * batch);
  for (int k = 0; k < batch; ++k) {
    if (k == 1)
      continue;
    cudaMemcpy((T*)d_src + k * nbytes, nv12.data, nbytes, ::cudaMemcpyHostToDevice);
  }
  // clang-format off
  void* inputs_luma[]{      d_src,         (T*)d_src + nbytes,         (T*)d_src + 2 * nbytes};
  void* inputs_chroma[]{(T*)d_src + h * w, (T*)d_src + nbytes + h * w, (T*)d_src + 2 * nbytes + h * w};
  // clang-format on

  void** luma{nullptr};
  void** chroma{nullptr};

  cudaMalloc(&luma, batch * sizeof(void*));
  cudaMalloc(&chroma, batch * sizeof(void*));

  cudaMemcpy(luma, inputs_luma, batch * sizeof(void*), cudaMemcpyHostToDevice);
  cudaMemcpy(chroma, inputs_chroma, batch * sizeof(void*), cudaMemcpyHostToDevice);

  void* d_dst;
  cudaMalloc(&d_dst, pad_h * 3 * new_w * batch * sizeof(OT));

  int input_width = w, input_height = h, input_stride = w_pitch, input_batch = batch;
  int scaled_width = new_w, scaled_height = new_h, output_xoffset = 0, output_yoffset = 0;
  FillColor fillcolor{0, 0, 0};
  void* out_ptr = d_dst;
  int out_width = new_w, out_height = pad_h, out_stride = new_w * 3;
  // float mean = 128.0f, scale = 1.0f / 128.0f;
  float mean = 0.f, scale = 1.0f;
  printf("start @%d\n", __LINE__);

  batched_convert_yuv_to_rgb_v2(
    (const void**)luma,
    (const void**)chroma,
    input_width,
    input_stride,
    input_height,
    input_batch,
    YUVFormat::NV12PitchLinear,
    scaled_width,
    scaled_height,
    output_xoffset,
    output_yoffset,
    fillcolor,
    out_ptr,
    out_width,
    out_stride,
    out_height,
    // DataType::Float32,
    // PixelLayout::NCHW_BGR,

    DataType::Int8,
    PixelLayout::NHWC_BGR,
    Interpolation::Bilinear,
    mean,
    mean,
    mean,
    scale,
    scale,
    scale,
    0);
  cudaDeviceSynchronize();
  printf("done @%d\n", __LINE__);

  int neles = out_height * 3 * out_width * batch;
  std::vector<OT> h_out(neles);
  cudaMemcpy(h_out.data(), out_ptr, neles * sizeof(OT), ::cudaMemcpyDeviceToHost);

  cv::Mat img_bgr_ref(out_height, out_width, CV_8UC3, cv::Scalar::all(0));
  cv::Mat img_bgr_ref2(out_height, out_width, CV_8UC3, cv::Scalar::all(0));

  memcpy(img_bgr_ref.data, h_out.data(), out_height * 3 * out_width);
  memcpy(img_bgr_ref2.data, h_out.data() + out_height * 3 * out_width * 2, out_height * 3 * out_width);

  cv::imwrite("test_cu_v2a.png", img_bgr_ref);
  cv::imwrite("test_cu_v2b.png", img_bgr_ref2);

  // for (int i = 0; i < batch; i++) {
  //   for (int j = 0; j < out_height * 3 * out_width; ++j)
  //     printf("%f ", h_out[i * out_height * 3 * out_width + j]);
  //   printf("\n");
  // }

  cudaFree(luma);
  cudaFree(chroma);
  cudaFree(d_dst);
  cudaFree(d_src);

  return 0;
}

int t_yuv2() {
  auto img_bgr = cv::imread("1920x1080.jpg"); //  960*540  PAD_H = 544

  {
    int w = img_bgr.cols;
    int h = img_bgr.rows;
    int c = img_bgr.channels();
    printf("------whc %d  %d  %d\n", w, h, c);
  }

  const int h = 1080, w = 1920, new_h = 540, new_w = 960, batch = 3, pad_h = 544;
  const int w_pitch = 1920; // 2048;

  ////// Convert from BGR to YUYV
  cv::Mat img_yuv422(h, w, CV_8UC2, cv::Scalar::all(0));
  cv::cvtColor(img_bgr, img_yuv422, cv::COLOR_BGR2YUV_YUYV);

  ////// YUYV TO BGR
  // cv::Mat img_bgr_gt;
  // cv::cvtColor(img_yuv422, img_bgr_gt, cv::COLOR_YUV2BGR_YUYV);
  // cv::imwrite("img_bgr_gt.png", img_bgr_gt);

  ////////////////////////////////// Byself
  using T = uint8_t;
  using OT = uint8_t;

  void* d_src;
  cudaMalloc(&d_src, h * 2 * w * batch);
  int nbytes = h * 2 * w;
  for (int k = 0; k < batch; ++k) {
    cudaMemcpy((T*)d_src + k * nbytes, img_yuv422.data, nbytes, ::cudaMemcpyHostToDevice);
  }
  void** luma{nullptr};
  void* inputs[]{d_src, (T*)d_src + nbytes, (T*)d_src + 2 * nbytes};
  cudaMalloc(&luma, batch * sizeof(void*));
  cudaMemcpy(luma, inputs, batch * sizeof(void*), cudaMemcpyHostToDevice);

  void* d_dst;
  cudaMalloc(&d_dst, pad_h * 3 * new_w * batch * sizeof(OT));

  const void** chroma = nullptr;
  int input_width = w, input_height = h, input_stride = w_pitch * 2, input_batch = batch;
  int scaled_width = new_w, scaled_height = new_h, output_xoffset = 0, output_yoffset = 0;
  FillColor fillcolor{0, 0, 0};
  void* out_ptr = d_dst;
  int out_width = new_w, out_height = pad_h, out_stride = new_w * 3;
  // float mean = 128.0f, scale = 1.0f / 128.0f;
  float mean = 0.f, scale = 1.f;
  batched_convert_yuv_to_rgb_v2(
    (const void**)luma,
    chroma,
    input_width,
    input_stride,
    input_height,
    input_batch,
    YUVFormat::YUV422Packed_YUYV,
    scaled_width,
    scaled_height,
    output_xoffset,
    output_yoffset,
    fillcolor,
    out_ptr,
    out_width,
    out_stride,
    out_height,
    // DataType::Float32,
    // PixelLayout::NCHW_BGR,
    DataType::Int8,
    PixelLayout::NHWC_BGR,
    Interpolation::Bilinear,
    mean,
    mean,
    mean,
    scale,
    scale,
    scale,
    0);
  cudaDeviceSynchronize();

  int neles = out_height * 3 * out_width * batch;
  std::vector<OT> h_out(neles);
  cudaMemcpy(h_out.data(), out_ptr, neles * sizeof(OT), ::cudaMemcpyDeviceToHost);

  cv::Mat img_bgr_ref(out_height, out_width, CV_8UC3, cv::Scalar::all(0));
  cv::Mat img_bgr_ref2(out_height, out_width, CV_8UC3, cv::Scalar::all(0));

  memcpy(img_bgr_ref.data, h_out.data(), out_height * 3 * out_width);
  memcpy(img_bgr_ref2.data, h_out.data() + out_height * 3 * out_width * 2, out_height * 3 * out_width);

  cv::imwrite("test_cu_v2a.png", img_bgr_ref);
  cv::imwrite("test_cu_v2c.png", img_bgr_ref2);

  // for (int i = 0; i < batch; i++) {
  //   for (int j = 0; j < out_height * 3 * out_width; ++j)
  //     printf("%f ", h_out[i * out_height * 3 * out_width + j]);
  //   printf("\n");
  // }
  cudaFree(luma);
  cudaFree(d_dst);
  cudaFree(d_src);

  return 0;
}

int t_nv12_bin() {
  printf("@%d\n", __LINE__);

  const int h = 2160, w = 3840, new_h = 540, new_w = 960, batch = 2, pad_h = 544;
  const int w_pad = 4096;
  const int h_pad = 2176;
  using T = uint8_t;
  using OT = uint8_t; // float;

  int ele_num = h_pad * w_pad * 1.5;

  //  13369344
  std::vector<std::vector<OT>> nv12(batch);
  void* d_src[batch];

  int i = 0;
  for (auto& it : nv12) {
    it.resize(ele_num);
    utils::read_bin<T>("img2d_cuda_" + std::to_string(2 + i) + ".yuv", (T*)it.data(), ele_num);

    cudaMalloc(&d_src[i], ele_num);
    cudaMemcpy(d_src[i++], it.data(), ele_num, ::cudaMemcpyHostToDevice);
  }

  // clang-format off
  void* inputs_luma[]{      d_src[0],                 (T*)d_src[1],               };
  void* inputs_chroma[]{(T*)d_src[0] + h_pad * w_pad, (T*)d_src[1] + h_pad * w_pad};
  // clang-format on

  void** luma{nullptr};
  void** chroma{nullptr};

  cudaMalloc(&luma, batch * sizeof(void*));
  cudaMalloc(&chroma, batch * sizeof(void*));

  cudaMemcpy(luma, inputs_luma, batch * sizeof(void*), cudaMemcpyHostToDevice);
  cudaMemcpy(chroma, inputs_chroma, batch * sizeof(void*), cudaMemcpyHostToDevice);

  void* d_dst;
  cudaMalloc(&d_dst, pad_h * 3 * new_w * batch * sizeof(OT));

  int input_width = w, input_height = h, input_stride = w_pad, /* ! */ input_batch = batch;
  int scaled_width = new_w, scaled_height = new_h, output_xoffset = 0, output_yoffset = 0;
  FillColor fillcolor{0, 0, 0};
  void* out_ptr = d_dst;
  int out_width = new_w, out_height = pad_h, out_stride = new_w * 3;
  // float mean = 128.0f, scale = 1.0f / 128.0f;
  float mean = 0.f, scale = 1.0f;
  printf("start @%d\n", __LINE__);

  batched_convert_yuv_to_rgb_v2(
    (const void**)luma,
    (const void**)chroma,
    input_width,
    input_stride,
    input_height,
    input_batch,
    YUVFormat::NV12PitchLinear,
    scaled_width,
    scaled_height,
    output_xoffset,
    output_yoffset,
    fillcolor,
    out_ptr,
    out_width,
    out_stride,
    out_height,
    // DataType::Float32,
    // PixelLayout::NCHW_BGR,

    DataType::Int8,
    PixelLayout::NHWC_BGR,
    Interpolation::Bilinear,
    mean,
    mean,
    mean,
    scale,
    scale,
    scale,
    0);
  cudaDeviceSynchronize();
  printf("done @%d\n", __LINE__);

  int neles = out_height * 3 * out_width * batch;
  std::vector<OT> h_out(neles);
  cudaMemcpy(h_out.data(), out_ptr, neles * sizeof(OT), ::cudaMemcpyDeviceToHost);

  for (int i = 0; i < batch; ++i) {
    cv::Mat it(out_height, out_width, CV_8UC3, cv::Scalar::all(0));
    memcpy(it.data, h_out.data() + out_height * 3 * out_width * i, out_height * 3 * out_width);
    cv::imwrite("test_cu_nv12-" + std::to_string(i) + ".png", it);
  }

  // for (int i = 0; i < batch; i++) {
  //   for (int j = 0; j < out_height * 3 * out_width; ++j)
  //     printf("%f ", h_out[i * out_height * 3 * out_width + j]);
  //   printf("\n");
  // }

  cudaFree(luma);
  cudaFree(chroma);
  cudaFree(d_dst);
  for (auto& it : d_src)
    cudaFree(it);
  return 0;
}

int t_yuv2_bin() {
  printf("@%d\n", __LINE__);

  constexpr int h = 1080, w = 1920, new_h = 540, new_w = 960, batch = 5, pad_h = 544;
  const int w_pad = 2048, h_pad = 1088;
  using T = uint8_t;
  using OT = uint8_t; // float;

  int ele_num = h_pad * w_pad * 2;

  std::vector<std::vector<OT>> img_yuv422(batch);
  void* d_src[batch];

  int i = 0;
  for (auto& it : img_yuv422) {
    it.resize(ele_num);
    if (i == 4) {
      utils::read_bin<T>("img2d_cuda_" + std::to_string(0) + ".yuv", (T*)it.data(), ele_num);
    } else {
      utils::read_bin<T>("img2d_cuda_" + std::to_string(4 + i) + ".yuv", (T*)it.data(), ele_num);
    }
    cudaMalloc(&d_src[i], ele_num);
    cudaMemcpy(d_src[i++], it.data(), ele_num, ::cudaMemcpyHostToDevice);
  }

  void** luma{nullptr};
  // clang-format off
  void* inputs[]{(T*)d_src[0] + w_pad * 2, 
                 (T*)d_src[1] + w_pad * 2, 
                 (T*)d_src[2] + w_pad * 2, 
                 (T*)d_src[3] + w_pad * 2,
                 (T*)d_src[4] + w_pad * 2};
  // clang-format on

  cudaMalloc(&luma, batch * sizeof(void*));
  cudaMemcpy(luma, inputs, batch * sizeof(void*), cudaMemcpyHostToDevice);

  void* d_dst;
  cudaMalloc(&d_dst, pad_h * 3 * new_w * batch * sizeof(OT));

  const void** chroma = nullptr;
  int input_width = w, input_height = h, input_stride = w_pad * 2, input_batch = batch;
  int scaled_width = new_w, scaled_height = new_h, output_xoffset = 0, output_yoffset = 0;
  FillColor fillcolor{0, 0, 0};
  void* out_ptr = d_dst;
  int out_width = new_w, out_height = pad_h, out_stride = new_w * 3;
  // float mean = 128.0f, scale = 1.0f / 128.0f;
  float mean = 0.f, scale = 1.f;
  batched_convert_yuv_to_rgb_v2(
    (const void**)luma,
    chroma,
    input_width,
    input_stride,
    input_height,
    input_batch,
    YUVFormat::YUV422Packed_YUYV,
    scaled_width,
    scaled_height,
    output_xoffset,
    output_yoffset,
    fillcolor,
    out_ptr,
    out_width,
    out_stride,
    out_height,
    // DataType::Float32,
    // PixelLayout::NCHW_BGR,
    DataType::Int8,
    PixelLayout::NHWC_BGR,
    Interpolation::Bilinear,
    mean,
    mean,
    mean,
    scale,
    scale,
    scale,
    0);
  cudaDeviceSynchronize();

  int neles = out_height * 3 * out_width * batch;
  std::vector<OT> h_out(neles);
  cudaMemcpy(h_out.data(), out_ptr, neles * sizeof(OT), ::cudaMemcpyDeviceToHost);

  for (int i = 0; i < batch; ++i) {
    cv::Mat it(out_height, out_width, CV_8UC3, cv::Scalar::all(0));
    memcpy(it.data, h_out.data() + out_height * 3 * out_width * i, out_height * 3 * out_width);
    cv::imwrite("test_cu_yuyv-" + std::to_string(i) + ".png", it);
  }

  cudaFree(luma);
  cudaFree(d_dst);
  for (auto& it : d_src)
    cudaFree(it);

  return 0;
}

int main() {
  t_nv12_bin();
  t_yuv2_bin();
  return 0;
}
#endif

/*

nvcc -std=c++17 -O2 ./yuv_to_rgb_kernel_v2.cu  \
-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs \
-I/usr/local/include/opencv4 \
-L/usr/local/lib  -o v2


*/