// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
//
// Single-precision cuBLAS (SGEMM) accelerator for the per-edge r_U-gradient loop
// of EigenstateSynthesis::residualForPeriodsGradient.
//
// The loop-invariant matrices are uploaded once; each edge streams its low-rank
// dM/dl² factors fa, fb. The heavy GEMMs run in single precision on the device;
// the double-precision host path stays the correctness oracle. Column-major
// throughout, matching Eigen's default storage and cuBLAS.

#include "cuda/eigenstate_cuda.h"

#include <cstdio>
#include <stdexcept>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      char msg[256];                                                         \
      snprintf(msg, sizeof(msg), "CUDA error at %s:%d: %s", __FILE__,        \
               __LINE__, cudaGetErrorString(err));                          \
      throw std::runtime_error(msg);                                         \
    }                                                                        \
  } while (0)

#define CUBLAS_CHECK(call)                                                   \
  do {                                                                       \
    cublasStatus_t st = (call);                                              \
    if (st != CUBLAS_STATUS_SUCCESS) {                                       \
      char msg[256];                                                         \
      snprintf(msg, sizeof(msg), "cuBLAS error at %s:%d: status %d",         \
               __FILE__, __LINE__, static_cast<int>(st));                    \
      throw std::runtime_error(msg);                                         \
    }                                                                        \
  } while (0)

namespace tessera {
namespace cuda {

namespace {
// Allocate n floats on the device and, if `host` is non-null, upload n floats into them.
float* devAlloc(int n, const float* host = nullptr) {
  float* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, static_cast<size_t>(n) * sizeof(float)));
  if (host != nullptr)
    CUDA_CHECK(cudaMemcpy(d, host, static_cast<size_t>(n) * sizeof(float),
                          cudaMemcpyHostToDevice));
  return d;
}
}  // namespace

RuGradientGpu::RuGradientGpu(int N, int nnd, int nd, int rmax,
                             const float* Unn, const float* UnnS,
                             const float* Un, const float* M, const float* p2)
    : N_(N), nnd_(nnd), nd_(nd), rmax_(rmax) {
  cublasHandle_t h = nullptr;
  CUBLAS_CHECK(cublasCreate(&h));
  handle_ = h;

  // Loop-invariant constants.
  dUnn_ = devAlloc(N * nnd, Unn);
  dUnnS_ = devAlloc(N * nnd, UnnS);
  dUn_ = devAlloc(N * nd, Un);
  dM_ = devAlloc(N * N, M);
  dP2_ = devAlloc(N * 2, p2);

  // Per-edge scratch, sized by the largest rank rmax.
  dFa_ = devAlloc(N * rmax);
  dFb_ = devAlloc(N * rmax);
  dL_ = devAlloc(nnd * rmax);
  dR_ = devAlloc(rmax * nd);
  dCore_ = devAlloc(nnd * nd);
  dDUn_ = devAlloc(N * nd);
  dT2_ = devAlloc(rmax * 2);
  dDMp2_ = devAlloc(N * 2);
  dDpsi2_ = devAlloc(N * 2);
  dMdpsi2_ = devAlloc(N * 2);
}

RuGradientGpu::~RuGradientGpu() {
  cudaFree(dUnn_); cudaFree(dUnnS_); cudaFree(dUn_); cudaFree(dM_);
  cudaFree(dP2_);
  cudaFree(dFa_); cudaFree(dFb_); cudaFree(dL_); cudaFree(dR_);
  cudaFree(dCore_); cudaFree(dDUn_); cudaFree(dT2_); cudaFree(dDMp2_);
  cudaFree(dDpsi2_); cudaFree(dMdpsi2_);
  if (handle_ != nullptr)
    cublasDestroy(reinterpret_cast<cublasHandle_t>(handle_));
}

void RuGradientGpu::edgeStage1(const float* fa, const float* fb, int r,
                               float* dUn, float* dMp2) {
  cublasHandle_t h = reinterpret_cast<cublasHandle_t>(handle_);
  const float one = 1.0f, zero = 0.0f;
  const int N = N_, nnd = nnd_, nd = nd_;

  CUDA_CHECK(cudaMemcpy(dFa_, fa, static_cast<size_t>(N) * r * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dFb_, fb, static_cast<size_t>(N) * r * sizeof(float),
                        cudaMemcpyHostToDevice));

  // L = Unnᵀ·fa            (nnd×r)
  CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, nnd, r, N, &one, dUnn_,
                           N, dFa_, N, &zero, dL_, nnd));
  // R = fbᵀ·Un             (r×nd)
  CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, r, nd, N, &one, dFb_, N,
                           dUn_, N, &zero, dR_, r));
  // core = L·R             (nnd×nd)
  CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, nnd, nd, r, &one, dL_,
                           nnd, dR_, r, &zero, dCore_, nnd));
  // dUn = UnnS·core        (N×nd)   [UnnS folds the invlam diagonal]
  CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, nd, nnd, &one,
                           dUnnS_, N, dCore_, nnd, &zero, dDUn_, N));
  // T2 = fbᵀ·P2            (r×2)
  CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, r, 2, N, &one, dFb_, N,
                           dP2_, N, &zero, dT2_, r));
  // dMp2 = fa·T2           (N×2)
  CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, 2, r, &one, dFa_, N,
                           dT2_, r, &zero, dDMp2_, N));

  CUDA_CHECK(cudaMemcpy(dUn, dDUn_, static_cast<size_t>(N) * nd * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(dMp2, dDMp2_, static_cast<size_t>(N) * 2 * sizeof(float),
                        cudaMemcpyDeviceToHost));
}

void RuGradientGpu::edgeStage2(const float* dpsi2, float* Mdpsi2) {
  cublasHandle_t h = reinterpret_cast<cublasHandle_t>(handle_);
  const float one = 1.0f, zero = 0.0f;
  const int N = N_;
  CUDA_CHECK(cudaMemcpy(dDpsi2_, dpsi2, static_cast<size_t>(N) * 2 * sizeof(float),
                        cudaMemcpyHostToDevice));
  // Mdpsi2 = M·dpsi2       (N×2)
  CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, 2, N, &one, dM_, N,
                           dDpsi2_, N, &zero, dMdpsi2_, N));
  CUDA_CHECK(cudaMemcpy(Mdpsi2, dMdpsi2_, static_cast<size_t>(N) * 2 * sizeof(float),
                        cudaMemcpyDeviceToHost));
}

}  // namespace cuda
}  // namespace tessera
