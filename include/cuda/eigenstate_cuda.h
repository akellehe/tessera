// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
#pragma once

namespace tessera {
namespace cuda {

/// Single-precision cuBLAS (SGEMM) accelerator for the per-edge analytic r_U-gradient loop
/// of `EigenstateSynthesis::residualForPeriodsGradient`, which dominates level-2 relaxation
/// (about 82 s per call at n1 = 2724 edges, against a 16-core double-precision host loop).
///
/// The loop-invariant matrices — the non-null and null eigenvector blocks of the metric
/// Laplacian M = L1, M itself, and the unit carried representative p — are uploaded once in
/// single precision. Each edge then streams only its low-rank dM/dl² factors `fa` and `fb`,
/// and the heavy GEMMs run on the device. The dense eigensolve and the small per-edge algebra
/// stay on the host in double precision, so the single-precision GEMMs are the only
/// approximation: about 1e-5 relative error against the double-precision result at level 2,
/// with an unchanged descent direction. The double-precision host path stays the default and
/// the correctness oracle.
///
/// All matrices are column-major, matching both Eigen's default storage and cuBLAS, so the
/// caller passes `Eigen::MatrixXf::data()` directly. Complex N-vectors are packed as an N×2
/// column-major block `[Re | Im]`.
///
/// Host/device contract: the object owns its cuBLAS handle and every device allocation, and
/// frees them in the destructor; the caller owns all host buffers and they need not be
/// pinned. There are no custom kernels, so no launch configuration — the work is cuBLAS GEMM
/// calls on the default stream. Each call is synchronous: it uploads its inputs, runs, and
/// copies the results back before returning. The per-edge device scratch is shared across
/// calls, so one instance must not be used from two threads at once. The constructor and both
/// stage calls throw `std::runtime_error` on a CUDA or cuBLAS failure.
class RuGradientGpu {
 public:
  /// Upload the loop-invariant matrices (column-major, single precision):
  ///   `Unn`  (N×nnd) — non-null eigenvectors of M;
  ///   `UnnS` (N×nnd) — `Unn` with column r scaled by invlam[r] = −1/λ_nn[r]
  ///                    (so dUn = UnnS·core hoists the diagonal, by associativity);
  ///   `Un`   (N×nd)  — null/harmonic eigenvectors of M;
  ///   `M`    (N×N)   — the real metric Laplacian L1;
  ///   `p2`   (N×2)   — `[Re p | Im p]`, the unit carried representative.
  /// `rmax` is the largest per-edge rank (columns of `fa`/`fb`); it sizes the device scratch,
  /// so every later call must pass `r <= rmax`.
  RuGradientGpu(int N, int nnd, int nd, int rmax,
                const float* Unn, const float* UnnS, const float* Un,
                const float* M, const float* p2);
  ~RuGradientGpu();
  RuGradientGpu(const RuGradientGpu&) = delete;
  RuGradientGpu& operator=(const RuGradientGpu&) = delete;

  /// Per-edge stage 1, the dominant GEMMs. Given `fa` and `fb` (both N×r, column-major,
  /// single precision), compute on the device and copy back into the caller's buffers:
  ///   `dUn`  (N×nd)  = UnnS · ((Unnᵀ·fa)·(fbᵀ·Un));
  ///   `dMp2` (N×2)   = fa · (fbᵀ · p2)            — the complex dM·p, packed [Re|Im].
  void edgeStage1(const float* fa, const float* fb, int r,
                  float* dUn, float* dMp2);

  /// Per-edge stage 2: `Mdpsi2` (N×2) = M · `dpsi2` (N×2), the dense product against the
  /// post-leak perturbed cochain, both packed as [Re | Im].
  void edgeStage2(const float* dpsi2, float* Mdpsi2);

 private:
  int N_, nnd_, nd_, rmax_;
  void* handle_;  // cublasHandle_t, opaque so the header stays cuBLAS-free
  // Loop-invariant device constants.
  float *dUnn_, *dUnnS_, *dUn_, *dM_, *dP2_;
  // Per-edge device scratch (sized by N, nnd, nd, rmax in the ctor).
  float *dFa_, *dFb_, *dL_, *dR_, *dCore_, *dDUn_, *dT2_, *dDMp2_;
  float *dDpsi2_, *dMdpsi2_;
};

}  // namespace cuda
}  // namespace tessera
