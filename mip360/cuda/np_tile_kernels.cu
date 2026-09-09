// Tile-based MLP splatting kernel with per-pixel sorting.
// One block = one tile (TW x TH threads). Each thread (pixel) gathers
// intersections from a per-tile primitive list, sorts them by depth,
// then evaluates the MLP and alpha-blends in the correct order.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Block radix sort (CUB)
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>

// ---- helpers for stats (tilecenter only) ----
__device__ __forceinline__ void atomicMaxFloatPos(float* addr, float val) {
  // For non-negative floats, bitwise ordering matches numeric ordering.
  int* addr_i = reinterpret_cast<int*>(addr);
  int old = *addr_i;
  int assumed;
  const int v = __float_as_int(val);
  do {
    assumed = old;
    if (assumed >= v) break;
    old = atomicCAS(addr_i, assumed, v);
  } while (assumed != old);
}

#ifndef MLPSP_MAX_TILE_PRIMS
#define MLPSP_MAX_TILE_PRIMS 768
#endif

#ifndef MLPSP_MAX_H
#define MLPSP_MAX_H 32
#endif

// EN: Fast sigmoid approximation/helper used in MLP output activation.
__device__ __forceinline__ float fast_sigmoid(float x) {
  return 1.0f / (1.0f + __expf(-x));
}
// EN: Numerically stable softplus helper for density activation.
__device__ __forceinline__ float fast_softplus(float x) {
  if (x > 20.f) return x;
  if (x < -20.f) return __expf(x);
  return log1pf(__expf(x));
}

// EN: Computes 3D dot product with FMA for better precision/perf.
__device__ __forceinline__ float dot3(float ax, float ay, float az, float bx, float by, float bz) {
  return fmaf(ax, bx, fmaf(ay, by, az * bz));
}

__device__ __forceinline__ float clamp_m11(float x) {
  return fminf(1.f, fmaxf(-1.f, x));
}

// Local-coordinate tetrahedral gating used by semantic slot mixing.
// feat_slots=4 : f = sum_k g_k * slot_k
// feat_slots=5 : f = base + sum_k g_k * delta_k
__device__ __forceinline__ void semantic_tetra_gates(
    const float qlx, const float qly, const float qlz,
    const float lgx, const float lgy, const float lgz,
    const float norm_beta,
    const float logit_temp,
    float& g0, float& g1, float& g2, float& g3) {
  const float sx = __expf(lgx);
  const float sy = __expf(lgy);
  const float sz = __expf(lgz);
  const float ux = clamp_m11(qlx / (norm_beta * sx + 1e-6f));
  const float uy = clamp_m11(qly / (norm_beta * sy + 1e-6f));
  const float uz = clamp_m11(qlz / (norm_beta * sz + 1e-6f));

  const float l0 = logit_temp * ( ux + uy + uz);
  const float l1 = logit_temp * ( ux - uy - uz);
  const float l2 = logit_temp * (-ux + uy - uz);
  const float l3 = logit_temp * (-ux - uy + uz);
  const float lm = fmaxf(fmaxf(l0, l1), fmaxf(l2, l3));
  const float e0 = __expf(l0 - lm);
  const float e1 = __expf(l1 - lm);
  const float e2 = __expf(l2 - lm);
  const float e3 = __expf(l3 - lm);
  const float inv_sum = 1.f / fmaxf(e0 + e1 + e2 + e3, 1e-8f);
  g0 = e0 * inv_sum;
  g1 = e1 * inv_sum;
  g2 = e2 * inv_sum;
  g3 = e3 * inv_sum;
}

// EN: Copies float buffer with vectorized path when aligned.
__device__ __noinline__ void copy_float_strided_vec4(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int n,
    int tid,
    int stride) {
  const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);
  const uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
  const bool aligned16 = ((dst_addr & 0xF) == 0) && ((src_addr & 0xF) == 0);
  if (aligned16) {
    int i = tid * 4;
    const int step = stride * 4;
    for (; i + 3 < n; i += step) {
      *reinterpret_cast<float4*>(dst + i) = *reinterpret_cast<const float4*>(src + i);
    }
    for (; i < n; i += stride) dst[i] = src[i];
  } else {
    for (int i = tid; i < n; i += stride) dst[i] = src[i];
  }
}

// EN: Best-effort L2 prefetch for next primitive weights.
__device__ __noinline__ void prefetch_weights_l2(
    const float* __restrict__ W1,
    const float* __restrict__ b1,
    const float* __restrict__ W2,
    const float* __restrict__ b2,
    const float* __restrict__ W3,
    const float* __restrict__ b3,
    int p,
    int H,
    int IN) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  auto l2_prefetch = [](const float* ptr) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr));
  };
  if (p < 0) return;
  const float* W1p = W1 + static_cast<size_t>(p) * H * IN;
  const float* b1p = b1 + static_cast<size_t>(p) * H;
  const float* W2p = W2 + static_cast<size_t>(p) * H * H;
  const float* b2p = b2 + static_cast<size_t>(p) * H;
  // The output layer is [4, H] with a four-element bias per primitive.
  const float* W3p = W3 + static_cast<size_t>(p) * 4 * H;
  const float* b3p = b3 + static_cast<size_t>(p) * 4;

  // Prefetch a bounded window per tensor to avoid excess prefetch traffic.
  for (int i = 0; i < 256; i += 32) l2_prefetch(W1p + i);
  for (int i = 0; i < 64; i += 32) l2_prefetch(W2p + i);
  l2_prefetch(b1p);
  l2_prefetch(b2p);
  l2_prefetch(W3p);
  l2_prefetch(b3p);
#endif
}

template <typename T>
// EN: Converts scalar CUDA type to float (generic path).
__device__ __forceinline__ float scalar_to_float(T v) {
  return static_cast<float>(v);
}

template <>
// EN: Converts half precision scalar to float.
__device__ __forceinline__ float scalar_to_float<__half>(__half v) {
  return __half2float(v);
}

template <>
// EN: Converts bfloat16 scalar to float.
__device__ __forceinline__ float scalar_to_float<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T>
// EN: Loads one scalar from memory and returns it as float.
__device__ __forceinline__ float load_as_float(const T* p) {
  return scalar_to_float<T>(*p);
}

template <typename WType>
// EN: Computes row-vector dot product between weights and input vector.
__device__ __forceinline__ float row_dot_vec(const WType* w, const float* x, int n) {
  float acc = 0.f;
  for (int i = 0; i < n; ++i) acc = fmaf(load_as_float(w + i), x[i], acc);
  return acc;
}

template <>
// EN: Optimized row dot product for half weights using half2.
__device__ __forceinline__ float row_dot_vec<__half>(const __half* w, const float* x, int n) {
  float acc = 0.f;
  int i = 0;
  for (; i + 1 < n; i += 2) {
    const __half2 hv = *reinterpret_cast<const __half2*>(w + i);
    const float2 wf = __half22float2(hv);
    acc = fmaf(wf.x, x[i], acc);
    acc = fmaf(wf.y, x[i + 1], acc);
  }
  if (i < n) acc = fmaf(__half2float(w[i]), x[i], acc);
  return acc;
}

template <>
// EN: Optimized row dot product for bfloat16 weights using vectorized loads.
__device__ __forceinline__ float row_dot_vec<__nv_bfloat16>(const __nv_bfloat16* w, const float* x, int n) {
  float acc = 0.f;
  int i = 0;
  for (; i + 1 < n; i += 2) {
    const __nv_bfloat162 bv = *reinterpret_cast<const __nv_bfloat162*>(w + i);
    const float2 wf = __bfloat1622float2(bv);
    acc = fmaf(wf.x, x[i], acc);
    acc = fmaf(wf.y, x[i + 1], acc);
  }
  if (i < n) acc = fmaf(__bfloat162float(w[i]), x[i], acc);
  return acc;
}

template <typename WType>
// EN: Strided copy from typed weights to float buffer.
__device__ __forceinline__ void cast_copy_strided(float* dst, const WType* src, int n, int tid, int stride) {
  for (int i = tid; i < n; i += stride) dst[i] = load_as_float(src + i);
}

template <>
// EN: Strided cast-copy specialized for half using half2 conversion.
__device__ __forceinline__ void cast_copy_strided<__half>(float* dst, const __half* src, int n, int tid, int stride) {
  int i = tid * 2;
  const int step = stride * 2;
  for (; i + 1 < n; i += step) {
    const __half2 hv = *reinterpret_cast<const __half2*>(src + i);
    const float2 vf = __half22float2(hv);
    dst[i] = vf.x;
    dst[i + 1] = vf.y;
  }
  if (i < n) dst[i] = __half2float(src[i]);
}

template <>
// EN: Strided cast-copy specialized for bfloat16 using packed conversion.
__device__ __forceinline__ void cast_copy_strided<__nv_bfloat16>(float* dst, const __nv_bfloat16* src, int n, int tid, int stride) {
  int i = tid * 2;
  const int step = stride * 2;
  for (; i + 1 < n; i += step) {
    const __nv_bfloat162 bv = *reinterpret_cast<const __nv_bfloat162*>(src + i);
    const float2 vf = __bfloat1622float2(bv);
    dst[i] = vf.x;
    dst[i + 1] = vf.y;
  }
  if (i < n) dst[i] = __bfloat162float(src[i]);
}

// EN: Warp-aggregated atomic add to reduce atomic contention on float grads.
// EN: Threads targeting the same address are reduced first, then one atomicAdd is issued.
// Architectures before Volta fall back to a plain atomicAdd.
__device__ __forceinline__ void atomic_add_warp_agg(float* addr, float value) {
#if __CUDA_ARCH__ >= 700
  const unsigned active = __activemask();
  const unsigned peers = __match_any_sync(active, (unsigned long long)addr);
  const int lane = (int)threadIdx.x & 31;
  const int leader = __ffs((int)peers) - 1;
  if ((peers & (peers - 1u)) == 0u) {
    atomicAdd(addr, value);
    return;
  }

  // `peers` is generally sparse: pixels in one warp can be backpropagating
  // different compact hit indices.  A fixed shfl-down tree only works for a
  // contiguous lane group and drops values for masks such as lanes {0, 3}.
  // Enumerating the participating source lanes is correct for every mask.  All
  // lanes in a peer group execute the same shuffles, as required by *_sync.
  float sum = 0.f;
  unsigned remaining = peers;
  while (remaining != 0u) {
    const int src_lane = __ffs((int)remaining) - 1;
    const float peer_value = __shfl_sync(peers, value, src_lane);
    if (lane == leader) sum += peer_value;
    remaining &= remaining - 1u;
  }
  if (lane == leader) atomicAdd(addr, sum);
#else
  atomicAdd(addr, value);
#endif
}

// EN: Evaluates the MLP for one 6D feature and returns RGB + raw sigma.
template <typename WType>
__device__ __forceinline__ void mlp_eval_rgb_sigma(
    const float* __restrict__ freq_bands,
    const float* __restrict__ attenuation,
    const int F,
    const WType* __restrict__ W1p, const WType* __restrict__ b1p,
    const WType* __restrict__ W2p, const WType* __restrict__ b2p,
    const WType* __restrict__ W3p, const WType* __restrict__ b3p,
    const int H, const int IN,
    const float v6[6],
    float &out_r, float &out_g, float &out_b, float &out_sigma_raw)
{
  // Uses the current four-layer MLP layout.
  float h1[MLPSP_MAX_H];
  const int Hcap = H > MLPSP_MAX_H ? MLPSP_MAX_H : H;
  for (int h_idx = 0; h_idx < Hcap; ++h_idx) h1[h_idx] = load_as_float(b1p + h_idx);

  const int cos_offset = IN / 2;
  for (int c = 0; c < 6; ++c) {
    for (int f = 0; f < F; ++f) {
      const float fb = freq_bands[f];
      const float att = attenuation[f];
      const float xproj = v6[c] * fb * att;
      const int j = c * F + f;
      const float s = __sinf(xproj);
      const float co = __cosf(xproj);
      for (int h_idx = 0; h_idx < Hcap; ++h_idx) {
        const WType* Wi = W1p + (size_t)h_idx * IN;
        h1[h_idx] = fmaf(load_as_float(Wi + j), s, h1[h_idx]);
        h1[h_idx] = fmaf(load_as_float(Wi + j + cos_offset), co, h1[h_idx]);
      }
    }
  }
  for (int h_idx = 0; h_idx < Hcap; ++h_idx) h1[h_idx] = h1[h_idx] > 0.f ? h1[h_idx] : 0.f;

  float h2[MLPSP_MAX_H];
  for (int o = 0; o < Hcap; ++o) {
    float acc = load_as_float(b2p + o);
    const WType* W2row = W2p + (size_t)o * H;
    acc += row_dot_vec<WType>(W2row, h1, Hcap);
    h2[o] = acc > 0.f ? acc : 0.f;
  }

  float o3 = load_as_float(b3p + 0), o4 = load_as_float(b3p + 1);
  float o5 = load_as_float(b3p + 2), o6 = load_as_float(b3p + 3);
  for (int h_idx = 0; h_idx < Hcap; ++h_idx) {
    const float hi = h2[h_idx];
    o3 = fmaf(load_as_float(W3p + 0 * H + h_idx), hi, o3);
    o4 = fmaf(load_as_float(W3p + 1 * H + h_idx), hi, o4);
    o5 = fmaf(load_as_float(W3p + 2 * H + h_idx), hi, o5);
    o6 = fmaf(load_as_float(W3p + 3 * H + h_idx), hi, o6);
  }

  out_r = fast_sigmoid(o3);
  out_g = fast_sigmoid(o4);
  out_b = fast_sigmoid(o5);
  out_sigma_raw = fast_softplus(o6);
}
// EN: Computes ray-to-primitive closest approach and Gaussian culling terms.
__device__ __forceinline__ bool hit_and_features(
    float rox, float roy, float roz,
    float rdx, float rdy, float rdz,
    float cx, float cy, float cz,
    float logs,
    float thr,
    float &t, float &dx, float &dy, float &dz,
    float &decay)
{
  const float vx = cx - rox;
  const float vy = cy - roy;
  const float vz = cz - roz;
  t = fmaf(vx, rdx, fmaf(vy, rdy, vz * rdz));
  if (t <= 0.f) return false;

  const float px = rox + t * rdx;
  const float py = roy + t * rdy;
  const float pz = roz + t * rdz;
  dx = cx - px; dy = cy - py; dz = cz - pz; // center - foot

  const float perp2 = fmaxf(dx*dx + dy*dy + dz*dz, 0.f);
  const float s = __expf(logs);
  const float s2 = s*s;
  const float score = __expf(-0.5f * perp2 / (s2 + 1e-12f));
  if (score <= thr) return false;

  decay = __expf(- perp2 / (2.f * s2 + 1e-12f));
  return true;
}


// ------------------------------
// Quaternion helpers (unit quaternion expected)
// ------------------------------
__device__ __forceinline__ float3 quat_rotate(
    float qw, float qx, float qy, float qz, float3 v) {
  // v' = q * v * q_conj using optimized form
  const float3 qv = make_float3(qx, qy, qz);
  const float3 t = make_float3(
      2.f * (qv.y * v.z - qv.z * v.y),
      2.f * (qv.z * v.x - qv.x * v.z),
      2.f * (qv.x * v.y - qv.y * v.x));
  return make_float3(
      v.x + qw * t.x + (qv.y * t.z - qv.z * t.y),
      v.y + qw * t.y + (qv.z * t.x - qv.x * t.z),
      v.z + qw * t.z + (qv.x * t.y - qv.y * t.x));
}

__device__ __forceinline__ float3 quat_rotate_conj(
    float qw, float qx, float qy, float qz, float3 v) {
  // rotate by q_conj (equivalently apply R(q)^T)
  // q_conj = (qw, -qx, -qy, -qz)
  return quat_rotate(qw, -qx, -qy, -qz, v);
}

// Compute dL/d(qw,qx,qy,qz) given dL/dR (3x3) where R is rotation matrix of (qw,qx,qy,qz).
// gR is in row-major: gR00..gR22 correspond to dL/dR_{row,col}.
__device__ __forceinline__ void rotmat_grad_to_quat_grad(
    float qw, float qx, float qy, float qz,
    float gR00, float gR01, float gR02,
    float gR10, float gR11, float gR12,
    float gR20, float gR21, float gR22,
    float &gqw, float &gqx, float &gqy, float &gqz) {

  // Rotation matrix (w,x,y,z):
  // R00 = 1 - 2(y^2 + z^2)
  // R01 = 2(xy - wz)
  // R02 = 2(xz + wy)
  // R10 = 2(xy + wz)
  // R11 = 1 - 2(x^2 + z^2)
  // R12 = 2(yz - wx)
  // R20 = 2(xz - wy)
  // R21 = 2(yz + wx)
  // R22 = 1 - 2(x^2 + y^2)

  // dR/dw
  gqw =
      gR01 * (-2.f * qz) + gR02 * ( 2.f * qy) +
      gR10 * ( 2.f * qz) + gR12 * (-2.f * qx) +
      gR20 * (-2.f * qy) + gR21 * ( 2.f * qx);

  // dR/dx
  gqx =
      gR01 * ( 2.f * qy) + gR02 * ( 2.f * qz) +
      gR10 * ( 2.f * qy) + gR11 * (-4.f * qx) + gR12 * (-2.f * qw) +
      gR20 * ( 2.f * qz) + gR21 * ( 2.f * qw) + gR22 * (-4.f * qx);

  // dR/dy
  gqy =
      gR00 * (-4.f * qy) + gR01 * ( 2.f * qx) + gR02 * ( 2.f * qw) +
      gR10 * ( 2.f * qx) + gR12 * ( 2.f * qz) +
      gR20 * (-2.f * qw) + gR21 * ( 2.f * qz) + gR22 * (-4.f * qy);

  // dR/dz
  gqz =
      gR00 * (-4.f * qz) + gR01 * (-2.f * qw) + gR02 * ( 2.f * qx) +
      gR10 * ( 2.f * qw) + gR11 * (-4.f * qz) + gR12 * ( 2.f * qy) +
      gR20 * ( 2.f * qx) + gR21 * ( 2.f * qy);
}


__device__ __forceinline__ float tstar_key_tilecenter(
    float crox, float croy, float croz,
    float crdx, float crdy, float crdz,
    float cx, float cy, float cz,
    float log_sx, float log_sy, float log_sz,
    float qw, float qx, float qy, float qz)
{
  // v = c - cro (world)
  const float vx = cx - crox;
  const float vy = cy - croy;
  const float vz = cz - croz;

  // M = diag(1/s^2)
  const float mx = __expf(-2.f * log_sx);
  const float my = __expf(-2.f * log_sy);
  const float mz = __expf(-2.f * log_sz);

  // v' = R^T v, d' = R^T crd
  const float3 v_loc  = quat_rotate_conj(qw, qx, qy, qz, make_float3(vx, vy, vz));
  const float3 rd_loc = quat_rotate_conj(qw, qx, qy, qz, make_float3(crdx, crdy, crdz));

  // denom = d'^T M d'
  const float denom = fmaf(rd_loc.x*rd_loc.x, mx,
                      fmaf(rd_loc.y*rd_loc.y, my,
                           (rd_loc.z*rd_loc.z) * mz));
  // numer = d'^T M v'
  const float numer = fmaf(rd_loc.x*v_loc.x, mx,
                      fmaf(rd_loc.y*v_loc.y, my,
                           (rd_loc.z*v_loc.z) * mz));

  // Robustness: if denom tiny -> treat as invalid (push to back)
  if (denom <= 1e-20f) return 1e30f;

  const float tstar = numer / denom;
  return (tstar > 0.f) ? tstar : 1e30f;
}



__device__ __forceinline__ void tstar_and_d2min(
    float rox, float roy, float roz,
    float rdx, float rdy, float rdz,
    float cx, float cy, float cz,
    float log_sx, float log_sy, float log_sz,
    float qw, float qx, float qy, float qz,
    float &tstar, float &denom, float &d2min)
{
  // v = c - ro (world)
  const float vx = cx - rox;
  const float vy = cy - roy;
  const float vz = cz - roz;

  // M = diag(1/s^2) where s = exp(log_s)
  const float mx = __expf(-2.f * log_sx);
  const float my = __expf(-2.f * log_sy);
  const float mz = __expf(-2.f * log_sz);

  // v' = R^T v, d' = R^T rd  (local coordinates)
  const float3 v_loc  = quat_rotate_conj(qw, qx, qy, qz, make_float3(vx, vy, vz));
  const float3 rd_loc = quat_rotate_conj(qw, qx, qy, qz, make_float3(rdx, rdy, rdz));

  // A = d'^T M d'
  denom = fmaf(rd_loc.x*rd_loc.x, mx,
          fmaf(rd_loc.y*rd_loc.y, my,
               (rd_loc.z*rd_loc.z) * mz));

  // B = d'^T M v'
  const float numer = fmaf(rd_loc.x*v_loc.x, mx,
                      fmaf(rd_loc.y*v_loc.y, my,
                           (rd_loc.z*v_loc.z) * mz));

  // Robustness: invalid if denom tiny
  if (denom <= 1e-20f) {
    tstar = 1e30f;
    d2min = 0.f;
    return;
  }

  // t* = B / A
  tstar = numer / denom;
  if (tstar <= 0.f) {
    tstar = 1e30f;
    d2min = 0.f;
    return;
  }

  // vMv = v'^T M v'
  const float vMv = fmaf(v_loc.x*v_loc.x, mx,
                    fmaf(v_loc.y*v_loc.y, my,
                         (v_loc.z*v_loc.z) * mz));

  // d2min = vMv - B^2 / A   (>=0 ideally; clamp for numeric safety)
  d2min = vMv - (numer * numer) / denom;
  d2min = fmaxf(d2min, 0.f);
}


__device__ __forceinline__ float tkey_tilecenter_enter_or_tc(
    float crox, float croy, float croz,
    float crdx, float crdy, float crdz,
    float cx, float cy, float cz,
    float log_sx, float log_sy, float log_sz,
    float qw, float qx, float qy, float qz,
    float thr)
{
  // v = center - ray_origin (world)
  const float vx = cx - crox;
  const float vy = cy - croy;
  const float vz = cz - croz;

  // tc = (x-c)·d  (assumes d is normalized; if not, still monotonic for sorting)
  const float tc = fmaf(vx, crdx, fmaf(vy, crdy, vz * crdz));
  if (tc <= 0.f) return 1e30f;  // behind camera -> push back

  // precision diag: 1/s^2
  const float mx = __expf(-2.f * log_sx);
  const float my = __expf(-2.f * log_sy);
  const float mz = __expf(-2.f * log_sz);

  // local frame: v' = R^T v, d' = R^T d
  const float3 v_loc  = quat_rotate_conj(qw, qx, qy, qz, make_float3(vx, vy, vz));
  const float3 rd_loc = quat_rotate_conj(qw, qx, qy, qz, make_float3(crdx, crdy, crdz));

  // A = d'^T M d'
  const float A = fmaf(rd_loc.x*rd_loc.x, mx,
                  fmaf(rd_loc.y*rd_loc.y, my,
                       (rd_loc.z*rd_loc.z) * mz));
  if (A <= 1e-20f) return tc;   // degenerate -> fallback to tc

  // B = d'^T M v'
  const float B = fmaf(rd_loc.x*v_loc.x, mx,
                  fmaf(rd_loc.y*v_loc.y, my,
                       (rd_loc.z*v_loc.z) * mz));

  // vMv = v'^T M v'
  const float vMv = fmaf(v_loc.x*v_loc.x, mx,
                    fmaf(v_loc.y*v_loc.y, my,
                         (v_loc.z*v_loc.z) * mz));

  // rho^2 from decay=exp(-0.5 r2) > thr  => r2 < -2 ln(thr)
  const float thr_c = fminf(fmaxf(thr, 1e-30f), 0.999999f);
  const float rho2  = -2.f * __logf(thr_c);

  // discriminant: Δ = B^2 - A (vMv - rho^2)
  float disc = fmaf(B, B, -A * (vMv - rho2));
  if (disc <= 0.f) return tc;   // no iso-surface intersection -> fallback to tc

  disc = fmaxf(disc, 0.f);
  const float sqrt_disc = __fsqrt_rn(disc);

  // entry root: (B - sqrt(Δ)) / A
  const float t_enter = (B - sqrt_disc) / A;

  // If entry is behind camera (e.g., ray starts inside iso-surface), fallback to tc
  return (t_enter > 0.f) ? t_enter : tc;
}


__device__ __forceinline__ void swapf(float &a, float &b) {
  float t = a; a = b; b = t;
}
__device__ __forceinline__ void sort2(float &a, float &b) {
  if (a > b) swapf(a,b);
}

// median of 5 via sorting network-ish (not fully minimal, but fine)
__device__ __forceinline__ float median5(float a, float b, float c, float d, float e) {
  sort2(a,b); sort2(c,d);
  sort2(a,c); sort2(b,d);
  sort2(b,c); sort2(d,e);
  sort2(c,d); sort2(b,c);
  return c; // now c is the median
}

// Rotated ellipsoid hit/decay (axis scales + quaternion rotation).
// Uses Mahalanobis-optimal t* (maximizes decay along the ray).
__device__ __forceinline__ bool hit_and_features_rot(
    float rox, float roy, float roz,
    float rdx, float rdy, float rdz,
    float cx, float cy, float cz,
    float log_sx, float log_sy, float log_sz,
    float qw, float qx, float qy, float qz,
    float thr,
    float &t, float &dx, float &dy, float &dz,
    float &qlx, float &qly, float &qlz,
    float &decay)
{
  // v = c - ro (world)
  const float vx = cx - rox;
  const float vy = cy - roy;
  const float vz = cz - roz;

  // precision diag: 1/s^2
  const float mx = __expf(-2.f * log_sx);
  const float my = __expf(-2.f * log_sy);
  const float mz = __expf(-2.f * log_sz);

  // rotate v and rd into local frame: v' = R^T v, d' = R^T rd
  const float3 v_loc  = quat_rotate_conj(qw, qx, qy, qz, make_float3(vx, vy, vz));
  const float3 rd_loc = quat_rotate_conj(qw, qx, qy, qz, make_float3(rdx, rdy, rdz));

  // t* = (d'^T M v') / (d'^T M d')
  const float denom = fmaf(rd_loc.x*rd_loc.x, mx,
                      fmaf(rd_loc.y*rd_loc.y, my,
                           (rd_loc.z*rd_loc.z) * mz));
  if (denom <= 1e-20f) return false;
  const float numer = fmaf(rd_loc.x*v_loc.x, mx,
                      fmaf(rd_loc.y*v_loc.y, my,
                           (rd_loc.z*v_loc.z) * mz));

  t = numer / denom;
  if (t <= 0.f) return false;

  // point on ray at t*
  const float px = rox + t * rdx;
  const float py = roy + t * rdy;
  const float pz = roz + t * rdz;

  // d = c - p (world)
  dx = cx - px; dy = cy - py; dz = cz - pz;

  // local offset q = R^T d
  const float3 qloc = quat_rotate_conj(qw, qx, qy, qz, make_float3(dx, dy, dz));
  qlx = qloc.x; qly = qloc.y; qlz = qloc.z;

  const float r2 = fmaxf(qlx*qlx*mx + qly*qly*my + qlz*qlz*mz, 0.f);
  decay = __expf(-0.5f * r2);
  return decay > thr;
}


// Kernel: render tiles with per-pixel sorting
__global__ void render_tiles_kernel_with_sort(
    // Rays and primitives
    const float* __restrict__ rays_o,
    const float* __restrict__ rays_d,
    const float* __restrict__ pos,
    const float* __restrict__ logs,
    const float* __restrict__ quats,
    const int32_t* __restrict__ culled_prim_indices,
    const int32_t* __restrict__ tile_offsets,
    // Encoding params
    const float* __restrict__ freq_bands,
    const float* __restrict__ attenuation,
    const int F,
    // MLP weights
    const float* __restrict__ W1, const float* __restrict__ b1,
    const float* __restrict__ W2, const float* __restrict__ b2,
    const float* __restrict__ W3, const float* __restrict__ b3,
    const int N, const int H, const int IN,
    // Image/tile
    const int Wimg, const int Himg, const int TW, const int TH,
    // Threshold
    const float thr,
    // Outputs
    float* __restrict__ out_rgb,
    float* __restrict__ out_alpha,
    int32_t* __restrict__ intersection_overflow
){
  const int tile_x = blockIdx.x;
  const int tile_y = blockIdx.y;
  const int x = tile_x * TW + threadIdx.x;
  const int y = tile_y * TH + threadIdx.y;
  if (x >= Wimg || y >= Himg) return;

  const int pix = y * Wimg + x;
  const int tile_id = tile_y * ((Wimg + TW - 1)/TW) + tile_x;

  // Load ray (already normalized from Python)
  const float rox = rays_o[pix*3+0];
  const float roy = rays_o[pix*3+1];
  const float roz = rays_o[pix*3+2];
  const float rdx = rays_d[pix*3+0];
  const float rdy = rays_d[pix*3+1];
  const float rdz = rays_d[pix*3+2];

  // --- Stage 1: Gather intersections for this pixel ---
  const int MAX_INTERSECTIONS = 256;
  float intersection_t[MAX_INTERSECTIONS];
  int   intersection_prim_id[MAX_INTERSECTIONS];
  float intersection_decay[MAX_INTERSECTIONS];
  float intersection_dx[MAX_INTERSECTIONS];
  float intersection_dy[MAX_INTERSECTIONS];
  float intersection_dz[MAX_INTERSECTIONS];
  float intersection_qlx[MAX_INTERSECTIONS];
  float intersection_qly[MAX_INTERSECTIONS];
  float intersection_qlz[MAX_INTERSECTIONS];
  int intersection_count = 0;

  const int32_t prim_start = tile_offsets[tile_id];
  const int32_t prim_end = tile_offsets[tile_id + 1];

  for (int k = prim_start; k < prim_end; ++k) {
    const int p = culled_prim_indices[k];
    float t, dx, dy, dz, decay;
    const float cx = pos[p*3+0];
    const float cy = pos[p*3+1];
    const float cz = pos[p*3+2];
    const float lgx = logs[p*3+0];
    const float lgy = logs[p*3+1];
    const float lgz = logs[p*3+2];
    const float qw  = quats[p*4+0];
    const float qx  = quats[p*4+1];
    const float qy  = quats[p*4+2];
    const float qz  = quats[p*4+3];
    float qlx, qly, qlz;

    if (hit_and_features_rot(rox, roy, roz, rdx, rdy, rdz,
                             cx, cy, cz,
                             lgx, lgy, lgz,
                             qw, qx, qy, qz,
                             thr,
                             t, dx, dy, dz,
                             qlx, qly, qlz,
                             decay)) {
      if (intersection_count < MAX_INTERSECTIONS) {
        intersection_t[intersection_count] = t;
        intersection_prim_id[intersection_count] = p;
        intersection_decay[intersection_count] = decay;
        intersection_dx[intersection_count] = dx;
        intersection_dy[intersection_count] = dy;
        intersection_dz[intersection_count] = dz;
        intersection_count++;
      } else {
        atomicExch(intersection_overflow, 1);
        break;
      }
    }
  }

  if (intersection_count == 0) {
      out_rgb[pix*3+0] = 0.f;
      out_rgb[pix*3+1] = 0.f;
      out_rgb[pix*3+2] = 0.f;
      out_alpha[pix] = 0.f;
      return;
  }

  // --- Stage 2: Sort intersections by t (Insertion Sort) ---
  for (int i = 1; i < intersection_count; i++) {
    float key_t = intersection_t[i];
    int   key_p = intersection_prim_id[i];
    float key_decay = intersection_decay[i];
    float key_dx = intersection_dx[i], key_dy = intersection_dy[i], key_dz = intersection_dz[i];
    int j = i - 1;
    while (j >= 0 && intersection_t[j] > key_t) {
      intersection_t[j+1] = intersection_t[j];
      intersection_prim_id[j+1] = intersection_prim_id[j];
      intersection_decay[j+1] = intersection_decay[j];
      intersection_dx[j+1] = intersection_dx[j];
      intersection_dy[j+1] = intersection_dy[j];
      intersection_dz[j+1] = intersection_dz[j];
      j--;
    }
    intersection_t[j+1] = key_t;
    intersection_prim_id[j+1] = key_p;
    intersection_decay[j+1] = key_decay;
    intersection_dx[j+1] = key_dx;
    intersection_dy[j+1] = key_dy;
    intersection_dz[j+1] = key_dz;
  }

  // --- Stage 3: Blend in sorted order ---
  float3 C = make_float3(0.f, 0.f, 0.f);
  float trans = 1.f;

  for (int i = 0; i < intersection_count; ++i) {
    const int p = intersection_prim_id[i];
    const float decay = intersection_decay[i];
    const float dx = intersection_dx[i];
    const float dy = intersection_dy[i];
    const float dz = intersection_dz[i];

    // Build 6D input vector
    float v6[6] = {rdx, rdy, rdz, dx, dy, dz};

    // MLP evaluation
    const float* W1p = W1 + (size_t)p * H * IN;
    const float* b1p = b1 + (size_t)p * H;
    const float* W2p = W2 + (size_t)p * H * H;
    const float* b2p = b2 + (size_t)p * H;
    const float* W3p = W3 + (size_t)p * 4 * H;
    const float* b3p = b3 + (size_t)p * 4;

    float h1[MLPSP_MAX_H];
    int Hcap = H > MLPSP_MAX_H ? MLPSP_MAX_H : H;
    for (int h_idx = 0; h_idx < Hcap; ++h_idx) h1[h_idx] = b1p[h_idx];

    const int cos_offset = IN / 2;
    for (int c = 0; c < 6; ++c) {
      for (int f = 0; f < F; ++f) {
        const float fb = freq_bands[f];
        const float att = attenuation[f];
        const float xproj = v6[c] * fb * att;
        const int j = c * F + f;
        const float s = __sinf(xproj);
        const float co = __cosf(xproj);
        for (int h_idx = 0; h_idx < Hcap; ++h_idx) {
          const float* Wi = W1p + (size_t)h_idx * IN;
          h1[h_idx] = fmaf(Wi[j], s, h1[h_idx]);
          h1[h_idx] = fmaf(Wi[j + cos_offset], co, h1[h_idx]);
        }
      }
    }
    for (int h_idx = 0; h_idx < Hcap; ++h_idx) h1[h_idx] = h1[h_idx] > 0.f ? h1[h_idx] : 0.f;

    float h2[MLPSP_MAX_H];
    for (int o = 0; o < Hcap; ++o) {
      float acc = b2p[o];
      const float* W2row = W2p + (size_t)o * H;
      for (int h_idx = 0; h_idx < Hcap; ++h_idx) acc = fmaf(W2row[h_idx], h1[h_idx], acc);
      h2[o] = acc > 0.f ? acc : 0.f;
    }

    float o3, o4, o5, o6;
    o3=b3p[0]; o4=b3p[1]; o5=b3p[2]; o6=b3p[3];
    for (int h_idx = 0; h_idx < Hcap; ++h_idx) {
      const float hi = h2[h_idx];
      o3 = fmaf(W3p[0*H + h_idx], hi, o3);
      o4 = fmaf(W3p[1*H + h_idx], hi, o4);
      o5 = fmaf(W3p[2*H + h_idx], hi, o5);
      o6 = fmaf(W3p[3*H + h_idx], hi, o6);
    }

    const float sigma_raw = fast_softplus(o6);
    const float sigma = sigma_raw * decay;
    const float alpha = 1.f - __expf(-sigma);
    const float w = trans * alpha;

    const float R = fast_sigmoid(o3);
    const float G = fast_sigmoid(o4);
    const float Bc = fast_sigmoid(o5);
    C.x = fmaf(w, R, C.x);
    C.y = fmaf(w, G, C.y);
    C.z = fmaf(w, Bc, C.z);
    trans *= (1.f - alpha);
    if (trans < 1e-3f) break;
  }

  out_rgb[pix*3+0] = C.x;
  out_rgb[pix*3+1] = C.y;
  out_rgb[pix*3+2] = C.z;
  out_alpha[pix]   = 1.f - trans;
}

// Kernel: render tiles with per-pixel sorting + feature accumulation.
__global__ void render_tiles_kernel_with_sort_and_feat(
    // Rays and primitives
    const float* __restrict__ rays_o,
    const float* __restrict__ rays_d,
    const float* __restrict__ pos,
    const float* __restrict__ logs,
    const float* __restrict__ quats,
    const int32_t* __restrict__ culled_prim_indices,
    const int32_t* __restrict__ tile_offsets,
    // Encoding params
    const float* __restrict__ freq_bands,
    const float* __restrict__ attenuation,
    const int F,
    // MLP weights
    const float* __restrict__ W1, const float* __restrict__ b1,
    const float* __restrict__ W2, const float* __restrict__ b2,
    const float* __restrict__ W3, const float* __restrict__ b3,
    // Semantic feature attrs
    const float* __restrict__ feat_attr,
    const int feat_slots,
    const int Cfeat,
    const float gating_beta,
    const float gating_temp,
    const int N, const int H, const int IN,
    // Image/tile
    const int Wimg, const int Himg, const int TW, const int TH,
    // Threshold
    const float thr,
    // Outputs
    float* __restrict__ out_rgb,
    float* __restrict__ out_alpha,
    float* __restrict__ out_feat,
    int32_t* __restrict__ intersection_overflow
){
  const int tile_x = blockIdx.x;
  const int tile_y = blockIdx.y;
  const int x = tile_x * TW + threadIdx.x;
  const int y = tile_y * TH + threadIdx.y;
  if (x >= Wimg || y >= Himg) return;

  const int pix = y * Wimg + x;
  const int tile_id = tile_y * ((Wimg + TW - 1)/TW) + tile_x;

  // Load ray (already normalized from Python)
  const float rox = rays_o[pix*3+0];
  const float roy = rays_o[pix*3+1];
  const float roz = rays_o[pix*3+2];
  const float rdx = rays_d[pix*3+0];
  const float rdy = rays_d[pix*3+1];
  const float rdz = rays_d[pix*3+2];

  // --- Stage 1: Gather intersections for this pixel ---
  const int MAX_INTERSECTIONS = 256;
  float intersection_t[MAX_INTERSECTIONS];
  int   intersection_prim_id[MAX_INTERSECTIONS];
  float intersection_decay[MAX_INTERSECTIONS];
  float intersection_dx[MAX_INTERSECTIONS];
  float intersection_dy[MAX_INTERSECTIONS];
  float intersection_dz[MAX_INTERSECTIONS];
  float intersection_qlx[MAX_INTERSECTIONS];
  float intersection_qly[MAX_INTERSECTIONS];
  float intersection_qlz[MAX_INTERSECTIONS];
  int intersection_count = 0;

  const int32_t prim_start = tile_offsets[tile_id];
  const int32_t prim_end = tile_offsets[tile_id + 1];

  for (int k = prim_start; k < prim_end; ++k) {
    const int p = culled_prim_indices[k];
    float t, dx, dy, dz, qlx, qly, qlz, decay;
    const float cx = pos[p*3+0];
    const float cy = pos[p*3+1];
    const float cz = pos[p*3+2];
    const float lgx = logs[p*3+0];
    const float lgy = logs[p*3+1];
    const float lgz = logs[p*3+2];
    const float qw  = quats[p*4+0];
    const float qx  = quats[p*4+1];
    const float qy  = quats[p*4+2];
    const float qz  = quats[p*4+3];

    if (hit_and_features_rot(rox, roy, roz, rdx, rdy, rdz,
                             cx, cy, cz,
                             lgx, lgy, lgz,
                             qw, qx, qy, qz,
                             thr,
                             t, dx, dy, dz,
                             qlx, qly, qlz,
                             decay)) {
      if (intersection_count < MAX_INTERSECTIONS) {
        intersection_t[intersection_count] = t;
        intersection_prim_id[intersection_count] = p;
        intersection_decay[intersection_count] = decay;
        intersection_dx[intersection_count] = dx;
        intersection_dy[intersection_count] = dy;
        intersection_dz[intersection_count] = dz;
        intersection_qlx[intersection_count] = qlx;
        intersection_qly[intersection_count] = qly;
        intersection_qlz[intersection_count] = qlz;
        intersection_count++;
      } else {
        atomicExch(intersection_overflow, 1);
        break;
      }
    }
  }

  // Always initialize output feature for this pixel.
  float* dst_feat = out_feat + (size_t)pix * Cfeat;
  for (int c = 0; c < Cfeat; ++c) dst_feat[c] = 0.f;

  if (intersection_count == 0) {
      out_rgb[pix*3+0] = 0.f;
      out_rgb[pix*3+1] = 0.f;
      out_rgb[pix*3+2] = 0.f;
      out_alpha[pix] = 0.f;
      return;
  }

  // --- Stage 2: Sort intersections by t (Insertion Sort) ---
  for (int i = 1; i < intersection_count; i++) {
    float key_t = intersection_t[i];
    int   key_p = intersection_prim_id[i];
    float key_decay = intersection_decay[i];
    float key_dx = intersection_dx[i], key_dy = intersection_dy[i], key_dz = intersection_dz[i];
    float key_qlx = intersection_qlx[i], key_qly = intersection_qly[i], key_qlz = intersection_qlz[i];
    int j = i - 1;
    while (j >= 0 && intersection_t[j] > key_t) {
      intersection_t[j+1] = intersection_t[j];
      intersection_prim_id[j+1] = intersection_prim_id[j];
      intersection_decay[j+1] = intersection_decay[j];
      intersection_dx[j+1] = intersection_dx[j];
      intersection_dy[j+1] = intersection_dy[j];
      intersection_dz[j+1] = intersection_dz[j];
      intersection_qlx[j+1] = intersection_qlx[j];
      intersection_qly[j+1] = intersection_qly[j];
      intersection_qlz[j+1] = intersection_qlz[j];
      j--;
    }
    intersection_t[j+1] = key_t;
    intersection_prim_id[j+1] = key_p;
    intersection_decay[j+1] = key_decay;
    intersection_dx[j+1] = key_dx;
    intersection_dy[j+1] = key_dy;
    intersection_dz[j+1] = key_dz;
    intersection_qlx[j+1] = key_qlx;
    intersection_qly[j+1] = key_qly;
    intersection_qlz[j+1] = key_qlz;
  }

  // --- Stage 3: Blend in sorted order ---
  float3 C = make_float3(0.f, 0.f, 0.f);
  float trans = 1.f;

  for (int i = 0; i < intersection_count; ++i) {
    const int p = intersection_prim_id[i];
    const float decay = intersection_decay[i];
    const float dx = intersection_dx[i];
    const float dy = intersection_dy[i];
    const float dz = intersection_dz[i];
    const float qlx = intersection_qlx[i];
    const float qly = intersection_qly[i];
    const float qlz = intersection_qlz[i];

    // Build 6D input vector
    float v6[6] = {rdx, rdy, rdz, dx, dy, dz};

    // MLP evaluation
    const float* W1p = W1 + (size_t)p * H * IN;
    const float* b1p = b1 + (size_t)p * H;
    const float* W2p = W2 + (size_t)p * H * H;
    const float* b2p = b2 + (size_t)p * H;
    const float* W3p = W3 + (size_t)p * 4 * H;
    const float* b3p = b3 + (size_t)p * 4;

    float h1[MLPSP_MAX_H];
    int Hcap = H > MLPSP_MAX_H ? MLPSP_MAX_H : H;
    for (int h_idx = 0; h_idx < Hcap; ++h_idx) h1[h_idx] = b1p[h_idx];

    const int cos_offset = IN / 2;
    for (int c = 0; c < 6; ++c) {
      for (int f = 0; f < F; ++f) {
        const float fb = freq_bands[f];
        const float att = attenuation[f];
        const float xproj = v6[c] * fb * att;
        const int j = c * F + f;
        const float s = __sinf(xproj);
        const float co = __cosf(xproj);
        for (int h_idx = 0; h_idx < Hcap; ++h_idx) {
          const float* Wi = W1p + (size_t)h_idx * IN;
          h1[h_idx] = fmaf(Wi[j], s, h1[h_idx]);
          h1[h_idx] = fmaf(Wi[j + cos_offset], co, h1[h_idx]);
        }
      }
    }
    for (int h_idx = 0; h_idx < Hcap; ++h_idx) h1[h_idx] = h1[h_idx] > 0.f ? h1[h_idx] : 0.f;

    float h2[MLPSP_MAX_H];
    for (int o = 0; o < Hcap; ++o) {
      float acc = b2p[o];
      const float* W2row = W2p + (size_t)o * H;
      for (int h_idx = 0; h_idx < Hcap; ++h_idx) acc = fmaf(W2row[h_idx], h1[h_idx], acc);
      h2[o] = acc > 0.f ? acc : 0.f;
    }

    float o3, o4, o5, o6;
    o3=b3p[0]; o4=b3p[1]; o5=b3p[2]; o6=b3p[3];
    for (int h_idx = 0; h_idx < Hcap; ++h_idx) {
      const float hi = h2[h_idx];
      o3 = fmaf(W3p[0*H + h_idx], hi, o3);
      o4 = fmaf(W3p[1*H + h_idx], hi, o4);
      o5 = fmaf(W3p[2*H + h_idx], hi, o5);
      o6 = fmaf(W3p[3*H + h_idx], hi, o6);
    }

    const float sigma_raw = fast_softplus(o6);
    const float sigma = sigma_raw * decay;
    const float alpha = 1.f - __expf(-sigma);
    const float w = trans * alpha;

    const float R = fast_sigmoid(o3);
    const float G = fast_sigmoid(o4);
    const float Bc = fast_sigmoid(o5);
    C.x = fmaf(w, R, C.x);
    C.y = fmaf(w, G, C.y);
    C.z = fmaf(w, Bc, C.z);

    if (feat_slots <= 1) {
      const float* src_feat = feat_attr + (size_t)p * Cfeat;
      for (int c = 0; c < Cfeat; ++c) dst_feat[c] = fmaf(w, src_feat[c], dst_feat[c]);
    } else if (feat_slots == 5) {
      float g0, g1, g2, g3;
      const float lgx = logs[p*3+0];
      const float lgy = logs[p*3+1];
      const float lgz = logs[p*3+2];
      semantic_tetra_gates(qlx, qly, qlz, lgx, lgy, lgz, gating_beta, gating_temp, g0, g1, g2, g3);
      const float* base = feat_attr + ((size_t)p * feat_slots + 0) * Cfeat;
      const float* d0 = feat_attr + ((size_t)p * feat_slots + 1) * Cfeat;
      const float* d1 = feat_attr + ((size_t)p * feat_slots + 2) * Cfeat;
      const float* d2 = feat_attr + ((size_t)p * feat_slots + 3) * Cfeat;
      const float* d3 = feat_attr + ((size_t)p * feat_slots + 4) * Cfeat;
      for (int c = 0; c < Cfeat; ++c) {
        float mix = base[c];
        mix = fmaf(g0, d0[c], mix);
        mix = fmaf(g1, d1[c], mix);
        mix = fmaf(g2, d2[c], mix);
        mix = fmaf(g3, d3[c], mix);
        dst_feat[c] = fmaf(w, mix, dst_feat[c]);
      }
    } else if (feat_slots == 4) {
      float g0, g1, g2, g3;
      const float lgx = logs[p*3+0];
      const float lgy = logs[p*3+1];
      const float lgz = logs[p*3+2];
      semantic_tetra_gates(qlx, qly, qlz, lgx, lgy, lgz, gating_beta, gating_temp, g0, g1, g2, g3);
      const float* s0 = feat_attr + ((size_t)p * feat_slots + 0) * Cfeat;
      const float* s1 = feat_attr + ((size_t)p * feat_slots + 1) * Cfeat;
      const float* s2 = feat_attr + ((size_t)p * feat_slots + 2) * Cfeat;
      const float* s3 = feat_attr + ((size_t)p * feat_slots + 3) * Cfeat;
      for (int c = 0; c < Cfeat; ++c) {
        float mix = 0.f;
        mix = fmaf(g0, s0[c], mix);
        mix = fmaf(g1, s1[c], mix);
        mix = fmaf(g2, s2[c], mix);
        mix = fmaf(g3, s3[c], mix);
        dst_feat[c] = fmaf(w, mix, dst_feat[c]);
      }
    } else {
      const float* src_feat = feat_attr + (size_t)p * feat_slots * Cfeat;
      for (int c = 0; c < Cfeat; ++c) dst_feat[c] = fmaf(w, src_feat[c], dst_feat[c]);
    }
    trans *= (1.f - alpha);
    if (trans < 1e-3f) break;
  }

  out_rgb[pix*3+0] = C.x;
  out_rgb[pix*3+1] = C.y;
  out_rgb[pix*3+2] = C.z;
  out_alpha[pix]   = 1.f - trans;
}

// Backward (per-pixel): feat_attr-only gradient accumulation.
__global__ void render_tiles_backward_featattr_per_pixel(
    const float* __restrict__ rays_o,
    const float* __restrict__ rays_d,
    const float* __restrict__ pos,
    const float* __restrict__ log_scales,
    const float* __restrict__ quats,
    const int32_t* __restrict__ culled_prim_indices,
    const int32_t* __restrict__ tile_offsets,
    const float* __restrict__ freq_bands,
    const float* __restrict__ attenuation,
    const int F,
    const float* __restrict__ W1, const float* __restrict__ b1,
    const float* __restrict__ W2, const float* __restrict__ b2,
    const float* __restrict__ W3, const float* __restrict__ b3,
    const int N, const int H, const int IN,
    const int Wimg, const int Himg, const int TW, const int TH,
    const float thr,
    const float* __restrict__ grad_out_feat,
    const int feat_slots,
    const int Cfeat,
    const float gating_beta,
    const float gating_temp,
    float* __restrict__ grad_feat_attr,
    int32_t* __restrict__ intersection_overflow)
{
  const int tile_x = blockIdx.x;
  const int tile_y = blockIdx.y;
  const int x = tile_x * TW + threadIdx.x;
  const int y = tile_y * TH + threadIdx.y;
  if (x >= Wimg || y >= Himg) return;
  const int pix = y * Wimg + x;
  const int tile_id = tile_y * ((Wimg + TW - 1)/TW) + tile_x;

  const float rox = rays_o[pix*3+0];
  const float roy = rays_o[pix*3+1];
  const float roz = rays_o[pix*3+2];
  const float rdx = rays_d[pix*3+0];
  const float rdy = rays_d[pix*3+1];
  const float rdz = rays_d[pix*3+2];

  const int32_t prim_start = tile_offsets[tile_id];
  const int32_t prim_end = tile_offsets[tile_id + 1];

  const int MAX_INTERSECTIONS = 256;
  float intersection_t[MAX_INTERSECTIONS];
  int   intersection_prim_id[MAX_INTERSECTIONS];
  float intersection_decay[MAX_INTERSECTIONS];
  float intersection_dx[MAX_INTERSECTIONS];
  float intersection_dy[MAX_INTERSECTIONS];
  float intersection_dz[MAX_INTERSECTIONS];
  float intersection_qlx[MAX_INTERSECTIONS];
  float intersection_qly[MAX_INTERSECTIONS];
  float intersection_qlz[MAX_INTERSECTIONS];
  int intersection_count = 0;

  for (int k = prim_start; k < prim_end; ++k) {
    const int p = culled_prim_indices[k];
    float t, dx, dy, dz, qlx, qly, qlz, decay;
    const float cx = pos[p*3+0];
    const float cy = pos[p*3+1];
    const float cz = pos[p*3+2];
    const float lgx = log_scales[p*3+0];
    const float lgy = log_scales[p*3+1];
    const float lgz = log_scales[p*3+2];
    const float qw  = quats[p*4+0];
    const float qx  = quats[p*4+1];
    const float qy  = quats[p*4+2];
    const float qz  = quats[p*4+3];

    if (hit_and_features_rot(rox, roy, roz, rdx, rdy, rdz,
                             cx, cy, cz,
                             lgx, lgy, lgz,
                             qw, qx, qy, qz,
                             thr,
                             t, dx, dy, dz,
                             qlx, qly, qlz,
                             decay)) {
      if (intersection_count < MAX_INTERSECTIONS) {
        intersection_t[intersection_count] = t;
        intersection_prim_id[intersection_count] = p;
        intersection_decay[intersection_count] = decay;
        intersection_dx[intersection_count] = dx;
        intersection_dy[intersection_count] = dy;
        intersection_dz[intersection_count] = dz;
        intersection_qlx[intersection_count] = qlx;
        intersection_qly[intersection_count] = qly;
        intersection_qlz[intersection_count] = qlz;
        intersection_count++;
      } else {
        atomicExch(intersection_overflow, 1);
        break;
      }
    }
  }
  if (intersection_count == 0) return;

  for (int i = 1; i < intersection_count; i++) {
    float key_t = intersection_t[i];
    int   key_p = intersection_prim_id[i];
    float key_decay = intersection_decay[i];
    float key_dx = intersection_dx[i], key_dy = intersection_dy[i], key_dz = intersection_dz[i];
    float key_qlx = intersection_qlx[i], key_qly = intersection_qly[i], key_qlz = intersection_qlz[i];
    int j = i - 1;
    while (j >= 0 && intersection_t[j] > key_t) {
      intersection_t[j+1] = intersection_t[j];
      intersection_prim_id[j+1] = intersection_prim_id[j];
      intersection_decay[j+1] = intersection_decay[j];
      intersection_dx[j+1] = intersection_dx[j];
      intersection_dy[j+1] = intersection_dy[j];
      intersection_dz[j+1] = intersection_dz[j];
      intersection_qlx[j+1] = intersection_qlx[j];
      intersection_qly[j+1] = intersection_qly[j];
      intersection_qlz[j+1] = intersection_qlz[j];
      j--;
    }
    intersection_t[j+1] = key_t;
    intersection_prim_id[j+1] = key_p;
    intersection_decay[j+1] = key_decay;
    intersection_dx[j+1] = key_dx;
    intersection_dy[j+1] = key_dy;
    intersection_dz[j+1] = key_dz;
    intersection_qlx[j+1] = key_qlx;
    intersection_qly[j+1] = key_qly;
    intersection_qlz[j+1] = key_qlz;
  }

  float trans = 1.f;
  const float* gfeat_pix = grad_out_feat + (size_t)pix * Cfeat;

  for (int i = 0; i < intersection_count; ++i) {
    const int p = intersection_prim_id[i];
    const float decay = intersection_decay[i];
    const float dx = intersection_dx[i];
    const float dy = intersection_dy[i];
    const float dz = intersection_dz[i];
    const float qlx = intersection_qlx[i];
    const float qly = intersection_qly[i];
    const float qlz = intersection_qlz[i];
    float v6[6] = {rdx, rdy, rdz, dx, dy, dz};

    const float* W1p = W1 + (size_t)p * H * IN;
    const float* b1p = b1 + (size_t)p * H;
    const float* W2p = W2 + (size_t)p * H * H;
    const float* b2p = b2 + (size_t)p * H;
    const float* W3p = W3 + (size_t)p * 4 * H;
    const float* b3p = b3 + (size_t)p * 4;
    float R, G, Bc, sigma_raw;
    mlp_eval_rgb_sigma(freq_bands, attenuation, F, W1p, b1p, W2p, b2p, W3p, b3p, H, IN, v6, R, G, Bc, sigma_raw);

    const float sigma = sigma_raw * decay;
    const float alpha = 1.f - __expf(-sigma);
    const float w = trans * alpha;
    if (w <= 0.f) {
      trans *= (1.f - alpha);
      if (trans < 1e-3f) break;
      continue;
    }

    if (feat_slots <= 1) {
      float* gfeat_prim = grad_feat_attr + (size_t)p * Cfeat;
      for (int c = 0; c < Cfeat; ++c) {
        atomicAdd(&gfeat_prim[c], w * gfeat_pix[c]);
      }
    } else if (feat_slots == 5) {
      const float lgx = log_scales[p*3+0];
      const float lgy = log_scales[p*3+1];
      const float lgz = log_scales[p*3+2];
      float g0, g1, g2, g3;
      semantic_tetra_gates(qlx, qly, qlz, lgx, lgy, lgz, gating_beta, gating_temp, g0, g1, g2, g3);
      float* gbase = grad_feat_attr + ((size_t)p * feat_slots + 0) * Cfeat;
      float* gd0 = grad_feat_attr + ((size_t)p * feat_slots + 1) * Cfeat;
      float* gd1 = grad_feat_attr + ((size_t)p * feat_slots + 2) * Cfeat;
      float* gd2 = grad_feat_attr + ((size_t)p * feat_slots + 3) * Cfeat;
      float* gd3 = grad_feat_attr + ((size_t)p * feat_slots + 4) * Cfeat;
      for (int c = 0; c < Cfeat; ++c) {
        const float gc = w * gfeat_pix[c];
        atomicAdd(&gbase[c], gc);
        atomicAdd(&gd0[c], gc * g0);
        atomicAdd(&gd1[c], gc * g1);
        atomicAdd(&gd2[c], gc * g2);
        atomicAdd(&gd3[c], gc * g3);
      }
    } else if (feat_slots == 4) {
      const float lgx = log_scales[p*3+0];
      const float lgy = log_scales[p*3+1];
      const float lgz = log_scales[p*3+2];
      float g0, g1, g2, g3;
      semantic_tetra_gates(qlx, qly, qlz, lgx, lgy, lgz, gating_beta, gating_temp, g0, g1, g2, g3);
      float* gs0 = grad_feat_attr + ((size_t)p * feat_slots + 0) * Cfeat;
      float* gs1 = grad_feat_attr + ((size_t)p * feat_slots + 1) * Cfeat;
      float* gs2 = grad_feat_attr + ((size_t)p * feat_slots + 2) * Cfeat;
      float* gs3 = grad_feat_attr + ((size_t)p * feat_slots + 3) * Cfeat;
      for (int c = 0; c < Cfeat; ++c) {
        const float gc = w * gfeat_pix[c];
        atomicAdd(&gs0[c], gc * g0);
        atomicAdd(&gs1[c], gc * g1);
        atomicAdd(&gs2[c], gc * g2);
        atomicAdd(&gs3[c], gc * g3);
      }
    } else {
      float* gfeat_prim = grad_feat_attr + (size_t)p * feat_slots * Cfeat;
      for (int c = 0; c < Cfeat; ++c) {
        atomicAdd(&gfeat_prim[c], w * gfeat_pix[c]);
      }
    }

    trans *= (1.f - alpha);
    if (trans < 1e-3f) break;
  }
}

// Backward (per-pixel): count pass
// EN: Backward prepass kernel that counts valid contributions per pixel (per-pixel order).
__global__ void render_tiles_backward_count_per_pixel(
    const float* __restrict__ rays_o,
    const float* __restrict__ rays_d,
    const float* __restrict__ pos,
    const float* __restrict__ log_scales,
    const float* __restrict__ quats,
    const int32_t* __restrict__ culled_prim_indices,
    const int32_t* __restrict__ tile_offsets,
    const float* __restrict__ freq_bands,
    const float* __restrict__ attenuation,
    const int F,
    const float* __restrict__ W1, const float* __restrict__ b1,
    const float* __restrict__ W2, const float* __restrict__ b2,
    const float* __restrict__ W3, const float* __restrict__ b3,
    const int N, const int H, const int IN,
    const int Wimg, const int Himg, const int TW, const int TH,
    const float thr,
    int32_t* __restrict__ out_counts,
    int32_t* __restrict__ intersection_overflow)
{
  const int tile_x = blockIdx.x;
  const int tile_y = blockIdx.y;
  const int x = tile_x * TW + threadIdx.x;
  const int y = tile_y * TH + threadIdx.y;
  if (x >= Wimg || y >= Himg) return;
  const int pix = y * Wimg + x;
  const int tile_id = tile_y * ((Wimg + TW - 1)/TW) + tile_x;

  const float rox = rays_o[pix*3+0];
  const float roy = rays_o[pix*3+1];
  const float roz = rays_o[pix*3+2];
  const float rdx = rays_d[pix*3+0];
  const float rdy = rays_d[pix*3+1];
  const float rdz = rays_d[pix*3+2];

  const int MAX_INTERSECTIONS = 256;
  float intersection_t[MAX_INTERSECTIONS];
  int   intersection_prim_id[MAX_INTERSECTIONS];
  float intersection_dx[MAX_INTERSECTIONS];
  float intersection_dy[MAX_INTERSECTIONS];
  float intersection_dz[MAX_INTERSECTIONS];
  float intersection_decay[MAX_INTERSECTIONS];
  int intersection_count = 0;

  const int32_t prim_start = tile_offsets[tile_id];
  const int32_t prim_end = tile_offsets[tile_id + 1];

  for (int k = prim_start; k < prim_end; ++k) {
    const int p = culled_prim_indices[k];
    float t, dx, dy, dz, qlx, qly, qlz, decay;
    const float cx = pos[p*3+0];
    const float cy = pos[p*3+1];
    const float cz = pos[p*3+2];
    const float lgx = log_scales[p*3+0];
    const float lgy = log_scales[p*3+1];
    const float lgz = log_scales[p*3+2];
    const float qw  = quats[p*4+0];
    const float qx  = quats[p*4+1];
    const float qy  = quats[p*4+2];
    const float qz  = quats[p*4+3];
    if (hit_and_features_rot(rox, roy, roz, rdx, rdy, rdz,
                             cx, cy, cz,
                             lgx, lgy, lgz,
                             qw, qx, qy, qz,
                             thr,
                             t, dx, dy, dz,
                             qlx, qly, qlz,
                             decay)) {
      if (intersection_count < MAX_INTERSECTIONS) {
        intersection_t[intersection_count] = t;
        intersection_prim_id[intersection_count] = p;
        intersection_dx[intersection_count] = dx;
        intersection_dy[intersection_count] = dy;
        intersection_dz[intersection_count] = dz;
        intersection_decay[intersection_count] = decay;
        intersection_count++;
      } else {
        atomicExch(intersection_overflow, 1);
        break;
      }
    }
  }

  if (intersection_count == 0) {
    out_counts[pix] = 0;
    return;
  }

  // sort by t
  for (int i = 1; i < intersection_count; i++) {
    float key_t = intersection_t[i];
    int key_p = intersection_prim_id[i];
    float key_dx = intersection_dx[i], key_dy = intersection_dy[i], key_dz = intersection_dz[i];
    float key_decay = intersection_decay[i];
    int j = i - 1;
    while (j >= 0 && intersection_t[j] > key_t) {
      intersection_t[j+1] = intersection_t[j];
      intersection_prim_id[j+1] = intersection_prim_id[j];
      intersection_dx[j+1] = intersection_dx[j];
      intersection_dy[j+1] = intersection_dy[j];
      intersection_dz[j+1] = intersection_dz[j];
      intersection_decay[j+1] = intersection_decay[j];
      j--;
    }
    intersection_t[j+1] = key_t;
    intersection_prim_id[j+1] = key_p;
    intersection_dx[j+1] = key_dx;
    intersection_dy[j+1] = key_dy;
    intersection_dz[j+1] = key_dz;
    intersection_decay[j+1] = key_decay;
  }

  // Reduce per-thread local scratch footprint to curb local-memory traffic and spilling.
  constexpr int MAX_HITS = 256;
  float trans = 1.f;
  int used = 0;
  for (int i = 0; i < intersection_count; ++i) {
    const int p = intersection_prim_id[i];
    const float dx = intersection_dx[i];
    const float dy = intersection_dy[i];
    const float dz = intersection_dz[i];
    const float decay = intersection_decay[i];
    const float v6[6] = {rdx, rdy, rdz, dx, dy, dz};

    const float* W1p = W1 + (size_t)p * H * IN;
    const float* b1p = b1 + (size_t)p * H;
    const float* W2p = W2 + (size_t)p * H * H;
    const float* b2p = b2 + (size_t)p * H;
    const float* W3p = W3 + (size_t)p * 4 * H;
    const float* b3p = b3 + (size_t)p * 4;

    float R, G, Bc, sigma_raw;
    mlp_eval_rgb_sigma(freq_bands, attenuation, F, W1p, b1p, W2p, b2p, W3p, b3p, H, IN, v6, R, G, Bc, sigma_raw);
    const float sigma = sigma_raw * decay;
    const float alpha = 1.f - __expf(-sigma);
    trans *= (1.f - alpha);
    used++;
    if (trans < 1e-3f) break;
  }
  out_counts[pix] = used;
}

// Backward (per-pixel): emit pass
// EN: Backward emit kernel that materializes per-hit records and geometry grads (per-pixel order).
__global__ void render_tiles_backward_emit_per_pixel(
    const float* __restrict__ rays_o,
    const float* __restrict__ rays_d,
    const float* __restrict__ pos,
    const float* __restrict__ log_scales,
    const float* __restrict__ quats,
    const int32_t* __restrict__ culled_prim_indices,
    const int32_t* __restrict__ tile_offsets,
    const float* __restrict__ freq_bands,
    const float* __restrict__ attenuation,
    const int F,
    const float* __restrict__ W1, const float* __restrict__ b1,
    const float* __restrict__ W2, const float* __restrict__ b2,
    const float* __restrict__ W3, const float* __restrict__ b3,
    const int N, const int H, const int IN,
    const int Wimg, const int Himg, const int TW, const int TH,
    const float thr,
    const float* __restrict__ grad_out_rgb,
    const float* __restrict__ grad_T_end,
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ offsets,
    int32_t* __restrict__ out_prim_id,
    float* __restrict__ out_x6,
    float* __restrict__ out_grad_rgb,
    float* __restrict__ out_grad_sigma_raw,
    float* __restrict__ grad_positions,
    float* __restrict__ grad_log_scales,
    float* __restrict__ grad_quats,
    int32_t* __restrict__ intersection_overflow)
{
  const int tile_x = blockIdx.x;
  const int tile_y = blockIdx.y;
  const int x = tile_x * TW + threadIdx.x;
  const int y = tile_y * TH + threadIdx.y;
  if (x >= Wimg || y >= Himg) return;
  const int pix = y * Wimg + x;
  const int tile_id = tile_y * ((Wimg + TW - 1)/TW) + tile_x;

  const float rox = rays_o[pix*3+0];
  const float roy = rays_o[pix*3+1];
  const float roz = rays_o[pix*3+2];
  const float rdx = rays_d[pix*3+0];
  const float rdy = rays_d[pix*3+1];
  const float rdz = rays_d[pix*3+2];

  const float grad_r = grad_out_rgb[pix*3+0];
  const float grad_g = grad_out_rgb[pix*3+1];
  const float grad_b = grad_out_rgb[pix*3+2];
  const float gT_end = grad_T_end[pix];

  const int MAX_INTERSECTIONS = 256;
  float intersection_t[MAX_INTERSECTIONS];
  int   intersection_prim_id[MAX_INTERSECTIONS];
  float intersection_dx[MAX_INTERSECTIONS];
  float intersection_dy[MAX_INTERSECTIONS];
  float intersection_dz[MAX_INTERSECTIONS];
  float intersection_decay[MAX_INTERSECTIONS];
  int intersection_count = 0;

  const int32_t prim_start = tile_offsets[tile_id];
  const int32_t prim_end = tile_offsets[tile_id + 1];

  for (int k = prim_start; k < prim_end; ++k) {
    const int p = culled_prim_indices[k];
    float t, dx, dy, dz, qlx, qly, qlz, decay;
    const float cx = pos[p*3+0];
    const float cy = pos[p*3+1];
    const float cz = pos[p*3+2];
    const float lgx = log_scales[p*3+0];
    const float lgy = log_scales[p*3+1];
    const float lgz = log_scales[p*3+2];
    const float qw  = quats[p*4+0];
    const float qx  = quats[p*4+1];
    const float qy  = quats[p*4+2];
    const float qz  = quats[p*4+3];
    if (hit_and_features_rot(rox, roy, roz, rdx, rdy, rdz,
                             cx, cy, cz,
                             lgx, lgy, lgz,
                             qw, qx, qy, qz,
                             thr,
                             t, dx, dy, dz,
                             qlx, qly, qlz,
                             decay)) {
      if (intersection_count < MAX_INTERSECTIONS) {
        intersection_t[intersection_count] = t;
        intersection_prim_id[intersection_count] = p;
        intersection_dx[intersection_count] = dx;
        intersection_dy[intersection_count] = dy;
        intersection_dz[intersection_count] = dz;
        intersection_decay[intersection_count] = decay;
        intersection_count++;
      } else {
        atomicExch(intersection_overflow, 1);
        break;
      }
    }
  }
  if (intersection_count == 0) return;

  // sort by t
  for (int i = 1; i < intersection_count; i++) {
    float key_t = intersection_t[i];
    int key_p = intersection_prim_id[i];
    float key_dx = intersection_dx[i], key_dy = intersection_dy[i], key_dz = intersection_dz[i];
    float key_decay = intersection_decay[i];
    int j = i - 1;
    while (j >= 0 && intersection_t[j] > key_t) {
      intersection_t[j+1] = intersection_t[j];
      intersection_prim_id[j+1] = intersection_prim_id[j];
      intersection_dx[j+1] = intersection_dx[j];
      intersection_dy[j+1] = intersection_dy[j];
      intersection_dz[j+1] = intersection_dz[j];
      intersection_decay[j+1] = intersection_decay[j];
      j--;
    }
    intersection_t[j+1] = key_t;
    intersection_prim_id[j+1] = key_p;
    intersection_dx[j+1] = key_dx;
    intersection_dy[j+1] = key_dy;
    intersection_dz[j+1] = key_dz;
    intersection_decay[j+1] = key_decay;
  }

  float trans_arr[MAX_INTERSECTIONS];
  float alpha_arr[MAX_INTERSECTIONS];
  float sigma_raw_arr[MAX_INTERSECTIONS];
  float decay_arr[MAX_INTERSECTIONS];
  float dotg_arr[MAX_INTERSECTIONS];
  int prim_arr[MAX_INTERSECTIONS];
  float dx_arr[MAX_INTERSECTIONS];
  float dy_arr[MAX_INTERSECTIONS];
  float dz_arr[MAX_INTERSECTIONS];

  float trans = 1.f;
  int used = 0;
  for (int i = 0; i < intersection_count; ++i) {
    const int p = intersection_prim_id[i];
    const float dx = intersection_dx[i];
    const float dy = intersection_dy[i];
    const float dz = intersection_dz[i];
    const float decay = intersection_decay[i];
    const float v6[6] = {rdx, rdy, rdz, dx, dy, dz};

    const float* W1p = W1 + (size_t)p * H * IN;
    const float* b1p = b1 + (size_t)p * H;
    const float* W2p = W2 + (size_t)p * H * H;
    const float* b2p = b2 + (size_t)p * H;
    const float* W3p = W3 + (size_t)p * 4 * H;
    const float* b3p = b3 + (size_t)p * 4;

    float R, G, Bc, sigma_raw;
    mlp_eval_rgb_sigma(freq_bands, attenuation, F, W1p, b1p, W2p, b2p, W3p, b3p, H, IN, v6, R, G, Bc, sigma_raw);
    const float sigma = sigma_raw * decay;
    const float alpha = 1.f - __expf(-sigma);
    const float dotg = dot3(grad_r, grad_g, grad_b, R, G, Bc);

    trans_arr[used] = trans;
    alpha_arr[used] = alpha;
    sigma_raw_arr[used] = sigma_raw;
    decay_arr[used] = decay;
    dotg_arr[used] = dotg;
    prim_arr[used] = p;
    dx_arr[used] = dx;
    dy_arr[used] = dy;
    dz_arr[used] = dz;

    trans *= (1.f - alpha);
    used++;
    if (trans < 1e-3f) break;
  }

  const int max_emit = counts[pix];
  if (used > max_emit) used = max_emit;

  // reverse pass for grad_alpha and geometry
  const int base = offsets[pix];
  float gT_next = gT_end;
  for (int i = used - 1; i >= 0; --i) {
    const float trans_i = trans_arr[i];
    const float alpha_i = alpha_arr[i];
    const float dotg = dotg_arr[i];
    const int p = prim_arr[i];

    const float grad_alpha = trans_i * (dotg - gT_next);
    const float gT_i = gT_next * (1.f - alpha_i) + dotg * alpha_i;
    gT_next = gT_i;

    const float sigma_raw = sigma_raw_arr[i];
    const float decay = decay_arr[i];
    const float sigma = sigma_raw * decay;
    const float exp_neg_sigma = __expf(-sigma);
    const float grad_sigma = grad_alpha * exp_neg_sigma;
    const float grad_sigma_raw = grad_sigma * decay;

    out_grad_sigma_raw[base + i] = grad_sigma_raw;

    // geometry grads (rotated ellipsoid)
    const float lgx = log_scales[p*3+0];
    const float lgy = log_scales[p*3+1];
    const float lgz = log_scales[p*3+2];
    const float qw  = quats[p*4+0];
    const float qx  = quats[p*4+1];
    const float qy  = quats[p*4+2];
    const float qz  = quats[p*4+3];
    const float mx = __expf(-2.f * lgx);
    const float my = __expf(-2.f * lgy);
    const float mz = __expf(-2.f * lgz);
    const float grad_decay = grad_sigma * sigma_raw;
    const float3 d = make_float3(dx_arr[i], dy_arr[i], dz_arr[i]);
    const float3 qloc = quat_rotate_conj(qw, qx, qy, qz, d);
    const float qlx = qloc.x;
    const float qly = qloc.y;
    const float qlz = qloc.z;

    // dL/d(qloc)
    const float gqx_loc = -grad_decay * decay * mx * qlx;
    const float gqy_loc = -grad_decay * decay * my * qly;
    const float gqz_loc = -grad_decay * decay * mz * qlz;

    // dL/dd = R * dL/d(qloc)
    const float3 grad_loc = make_float3(gqx_loc, gqy_loc, gqz_loc);
    const float3 grad_d = quat_rotate(qw, qx, qy, qz, grad_loc);
    atomicAdd(&grad_positions[p*3+0], grad_d.x);
    atomicAdd(&grad_positions[p*3+1], grad_d.y);
    atomicAdd(&grad_positions[p*3+2], grad_d.z);

    // dL/dlog_scales
    atomicAdd(&grad_log_scales[p*3+0], grad_decay * decay * mx * qlx * qlx);
    atomicAdd(&grad_log_scales[p*3+1], grad_decay * decay * my * qly * qly);
    atomicAdd(&grad_log_scales[p*3+2], grad_decay * decay * mz * qlz * qlz);

    // dL/dquats
    const float gR00 = d.x * gqx_loc; const float gR01 = d.x * gqy_loc; const float gR02 = d.x * gqz_loc;
    const float gR10 = d.y * gqx_loc; const float gR11 = d.y * gqy_loc; const float gR12 = d.y * gqz_loc;
    const float gR20 = d.z * gqx_loc; const float gR21 = d.z * gqy_loc; const float gR22 = d.z * gqz_loc;
    float gqw, ggqx, ggqy, ggqz;
    rotmat_grad_to_quat_grad(qw, qx, qy, qz,
                             gR00, gR01, gR02,
                             gR10, gR11, gR12,
                             gR20, gR21, gR22,
                             gqw, ggqx, ggqy, ggqz);
    atomicAdd(&grad_quats[p*4+0], gqw);
    atomicAdd(&grad_quats[p*4+1], ggqx);
    atomicAdd(&grad_quats[p*4+2], ggqy);
    atomicAdd(&grad_quats[p*4+3], ggqz);
  }

  // emit records (forward order)
  for (int i = 0; i < used; ++i) {
    const int p = prim_arr[i];
    out_prim_id[base + i] = p;
    const float trans_i = trans_arr[i];
    const float alpha_i = alpha_arr[i];
    const float w = trans_i * alpha_i;
    out_grad_rgb[(base + i)*3 + 0] = grad_r * w;
    out_grad_rgb[(base + i)*3 + 1] = grad_g * w;
    out_grad_rgb[(base + i)*3 + 2] = grad_b * w;
    out_x6[(base + i)*6 + 0] = rdx;
    out_x6[(base + i)*6 + 1] = rdy;
    out_x6[(base + i)*6 + 2] = rdz;
    out_x6[(base + i)*6 + 3] = dx_arr[i];
    out_x6[(base + i)*6 + 4] = dy_arr[i];
    out_x6[(base + i)*6 + 5] = dz_arr[i];
  }
}

// Kernel: tile-level approximate ordering (tile-center ray) + prim-major traversal.
// NOTE: Blending order becomes tile-shared (approximate). Per-pixel t is still used inside hit_and_features
// for distance/feature computation (not for ordering).
template <typename WType>
// EN: Forward rendering kernel using tile-center primitive ordering with shared-weight staging.
__global__ void render_tiles_kernel_tilecenter_order(
    const float* __restrict__ rays_o,
    const float* __restrict__ rays_d,
    const float* __restrict__ pos,
    const float* __restrict__ log_scales,
    const float* __restrict__ quats,
    const int32_t* __restrict__ culled_prim_indices,
    const int32_t* __restrict__ tile_offsets,
    const float* __restrict__ freq_bands,
    const float* __restrict__ attenuation,
    const int F,
    const WType* __restrict__ W1, const WType* __restrict__ b1,
    const WType* __restrict__ W2, const WType* __restrict__ b2,
    const WType* __restrict__ W3, const WType* __restrict__ b3,
    const int N, const int H, const int IN,
    const int Wimg, const int Himg, const int TW, const int TH,
    const float thr,
    float* __restrict__ out_rgb,
    float* __restrict__ out_alpha
){
  const int tile_x = blockIdx.x;
  const int tile_y = blockIdx.y;
  // Use a 1D block of (TW*TH) threads so CUB block primitives can be used safely.
  const int tid = (int)threadIdx.x;
  const int tile_threads = TW * TH;
  const bool in_tile_thread = (tid < tile_threads);
  const int lx = in_tile_thread ? (tid % TW) : 0;
  const int ly = in_tile_thread ? (tid / TW) : 0;
  const int x = tile_x * TW + lx;
  const int y = tile_y * TH + ly;
  const bool valid_pix = in_tile_thread && (x < Wimg) && (y < Himg);
  const int pix = valid_pix ? (y * Wimg + x) : 0;

  const int num_tiles_x = (Wimg + TW - 1) / TW;
  const int tile_id = tile_y * num_tiles_x + tile_x;

  // per-pixel ray
  const float rox = valid_pix ? rays_o[pix*3+0] : 0.f;
  const float roy = valid_pix ? rays_o[pix*3+1] : 0.f;
  const float roz = valid_pix ? rays_o[pix*3+2] : 0.f;
  const float rdx = valid_pix ? rays_d[pix*3+0] : 0.f;
  const float rdy = valid_pix ? rays_d[pix*3+1] : 0.f;
  const float rdz = valid_pix ? rays_d[pix*3+2] : 0.f;

  // tile-center ray for ordering key
  const int cxp = min(tile_x * TW + (TW >> 1), Wimg - 1);
  const int cyp = min(tile_y * TH + (TH >> 1), Himg - 1);
  const int pix_c = cyp * Wimg + cxp;
  const float crox = rays_o[pix_c*3+0];
  const float croy = rays_o[pix_c*3+1];
  const float croz = rays_o[pix_c*3+2];
  const float crdx = rays_d[pix_c*3+0];
  const float crdy = rays_d[pix_c*3+1];
  const float crdz = rays_d[pix_c*3+2];


  const int32_t prim_start = tile_offsets[tile_id];
  const int32_t prim_end = tile_offsets[tile_id + 1];
  const int tile_len = (int)(prim_end - prim_start);

  float3 C = make_float3(0.f, 0.f, 0.f);
  float trans = 1.f;

  if (tile_len <= 0) {
    if (valid_pix) {
      out_rgb[pix*3+0] = 0.f;
      out_rgb[pix*3+1] = 0.f;
      out_rgb[pix*3+2] = 0.f;
      out_alpha[pix] = 0.f;
    }
    return;
  }

  // Block-level radix sort uses a fixed 256-thread block; only the first TW*TH threads map to pixels
  // while the remaining threads participate in sort/staging work.
  constexpr int MAX_TILE_PRIMS = MLPSP_MAX_TILE_PRIMS;
  constexpr int BLOCK_THREADS = 256;
  constexpr int ITEMS_PER_THREAD = MAX_TILE_PRIMS / BLOCK_THREADS; // 3
  static_assert(MAX_TILE_PRIMS % BLOCK_THREADS == 0, "MAX_TILE_PRIMS must be divisible by BLOCK_THREADS");

  __shared__ int s_pid[MAX_TILE_PRIMS];
  using BlockRadixSortT = cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD, int>;
  __shared__ typename BlockRadixSortT::TempStorage sort_storage;

  const int num_threads = (int)blockDim.x;
  const int load_n = tile_len < MAX_TILE_PRIMS ? tile_len : MAX_TILE_PRIMS;

  // Load tile list + compute key on tile-center ray, then radix-sort by key (ascending).
  // We use CUB's row-major ("blocked") arrangement: thread 0 owns items [0,1], thread 1 owns [2,3], ...
  // Use float keys directly.
  float keys[ITEMS_PER_THREAD];
  int vals[ITEMS_PER_THREAD];

#pragma unroll
  for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
    const int i = tid * ITEMS_PER_THREAD + it;
    if (i < load_n) {
      const int p = (int)culled_prim_indices[prim_start + i];
      vals[it] = p;

      /*
      const float vx = pos[p*3+0] - crox;
      const float vy = pos[p*3+1] - croy;
      const float vz = pos[p*3+2] - croz;
      const float tkey = fmaf(vx, crdx, fmaf(vy, crdy, vz * crdz));
      */


      const float cx = pos[p*3+0];
      const float cy = pos[p*3+1];
      const float cz = pos[p*3+2];
      const float lgx = log_scales[p*3+0];
      const float lgy = log_scales[p*3+1];
      const float lgz = log_scales[p*3+2];
      const float qw  = quats[p*4+0];
      const float qx  = quats[p*4+1];
      const float qy  = quats[p*4+2];
      const float qz  = quats[p*4+3];
      /*
      const float tkey = tstar_key_tilecenter(
          crox, croy, croz, crdx, crdy, crdz,
          cx, cy, cz,
          lgx, lgy, lgz,
          qw, qx, qy, qz
      );

      keys[it] = (tkey > 0.f) ? tkey : 1e30f;*/

      const float tkey = tstar_key_tilecenter(
          crox, croy, croz, crdx, crdy, crdz,
          cx, cy, cz,
          lgx, lgy, lgz,
          qw, qx, qy, qz
      );
      keys[it] =(tkey > 0.f) ? tkey : 1e30f;
    } else {
      vals[it] = -1;
      keys[it] = 1e30f;
    }
  }


  // Radix sort across the block; output remains in row-major ("blocked") layout.
  BlockRadixSortT(sort_storage).Sort(keys, vals);
  __syncthreads();

#pragma unroll
  for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
    const int i = tid * ITEMS_PER_THREAD + it;
    s_pid[i] = vals[it];
  }
  __syncthreads();

  // dynamic shared memory for weights
  extern __shared__ float sW[];
  const int Hcap = H > MLPSP_MAX_H ? MLPSP_MAX_H : H;
  const int W1n = Hcap * IN;
  const int b1n = Hcap;
  const int W2n = Hcap * Hcap;
  const int b2n = Hcap;
  const int W3n = 4 * Hcap;
  const int b3n = 4;
  float* sW1 = sW;
  float* sb1 = sW1 + W1n;
  float* sW2 = sb1 + b1n;
  float* sb2 = sW2 + W2n;
  float* sW3 = sb2 + b2n;
  float* sb3 = sW3 + W3n;

  auto load_weights = [&](int p) {
    const WType* W1p = W1 + (size_t)p * H * IN;
    const WType* b1p = b1 + (size_t)p * H;
    const WType* W2p = W2 + (size_t)p * H * H;
    const WType* b2p = b2 + (size_t)p * H;
    const WType* W3p = W3 + (size_t)p * 4 * H;
    const WType* b3p = b3 + (size_t)p * 4;

    cast_copy_strided<WType>(sW1, W1p, W1n, tid, num_threads);
    cast_copy_strided<WType>(sb1, b1p, b1n, tid, num_threads);
    cast_copy_strided<WType>(sW2, W2p, W2n, tid, num_threads);
    cast_copy_strided<WType>(sb2, b2p, b2n, tid, num_threads);
    cast_copy_strided<WType>(sW3, W3p, W3n, tid, num_threads);
    cast_copy_strided<WType>(sb3, b3p, b3n, tid, num_threads);
    __syncthreads();
  };

  auto eval_and_blend = [&](int p) {
    if (valid_pix && (trans >= 1e-3f)) {
      float t, dx, dy, dz, decay;
      const float cx = pos[p*3+0];
      const float cy = pos[p*3+1];
      const float cz = pos[p*3+2];
      const float lgx = log_scales[p*3+0];
      const float lgy = log_scales[p*3+1];
      const float lgz = log_scales[p*3+2];
      const float qw  = quats[p*4+0];
      const float qx  = quats[p*4+1];
      const float qy  = quats[p*4+2];
      const float qz  = quats[p*4+3];
      float qlx, qly, qlz;
      if (hit_and_features_rot(rox, roy, roz, rdx, rdy, rdz,
                               cx, cy, cz,
                               lgx, lgy, lgz,
                               qw, qx, qy, qz,
                               thr,
                               t, dx, dy, dz,
                               qlx, qly, qlz,
                               decay)) {
        float v6[6] = {rdx, rdy, rdz, dx, dy, dz};
        // ---- layer1: IN -> H (positional-encoding input) ----
        float h1[MLPSP_MAX_H];
        for (int h_idx = 0; h_idx < Hcap; ++h_idx) h1[h_idx] = sb1[h_idx];

        const int cos_offset = IN / 2;
        for (int c = 0; c < 6; ++c) {
          for (int f = 0; f < F; ++f) {
            const float fb = freq_bands[f];
            const float att = attenuation[f];
            const float xproj = v6[c] * fb * att;
            const int j = c * F + f;
            const float s  = __sinf(xproj);
            const float co = __cosf(xproj);

            // accumulate into h1
            for (int h_idx = 0; h_idx < Hcap; ++h_idx) {
              const float* Wi = sW1 + (size_t)h_idx * IN;
              h1[h_idx] = fmaf(Wi[j],            s,  h1[h_idx]);
              h1[h_idx] = fmaf(Wi[j + cos_offset], co, h1[h_idx]);
            }
          }
        }
        for (int h_idx = 0; h_idx < Hcap; ++h_idx) h1[h_idx] = h1[h_idx] > 0.f ? h1[h_idx] : 0.f;

        // ---- layer2: H -> H ----
        float h2[MLPSP_MAX_H];
        for (int o = 0; o < Hcap; ++o) {
          float acc = sb2[o];
          const float* W2row = sW2 + (size_t)o * Hcap;
          for (int h_idx = 0; h_idx < Hcap; ++h_idx) acc = fmaf(W2row[h_idx], h1[h_idx], acc);
          h2[o] = acc > 0.f ? acc : 0.f;
        }

        // ---- layer3: H -> OUT(4) ----
        float o3 = sb3[0], o4 = sb3[1], o5 = sb3[2], o6 = sb3[3];
        for (int h_idx = 0; h_idx < Hcap; ++h_idx) {
          const float hi = h2[h_idx];
          o3 = fmaf(sW3[0*Hcap + h_idx], hi, o3);
          o4 = fmaf(sW3[1*Hcap + h_idx], hi, o4);
          o5 = fmaf(sW3[2*Hcap + h_idx], hi, o5);
          o6 = fmaf(sW3[3*Hcap + h_idx], hi, o6);
        }

        // ---- activations & compositing ----
        const float sigma_raw = fast_softplus(o6);
        const float sigma = sigma_raw * decay;
        const float alpha = 1.f - __expf(-sigma);
        const float w = trans * alpha;

        const float R  = fast_sigmoid(o3);
        const float G  = fast_sigmoid(o4);
        const float Bc = fast_sigmoid(o5);
        C.x = fmaf(w, R, C.x);
        C.y = fmaf(w, G, C.y);
        C.z = fmaf(w, Bc, C.z);
        trans *= (1.f - alpha);
      }
    }
    __syncthreads();
  };

  // sorted prefix
  for (int i = 0; i < load_n; ++i) {
    const int p = s_pid[i];
    if (p < 0 || p >= N) continue;
    load_weights(p);
    eval_and_blend(p);
    const int active = (valid_pix && (trans >= 1e-3f)) ? 1 : 0;
    if (__syncthreads_or(active) == 0) break;
  }

  // unsorted tail (if any)
  if (tile_len > load_n) {
    for (int i = prim_start + load_n; i < prim_end; ++i) {
      const int p = (int)culled_prim_indices[i];
      if (p < 0 || p >= N) continue;
      load_weights(p);
      eval_and_blend(p);
      const int active = (valid_pix && (trans >= 1e-3f)) ? 1 : 0;
      if (__syncthreads_or(active) == 0) break;
    }
  }

  if (valid_pix) {
    out_rgb[pix*3+0] = C.x;
    out_rgb[pix*3+1] = C.y;
    out_rgb[pix*3+2] = C.z;
    out_alpha[pix]   = 1.f - trans;
  }
}

__global__ void render_tiles_kernel_tilecenter_order_stats(
    const float* __restrict__ rays_o,
    const float* __restrict__ rays_d,
    const float* __restrict__ pos,
    const float* __restrict__ log_scales,
    const float* __restrict__ quats,
    const int32_t* __restrict__ culled_prim_indices,
    const int32_t* __restrict__ tile_offsets,
    const float* __restrict__ freq_bands,
    const float* __restrict__ attenuation,
    const int F,
    const float* __restrict__ W1, const float* __restrict__ b1,
    const float* __restrict__ W2, const float* __restrict__ b2,
    const float* __restrict__ W3, const float* __restrict__ b3,
    const int N, const int H, const int IN,
    const int Wimg, const int Himg, const int TW, const int TH,
    const float thr,
    const float* __restrict__ gt_rgb,              // ✅ NEW: (Wimg*Himg*3)
    float* __restrict__ out_rgb,
    float* __restrict__ out_alpha,
    float* __restrict__ stat_sum_w,
    float* __restrict__ stat_max_w,
    int32_t* __restrict__ stat_hits,
    float* __restrict__ stat_sum_werr,             // ✅ NEW: (N)
    int32_t* __restrict__ stat_contrib_count       // ✅ NEW: (N)
){
  constexpr float W_EPS = 1e-1f;

  const int tile_x = blockIdx.x;
  const int tile_y = blockIdx.y;
  const int tid = (int)threadIdx.x;
  const int tile_threads = TW * TH;
  const bool in_tile_thread = (tid < tile_threads);
  const int lx = in_tile_thread ? (tid % TW) : 0;
  const int ly = in_tile_thread ? (tid / TW) : 0;
  const int x = tile_x * TW + lx;
  const int y = tile_y * TH + ly;
  const bool valid_pix = in_tile_thread && (x < Wimg) && (y < Himg);
  const int pix = valid_pix ? (y * Wimg + x) : 0;

  const int num_tiles_x = (Wimg + TW - 1) / TW;
  const int tile_id = tile_y * num_tiles_x + tile_x;

  // per-pixel ray
  const float rox = valid_pix ? rays_o[pix*3+0] : 0.f;
  const float roy = valid_pix ? rays_o[pix*3+1] : 0.f;
  const float roz = valid_pix ? rays_o[pix*3+2] : 0.f;
  const float rdx = valid_pix ? rays_d[pix*3+0] : 0.f;
  const float rdy = valid_pix ? rays_d[pix*3+1] : 0.f;
  const float rdz = valid_pix ? rays_d[pix*3+2] : 0.f;

  // tile-center ray for ordering key
  const int cxp = min(tile_x * TW + (TW >> 1), Wimg - 1);
  const int cyp = min(tile_y * TH + (TH >> 1), Himg - 1);
  const int pix_c = cyp * Wimg + cxp;
  const float crox = rays_o[pix_c*3+0];
  const float croy = rays_o[pix_c*3+1];
  const float croz = rays_o[pix_c*3+2];
  const float crdx = rays_d[pix_c*3+0];
  const float crdy = rays_d[pix_c*3+1];
  const float crdz = rays_d[pix_c*3+2];

  // GT for this pixel (for pixel_error)
  const float gtR = valid_pix ? gt_rgb[pix*3+0] : 0.f;
  const float gtG = valid_pix ? gt_rgb[pix*3+1] : 0.f;
  const float gtB = valid_pix ? gt_rgb[pix*3+2] : 0.f;

  const int32_t prim_start = tile_offsets[tile_id];
  const int32_t prim_end = tile_offsets[tile_id + 1];
  const int tile_len = (int)(prim_end - prim_start);

  float3 C = make_float3(0.f, 0.f, 0.f);
  float trans = 1.f;

  if (tile_len <= 0) {
    if (valid_pix) {
      out_rgb[pix*3+0] = 0.f;
      out_rgb[pix*3+1] = 0.f;
      out_rgb[pix*3+2] = 0.f;
      out_alpha[pix] = 0.f;
    }
    return;
  }

  constexpr int MAX_TILE_PRIMS = MLPSP_MAX_TILE_PRIMS;
  constexpr int BLOCK_THREADS = 256;
  constexpr int ITEMS_PER_THREAD = MAX_TILE_PRIMS / BLOCK_THREADS; // 3
  static_assert(MAX_TILE_PRIMS % BLOCK_THREADS == 0, "MAX_TILE_PRIMS must be divisible by BLOCK_THREADS");

  __shared__ int s_pid[MAX_TILE_PRIMS];
  using BlockRadixSortT = cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD, int>;
  __shared__ typename BlockRadixSortT::TempStorage sort_storage;

  using BlockReduceF = cub::BlockReduce<float, BLOCK_THREADS>;
  using BlockReduceI = cub::BlockReduce<int,   BLOCK_THREADS>;
  __shared__ typename BlockReduceF::TempStorage reduce_storage_f;
  __shared__ typename BlockReduceI::TempStorage reduce_storage_i;

  const int num_threads = (int)blockDim.x;
  const int load_n = tile_len < MAX_TILE_PRIMS ? tile_len : MAX_TILE_PRIMS;

  float keys[ITEMS_PER_THREAD];
  int vals[ITEMS_PER_THREAD];

#pragma unroll
  for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
    const int i = tid * ITEMS_PER_THREAD + it;
    if (i < load_n) {
      const int p = (int)culled_prim_indices[prim_start + i];
      vals[it] = p;
      const float vx = pos[p*3+0] - crox;
      const float vy = pos[p*3+1] - croy;
      const float vz = pos[p*3+2] - croz;
      const float tkey = fmaf(vx, crdx, fmaf(vy, crdy, vz * crdz));
      keys[it] = (tkey > 0.f) ? tkey : 1e30f;
    } else {
      vals[it] = -1;
      keys[it] = 1e30f;
    }
  }

  BlockRadixSortT(sort_storage).Sort(keys, vals);
  __syncthreads();

#pragma unroll
  for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
    const int i = tid * ITEMS_PER_THREAD + it;
    s_pid[i] = vals[it];
  }
  __syncthreads();

  // dynamic shared memory for weights
  extern __shared__ float sW[];
  const int Hcap = H > 64 ? 64 : H;
  const int W1n = Hcap * IN;
  const int b1n = Hcap;
  const int W2n = Hcap * Hcap;
  const int b2n = Hcap;
  const int W3n = 4 * Hcap;
  const int b3n = 4;
  float* sW1 = sW;
  float* sb1 = sW1 + W1n;
  float* sW2 = sb1 + b1n;
  float* sb2 = sW2 + W2n;
  float* sW3 = sb2 + b2n;
  float* sb3 = sW3 + W3n;
  const int total_floats = W1n + b1n + W2n + b2n + W3n + b3n;

  auto load_weights = [&](int p) {
    const float* W1p = W1 + (size_t)p * H * IN;
    const float* b1p = b1 + (size_t)p * H;
    const float* W2p = W2 + (size_t)p * H * H;
    const float* b2p = b2 + (size_t)p * H;
    const float* W3p = W3 + (size_t)p * 4 * H;
    const float* b3p = b3 + (size_t)p * 4;

    for (int idx = tid; idx < total_floats; idx += num_threads) {
      int off = idx;
      if (off < W1n) { sW1[off] = W1p[off]; continue; }
      off -= W1n;
      if (off < b1n) { sb1[off] = b1p[off]; continue; }
      off -= b1n;
      if (off < W2n) { sW2[off] = W2p[off]; continue; }
      off -= W2n;
      if (off < b2n) { sb2[off] = b2p[off]; continue; }
      off -= b2n;
      if (off < W3n) { sW3[off] = W3p[off]; continue; }
      off -= W3n;
      if (off < b3n) { sb3[off] = b3p[off]; }
    }
    __syncthreads();
  };

  // Accumulate error-weighted and significant-contribution statistics.
  auto eval_and_blend = [&](int p,
                            float &local_w,
                            float &local_werr,
                            int   &local_hit,
                            int   &local_contrib) {
    local_w = 0.f;
    local_werr = 0.f;
    local_hit = 0;
    local_contrib = 0;

    if (valid_pix && (trans >= 1e-3f)) {
      float t, dx, dy, dz, decay;
      const float cx = pos[p*3+0];
      const float cy = pos[p*3+1];
      const float cz = pos[p*3+2];
      const float lgx = log_scales[p*3+0];
      const float lgy = log_scales[p*3+1];
      const float lgz = log_scales[p*3+2];
      const float qw  = quats[p*4+0];
      const float qx  = quats[p*4+1];
      const float qy  = quats[p*4+2];
      const float qz  = quats[p*4+3];
      float qlx, qly, qlz;

      if (hit_and_features_rot(rox, roy, roz, rdx, rdy, rdz,
                               cx, cy, cz,
                               lgx, lgy, lgz,
                               qw, qx, qy, qz,
                               thr,
                               t, dx, dy, dz,
                               qlx, qly, qlz,
                               decay)) {

        float v6[6] = {rdx, rdy, rdz, dx, dy, dz};

        float h1[32];
        for (int h_idx = 0; h_idx < Hcap; ++h_idx) h1[h_idx] = sb1[h_idx];

        const int cos_offset = IN / 2;
        for (int c = 0; c < 6; ++c) {
          for (int f = 0; f < F; ++f) {
            const float fb = freq_bands[f];
            const float att = attenuation[f];
            const float xproj = v6[c] * fb * att;
            const int j = c * F + f;
            const float s = __sinf(xproj);
            const float co = __cosf(xproj);
            for (int h_idx = 0; h_idx < Hcap; ++h_idx) {
              const float* Wi = sW1 + (size_t)h_idx * IN;
              h1[h_idx] = fmaf(Wi[j], s, h1[h_idx]);
              h1[h_idx] = fmaf(Wi[j + cos_offset], co, h1[h_idx]);
            }
          }
        }
        for (int h_idx = 0; h_idx < Hcap; ++h_idx) h1[h_idx] = h1[h_idx] > 0.f ? h1[h_idx] : 0.f;

        float h2[32];
        for (int o = 0; o < Hcap; ++o) {
          float acc = sb2[o];
          const float* W2row = sW2 + (size_t)o * Hcap;
          for (int h_idx = 0; h_idx < Hcap; ++h_idx) acc = fmaf(W2row[h_idx], h1[h_idx], acc);
          h2[o] = acc > 0.f ? acc : 0.f;
        }

        float o3 = sb3[0], o4 = sb3[1], o5 = sb3[2], o6 = sb3[3];
        for (int h_idx = 0; h_idx < Hcap; ++h_idx) {
          const float hi = h2[h_idx];
          o3 = fmaf(sW3[0*Hcap + h_idx], hi, o3);
          o4 = fmaf(sW3[1*Hcap + h_idx], hi, o4);
          o5 = fmaf(sW3[2*Hcap + h_idx], hi, o5);
          o6 = fmaf(sW3[3*Hcap + h_idx], hi, o6);
        }

        const float sigma_raw = fast_softplus(o6);
        const float sigma = sigma_raw * decay;
        const float alpha = 1.f - __expf(-sigma);
        const float w = trans * alpha;

        // ---- stats ----
        local_w = w;
        local_hit = (w > 1e-8f) ? 1 : 0;
        local_contrib = (w > W_EPS) ? 1 : 0;

        // Prediction after the current compositing step, used for pixel error.
        const float R = fast_sigmoid(o3);
        const float G = fast_sigmoid(o4);
        const float Bc = fast_sigmoid(o5);

        const float predR = fmaf(w, R, C.x);
        const float predG = fmaf(w, G, C.y);
        const float predB = fmaf(w, Bc, C.z);

        // pixel_error: L1 (sum abs over RGB)
        const float err = fabsf(predR - gtR) + fabsf(predG - gtG) + fabsf(predB - gtB);

        // Accumulate w * pixel_error only for significant contributions.
        if (w > W_EPS) local_werr = w * err;

        // ---- blend ----
        C.x = predR;
        C.y = predG;
        C.z = predB;
        trans *= (1.f - alpha);
      }
    }
    __syncthreads();
  };

  // sorted prefix
  for (int i = 0; i < load_n; ++i) {
    const int p = s_pid[i];
    if (p < 0 || p >= N) continue;

    load_weights(p);

    float local_w = 0.f;
    float local_werr = 0.f;
    int local_hit = 0;
    int local_contrib = 0;

    eval_and_blend(p, local_w, local_werr, local_hit, local_contrib);

    if (stat_sum_w != nullptr) {
      const float tile_sum_w = BlockReduceF(reduce_storage_f).Sum(local_w);
      __syncthreads();
      const float tile_max_w = BlockReduceF(reduce_storage_f).Reduce(local_w, cub::Max());
      __syncthreads();
      const int tile_hits = BlockReduceI(reduce_storage_i).Sum(local_hit);
      __syncthreads();

      // ✅ NEW
      const float tile_sum_werr = BlockReduceF(reduce_storage_f).Sum(local_werr);
      __syncthreads();
      const int tile_contrib = BlockReduceI(reduce_storage_i).Sum(local_contrib);
      __syncthreads();

      if (tid == 0) {
        atomicAdd(&stat_sum_w[p], tile_sum_w);
        atomicMaxFloatPos(&stat_max_w[p], tile_max_w);
        atomicAdd(&stat_hits[p], tile_hits);

        // ✅ NEW
        if (stat_sum_werr) atomicAdd(&stat_sum_werr[p], tile_sum_werr);
        if (stat_contrib_count) atomicAdd(&stat_contrib_count[p], tile_contrib);
      }
    }

    const int active = (valid_pix && (trans >= 1e-3f)) ? 1 : 0;
    if (__syncthreads_or(active) == 0) break;
  }

  // unsorted tail (if any)
  if (tile_len > load_n) {
    for (int i = prim_start + load_n; i < prim_end; ++i) {
      const int p = (int)culled_prim_indices[i];
      if (p < 0 || p >= N) continue;

      load_weights(p);

      float local_w = 0.f;
      float local_werr = 0.f;
      int local_hit = 0;
      int local_contrib = 0;

      eval_and_blend(p, local_w, local_werr, local_hit, local_contrib);

      if (stat_sum_w != nullptr) {
        const float tile_sum_w = BlockReduceF(reduce_storage_f).Sum(local_w);
        __syncthreads();
        const float tile_max_w = BlockReduceF(reduce_storage_f).Reduce(local_w, cub::Max());
        __syncthreads();
        const int tile_hits = BlockReduceI(reduce_storage_i).Sum(local_hit);
        __syncthreads();

        // ✅ NEW
        const float tile_sum_werr = BlockReduceF(reduce_storage_f).Sum(local_werr);
        __syncthreads();
        const int tile_contrib = BlockReduceI(reduce_storage_i).Sum(local_contrib);
        __syncthreads();

        if (tid == 0) {
          atomicAdd(&stat_sum_w[p], tile_sum_w);
          atomicMaxFloatPos(&stat_max_w[p], tile_max_w);
          atomicAdd(&stat_hits[p], tile_hits);

          // ✅ NEW
          if (stat_sum_werr) atomicAdd(&stat_sum_werr[p], tile_sum_werr);
          if (stat_contrib_count) atomicAdd(&stat_contrib_count[p], tile_contrib);
        }
      }

      const int active = (valid_pix && (trans >= 1e-3f)) ? 1 : 0;
      if (__syncthreads_or(active) == 0) break;
    }
  }

  if (valid_pix) {
    out_rgb[pix*3+0] = C.x;
    out_rgb[pix*3+1] = C.y;
    out_rgb[pix*3+2] = C.z;
    out_alpha[pix]   = 1.f - trans;
  }
}


// Backward (tilecenter): count pass
// EN: Backward prepass kernel that counts valid contributions in tile-center traversal order.
__global__ void render_tiles_backward_count_tilecenter(
    const float* __restrict__ rays_o,
    const float* __restrict__ rays_d,
    const float* __restrict__ pos,
    const float* __restrict__ log_scales,
    const float* __restrict__ quats,
    const int32_t* __restrict__ culled_prim_indices,
    const int32_t* __restrict__ tile_offsets,
    const float* __restrict__ freq_bands,
    const float* __restrict__ attenuation,
    const int F,
    const float* __restrict__ W1, const float* __restrict__ b1,
    const float* __restrict__ W2, const float* __restrict__ b2,
    const float* __restrict__ W3, const float* __restrict__ b3,
    const int N, const int H, const int IN,
    const int Wimg, const int Himg, const int TW, const int TH,
    const float thr,
    int32_t* __restrict__ out_counts)
{
  const int tile_x = blockIdx.x;
  const int tile_y = blockIdx.y;
  const int tid = (int)threadIdx.x;
  const int tile_threads = TW * TH;
  const bool in_tile_thread = (tid < tile_threads);
  const int lx = in_tile_thread ? (tid % TW) : 0;
  const int ly = in_tile_thread ? (tid / TW) : 0;
  const int x = tile_x * TW + lx;
  const int y = tile_y * TH + ly;
  const bool valid_pix = in_tile_thread && (x < Wimg) && (y < Himg);
  const int pix = valid_pix ? (y * Wimg + x) : 0;

  const int num_tiles_x = (Wimg + TW - 1) / TW;
  const int tile_id = tile_y * num_tiles_x + tile_x;

  const float rox = valid_pix ? rays_o[pix*3+0] : 0.f;
  const float roy = valid_pix ? rays_o[pix*3+1] : 0.f;
  const float roz = valid_pix ? rays_o[pix*3+2] : 0.f;
  const float rdx = valid_pix ? rays_d[pix*3+0] : 0.f;
  const float rdy = valid_pix ? rays_d[pix*3+1] : 0.f;
  const float rdz = valid_pix ? rays_d[pix*3+2] : 0.f;

  const int cxp = min(tile_x * TW + (TW >> 1), Wimg - 1);
  const int cyp = min(tile_y * TH + (TH >> 1), Himg - 1);
  const int pix_c = cyp * Wimg + cxp;
  const float crox = rays_o[pix_c*3+0];
  const float croy = rays_o[pix_c*3+1];
  const float croz = rays_o[pix_c*3+2];
  const float crdx = rays_d[pix_c*3+0];
  const float crdy = rays_d[pix_c*3+1];
  const float crdz = rays_d[pix_c*3+2];

  const int32_t prim_start = tile_offsets[tile_id];
  const int32_t prim_end = tile_offsets[tile_id + 1];
  const int tile_len = (int)(prim_end - prim_start);

  if (tile_len <= 0) {
    if (valid_pix) out_counts[pix] = 0;
    return;
  }

  constexpr int MAX_TILE_PRIMS = MLPSP_MAX_TILE_PRIMS;
  constexpr int BLOCK_THREADS = 256;
  constexpr int ITEMS_PER_THREAD = MAX_TILE_PRIMS / BLOCK_THREADS;
  __shared__ int s_pid[MAX_TILE_PRIMS];
  using BlockRadixSortT = cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD, int>;
  __shared__ typename BlockRadixSortT::TempStorage sort_storage;

  const int num_threads = (int)blockDim.x;
  const int load_n = tile_len < MAX_TILE_PRIMS ? tile_len : MAX_TILE_PRIMS;

  float keys[ITEMS_PER_THREAD];
  int vals[ITEMS_PER_THREAD];
#pragma unroll
  for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
    const int i = tid * ITEMS_PER_THREAD + it;
    if (i < load_n) {
      const int p = (int)culled_prim_indices[prim_start + i];
      vals[it] = p;
      const float vx = pos[p*3+0] - crox;
      const float vy = pos[p*3+1] - croy;
      const float vz = pos[p*3+2] - croz;
      const float tkey = fmaf(vx, crdx, fmaf(vy, crdy, vz * crdz));
      keys[it] = (tkey > 0.f) ? tkey : 1e30f;
    } else {
      vals[it] = -1;
      keys[it] = 1e30f;
    }
  }
  BlockRadixSortT(sort_storage).Sort(keys, vals);
  __syncthreads();
#pragma unroll
  for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
    const int i = tid * ITEMS_PER_THREAD + it;
    s_pid[i] = vals[it];
  }
  __syncthreads();

  // Keep no-atomic profiling kernel aligned with full-kernel traversal state size.
  constexpr int MAX_HITS = 256;
  float trans = 1.f;
  int used = 0;

  auto process_prim = [&](int p) {
    if (!valid_pix || trans < 1e-3f || used >= MAX_HITS) return;
    float t, dx, dy, dz, decay;
    const float cx = pos[p*3+0];
    const float cy = pos[p*3+1];
    const float cz = pos[p*3+2];
    const float lgx = log_scales[p*3+0];
      const float lgy = log_scales[p*3+1];
      const float lgz = log_scales[p*3+2];
      const float qw  = quats[p*4+0];
      const float qx  = quats[p*4+1];
      const float qy  = quats[p*4+2];
      const float qz  = quats[p*4+3];
    float qlx, qly, qlz;
      if (hit_and_features_rot(rox, roy, roz, rdx, rdy, rdz,
                               cx, cy, cz,
                               lgx, lgy, lgz,
                               qw, qx, qy, qz,
                               thr,
                               t, dx, dy, dz,
                               qlx, qly, qlz,
                               decay)) {
      const float v6[6] = {rdx, rdy, rdz, dx, dy, dz};
      const float* W1p = W1 + (size_t)p * H * IN;
      const float* b1p = b1 + (size_t)p * H;
      const float* W2p = W2 + (size_t)p * H * H;
      const float* b2p = b2 + (size_t)p * H;
      const float* W3p = W3 + (size_t)p * 4 * H;
      const float* b3p = b3 + (size_t)p * 4;
      float R, G, Bc, sigma_raw;
      mlp_eval_rgb_sigma(freq_bands, attenuation, F, W1p, b1p, W2p, b2p, W3p, b3p, H, IN, v6, R, G, Bc, sigma_raw);
      const float sigma = sigma_raw * decay;
      const float alpha = 1.f - __expf(-sigma);
      trans *= (1.f - alpha);
      used++;
    }
  };

  for (int i = 0; i < load_n; ++i) {
    const int p = s_pid[i];
    if (p >= 0 && p < N) process_prim(p);
    const int active = (valid_pix && (trans >= 1e-3f)) ? 1 : 0;
    if (__syncthreads_or(active) == 0 || used >= MAX_HITS) break;
  }
  if (tile_len > load_n) {
    for (int i = prim_start + load_n; i < prim_end; ++i) {
      const int p = (int)culled_prim_indices[i];
      if (p < 0 || p >= N) continue;
      process_prim(p);
      const int active = (valid_pix && (trans >= 1e-3f)) ? 1 : 0;
      if (__syncthreads_or(active) == 0 || used >= MAX_HITS) break;
    }
  }

  if (valid_pix) out_counts[pix] = used;
}

// Backward (tilecenter): emit pass
// EN: Backward emit kernel for tile-center traversal that outputs hit-level gradients.
__global__ void render_tiles_backward_emit_tilecenter(
    const float* __restrict__ rays_o,
    const float* __restrict__ rays_d,
    const float* __restrict__ pos,
    const float* __restrict__ logs,
    const float* __restrict__ quats,
    const int32_t* __restrict__ culled_prim_indices,
    const int32_t* __restrict__ tile_offsets,
    const float* __restrict__ freq_bands,
    const float* __restrict__ attenuation,
    const int F,
    const float* __restrict__ W1, const float* __restrict__ b1,
    const float* __restrict__ W2, const float* __restrict__ b2,
    const float* __restrict__ W3, const float* __restrict__ b3,
    const int N, const int H, const int IN,
    const int Wimg, const int Himg, const int TW, const int TH,
    const float thr,
    const float* __restrict__ grad_out_rgb,
    const float* __restrict__ grad_T_end,
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ offsets,
    int32_t* __restrict__ out_prim_id,
    float* __restrict__ out_x6,
    float* __restrict__ out_grad_rgb,
    float* __restrict__ out_grad_sigma_raw,
    float* __restrict__ grad_positions,
    float* __restrict__ grad_log_scales)
{
  const int tile_x = blockIdx.x;
  const int tile_y = blockIdx.y;
  const int tid = (int)threadIdx.x;
  const int tile_threads = TW * TH;
  const bool in_tile_thread = (tid < tile_threads);
  const int lx = in_tile_thread ? (tid % TW) : 0;
  const int ly = in_tile_thread ? (tid / TW) : 0;
  const int x = tile_x * TW + lx;
  const int y = tile_y * TH + ly;
  const bool valid_pix = in_tile_thread && (x < Wimg) && (y < Himg);
  const int pix = valid_pix ? (y * Wimg + x) : 0;

  const int num_tiles_x = (Wimg + TW - 1) / TW;
  const int tile_id = tile_y * num_tiles_x + tile_x;

  const float rox = valid_pix ? rays_o[pix*3+0] : 0.f;
  const float roy = valid_pix ? rays_o[pix*3+1] : 0.f;
  const float roz = valid_pix ? rays_o[pix*3+2] : 0.f;
  const float rdx = valid_pix ? rays_d[pix*3+0] : 0.f;
  const float rdy = valid_pix ? rays_d[pix*3+1] : 0.f;
  const float rdz = valid_pix ? rays_d[pix*3+2] : 0.f;

  const float grad_r = valid_pix ? grad_out_rgb[pix*3+0] : 0.f;
  const float grad_g = valid_pix ? grad_out_rgb[pix*3+1] : 0.f;
  const float grad_b = valid_pix ? grad_out_rgb[pix*3+2] : 0.f;
  const float gT_end = valid_pix ? grad_T_end[pix] : 0.f;

  const int cxp = min(tile_x * TW + (TW >> 1), Wimg - 1);
  const int cyp = min(tile_y * TH + (TH >> 1), Himg - 1);
  const int pix_c = cyp * Wimg + cxp;
  const float crox = rays_o[pix_c*3+0];
  const float croy = rays_o[pix_c*3+1];
  const float croz = rays_o[pix_c*3+2];
  const float crdx = rays_d[pix_c*3+0];
  const float crdy = rays_d[pix_c*3+1];
  const float crdz = rays_d[pix_c*3+2];

  const int32_t prim_start = tile_offsets[tile_id];
  const int32_t prim_end = tile_offsets[tile_id + 1];
  const int tile_len = (int)(prim_end - prim_start);

  if (tile_len <= 0) return;

  constexpr int MAX_TILE_PRIMS = MLPSP_MAX_TILE_PRIMS;
  constexpr int BLOCK_THREADS = 256;
  constexpr int ITEMS_PER_THREAD = MAX_TILE_PRIMS / BLOCK_THREADS;
  __shared__ int s_pid[MAX_TILE_PRIMS];
  using BlockRadixSortT = cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD, int>;
  __shared__ typename BlockRadixSortT::TempStorage sort_storage;

  const int num_threads = (int)blockDim.x;
  const int load_n = tile_len < MAX_TILE_PRIMS ? tile_len : MAX_TILE_PRIMS;

  float keys[ITEMS_PER_THREAD];
  int vals[ITEMS_PER_THREAD];
#pragma unroll
  for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
    const int i = tid * ITEMS_PER_THREAD + it;
    if (i < load_n) {
      const int p = (int)culled_prim_indices[prim_start + i];
      vals[it] = p;
      const float vx = pos[p*3+0] - crox;
      const float vy = pos[p*3+1] - croy;
      const float vz = pos[p*3+2] - croz;
      const float tkey = fmaf(vx, crdx, fmaf(vy, crdy, vz * crdz));
      keys[it] = (tkey > 0.f) ? tkey : 1e30f;
    } else {
      vals[it] = -1;
      keys[it] = 1e30f;
    }
  }
  BlockRadixSortT(sort_storage).Sort(keys, vals);
  __syncthreads();
#pragma unroll
  for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
    const int i = tid * ITEMS_PER_THREAD + it;
    s_pid[i] = vals[it];
  }
  __syncthreads();

  const int MAX_HITS = 512;
  float trans_arr[MAX_HITS];
  float alpha_arr[MAX_HITS];
  int prim_arr[MAX_HITS];
  float dx_arr[MAX_HITS];
  float dy_arr[MAX_HITS];
  float dz_arr[MAX_HITS];

  float trans = 1.f;
  int used = 0;

  auto process_prim = [&](int p) {
    if (trans < 1e-3f || used >= MAX_HITS) return;
    float t, dx, dy, dz, decay;
    const float cx = pos[p*3+0];
    const float cy = pos[p*3+1];
    const float cz = pos[p*3+2];
    const float lg = logs[p];
    if (hit_and_features(rox, roy, roz, rdx, rdy, rdz, cx, cy, cz, lg, thr, t, dx, dy, dz, decay)) {
      const float v6[6] = {rdx, rdy, rdz, dx, dy, dz};
      const float* W1p = W1 + (size_t)p * H * IN;
      const float* b1p = b1 + (size_t)p * H;
      const float* W2p = W2 + (size_t)p * H * H;
      const float* b2p = b2 + (size_t)p * H;
      const float* W3p = W3 + (size_t)p * 4 * H;
      const float* b3p = b3 + (size_t)p * 4;
      float R, G, Bc, sigma_raw;
      mlp_eval_rgb_sigma(freq_bands, attenuation, F, W1p, b1p, W2p, b2p, W3p, b3p, H, IN, v6, R, G, Bc, sigma_raw);
      const float sigma = sigma_raw * decay;
      const float alpha = 1.f - __expf(-sigma);

      trans_arr[used] = trans;
      alpha_arr[used] = alpha;
      prim_arr[used] = p;
      dx_arr[used] = dx;
      dy_arr[used] = dy;
      dz_arr[used] = dz;

      trans *= (1.f - alpha);
      used++;
    }
  };

  for (int i = 0; i < load_n; ++i) {
    const int p = s_pid[i];
    if (p >= 0 && p < N) process_prim(p);
    const int active = (valid_pix && (trans >= 1e-3f)) ? 1 : 0;
    if (__syncthreads_or(active) == 0) break;
  }
  if (tile_len > load_n) {
    for (int i = prim_start + load_n; i < prim_end; ++i) {
      const int p = (int)culled_prim_indices[i];
      if (p < 0 || p >= N) continue;
      process_prim(p);
      const int active = (valid_pix && (trans >= 1e-3f)) ? 1 : 0;
      if (__syncthreads_or(active) == 0) break;
    }
  }

  const int max_emit = counts[pix];
  if (used > max_emit) used = max_emit;

  // reverse pass
  float gT_next = gT_end;
  const int base = offsets[pix];
  for (int i = used - 1; i >= 0; --i) {
    const float trans_i = trans_arr[i];
    const float alpha_i = alpha_arr[i];
    const int p = prim_arr[i];

    float t, dx, dy, dz, decay;
    const float cx = pos[p*3+0];
    const float cy = pos[p*3+1];
    const float cz = pos[p*3+2];
    const float lg = logs[p];
    if (!hit_and_features(rox, roy, roz, rdx, rdy, rdz, cx, cy, cz, lg, thr, t, dx, dy, dz, decay)) {
      continue;
    }

    const float v6[6] = {rdx, rdy, rdz, dx, dy, dz};
    const float* W1p = W1 + (size_t)p * H * IN;
    const float* b1p = b1 + (size_t)p * H;
    const float* W2p = W2 + (size_t)p * H * H;
    const float* b2p = b2 + (size_t)p * H;
    const float* W3p = W3 + (size_t)p * 4 * H;
    const float* b3p = b3 + (size_t)p * 4;
    float R, G, Bc, sigma_raw;
    mlp_eval_rgb_sigma(freq_bands, attenuation, F, W1p, b1p, W2p, b2p, W3p, b3p, H, IN, v6, R, G, Bc, sigma_raw);
    const float dotg = dot3(grad_r, grad_g, grad_b, R, G, Bc);

    const float grad_alpha = trans_i * (dotg - gT_next);
    const float gT_i = gT_next * (1.f - alpha_i) + dotg * alpha_i;
    gT_next = gT_i;

    const float sigma = sigma_raw * decay;
    const float exp_neg_sigma = __expf(-sigma);
    const float grad_sigma = grad_alpha * exp_neg_sigma;
    const float sigmoid_o6 = 1.f - __expf(-sigma_raw);
    const float grad_sigma_raw = grad_sigma * decay * sigmoid_o6;

    out_grad_sigma_raw[base + i] = grad_sigma_raw;

    const float s = __expf(logs[p]);
    const float s2 = s * s;
    const float perp2 = fmaxf(dx_arr[i]*dx_arr[i] + dy_arr[i]*dy_arr[i] + dz_arr[i]*dz_arr[i], 0.f);
    const float grad_decay = grad_sigma * sigma_raw;
    const float scale = -grad_decay * decay / (s2 + 1e-12f);

    atomicAdd(&grad_positions[p*3+0], scale * dx_arr[i]);
    atomicAdd(&grad_positions[p*3+1], scale * dy_arr[i]);
    atomicAdd(&grad_positions[p*3+2], scale * dz_arr[i]);

    const float grad_log_s = grad_decay * decay * perp2 / (s2 + 1e-12f);
    atomicAdd(&grad_log_scales[p], grad_log_s);
  }

  // emit records
  for (int i = 0; i < used; ++i) {
    const int p = prim_arr[i];
    out_prim_id[base + i] = p;
    const float trans_i = trans_arr[i];
    const float alpha_i = alpha_arr[i];
    const float w = trans_i * alpha_i;
    out_grad_rgb[(base + i)*3 + 0] = grad_r * w;
    out_grad_rgb[(base + i)*3 + 1] = grad_g * w;
    out_grad_rgb[(base + i)*3 + 2] = grad_b * w;
    out_x6[(base + i)*6 + 0] = rdx;
    out_x6[(base + i)*6 + 1] = rdy;
    out_x6[(base + i)*6 + 2] = rdz;
    out_x6[(base + i)*6 + 3] = dx_arr[i];
    out_x6[(base + i)*6 + 4] = dy_arr[i];
    out_x6[(base + i)*6 + 5] = dz_arr[i];
  }
}

// Backward (tilecenter full): compute MLP grads in-kernel (option A)
template <typename WType>
// EN: Full tile-center backward kernel computing MLP and geometry gradients in one pass.
// The only effective backward kernel for tilecenter mode.
__global__ __launch_bounds__(256, 2) void render_tiles_backward_tilecenter_full(
    const float* __restrict__ rays_o,
    const float* __restrict__ rays_d,
    const float* __restrict__ pos,
    const float* __restrict__ log_scales,
    const float* __restrict__ quats,
    const int32_t* __restrict__ culled_prim_indices,
    const int32_t* __restrict__ tile_offsets,
    const float* __restrict__ freq_bands,
    const float* __restrict__ attenuation,
    const int F,
    const WType* __restrict__ W1, const WType* __restrict__ b1,
    const WType* __restrict__ W2, const WType* __restrict__ b2,
    const WType* __restrict__ W3, const WType* __restrict__ b3,
    const int N, const int H, const int IN,
    const int Wimg, const int Himg, const int TW, const int TH,
    const float thr,
    const float* __restrict__ grad_out_rgb,
    const float* __restrict__ grad_T_end,
    float* __restrict__ grad_W1, float* __restrict__ grad_b1,
    float* __restrict__ grad_W2, float* __restrict__ grad_b2,
    float* __restrict__ grad_W3, float* __restrict__ grad_b3,
    float* __restrict__ grad_positions,
    float* __restrict__ grad_log_scales,
    float* __restrict__ grad_quats)
{
  const int tile_x = blockIdx.x;
  const int tile_y = blockIdx.y;
  const int tid = (int)threadIdx.x;
  const int tile_threads = TW * TH;
  const bool in_tile_thread = (tid < tile_threads);
  const int lx = in_tile_thread ? (tid % TW) : 0;
  const int ly = in_tile_thread ? (tid / TW) : 0;
  const int x = tile_x * TW + lx;
  const int y = tile_y * TH + ly;
  const bool valid_pix = in_tile_thread && (x < Wimg) && (y < Himg);
  const int pix = valid_pix ? (y * Wimg + x) : 0;

  const int num_tiles_x = (Wimg + TW - 1) / TW;
  const int tile_id = tile_y * num_tiles_x + tile_x;

  const float rox = valid_pix ? rays_o[pix*3+0] : 0.f;
  const float roy = valid_pix ? rays_o[pix*3+1] : 0.f;
  const float roz = valid_pix ? rays_o[pix*3+2] : 0.f;
  const float rdx = valid_pix ? rays_d[pix*3+0] : 0.f;
  const float rdy = valid_pix ? rays_d[pix*3+1] : 0.f;
  const float rdz = valid_pix ? rays_d[pix*3+2] : 0.f;

  const float grad_r = valid_pix ? grad_out_rgb[pix*3+0] : 0.f;
  const float grad_g = valid_pix ? grad_out_rgb[pix*3+1] : 0.f;
  const float grad_b = valid_pix ? grad_out_rgb[pix*3+2] : 0.f;
  const float gT_end = valid_pix ? grad_T_end[pix] : 0.f;

  const int cxp = min(tile_x * TW + (TW >> 1), Wimg - 1);
  const int cyp = min(tile_y * TH + (TH >> 1), Himg - 1);
  const int pix_c = cyp * Wimg + cxp;
  const float crox = rays_o[pix_c*3+0];
  const float croy = rays_o[pix_c*3+1];
  const float croz = rays_o[pix_c*3+2];
  const float crdx = rays_d[pix_c*3+0];
  const float crdy = rays_d[pix_c*3+1];
  const float crdz = rays_d[pix_c*3+2];


  const int32_t prim_start = tile_offsets[tile_id];
  const int32_t prim_end = tile_offsets[tile_id + 1];
  const int tile_len = (int)(prim_end - prim_start);

  if (tile_len <= 0) return;

  constexpr int MAX_TILE_PRIMS = MLPSP_MAX_TILE_PRIMS;
  constexpr int BLOCK_THREADS = 256;
  constexpr int ITEMS_PER_THREAD = MAX_TILE_PRIMS / BLOCK_THREADS;
  __shared__ int s_pid[MAX_TILE_PRIMS];
  using BlockRadixSortT = cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD, int>;
  __shared__ typename BlockRadixSortT::TempStorage sort_storage;

  const int num_threads = (int)blockDim.x;
  const int load_n = tile_len < MAX_TILE_PRIMS ? tile_len : MAX_TILE_PRIMS;

  float keys[ITEMS_PER_THREAD];
  int vals[ITEMS_PER_THREAD];

#pragma unroll
  for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
    const int i = tid * ITEMS_PER_THREAD + it;
    if (i < load_n) {
      const int p = (int)culled_prim_indices[prim_start + i];
      vals[it] = p;

      /*
      const float vx = pos[p*3+0] - crox;
      const float vy = pos[p*3+1] - croy;
      const float vz = pos[p*3+2] - croz;
      const float tkey = fmaf(vx, crdx, fmaf(vy, crdy, vz * crdz));
      */


      const float cx = pos[p*3+0];
      const float cy = pos[p*3+1];
      const float cz = pos[p*3+2];
      const float lgx = log_scales[p*3+0];
      const float lgy = log_scales[p*3+1];
      const float lgz = log_scales[p*3+2];
      const float qw  = quats[p*4+0];
      const float qx  = quats[p*4+1];
      const float qy  = quats[p*4+2];
      const float qz  = quats[p*4+3];
      /*
      const float tkey = tstar_key_tilecenter(
          crox, croy, croz, crdx, crdy, crdz,
          cx, cy, cz,
          lgx, lgy, lgz,
          qw, qx, qy, qz
      );
      keys[it] = (tkey > 0.f) ? tkey : 1e30f;*/

      const float tkey = tstar_key_tilecenter(
          crox, croy, croz, crdx, crdy, crdz,
          cx, cy, cz,
          lgx, lgy, lgz,
          qw, qx, qy, qz
      );
      keys[it] =(tkey > 0.f) ? tkey : 1e30f;
    } else {
      vals[it] = -1;
      keys[it] = 1e30f;
    }
  }



  BlockRadixSortT(sort_storage).Sort(keys, vals);
  __syncthreads();
#pragma unroll
  for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
    const int i = tid * ITEMS_PER_THREAD + it;
    s_pid[i] = vals[it];
  }
  __syncthreads();

  constexpr int MAX_HITS = 512;
  float trans_arr[MAX_HITS];
  int prim_arr[MAX_HITS];
  float dx_arr[MAX_HITS];
  float dy_arr[MAX_HITS];
  float dz_arr[MAX_HITS];

  float trans = 1.f;
  int used = 0;

  auto process_prim = [&](int p) {
    if (!valid_pix || trans < 1e-3f || used >= MAX_HITS) return;
    float t, dx, dy, dz, decay;
    const float cx = pos[p*3+0];
    const float cy = pos[p*3+1];
    const float cz = pos[p*3+2];
    const float lgx = log_scales[p*3+0];
    const float lgy = log_scales[p*3+1];
    const float lgz = log_scales[p*3+2];
    const float qw  = quats[p*4+0];
    const float qx  = quats[p*4+1];
    const float qy  = quats[p*4+2];
    const float qz  = quats[p*4+3];
    float qlx, qly, qlz;
    if (hit_and_features_rot(rox, roy, roz, rdx, rdy, rdz,
                             cx, cy, cz,
                             lgx, lgy, lgz,
                             qw, qx, qy, qz,
                             thr,
                             t, dx, dy, dz,
                             qlx, qly, qlz,
                             decay)) {
      const float v6[6] = {rdx, rdy, rdz, dx, dy, dz};
      const WType* W1p = W1 + (size_t)p * H * IN;
      const WType* b1p = b1 + (size_t)p * H;
      const WType* W2p = W2 + (size_t)p * H * H;
      const WType* b2p = b2 + (size_t)p * H;
      const WType* W3p = W3 + (size_t)p * 4 * H;
      const WType* b3p = b3 + (size_t)p * 4;
      float R, G, Bc, sigma_raw;
      mlp_eval_rgb_sigma(freq_bands, attenuation, F, W1p, b1p, W2p, b2p, W3p, b3p, H, IN, v6, R, G, Bc, sigma_raw);
      const float sigma = sigma_raw * decay;
      const float alpha = 1.f - __expf(-sigma);

      trans_arr[used] = trans;
      prim_arr[used] = p;
      dx_arr[used] = dx;
      dy_arr[used] = dy;
      dz_arr[used] = dz;

      trans *= (1.f - alpha);
      used++;
    }
  };

  for (int i = 0; i < load_n; ++i) {
    const int p = s_pid[i];
    if (p >= 0 && p < N) process_prim(p);
    const int active = (valid_pix && (trans >= 1e-3f)) ? 1 : 0;
    if (__syncthreads_or(active) == 0 || used >= MAX_HITS) break;
  }
  if (tile_len > load_n) {
    for (int i = prim_start + load_n; i < prim_end; ++i) {
      const int p = (int)culled_prim_indices[i];
      if (p < 0 || p >= N) continue;
      process_prim(p);
      const int active = (valid_pix && (trans >= 1e-3f)) ? 1 : 0;
      if (__syncthreads_or(active) == 0 || used >= MAX_HITS) break;
    }
  }

  // reverse pass
  float gT_next = gT_end;
  const int Hcap = H > MLPSP_MAX_H ? MLPSP_MAX_H : H;
  const int cos_offset = IN / 2;
  for (int i = used - 1; i >= 0; --i) {
    const float trans_i = trans_arr[i];
    const int p = prim_arr[i];
    const float dx = dx_arr[i];
    const float dy = dy_arr[i];
    const float dz = dz_arr[i];
    const float3 d = make_float3(dx, dy, dz);

    const float v6[6] = {rdx, rdy, rdz, dx, dy, dz};
    const WType* W1p = W1 + (size_t)p * H * IN;
    const WType* b1p = b1 + (size_t)p * H;
    const WType* W2p = W2 + (size_t)p * H * H;
    const WType* b2p = b2 + (size_t)p * H;
    const WType* W3p = W3 + (size_t)p * 4 * H;
    const WType* b3p = b3 + (size_t)p * 4;

    const float lgx = log_scales[p*3+0];
    const float lgy = log_scales[p*3+1];
    const float lgz = log_scales[p*3+2];
    const float qw  = quats[p*4+0];
    const float qx  = quats[p*4+1];
    const float qy  = quats[p*4+2];
    const float qz  = quats[p*4+3];
    // local coords: qloc = R^T d  (conjugate rotation)
    const float3 qloc = quat_rotate_conj(qw, qx, qy, qz, d);
    const float qlx = qloc.x;
    const float qly = qloc.y;
    const float qlz = qloc.z;

    // MLP forward (recompute) + backprop
    // ---------- Forward recompute ----------
    float h1_pre[MLPSP_MAX_H];
    for (int h_idx = 0; h_idx < Hcap; ++h_idx) h1_pre[h_idx] = load_as_float(b1p + h_idx);

    // PE accumulate into h1_pre using W1
    for (int c = 0; c < 6; ++c) {
      for (int f = 0; f < F; ++f) {
        const float fb  = freq_bands[f];
        const float att = attenuation[f];
        const float xproj = v6[c] * fb * att;
        const int j = c * F + f;
        const float s  = __sinf(xproj);
        const float co = __cosf(xproj);
        for (int h_idx = 0; h_idx < Hcap; ++h_idx) {
          const WType* Wi = W1p + (size_t)h_idx * IN;
          h1_pre[h_idx] = fmaf(load_as_float(Wi + j),            s,  h1_pre[h_idx]);
          h1_pre[h_idx] = fmaf(load_as_float(Wi + j + cos_offset), co, h1_pre[h_idx]);
        }
      }
    }

    float h1_relu[MLPSP_MAX_H];
    for (int h_idx = 0; h_idx < Hcap; ++h_idx) h1_relu[h_idx] = (h1_pre[h_idx] > 0.f) ? h1_pre[h_idx] : 0.f;

    // layer2 pre + relu
    float h2_pre[MLPSP_MAX_H];
    float h2_relu[MLPSP_MAX_H];
    for (int o = 0; o < Hcap; ++o) {
      float acc = load_as_float(b2p + o);
      const WType* W2row = W2p + (size_t)o * H;
      acc += row_dot_vec<WType>(W2row, h1_relu, Hcap);
      h2_pre[o]  = acc;
      h2_relu[o] = (acc > 0.f) ? acc : 0.f;
    }

    // out3 = b3 + W3 * h2_relu
    float o3 = load_as_float(b3p + 0);
    float o4 = load_as_float(b3p + 1);
    float o5 = load_as_float(b3p + 2);
    float o6 = load_as_float(b3p + 3);
    for (int h_idx = 0; h_idx < Hcap; ++h_idx) {
      const float hi = h2_relu[h_idx];
      o3 = fmaf(load_as_float(W3p + 0 * H + h_idx), hi, o3);
      o4 = fmaf(load_as_float(W3p + 1 * H + h_idx), hi, o4);
      o5 = fmaf(load_as_float(W3p + 2 * H + h_idx), hi, o5);
      o6 = fmaf(load_as_float(W3p + 3 * H + h_idx), hi, o6);
    }

    const float R = fast_sigmoid(o3);
    const float G = fast_sigmoid(o4);
    const float Bc = fast_sigmoid(o5);
    const float sigma_raw = fast_softplus(o6);
    const float mx = __expf(-2.f * lgx);
    const float my = __expf(-2.f * lgy);
    const float mz = __expf(-2.f * lgz);
    const float r2 = fmaxf(qlx*qlx*mx + qly*qly*my + qlz*qlz*mz, 0.f);
    const float decay = __expf(-0.5f * r2);
    const float alpha_i = 1.f - __expf(-sigma_raw * decay);
    const float dotg = dot3(grad_r, grad_g, grad_b, R, G, Bc);

    const float grad_alpha = trans_i * (dotg - gT_next);
    const float gT_i = gT_next * (1.f - alpha_i) + dotg * alpha_i;
    gT_next = gT_i;

    const float sigma = sigma_raw * decay;
    const float exp_neg_sigma = __expf(-sigma);
    const float grad_sigma = grad_alpha * exp_neg_sigma;

    // sigma_raw = softplus(o6) => d sigma_raw / d o6 = sigmoid(o6) = 1 - exp(-sigma_raw)
    const float sigmoid_o6 = 1.f - __expf(-sigma_raw);
    //const float grad_sigma_raw = grad_sigma * decay * sigmoid_o6;
    const float grad_sigma_raw = grad_sigma * decay;

    // geometry grads (rotated ellipsoid)
    const float grad_decay = grad_sigma * sigma_raw;

    // dL/d(qloc) where qloc = R^T d and d = center - (ro + t rd)
    const float gqx_loc = -grad_decay * decay * mx * qlx;
    const float gqy_loc = -grad_decay * decay * my * qly;
    const float gqz_loc = -grad_decay * decay * mz * qlz;

    // dL/dd = R * dL/d(qloc)
    const float3 grad_loc = make_float3(gqx_loc, gqy_loc, gqz_loc);
    const float3 grad_d = quat_rotate(qw, qx, qy, qz, grad_loc);

    atomic_add_warp_agg(&grad_positions[p*3+0], grad_d.x);
    atomic_add_warp_agg(&grad_positions[p*3+1], grad_d.y);
    atomic_add_warp_agg(&grad_positions[p*3+2], grad_d.z);

    // log-scales (3D): d r2 / d log_s = -2 * q^2 / s^2  -> d decay / d log_s = decay * (q^2 / s^2)
    atomic_add_warp_agg(&grad_log_scales[p*3+0], grad_decay * decay * mx * qlx * qlx);
    atomic_add_warp_agg(&grad_log_scales[p*3+1], grad_decay * decay * my * qly * qly);
    atomic_add_warp_agg(&grad_log_scales[p*3+2], grad_decay * decay * mz * qlz * qlz);

    // quaternion: accumulate dL/dR then map to dL/dq
    const float gR00 = dx * gqx_loc; const float gR01 = dx * gqy_loc; const float gR02 = dx * gqz_loc;
    const float gR10 = dy * gqx_loc; const float gR11 = dy * gqy_loc; const float gR12 = dy * gqz_loc;
    const float gR20 = dz * gqx_loc; const float gR21 = dz * gqy_loc; const float gR22 = dz * gqz_loc;

    float gqw, gqx, gqy, gqz;
    rotmat_grad_to_quat_grad(qw, qx, qy, qz,
                             gR00, gR01, gR02,
                             gR10, gR11, gR12,
                             gR20, gR21, gR22,
                             gqw, gqx, gqy, gqz);

    atomic_add_warp_agg(&grad_quats[p*4+0], gqw);
    atomic_add_warp_agg(&grad_quats[p*4+1], gqx);
    atomic_add_warp_agg(&grad_quats[p*4+2], gqy);
    atomic_add_warp_agg(&grad_quats[p*4+3], gqz);

    // RGB path grads
    const float grad_rgb_r = grad_r * (trans_i * alpha_i);
    const float grad_rgb_g = grad_g * (trans_i * alpha_i);
    const float grad_rgb_b = grad_b * (trans_i * alpha_i);

    const float grad_o3 = grad_rgb_r * R  * (1.f - R);
    const float grad_o4 = grad_rgb_g * G  * (1.f - G);
    const float grad_o5 = grad_rgb_b * Bc * (1.f - Bc);
    //const float grad_o6 = grad_sigma_raw * sigmoid_o6;
    const float d_softplus = fast_sigmoid(o6);            // ✅ sigmoid(o6)
    const float grad_o6 = grad_sigma_raw * d_softplus;    // ✅

    // ---------- Write grads ----------
    float* gW1p = grad_W1 + (size_t)p * H * IN;
    float* gb1p = grad_b1 + (size_t)p * H;
    float* gW2p = grad_W2 + (size_t)p * H * H;
    float* gb2p = grad_b2 + (size_t)p * H;
    float* gW3p = grad_W3 + (size_t)p * 4 * H;
    float* gb3p = grad_b3 + (size_t)p * 4;

    // W3/b3 grads
    for (int h_idx = 0; h_idx < Hcap; ++h_idx) {
      const float hi = h2_relu[h_idx];
      atomic_add_warp_agg(&gW3p[0*H + h_idx], grad_o3 * hi);
      atomic_add_warp_agg(&gW3p[1*H + h_idx], grad_o4 * hi);
      atomic_add_warp_agg(&gW3p[2*H + h_idx], grad_o5 * hi);
      atomic_add_warp_agg(&gW3p[3*H + h_idx], grad_o6 * hi);
    }
    atomic_add_warp_agg(&gb3p[0], grad_o3);
    atomic_add_warp_agg(&gb3p[1], grad_o4);
    atomic_add_warp_agg(&gb3p[2], grad_o5);
    atomic_add_warp_agg(&gb3p[3], grad_o6);

    // dh2 = W3^T * grad_out, then ReLU'(h2_pre)
    float dh2[MLPSP_MAX_H];
    for (int o = 0; o < Hcap; ++o) {
      float go = load_as_float(W3p + 0*H + o) * grad_o3
              + load_as_float(W3p + 1*H + o) * grad_o4
              + load_as_float(W3p + 2*H + o) * grad_o5
              + load_as_float(W3p + 3*H + o) * grad_o6;
      if (h2_pre[o] <= 0.f) go = 0.f;
      dh2[o] = go;
      atomic_add_warp_agg(&gb2p[o], go); // b2 (layer2 bias) grad
    }

    // (2) Layer2: W2 grads and accumulate dh1
    float dh1[MLPSP_MAX_H];
    for (int h_idx = 0; h_idx < Hcap; ++h_idx) dh1[h_idx] = 0.f;

    for (int o = 0; o < Hcap; ++o) {
      const float go = dh2[o];
      const WType* W2row = W2p + (size_t)o * H;
      float* gW2row = gW2p + (size_t)o * H;
      for (int h_idx = 0; h_idx < Hcap; ++h_idx) {
        atomic_add_warp_agg(&gW2row[h_idx], go * h1_relu[h_idx]);
        dh1[h_idx] = fmaf(load_as_float(W2row + h_idx), go, dh1[h_idx]);
      }
    }

    // apply ReLU'(h1_pre) and b1 grads
    for (int h_idx = 0; h_idx < Hcap; ++h_idx) {
      if (h1_pre[h_idx] <= 0.f) dh1[h_idx] = 0.f;
      atomic_add_warp_agg(&gb1p[h_idx], dh1[h_idx]);
    }

    // (4) W1 grads (PE sin/cos)
    for (int c = 0; c < 6; ++c) {
      for (int f = 0; f < F; ++f) {
        const float fb  = freq_bands[f];
        const float att = attenuation[f];
        const float xproj = v6[c] * fb * att;
        const int j = c * F + f;
        const float s  = __sinf(xproj);
        const float co = __cosf(xproj);
        for (int h_idx = 0; h_idx < Hcap; ++h_idx) {
          float* gWi = gW1p + (size_t)h_idx * IN;
          atomic_add_warp_agg(&gWi[j],             dh1[h_idx] * s);
          atomic_add_warp_agg(&gWi[j + cos_offset], dh1[h_idx] * co);
        }
      }
    }
  }
}

// Backward (tilecenter): feat_attr-only gradient accumulation without global hit-list buffers.
__global__ void render_tiles_backward_featattr_tilecenter(
    const float* __restrict__ rays_o,
    const float* __restrict__ rays_d,
    const float* __restrict__ pos,
    const float* __restrict__ log_scales,
    const float* __restrict__ quats,
    const int32_t* __restrict__ culled_prim_indices,
    const int32_t* __restrict__ tile_offsets,
    const float* __restrict__ freq_bands,
    const float* __restrict__ attenuation,
    const int F,
    const float* __restrict__ W1, const float* __restrict__ b1,
    const float* __restrict__ W2, const float* __restrict__ b2,
    const float* __restrict__ W3, const float* __restrict__ b3,
    const int N, const int H, const int IN,
    const int Wimg, const int Himg, const int TW, const int TH,
    const float thr,
    const float* __restrict__ grad_out_feat,
    const int feat_slots,
    const int Cfeat,
    const float gating_beta,
    const float gating_temp,
    float* __restrict__ grad_feat_attr)
{
  const int tile_x = blockIdx.x;
  const int tile_y = blockIdx.y;
  const int tid = (int)threadIdx.x;
  const int tile_threads = TW * TH;
  const bool in_tile_thread = (tid < tile_threads);
  const int lx = in_tile_thread ? (tid % TW) : 0;
  const int ly = in_tile_thread ? (tid / TW) : 0;
  const int x = tile_x * TW + lx;
  const int y = tile_y * TH + ly;
  const bool valid_pix = in_tile_thread && (x < Wimg) && (y < Himg);
  const int pix = valid_pix ? (y * Wimg + x) : 0;

  const int num_tiles_x = (Wimg + TW - 1) / TW;
  const int tile_id = tile_y * num_tiles_x + tile_x;
  const int32_t prim_start = tile_offsets[tile_id];
  const int32_t prim_end = tile_offsets[tile_id + 1];
  const int tile_len = (int)(prim_end - prim_start);
  if (tile_len <= 0) return;

  const float rox = valid_pix ? rays_o[pix*3+0] : 0.f;
  const float roy = valid_pix ? rays_o[pix*3+1] : 0.f;
  const float roz = valid_pix ? rays_o[pix*3+2] : 0.f;
  const float rdx = valid_pix ? rays_d[pix*3+0] : 0.f;
  const float rdy = valid_pix ? rays_d[pix*3+1] : 0.f;
  const float rdz = valid_pix ? rays_d[pix*3+2] : 0.f;

  const int cxp = min(tile_x * TW + (TW >> 1), Wimg - 1);
  const int cyp = min(tile_y * TH + (TH >> 1), Himg - 1);
  const int pix_c = cyp * Wimg + cxp;
  const float crox = rays_o[pix_c*3+0];
  const float croy = rays_o[pix_c*3+1];
  const float croz = rays_o[pix_c*3+2];
  const float crdx = rays_d[pix_c*3+0];
  const float crdy = rays_d[pix_c*3+1];
  const float crdz = rays_d[pix_c*3+2];

  constexpr int MAX_TILE_PRIMS = MLPSP_MAX_TILE_PRIMS;
  constexpr int BLOCK_THREADS = 256;
  constexpr int ITEMS_PER_THREAD = MAX_TILE_PRIMS / BLOCK_THREADS;
  __shared__ int s_pid[MAX_TILE_PRIMS];
  using BlockRadixSortT = cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD, int>;
  __shared__ typename BlockRadixSortT::TempStorage sort_storage;

  const int load_n = tile_len < MAX_TILE_PRIMS ? tile_len : MAX_TILE_PRIMS;
  float keys[ITEMS_PER_THREAD];
  int vals[ITEMS_PER_THREAD];
#pragma unroll
  for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
    const int i = tid * ITEMS_PER_THREAD + it;
    if (i < load_n) {
      const int p = (int)culled_prim_indices[prim_start + i];
      vals[it] = p;
      const float cx = pos[p*3+0];
      const float cy = pos[p*3+1];
      const float cz = pos[p*3+2];
      const float lgx = log_scales[p*3+0];
      const float lgy = log_scales[p*3+1];
      const float lgz = log_scales[p*3+2];
      const float qw  = quats[p*4+0];
      const float qx  = quats[p*4+1];
      const float qy  = quats[p*4+2];
      const float qz  = quats[p*4+3];
      const float tkey = tstar_key_tilecenter(
          crox, croy, croz, crdx, crdy, crdz,
          cx, cy, cz,
          lgx, lgy, lgz,
          qw, qx, qy, qz
      );
      keys[it] = (tkey > 0.f) ? tkey : 1e30f;
    } else {
      vals[it] = -1;
      keys[it] = 1e30f;
    }
  }
  BlockRadixSortT(sort_storage).Sort(keys, vals);
  __syncthreads();
#pragma unroll
  for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
    const int i = tid * ITEMS_PER_THREAD + it;
    s_pid[i] = vals[it];
  }
  __syncthreads();

  float trans = 1.f;
  auto process_prim = [&](int p) {
    if (!valid_pix || trans < 1e-3f) return;
    float t, dx, dy, dz, decay;
    const float cx = pos[p*3+0];
    const float cy = pos[p*3+1];
    const float cz = pos[p*3+2];
    const float lgx = log_scales[p*3+0];
    const float lgy = log_scales[p*3+1];
    const float lgz = log_scales[p*3+2];
    const float qw  = quats[p*4+0];
    const float qx  = quats[p*4+1];
    const float qy  = quats[p*4+2];
    const float qz  = quats[p*4+3];
    float qlx, qly, qlz;
    if (hit_and_features_rot(rox, roy, roz, rdx, rdy, rdz,
                             cx, cy, cz,
                             lgx, lgy, lgz,
                             qw, qx, qy, qz,
                             thr,
                             t, dx, dy, dz,
                             qlx, qly, qlz,
                             decay)) {
      const float v6[6] = {rdx, rdy, rdz, dx, dy, dz};
      const float* W1p = W1 + (size_t)p * H * IN;
      const float* b1p = b1 + (size_t)p * H;
      const float* W2p = W2 + (size_t)p * H * H;
      const float* b2p = b2 + (size_t)p * H;
      const float* W3p = W3 + (size_t)p * 4 * H;
      const float* b3p = b3 + (size_t)p * 4;
      float R, G, Bc, sigma_raw;
      mlp_eval_rgb_sigma(freq_bands, attenuation, F, W1p, b1p, W2p, b2p, W3p, b3p, H, IN, v6, R, G, Bc, sigma_raw);
      const float sigma = sigma_raw * decay;
      const float alpha = 1.f - __expf(-sigma);
      const float w = trans * alpha;
      const float* gfeat_pix = grad_out_feat + (size_t)pix * Cfeat;
      if (feat_slots <= 1) {
        float* gfeat_prim = grad_feat_attr + (size_t)p * Cfeat;
        for (int c = 0; c < Cfeat; ++c) {
          atomic_add_warp_agg(&gfeat_prim[c], w * gfeat_pix[c]);
        }
      } else if (feat_slots == 5) {
        float g0, g1, g2, g3;
        semantic_tetra_gates(qlx, qly, qlz, lgx, lgy, lgz, gating_beta, gating_temp, g0, g1, g2, g3);
        float* gbase = grad_feat_attr + ((size_t)p * feat_slots + 0) * Cfeat;
        float* gd0 = grad_feat_attr + ((size_t)p * feat_slots + 1) * Cfeat;
        float* gd1 = grad_feat_attr + ((size_t)p * feat_slots + 2) * Cfeat;
        float* gd2 = grad_feat_attr + ((size_t)p * feat_slots + 3) * Cfeat;
        float* gd3 = grad_feat_attr + ((size_t)p * feat_slots + 4) * Cfeat;
        for (int c = 0; c < Cfeat; ++c) {
          const float gc = w * gfeat_pix[c];
          atomic_add_warp_agg(&gbase[c], gc);
          atomic_add_warp_agg(&gd0[c], gc * g0);
          atomic_add_warp_agg(&gd1[c], gc * g1);
          atomic_add_warp_agg(&gd2[c], gc * g2);
          atomic_add_warp_agg(&gd3[c], gc * g3);
        }
      } else if (feat_slots == 4) {
        float g0, g1, g2, g3;
        semantic_tetra_gates(qlx, qly, qlz, lgx, lgy, lgz, gating_beta, gating_temp, g0, g1, g2, g3);
        float* gs0 = grad_feat_attr + ((size_t)p * feat_slots + 0) * Cfeat;
        float* gs1 = grad_feat_attr + ((size_t)p * feat_slots + 1) * Cfeat;
        float* gs2 = grad_feat_attr + ((size_t)p * feat_slots + 2) * Cfeat;
        float* gs3 = grad_feat_attr + ((size_t)p * feat_slots + 3) * Cfeat;
        for (int c = 0; c < Cfeat; ++c) {
          const float gc = w * gfeat_pix[c];
          atomic_add_warp_agg(&gs0[c], gc * g0);
          atomic_add_warp_agg(&gs1[c], gc * g1);
          atomic_add_warp_agg(&gs2[c], gc * g2);
          atomic_add_warp_agg(&gs3[c], gc * g3);
        }
      } else {
        float* gfeat_prim = grad_feat_attr + (size_t)p * feat_slots * Cfeat;
        for (int c = 0; c < Cfeat; ++c) {
          atomic_add_warp_agg(&gfeat_prim[c], w * gfeat_pix[c]);
        }
      }
      trans *= (1.f - alpha);
    }
  };

  for (int i = 0; i < load_n; ++i) {
    const int p = s_pid[i];
    if (p >= 0 && p < N) process_prim(p);
    const int active = (valid_pix && (trans >= 1e-3f)) ? 1 : 0;
    if (__syncthreads_or(active) == 0) break;
  }
  if (tile_len > load_n) {
    for (int i = prim_start + load_n; i < prim_end; ++i) {
      const int p = (int)culled_prim_indices[i];
      if (p < 0 || p >= N) continue;
      process_prim(p);
      const int active = (valid_pix && (trans >= 1e-3f)) ? 1 : 0;
      if (__syncthreads_or(active) == 0) break;
    }
  }
}

// Backward (tilecenter backprop-only, no atomics): profile compute cost without atomic accumulation.
// EN: Tile-center backprop-only profiling kernel without atomic gradient accumulation.
__global__ __launch_bounds__(256, 2) void render_tiles_backward_tilecenter_backprop_noatomic(
    const float* __restrict__ rays_o,
    const float* __restrict__ rays_d,
    const float* __restrict__ pos,
    const float* __restrict__ log_scales,
    const float* __restrict__ quats,
    const int32_t* __restrict__ culled_prim_indices,
    const int32_t* __restrict__ tile_offsets,
    const float* __restrict__ freq_bands,
    const float* __restrict__ attenuation,
    const int F,
    const float* __restrict__ W1, const float* __restrict__ b1,
    const float* __restrict__ W2, const float* __restrict__ b2,
    const float* __restrict__ W3, const float* __restrict__ b3,
    const int N, const int H, const int IN,
    const int Wimg, const int Himg, const int TW, const int TH,
    const float thr,
    const float* __restrict__ grad_out_rgb,
    const float* __restrict__ grad_T_end,
    float* __restrict__ out_sink)
{
  const int tile_x = blockIdx.x;
  const int tile_y = blockIdx.y;
  const int tid = (int)threadIdx.x;
  const int tile_threads = TW * TH;
  const bool in_tile_thread = (tid < tile_threads);
  const int lx = in_tile_thread ? (tid % TW) : 0;
  const int ly = in_tile_thread ? (tid / TW) : 0;
  const int x = tile_x * TW + lx;
  const int y = tile_y * TH + ly;
  const bool valid_pix = in_tile_thread && (x < Wimg) && (y < Himg);
  const int pix = valid_pix ? (y * Wimg + x) : 0;

  const int num_tiles_x = (Wimg + TW - 1) / TW;
  const int tile_id = tile_y * num_tiles_x + tile_x;

  const float rox = valid_pix ? rays_o[pix*3+0] : 0.f;
  const float roy = valid_pix ? rays_o[pix*3+1] : 0.f;
  const float roz = valid_pix ? rays_o[pix*3+2] : 0.f;
  const float rdx = valid_pix ? rays_d[pix*3+0] : 0.f;
  const float rdy = valid_pix ? rays_d[pix*3+1] : 0.f;
  const float rdz = valid_pix ? rays_d[pix*3+2] : 0.f;

  const float grad_r = valid_pix ? grad_out_rgb[pix*3+0] : 0.f;
  const float grad_g = valid_pix ? grad_out_rgb[pix*3+1] : 0.f;
  const float grad_b = valid_pix ? grad_out_rgb[pix*3+2] : 0.f;
  const float gT_end = valid_pix ? grad_T_end[pix] : 0.f;

  const int cxp = min(tile_x * TW + (TW >> 1), Wimg - 1);
  const int cyp = min(tile_y * TH + (TH >> 1), Himg - 1);
  const int pix_c = cyp * Wimg + cxp;
  const float crox = rays_o[pix_c*3+0];
  const float croy = rays_o[pix_c*3+1];
  const float croz = rays_o[pix_c*3+2];
  const float crdx = rays_d[pix_c*3+0];
  const float crdy = rays_d[pix_c*3+1];
  const float crdz = rays_d[pix_c*3+2];

  const int32_t prim_start = tile_offsets[tile_id];
  const int32_t prim_end = tile_offsets[tile_id + 1];
  const int tile_len = (int)(prim_end - prim_start);
  if (tile_len <= 0) return;

  constexpr int MAX_TILE_PRIMS = MLPSP_MAX_TILE_PRIMS;
  constexpr int BLOCK_THREADS = 256;
  constexpr int ITEMS_PER_THREAD = MAX_TILE_PRIMS / BLOCK_THREADS;
  __shared__ int s_pid[MAX_TILE_PRIMS];
  using BlockRadixSortT = cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD, int>;
  __shared__ typename BlockRadixSortT::TempStorage sort_storage;

  const int load_n = tile_len < MAX_TILE_PRIMS ? tile_len : MAX_TILE_PRIMS;
  float keys[ITEMS_PER_THREAD];
  int vals[ITEMS_PER_THREAD];
#pragma unroll
  for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
    const int i = tid * ITEMS_PER_THREAD + it;
    if (i < load_n) {
      const int p = (int)culled_prim_indices[prim_start + i];
      vals[it] = p;
      const float vx = pos[p*3+0] - crox;
      const float vy = pos[p*3+1] - croy;
      const float vz = pos[p*3+2] - croz;
      const float tkey = fmaf(vx, crdx, fmaf(vy, crdy, vz * crdz));
      keys[it] = (tkey > 0.f) ? tkey : 1e30f;
    } else {
      vals[it] = -1;
      keys[it] = 1e30f;
    }
  }
  BlockRadixSortT(sort_storage).Sort(keys, vals);
  __syncthreads();
#pragma unroll
  for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
    const int i = tid * ITEMS_PER_THREAD + it;
    s_pid[i] = vals[it];
  }
  __syncthreads();

  constexpr int MAX_HITS = 512;
  float trans_arr[MAX_HITS];
  int prim_arr[MAX_HITS];
  float dx_arr[MAX_HITS];
  float dy_arr[MAX_HITS];
  float dz_arr[MAX_HITS];

  float trans = 1.f;
  int used = 0;

  auto process_prim = [&](int p) {
    if (!valid_pix || trans < 1e-3f || used >= MAX_HITS) return;
    float t, dx, dy, dz, decay;
    const float cx = pos[p*3+0];
    const float cy = pos[p*3+1];
    const float cz = pos[p*3+2];
    const float lgx = log_scales[p*3+0];
    const float lgy = log_scales[p*3+1];
    const float lgz = log_scales[p*3+2];
    const float qw  = quats[p*4+0];
    const float qx  = quats[p*4+1];
    const float qy  = quats[p*4+2];
    const float qz  = quats[p*4+3];
    float qlx, qly, qlz;
    if (hit_and_features_rot(rox, roy, roz, rdx, rdy, rdz,
                             cx, cy, cz,
                             lgx, lgy, lgz,
                             qw, qx, qy, qz,
                             thr,
                             t, dx, dy, dz,
                             qlx, qly, qlz,
                             decay)) {
      const float v6[6] = {rdx, rdy, rdz, dx, dy, dz};
      const float* W1p = W1 + (size_t)p * H * IN;
      const float* b1p = b1 + (size_t)p * H;
      const float* W2p = W2 + (size_t)p * H * H;
      const float* b2p = b2 + (size_t)p * H;
      const float* W3p = W3 + (size_t)p * 4 * H;
      const float* b3p = b3 + (size_t)p * 4;
      float R, G, Bc, sigma_raw;
      mlp_eval_rgb_sigma(freq_bands, attenuation, F, W1p, b1p, W2p, b2p, W3p, b3p, H, IN, v6, R, G, Bc, sigma_raw);
      const float sigma = sigma_raw * decay;
      const float alpha = 1.f - __expf(-sigma);

      trans_arr[used] = trans;
      prim_arr[used] = p;
      dx_arr[used] = dx;
      dy_arr[used] = dy;
      dz_arr[used] = dz;

      trans *= (1.f - alpha);
      used++;
    }
  };

  for (int i = 0; i < load_n; ++i) {
    const int p = s_pid[i];
    if (p >= 0 && p < N) process_prim(p);
    const int active = (valid_pix && (trans >= 1e-3f)) ? 1 : 0;
    if (__syncthreads_or(active) == 0 || used >= MAX_HITS) break;
  }
  if (tile_len > load_n) {
    for (int i = prim_start + load_n; i < prim_end; ++i) {
      const int p = (int)culled_prim_indices[i];
      if (p < 0 || p >= N) continue;
      process_prim(p);
      const int active = (valid_pix && (trans >= 1e-3f)) ? 1 : 0;
      if (__syncthreads_or(active) == 0 || used >= MAX_HITS) break;
    }
  }

  // reverse pass (compute only)
  float gT_next = gT_end;
  const int Hcap = H > 32 ? 32 : H;
  const int cos_offset = IN / 2;
  float sink = 0.f;
  for (int i = used - 1; i >= 0; --i) {
    const float trans_i = trans_arr[i];
    const int p = prim_arr[i];
    const float dx = dx_arr[i];
    const float dy = dy_arr[i];
    const float dz = dz_arr[i];
    const float3 d = make_float3(dx, dy, dz);

    const float v6[6] = {rdx, rdy, rdz, dx, dy, dz};
    const float* W1p = W1 + (size_t)p * H * IN;
    const float* b1p = b1 + (size_t)p * H;
    const float* W2p = W2 + (size_t)p * H * H;
    const float* b2p = b2 + (size_t)p * H;
    const float* W3p = W3 + (size_t)p * 4 * H;
    const float* b3p = b3 + (size_t)p * 4;

    const float lgx = log_scales[p*3+0];
    const float lgy = log_scales[p*3+1];
    const float lgz = log_scales[p*3+2];
    const float qw  = quats[p*4+0];
    const float qx  = quats[p*4+1];
    const float qy  = quats[p*4+2];
    const float qz  = quats[p*4+3];
    // local coords: qloc = R^T d  (conjugate rotation)
    const float3 qloc = quat_rotate_conj(qw, qx, qy, qz, d);
    const float qlx = qloc.x;
    const float qly = qloc.y;
    const float qlz = qloc.z;

    float h1_pre[32];
    for (int h_idx = 0; h_idx < Hcap; ++h_idx) h1_pre[h_idx] = b1p[h_idx];
    for (int c = 0; c < 6; ++c) {
      for (int f = 0; f < F; ++f) {
        const float fb = freq_bands[f];
        const float att = attenuation[f];
        const float xproj = v6[c] * fb * att;
        const int j = c * F + f;
        const float s = __sinf(xproj);
        const float co = __cosf(xproj);
        for (int h_idx = 0; h_idx < Hcap; ++h_idx) {
          const float* Wi = W1p + (size_t)h_idx * IN;
          h1_pre[h_idx] = fmaf(Wi[j], s, h1_pre[h_idx]);
          h1_pre[h_idx] = fmaf(Wi[j + cos_offset], co, h1_pre[h_idx]);
        }
      }
    }
    float h2_pre[32];
    for (int o = 0; o < Hcap; ++o) {
      float acc = b2p[o];
      const float* W2row = W2p + (size_t)o * H;
      for (int h_idx = 0; h_idx < Hcap; ++h_idx) {
        const float h1_relu = h1_pre[h_idx] > 0.f ? h1_pre[h_idx] : 0.f;
        acc = fmaf(W2row[h_idx], h1_relu, acc);
      }
      h2_pre[o] = acc;
    }

    float o3=b3p[0], o4=b3p[1], o5=b3p[2], o6=b3p[3];
    for (int h_idx = 0; h_idx < Hcap; ++h_idx) {
      const float hi = h2_pre[h_idx] > 0.f ? h2_pre[h_idx] : 0.f;
      o3 = fmaf(W3p[0*H + h_idx], hi, o3);
      o4 = fmaf(W3p[1*H + h_idx], hi, o4);
      o5 = fmaf(W3p[2*H + h_idx], hi, o5);
      o6 = fmaf(W3p[3*H + h_idx], hi, o6);
    }

    const float R = fast_sigmoid(o3);
    const float G = fast_sigmoid(o4);
    const float Bc = fast_sigmoid(o5);
    const float sigma_raw = fast_softplus(o6);
    const float mx = __expf(-2.f * lgx);
    const float my = __expf(-2.f * lgy);
    const float mz = __expf(-2.f * lgz);
    const float r2 = fmaxf(qlx*qlx*mx + qly*qly*my + qlz*qlz*mz, 0.f);
    const float decay = __expf(-0.5f * r2);
    const float alpha_i = 1.f - __expf(-sigma_raw * decay);
    const float dotg = dot3(grad_r, grad_g, grad_b, R, G, Bc);

    const float grad_alpha = trans_i * (dotg - gT_next);
    const float gT_i = gT_next * (1.f - alpha_i) + dotg * alpha_i;
    gT_next = gT_i;

    const float sigma = sigma_raw * decay;
    const float exp_neg_sigma = __expf(-sigma);
    const float grad_sigma = grad_alpha * exp_neg_sigma;
    const float sigmoid_o6 = 1.f - __expf(-sigma_raw);
    const float grad_sigma_raw = grad_sigma * decay * sigmoid_o6;

    // geometry grads (no atomics)
    const float grad_decay = grad_sigma * sigma_raw;
    const float gqx_loc = -grad_decay * decay * mx * qlx;
    const float gqy_loc = -grad_decay * decay * my * qly;
    const float gqz_loc = -grad_decay * decay * mz * qlz;

    // (pretend accumulate)
    sink += gqx_loc * dx;
    sink += gqy_loc * dy;
    sink += gqz_loc * dz;
    sink += (grad_decay * decay * mx * qlx * qlx);
    sink += (grad_decay * decay * my * qly * qly);
    sink += (grad_decay * decay * mz * qlz * qlz);

    const float grad_rgb_r = grad_r * (trans_i * alpha_i);
    const float grad_rgb_g = grad_g * (trans_i * alpha_i);
    const float grad_rgb_b = grad_b * (trans_i * alpha_i);

    const float grad_o3 = grad_rgb_r * R * (1.f - R);
    const float grad_o4 = grad_rgb_g * G * (1.f - G);
    const float grad_o5 = grad_rgb_b * Bc * (1.f - Bc);
    const float grad_o6 = grad_sigma_raw * sigmoid_o6;

    // W3/b3 grads (no atomics)
    for (int h_idx = 0; h_idx < Hcap; ++h_idx) {
      const float hi = h2_pre[h_idx] > 0.f ? h2_pre[h_idx] : 0.f;
      sink += grad_o3 * hi;
      sink += grad_o4 * hi;
      sink += grad_o5 * hi;
      sink += grad_o6 * hi;
    }
    sink += grad_o3 + grad_o4 + grad_o5 + grad_o6;

    // W2/b2 grads and dh1
    float dh1[32];
    for (int h_idx = 0; h_idx < Hcap; ++h_idx) dh1[h_idx] = 0.f;
    for (int o = 0; o < Hcap; ++o) {
      float go = W3p[0*H + o] * grad_o3
               + W3p[1*H + o] * grad_o4
               + W3p[2*H + o] * grad_o5
               + W3p[3*H + o] * grad_o6;
      if (h2_pre[o] <= 0.f) go = 0.f;
      sink += go;
      const float* W2row = W2p + (size_t)o * H;
      for (int h_idx = 0; h_idx < Hcap; ++h_idx) {
        const float h1_relu = h1_pre[h_idx] > 0.f ? h1_pre[h_idx] : 0.f;
        sink += go * h1_relu;
        dh1[h_idx] = fmaf(W2row[h_idx], go, dh1[h_idx]);
      }
    }
    for (int h_idx = 0; h_idx < Hcap; ++h_idx) {
      if (h1_pre[h_idx] <= 0.f) dh1[h_idx] = 0.f;
      sink += dh1[h_idx];
    }

    // W1 grads (no atomics)
    for (int c = 0; c < 6; ++c) {
      for (int f = 0; f < F; ++f) {
        const float fb = freq_bands[f];
        const float att = attenuation[f];
        const float xproj = v6[c] * fb * att;
        const int j = c * F + f;
        const float s = __sinf(xproj);
        const float co = __cosf(xproj);
        for (int h_idx = 0; h_idx < Hcap; ++h_idx) {
          sink += dh1[h_idx] * s;
          sink += dh1[h_idx] * co;
        }
      }
    }
  }

  if (valid_pix) out_sink[pix] = sink;
}

// Backward (tilecenter traversal-only): profile traversal + MLP forward cost without backprop.
// EN: Tile-center traversal-only profiling kernel that reports usage/transmittance.
__global__ void render_tiles_backward_tilecenter_traversal(
    const float* __restrict__ rays_o,
    const float* __restrict__ rays_d,
    const float* __restrict__ pos,
    const float* __restrict__ log_scales,
    const float* __restrict__ quats,
    const int32_t* __restrict__ culled_prim_indices,
    const int32_t* __restrict__ tile_offsets,
    const float* __restrict__ freq_bands,
    const float* __restrict__ attenuation,
    const int F,
    const float* __restrict__ W1, const float* __restrict__ b1,
    const float* __restrict__ W2, const float* __restrict__ b2,
    const float* __restrict__ W3, const float* __restrict__ b3,
    const int N, const int H, const int IN,
    const int Wimg, const int Himg, const int TW, const int TH,
    const float thr,
    int32_t* __restrict__ out_used,
    float* __restrict__ out_trans)
{
  const int tile_x = blockIdx.x;
  const int tile_y = blockIdx.y;
  const int tid = (int)threadIdx.x;
  const int tile_threads = TW * TH;
  const bool in_tile_thread = (tid < tile_threads);
  const int lx = in_tile_thread ? (tid % TW) : 0;
  const int ly = in_tile_thread ? (tid / TW) : 0;
  const int x = tile_x * TW + lx;
  const int y = tile_y * TH + ly;
  const bool valid_pix = in_tile_thread && (x < Wimg) && (y < Himg);
  const int pix = valid_pix ? (y * Wimg + x) : 0;

  const int num_tiles_x = (Wimg + TW - 1) / TW;
  const int tile_id = tile_y * num_tiles_x + tile_x;

  const float rox = valid_pix ? rays_o[pix*3+0] : 0.f;
  const float roy = valid_pix ? rays_o[pix*3+1] : 0.f;
  const float roz = valid_pix ? rays_o[pix*3+2] : 0.f;
  const float rdx = valid_pix ? rays_d[pix*3+0] : 0.f;
  const float rdy = valid_pix ? rays_d[pix*3+1] : 0.f;
  const float rdz = valid_pix ? rays_d[pix*3+2] : 0.f;

  const int cxp = min(tile_x * TW + (TW >> 1), Wimg - 1);
  const int cyp = min(tile_y * TH + (TH >> 1), Himg - 1);
  const int pix_c = cyp * Wimg + cxp;
  const float crox = rays_o[pix_c*3+0];
  const float croy = rays_o[pix_c*3+1];
  const float croz = rays_o[pix_c*3+2];
  const float crdx = rays_d[pix_c*3+0];
  const float crdy = rays_d[pix_c*3+1];
  const float crdz = rays_d[pix_c*3+2];

  const int32_t prim_start = tile_offsets[tile_id];
  const int32_t prim_end = tile_offsets[tile_id + 1];
  const int tile_len = (int)(prim_end - prim_start);
  if (tile_len <= 0) {
    if (valid_pix) {
      out_used[pix] = 0;
      out_trans[pix] = 1.f;
    }
    return;
  }

  constexpr int MAX_TILE_PRIMS = MLPSP_MAX_TILE_PRIMS;
  constexpr int BLOCK_THREADS = 256;
  constexpr int ITEMS_PER_THREAD = MAX_TILE_PRIMS / BLOCK_THREADS;
  __shared__ int s_pid[MAX_TILE_PRIMS];
  using BlockRadixSortT = cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD, int>;
  __shared__ typename BlockRadixSortT::TempStorage sort_storage;

  const int load_n = tile_len < MAX_TILE_PRIMS ? tile_len : MAX_TILE_PRIMS;
  float keys[ITEMS_PER_THREAD];
  int vals[ITEMS_PER_THREAD];
#pragma unroll
  for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
    const int i = tid * ITEMS_PER_THREAD + it;
    if (i < load_n) {
      const int p = (int)culled_prim_indices[prim_start + i];
      vals[it] = p;
      const float vx = pos[p*3+0] - crox;
      const float vy = pos[p*3+1] - croy;
      const float vz = pos[p*3+2] - croz;
      const float tkey = fmaf(vx, crdx, fmaf(vy, crdy, vz * crdz));
      keys[it] = (tkey > 0.f) ? tkey : 1e30f;
    } else {
      vals[it] = -1;
      keys[it] = 1e30f;
    }
  }
  BlockRadixSortT(sort_storage).Sort(keys, vals);
  __syncthreads();
#pragma unroll
  for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
    const int i = tid * ITEMS_PER_THREAD + it;
    s_pid[i] = vals[it];
  }
  __syncthreads();

  float trans = 1.f;
  int used = 0;

  auto process_prim = [&](int p) {
    if (!valid_pix || trans < 1e-3f || used >= MAX_TILE_PRIMS) return;
    float t, dx, dy, dz, decay;
    const float cx = pos[p*3+0];
    const float cy = pos[p*3+1];
    const float cz = pos[p*3+2];
    const float lgx = log_scales[p*3+0];
    const float lgy = log_scales[p*3+1];
    const float lgz = log_scales[p*3+2];
    const float qw  = quats[p*4+0];
    const float qx  = quats[p*4+1];
    const float qy  = quats[p*4+2];
    const float qz  = quats[p*4+3];
    float qlx, qly, qlz;
    if (hit_and_features_rot(rox, roy, roz, rdx, rdy, rdz,
                             cx, cy, cz,
                             lgx, lgy, lgz,
                             qw, qx, qy, qz,
                             thr,
                             t, dx, dy, dz,
                             qlx, qly, qlz,
                             decay)) {
      const float v6[6] = {rdx, rdy, rdz, dx, dy, dz};
      const float* W1p = W1 + (size_t)p * H * IN;
      const float* b1p = b1 + (size_t)p * H;
      const float* W2p = W2 + (size_t)p * H * H;
      const float* b2p = b2 + (size_t)p * H;
      const float* W3p = W3 + (size_t)p * 4 * H;
      const float* b3p = b3 + (size_t)p * 4;
      float R, G, Bc, sigma_raw;
      mlp_eval_rgb_sigma(freq_bands, attenuation, F, W1p, b1p, W2p, b2p, W3p, b3p, H, IN, v6, R, G, Bc, sigma_raw);
      const float sigma = sigma_raw * decay;
      const float alpha = 1.f - __expf(-sigma);
      trans *= (1.f - alpha);
      used++;
    }
  };

  for (int i = 0; i < load_n; ++i) {
    const int p = s_pid[i];
    if (p >= 0 && p < N) process_prim(p);
    const int active = (valid_pix && (trans >= 1e-3f)) ? 1 : 0;
    if (__syncthreads_or(active) == 0 || used >= MAX_TILE_PRIMS) break;
  }
  if (tile_len > load_n) {
    for (int i = prim_start + load_n; i < prim_end; ++i) {
      const int p = (int)culled_prim_indices[i];
      if (p < 0 || p >= N) continue;
      process_prim(p);
      const int active = (valid_pix && (trans >= 1e-3f)) ? 1 : 0;
      if (__syncthreads_or(active) == 0 || used >= MAX_TILE_PRIMS) break;
    }
  }

  if (valid_pix) {
    out_used[pix] = used;
    out_trans[pix] = trans;
  }
}

// Kernel: prim-major traversal, but assumes per-tile primitive lists are ALREADY ORDERED (no in-kernel sort).
// Intended for "globaldepth" mode where the host sorts each tile's list by primitive center camera-space forward (view-space z).
// EN: Forward rendering kernel that consumes a pre-sorted per-tile primitive list.
// The only effective forward kernel for tilecenter_split mode.
__global__ __launch_bounds__(256, 4) void render_tiles_kernel_tilecenter_presorted(
    const float* __restrict__ rays_o,
    const float* __restrict__ rays_d,
    const float* __restrict__ pos,
    const float* __restrict__ log_scales,
    const float* __restrict__ quats,
    const int32_t* __restrict__ culled_prim_indices,
    const int32_t* __restrict__ tile_offsets,
    const float* __restrict__ freq_bands,
    const float* __restrict__ attenuation,
    const int F,
    const float* __restrict__ W1, const float* __restrict__ b1,
    const float* __restrict__ W2, const float* __restrict__ b2,
    const float* __restrict__ W3, const float* __restrict__ b3,
    const int N, const int H, const int IN,
    const int Wimg, const int Himg, const int TW, const int TH,
    const float thr,
    float* __restrict__ out_rgb,
    float* __restrict__ out_alpha
){
  const int tile_x = blockIdx.x;
  const int tile_y = blockIdx.y;
  // 1D block of (TW*TH) threads
  const int tid = (int)threadIdx.x;
  const int tile_threads = TW * TH;
  const bool in_tile_thread = (tid < tile_threads);
  const int lx = in_tile_thread ? (tid % TW) : 0;
  const int ly = in_tile_thread ? (tid / TW) : 0;
  const int x = tile_x * TW + lx;
  const int y = tile_y * TH + ly;
  const bool valid_pix = in_tile_thread && (x < Wimg) && (y < Himg);
  const int pix = valid_pix ? (y * Wimg + x) : 0;

  const int num_tiles_x = (Wimg + TW - 1) / TW;
  const int tile_id = tile_y * num_tiles_x + tile_x;

  // per-pixel ray
  const float rox = valid_pix ? rays_o[pix*3+0] : 0.f;
  const float roy = valid_pix ? rays_o[pix*3+1] : 0.f;
  const float roz = valid_pix ? rays_o[pix*3+2] : 0.f;
  const float rdx = valid_pix ? rays_d[pix*3+0] : 0.f;
  const float rdy = valid_pix ? rays_d[pix*3+1] : 0.f;
  const float rdz = valid_pix ? rays_d[pix*3+2] : 0.f;

  const int32_t prim_start = tile_offsets[tile_id];
  const int32_t prim_end = tile_offsets[tile_id + 1];
  const int tile_len = (int)(prim_end - prim_start);

  float3 C = make_float3(0.f, 0.f, 0.f);
  float trans = 1.f;

  if (tile_len <= 0) {
    if (valid_pix) {
      out_rgb[pix*3+0] = 0.f;
      out_rgb[pix*3+1] = 0.f;
      out_rgb[pix*3+2] = 0.f;
      out_alpha[pix] = 0.f;
    }
    return;
  }

  constexpr int MAX_TILE_PRIMS = MLPSP_MAX_TILE_PRIMS;
  __shared__ int s_pid[MAX_TILE_PRIMS];

  const int num_threads = (int)blockDim.x;
  const int load_n = tile_len < MAX_TILE_PRIMS ? tile_len : MAX_TILE_PRIMS;

  // Load presorted per-tile primitive ids
  for (int i = tid; i < MAX_TILE_PRIMS; i += num_threads) {
    if (i < load_n) {
      s_pid[i] = (int)culled_prim_indices[prim_start + i];
    } else {
      s_pid[i] = -1;
    }
  }
  __syncthreads();

  // dynamic shared memory for weights
  extern __shared__ float sW[];
  const int Hcap = H > MLPSP_MAX_H ? MLPSP_MAX_H : H;
  const int W1n = Hcap * IN;
  const int b1n = Hcap;
  const int W2n = Hcap * Hcap;
  const int b2n = Hcap;
  const int W3n = 4 * Hcap;
  const int b3n = 4;
  float* sW1 = sW;
  float* sb1 = sW1 + W1n;
  float* sW2 = sb1 + b1n;
  float* sb2 = sW2 + W2n;
  float* sW3 = sb2 + b2n;
  float* sb3 = sW3 + W3n;
  auto load_weights = [&](int p) {
    const float* W1p = W1 + (size_t)p * H * IN;
    const float* b1p = b1 + (size_t)p * H;
    const float* W2p = W2 + (size_t)p * H * H;
    const float* b2p = b2 + (size_t)p * H;
    const float* W3p = W3 + (size_t)p * 4 * H;
    const float* b3p = b3 + (size_t)p * 4;
    copy_float_strided_vec4(sW1, W1p, W1n, tid, num_threads);
    copy_float_strided_vec4(sb1, b1p, b1n, tid, num_threads);
    copy_float_strided_vec4(sW2, W2p, W2n, tid, num_threads);
    copy_float_strided_vec4(sb2, b2p, b2n, tid, num_threads);
    copy_float_strided_vec4(sW3, W3p, W3n, tid, num_threads);
    copy_float_strided_vec4(sb3, b3p, b3n, tid, num_threads);
    __syncthreads();
  };

  auto eval_and_blend = [&](int p) {
    if (valid_pix && (trans >= 1e-3f)) {
      float t, dx, dy, dz, decay;
      const float cx = pos[p*3+0];
      const float cy = pos[p*3+1];
      const float cz = pos[p*3+2];
      const float lgx = log_scales[p*3+0];
    const float lgy = log_scales[p*3+1];
    const float lgz = log_scales[p*3+2];
    const float qw  = quats[p*4+0];
    const float qx  = quats[p*4+1];
    const float qy  = quats[p*4+2];
    const float qz  = quats[p*4+3];
      float qlx, qly, qlz;
    if (hit_and_features_rot(rox, roy, roz, rdx, rdy, rdz,
                             cx, cy, cz,
                             lgx, lgy, lgz,
                             qw, qx, qy, qz,
                             thr,
                             t, dx, dy, dz,
                             qlx, qly, qlz,
                             decay)) {
        float v6[6] = {rdx, rdy, rdz, dx, dy, dz};

        // ---- layer1: IN -> H (positional-encoding input) ----
        float h1[MLPSP_MAX_H];
        for (int h_idx = 0; h_idx < Hcap; ++h_idx) h1[h_idx] = sb1[h_idx];

        const int cos_offset = IN / 2;
        for (int c = 0; c < 6; ++c) {
          for (int f = 0; f < F; ++f) {
            const float fb = freq_bands[f];
            const float a  = attenuation[f];
            // identical encoding math to tilecenter_order
            const float v  = v6[c] * fb * a;
            const float sv = __sinf(v);
            const float cv = __cosf(v);

            const int si = c * F + f;
            const int ci = cos_offset + c * F + f;

            const float* w1s = sW1 + (size_t)si;
            const float* w1c = sW1 + (size_t)ci;
            for (int h_idx = 0; h_idx < Hcap; ++h_idx) {
              h1[h_idx] = fmaf(w1s[(size_t)h_idx * IN], sv, h1[h_idx]);
              h1[h_idx] = fmaf(w1c[(size_t)h_idx * IN], cv, h1[h_idx]);
            }
          }
        }
        for (int h_idx = 0; h_idx < Hcap; ++h_idx) h1[h_idx] = h1[h_idx] > 0.f ? h1[h_idx] : 0.f;

        // ---- layer2: H -> H ----
        float h2[MLPSP_MAX_H];
        #pragma unroll 1
        for (int o = 0; o < Hcap; ++o) {
          float acc = sb2[o];
          const float* W2row = sW2 + (size_t)o * Hcap;
          for (int h_idx = 0; h_idx < Hcap; ++h_idx) acc = fmaf(W2row[h_idx], h1[h_idx], acc);
          h2[o] = acc > 0.f ? acc : 0.f;
        }

        // ---- layer3: H -> OUT(4) ----
        float out4[4] = {sb3[0], sb3[1], sb3[2], sb3[3]};

        #pragma unroll 1
        for (int o = 0; o < Hcap; ++o) {
          const float xo = h2[o];
          const float* wcol = sW3 + (size_t)o;   // column o; k*Hcap stride below
          out4[0] = fmaf(wcol[0 * Hcap], xo, out4[0]);
          out4[1] = fmaf(wcol[1 * Hcap], xo, out4[1]);
          out4[2] = fmaf(wcol[2 * Hcap], xo, out4[2]);
          out4[3] = fmaf(wcol[3 * Hcap], xo, out4[3]);
        }

        const float R  = fast_sigmoid(out4[0]);
        const float G  = fast_sigmoid(out4[1]);
        const float Bc = fast_sigmoid(out4[2]);
        const float sigma = fast_softplus(out4[3]) * decay;

        const float alpha = 1.f - __expf(-sigma);
        const float w = alpha * trans;
        C.x = fmaf(w, R, C.x);
        C.y = fmaf(w, G, C.y);
        C.z = fmaf(w, Bc, C.z);
        trans *= (1.f - alpha);
      }
    }
  };

  // prim-major traversal in the (already sorted) list order
  for (int i = 0; i < load_n; ++i) {
    const int p = s_pid[i];
    if (p < 0) break;
    if (i + 1 < load_n) {
      const int pn = s_pid[i + 1];
      prefetch_weights_l2(W1, b1, W2, b2, W3, b3, pn, H, IN);
    }
    load_weights(p);
    eval_and_blend(p);
    const int active = (valid_pix && (trans >= 1e-3f)) ? 1 : 0;
    if (__syncthreads_or(active) == 0) break;
    __syncthreads();
  }

  // Process tail entries beyond MAX_TILE_PRIMS using global-memory list order.
  // This mirrors tilecenter_order behavior and avoids silently dropping primitives.
  if (tile_len > load_n) {
    for (int i = prim_start + load_n; i < prim_end; ++i) {
      const int p = (int)culled_prim_indices[i];
      if (p < 0 || p >= N) continue;
      if (i + 1 < prim_end) {
        const int pn = (int)culled_prim_indices[i + 1];
        if (pn >= 0 && pn < N) prefetch_weights_l2(W1, b1, W2, b2, W3, b3, pn, H, IN);
      }
      load_weights(p);
      eval_and_blend(p);
      const int active = (valid_pix && (trans >= 1e-3f)) ? 1 : 0;
      if (__syncthreads_or(active) == 0) break;
      __syncthreads();
    }
  }

  if (valid_pix) {
    out_rgb[pix*3+0] = C.x;
    out_rgb[pix*3+1] = C.y;
    out_rgb[pix*3+2] = C.z;
    out_alpha[pix]   = 1.f - trans;
  }
}


// Validate the sparse tile list before rendering.  Tile-center kernels also
// pass their fixed shared-list capacity.  Per-pixel kernels pass zero here and
// report overflow from the actual per-ray hit gather instead, because a tile's
// conservative candidate list can legitimately be larger than the hit buffer.
__global__ void validate_tile_candidate_capacity_kernel(
    const int32_t* __restrict__ tile_offsets,
    const int num_tiles,
    const int num_candidates,
    const int capacity,
    int32_t* __restrict__ status)
{
  for (int tile = (int)(blockIdx.x * blockDim.x + threadIdx.x);
       tile < num_tiles;
       tile += (int)(blockDim.x * gridDim.x)) {
    const int start = tile_offsets[tile];
    const int end = tile_offsets[tile + 1];
    if (start < 0 || end < start || end > num_candidates ||
        (tile == 0 && start != 0) ||
        (tile == num_tiles - 1 && end != num_candidates)) {
      atomicOr(status, 1);
      continue;
    }
    const int count = end - start;
    if (capacity > 0 && count > capacity) {
      atomicOr(status, 2);
      atomicMax(status + 1, count);
    }
  }
}

static inline void validate_tile_candidate_capacity(
    const torch::Tensor& culled_prim_indices,
    const torch::Tensor& tile_offsets,
    const int64_t Wimg,
    const int64_t Himg,
    const int64_t TW,
    const int64_t TH,
    const int capacity,
    const char* path_name)
{
  TORCH_CHECK(Wimg > 0 && Himg > 0 && TW > 0 && TH > 0,
              path_name, ": image and tile dimensions must be positive");
  TORCH_CHECK(culled_prim_indices.is_cuda() && culled_prim_indices.dtype() == at::kInt &&
                  culled_prim_indices.dim() == 1 && culled_prim_indices.is_contiguous(),
              path_name, ": culled_prim_indices must be a contiguous CUDA int32 vector");
  TORCH_CHECK(tile_offsets.is_cuda() && tile_offsets.dtype() == at::kInt &&
                  tile_offsets.dim() == 1 && tile_offsets.is_contiguous(),
              path_name, ": tile_offsets must be a contiguous CUDA int32 vector");

  const int64_t tiles_x = (Wimg + TW - 1) / TW;
  const int64_t tiles_y = (Himg + TH - 1) / TH;
  const int64_t num_tiles64 = tiles_x * tiles_y;
  TORCH_CHECK(num_tiles64 > 0 && num_tiles64 <= 2147483647LL,
              path_name, ": tile grid is too large");
  TORCH_CHECK(tile_offsets.numel() == num_tiles64 + 1,
              path_name, ": tile_offsets length mismatch (got ", tile_offsets.numel(),
              ", expected ", num_tiles64 + 1, ")");
  TORCH_CHECK(culled_prim_indices.numel() <= 2147483647LL,
              path_name, ": candidate list is too large for int32 offsets");

  const int num_tiles = (int)num_tiles64;
  const int num_candidates = (int)culled_prim_indices.numel();
  auto status = torch::zeros({2}, tile_offsets.options());
  const int threads = 256;
  const int needed_blocks = (num_tiles + threads - 1) / threads;
  const int blocks = needed_blocks < 4096 ? needed_blocks : 4096;
  validate_tile_candidate_capacity_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      tile_offsets.data_ptr<int32_t>(), num_tiles, num_candidates, capacity,
      status.data_ptr<int32_t>());
  AT_CUDA_CHECK(cudaGetLastError());

  // A host-visible decision is required to raise instead of returning a
  // plausible-but-wrong image.  This is one 8-byte copy per checked public
  // entry point; the render kernels themselves remain unchanged on valid data.
  auto status_cpu = status.cpu();
  const int32_t* status_ptr = status_cpu.data_ptr<int32_t>();
  TORCH_CHECK((status_ptr[0] & 1) == 0,
              path_name, ": tile_offsets must start at zero, end at the candidate-list length, "
              "and be monotonic/in bounds");
  if (capacity > 0) {
    TORCH_CHECK((status_ptr[0] & 2) == 0,
                path_name, ": tile candidate capacity ", capacity,
                " exceeded (maximum observed tile length ", status_ptr[1],
                "); refusing to truncate or use an inconsistent forward/backward order");
  }
}

static inline void check_per_pixel_intersection_overflow(
    const torch::Tensor& overflow,
    const char* path_name)
{
  auto overflow_cpu = overflow.cpu();
  TORCH_CHECK(overflow_cpu.data_ptr<int32_t>()[0] == 0,
              path_name, ": per-pixel intersection capacity 256 exceeded; "
              "refusing to return a silently truncated result");
}


// Internal test hook for masks whose equal-address lanes are non-contiguous.
__global__ void test_atomic_add_warp_agg_kernel(
    const int32_t* __restrict__ indices,
    const float* __restrict__ values,
    const int n,
    const int out_size,
    float* __restrict__ out)
{
  const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;
  const int dst = indices[i];
  if (dst >= 0 && dst < out_size) atomic_add_warp_agg(out + dst, values[i]);
}

torch::Tensor test_atomic_add_warp_agg_wrapper(
    torch::Tensor indices,
    torch::Tensor values,
    int64_t out_size)
{
  TORCH_CHECK(indices.is_cuda() && indices.dtype() == at::kInt && indices.dim() == 1 && indices.is_contiguous(),
              "indices must be a contiguous CUDA int32 vector");
  TORCH_CHECK(values.is_cuda() && values.dtype() == at::kFloat && values.dim() == 1 && values.is_contiguous(),
              "values must be a contiguous CUDA float32 vector");
  TORCH_CHECK(indices.get_device() == values.get_device(), "indices and values must be on the same CUDA device");
  TORCH_CHECK(indices.numel() == values.numel(), "indices and values must have the same length");
  TORCH_CHECK(indices.numel() <= 2147483647LL, "test input is too large");
  TORCH_CHECK(out_size > 0 && out_size <= 2147483647LL, "out_size must be a positive int32 value");

  auto out = torch::zeros({out_size}, values.options());
  const int n = (int)indices.numel();
  if (n > 0) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    test_atomic_add_warp_agg_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        indices.data_ptr<int32_t>(), values.data_ptr<float>(), n, (int)out_size,
        out.data_ptr<float>());
    AT_CUDA_CHECK(cudaGetLastError());
  }
  return out;
}


class RenderTilesCUDASinglePass : public torch::autograd::Function<RenderTilesCUDASinglePass> {
public:
    // EN: Autograd forward entry for single-pass per-pixel sorted renderer.
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor rays_o,
        torch::Tensor rays_d,
        torch::Tensor positions,
        torch::Tensor log_scales,
        torch::Tensor quats,
        torch::Tensor culled_prim_indices,
        torch::Tensor tile_offsets,
        torch::Tensor freq_bands,
        torch::Tensor attenuation,
        torch::Tensor W1, torch::Tensor b1,
        torch::Tensor W2, torch::Tensor b2,
        torch::Tensor W3, torch::Tensor b3,
        int64_t Wimg, int64_t Himg,
        int64_t TW, int64_t TH,
        double threshold)
    {
        // For now, we don't save any tensors for backward.
        // ctx->save_for_backward({ ... });

        TORCH_CHECK(rays_o.is_cuda() && rays_o.dtype()==at::kFloat && rays_o.dim()==2 && rays_o.size(1)==3);
        TORCH_CHECK(rays_d.is_cuda() && rays_d.dtype()==at::kFloat && rays_d.dim()==2 && rays_d.size(1)==3);
        TORCH_CHECK(positions.is_cuda() && positions.dtype()==at::kFloat && positions.dim()==2 && positions.size(1)==3);
        TORCH_CHECK(log_scales.is_cuda() && log_scales.dtype()==at::kFloat && log_scales.dim()==2 && log_scales.size(1)==3);
        TORCH_CHECK(quats.is_cuda() && quats.dtype()==at::kFloat && quats.dim()==2 && quats.size(1)==4);
        TORCH_CHECK(culled_prim_indices.is_cuda() && culled_prim_indices.dtype()==at::kInt && culled_prim_indices.dim()==1);
        TORCH_CHECK(tile_offsets.is_cuda() && tile_offsets.dtype()==at::kInt && tile_offsets.dim()==1);
        TORCH_CHECK(freq_bands.is_cuda() && freq_bands.dtype()==at::kFloat && freq_bands.dim()==1);
        TORCH_CHECK(attenuation.is_cuda() && attenuation.dtype()==at::kFloat && attenuation.dim()==1);

        const int64_t B = rays_o.size(0);
        TORCH_CHECK(B == (Wimg * Himg), "B must equal W*H for tile renderer");
        validate_tile_candidate_capacity(
            culled_prim_indices, tile_offsets, Wimg, Himg, TW, TH, 0,
            "render_tiles_autograd");

        const int64_t N = positions.size(0);
        const int64_t Hh = W2.size(1);
        const int64_t IN = W1.size(2);
        const int64_t F = freq_bands.size(0);
        TORCH_CHECK(IN == 6 * F * 2, "Encoded dim must be 6*F*2");

        auto opts_f = rays_o.options().dtype(at::kFloat);
        auto rgb = torch::zeros({B, 3}, opts_f);
        auto alpha = torch::zeros({B, 1}, opts_f);
        auto overflow = torch::zeros({1}, rays_o.options().dtype(at::kInt));

        // Per-pixel kernel expects a 2D block layout (threadIdx.x/y map to pixel in tile).
        dim3 block((unsigned)TW, (unsigned)TH, 1);
        dim3 grid((unsigned)((Wimg + TW - 1)/TW), (unsigned)((Himg + TH - 1)/TH), 1);
        render_tiles_kernel_with_sort<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            rays_o.data_ptr<float>(),
            rays_d.data_ptr<float>(),
            positions.data_ptr<float>(),
            log_scales.data_ptr<float>(),
            quats.data_ptr<float>(),
            culled_prim_indices.data_ptr<int32_t>(),
            tile_offsets.data_ptr<int32_t>(),
            freq_bands.data_ptr<float>(),
            attenuation.data_ptr<float>(),
            (int)F,
            W1.data_ptr<float>(), b1.data_ptr<float>(),
            W2.data_ptr<float>(), b2.data_ptr<float>(),
            W3.data_ptr<float>(), b3.data_ptr<float>(),
            (int)N, (int)Hh, (int)IN,
            (int)Wimg, (int)Himg, (int)TW, (int)TH,
            (float)threshold,
            rgb.data_ptr<float>(), alpha.data_ptr<float>(),
            overflow.data_ptr<int32_t>());
        AT_CUDA_CHECK(cudaGetLastError());
        check_per_pixel_intersection_overflow(overflow, "render_tiles_autograd");

        return {rgb, alpha};
    }

    // EN: Placeholder autograd backward returning null grads for this path.
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs)
    {
        // Placeholder: return null gradients for all inputs that require grad.
        // The order must match the inputs to forward().
        return {
            torch::Tensor(), // rays_o (doesn't require grad)
            torch::Tensor(), // rays_d (doesn't require grad)
            torch::Tensor(), // positions
            torch::Tensor(), // log_scales
            torch::Tensor(), // quats
            torch::Tensor(), // culled_prim_indices (doesn't require grad)
            torch::Tensor(), // tile_offsets (doesn't require grad)
            torch::Tensor(), // freq_bands (doesn't require grad)
            torch::Tensor(), // attenuation (doesn't require grad)
            torch::Tensor(), // W1
            torch::Tensor(), // b1
            torch::Tensor(), // W2
            torch::Tensor(), // b2
            torch::Tensor(), // W3
            torch::Tensor(), // b3
            torch::Tensor(), // Wimg (doesn't require grad)
            torch::Tensor(), // Himg (doesn't require grad)
            torch::Tensor(), // TW (doesn't require grad)
            torch::Tensor(), // TH (doesn't require grad)
            torch::Tensor()  // threshold (doesn't require grad)
        };
    }
};

class RenderTilesCUDATileCenterOrder : public torch::autograd::Function<RenderTilesCUDATileCenterOrder> {
public:
    // EN: Autograd forward entry for tile-center ordered renderer.
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor rays_o,
        torch::Tensor rays_d,
        torch::Tensor positions,
        torch::Tensor log_scales,
        torch::Tensor quats,
        torch::Tensor culled_prim_indices,
        torch::Tensor tile_offsets,
        torch::Tensor freq_bands,
        torch::Tensor attenuation,
        torch::Tensor W1, torch::Tensor b1,
        torch::Tensor W2, torch::Tensor b2,
        torch::Tensor W3, torch::Tensor b3,
        int64_t Wimg, int64_t Himg,
        int64_t TW, int64_t TH,
        double threshold)
    {
        // This kernel uses a CUB BlockRadixSort specialization that assumes 256 threads per block.
        TORCH_CHECK(((int)TW * (int)TH) > 0 && ((int)TW * (int)TH) <= 256, "Tile-center order kernel currently supports 0 < TW*TH <= 256");

        TORCH_CHECK(rays_o.is_cuda() && rays_o.dtype()==at::kFloat && rays_o.dim()==2 && rays_o.size(1)==3);
        TORCH_CHECK(rays_d.is_cuda() && rays_d.dtype()==at::kFloat && rays_d.dim()==2 && rays_d.size(1)==3);
        TORCH_CHECK(positions.is_cuda() && positions.dtype()==at::kFloat && positions.dim()==2 && positions.size(1)==3);
        TORCH_CHECK(log_scales.is_cuda() && log_scales.dtype()==at::kFloat && log_scales.dim()==2 && log_scales.size(1)==3);
        TORCH_CHECK(quats.is_cuda() && quats.dtype()==at::kFloat && quats.dim()==2 && quats.size(1)==4);
        TORCH_CHECK(culled_prim_indices.is_cuda() && culled_prim_indices.dtype()==at::kInt && culled_prim_indices.dim()==1);
        TORCH_CHECK(tile_offsets.is_cuda() && tile_offsets.dtype()==at::kInt && tile_offsets.dim()==1);
        TORCH_CHECK(freq_bands.is_cuda() && freq_bands.dtype()==at::kFloat && freq_bands.dim()==1);
        TORCH_CHECK(attenuation.is_cuda() && attenuation.dtype()==at::kFloat && attenuation.dim()==1);

        const int64_t B = rays_o.size(0);
        TORCH_CHECK(B == (Wimg * Himg), "B must equal W*H for tile renderer");
        validate_tile_candidate_capacity(
            culled_prim_indices, tile_offsets, Wimg, Himg, TW, TH, MLPSP_MAX_TILE_PRIMS,
            "render_tiles_autograd_tilecenter");

        const int64_t N = positions.size(0);
        const int64_t Hh = W2.size(1);
        const int64_t IN = W1.size(2);
        const int64_t F = freq_bands.size(0);
        TORCH_CHECK(IN == 6 * F * 2, "Encoded dim must be 6*F*2");

        auto opts_f = rays_o.options().dtype(at::kFloat);
        auto rgb = torch::zeros({B, 3}, opts_f);
        auto alpha = torch::zeros({B, 1}, opts_f);

        // Tile-center order kernel uses a 1D block (TW*TH threads) so CUB block primitives work correctly.
        dim3 block(256, 1, 1);
        dim3 grid((unsigned)((Wimg + TW - 1)/TW), (unsigned)((Himg + TH - 1)/TH), 1);

        const int Hcap = (int)(Hh > MLPSP_MAX_H ? MLPSP_MAX_H : Hh);
        const int shared_floats = Hcap * (int)IN + Hcap + Hcap * Hcap + Hcap + 4 * Hcap + 4;
        const size_t shared_bytes = (size_t)shared_floats * sizeof(float);

        render_tiles_kernel_tilecenter_order<<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
            rays_o.data_ptr<float>(),
            rays_d.data_ptr<float>(),
            positions.data_ptr<float>(),
            log_scales.data_ptr<float>(),
            quats.data_ptr<float>(),
            culled_prim_indices.data_ptr<int32_t>(),
            tile_offsets.data_ptr<int32_t>(),
            freq_bands.data_ptr<float>(),
            attenuation.data_ptr<float>(),
            (int)F,
            W1.data_ptr<float>(), b1.data_ptr<float>(),
            W2.data_ptr<float>(), b2.data_ptr<float>(),
            W3.data_ptr<float>(), b3.data_ptr<float>(),
            (int)N, (int)Hh, (int)IN,
            (int)Wimg, (int)Himg, (int)TW, (int)TH,
            (float)threshold,
            rgb.data_ptr<float>(), alpha.data_ptr<float>());
        AT_CUDA_CHECK(cudaGetLastError());

        return {rgb, alpha};
    }

    // EN: Placeholder autograd backward for tile-center ordered forward.
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs)
    {
        return {
            torch::Tensor(), // rays_o
            torch::Tensor(), // rays_d
            torch::Tensor(), // positions
            torch::Tensor(), // log_scales
            torch::Tensor(), // quats
            torch::Tensor(), // culled_prim_indices
            torch::Tensor(), // tile_offsets
            torch::Tensor(), // freq_bands
            torch::Tensor(), // attenuation
            torch::Tensor(), // W1
            torch::Tensor(), // b1
            torch::Tensor(), // W2
            torch::Tensor(), // b2
            torch::Tensor(), // W3
            torch::Tensor(), // b3
            torch::Tensor(), // Wimg
            torch::Tensor(), // Himg
            torch::Tensor(), // TW
            torch::Tensor(), // TH
            torch::Tensor()  // threshold
        };
    }
};

class RenderTilesCUDATileCenterPresorted : public torch::autograd::Function<RenderTilesCUDATileCenterPresorted> {
public:
    // EN: Autograd forward entry for tile-center renderer with pre-sorted lists.
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor rays_o,
        torch::Tensor rays_d,
        torch::Tensor positions,
        torch::Tensor log_scales,
        torch::Tensor quats,
        torch::Tensor culled_prim_indices,
        torch::Tensor tile_offsets,
        torch::Tensor freq_bands,
        torch::Tensor attenuation,
        torch::Tensor W1, torch::Tensor b1,
        torch::Tensor W2, torch::Tensor b2,
        torch::Tensor W3, torch::Tensor b3,
        int64_t Wimg, int64_t Himg,
        int64_t TW, int64_t TH,
        double threshold)
    {
        TORCH_CHECK(((int)TW * (int)TH) <= 256, "Tile size too large");

        TORCH_CHECK(rays_o.is_cuda() && rays_o.dtype()==at::kFloat && rays_o.dim()==2 && rays_o.size(1)==3);
        TORCH_CHECK(rays_d.is_cuda() && rays_d.dtype()==at::kFloat && rays_d.dim()==2 && rays_d.size(1)==3);
        TORCH_CHECK(positions.is_cuda() && positions.dtype()==at::kFloat && positions.dim()==2 && positions.size(1)==3);
        TORCH_CHECK(log_scales.is_cuda() && log_scales.dtype()==at::kFloat && log_scales.dim()==2 && log_scales.size(1)==3);
        TORCH_CHECK(culled_prim_indices.is_cuda() && culled_prim_indices.dtype()==at::kInt && culled_prim_indices.dim()==1);
        TORCH_CHECK(tile_offsets.is_cuda() && tile_offsets.dtype()==at::kInt && tile_offsets.dim()==1);
        TORCH_CHECK(freq_bands.is_cuda() && freq_bands.dtype()==at::kFloat && freq_bands.dim()==1);
        TORCH_CHECK(attenuation.is_cuda() && attenuation.dtype()==at::kFloat && attenuation.dim()==1);

        const int64_t B = rays_o.size(0);
        TORCH_CHECK(B == (Wimg * Himg), "B must equal W*H for tile renderer");

        const int64_t N = positions.size(0);
        const int64_t Hh = W2.size(1);
        const int64_t IN = W1.size(2);
        const int64_t F = freq_bands.size(0);
        TORCH_CHECK(IN == 6 * F * 2, "Encoded dim must be 6*F*2");

        auto opts_f = rays_o.options().dtype(at::kFloat);
        auto rgb = torch::zeros({B, 3}, opts_f);
        auto alpha = torch::zeros({B, 1}, opts_f);

        dim3 block(256, 1, 1);
        dim3 grid((unsigned)((Wimg + TW - 1)/TW), (unsigned)((Himg + TH - 1)/TH), 1);

        const int Hcap = (int)(Hh > MLPSP_MAX_H ? MLPSP_MAX_H : Hh);
        const int shared_floats = Hcap * (int)IN + Hcap + Hcap * Hcap + Hcap + 4 * Hcap + 4;
        const size_t shared_bytes = (size_t)shared_floats * sizeof(float);

        render_tiles_kernel_tilecenter_presorted<<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
            rays_o.data_ptr<float>(),
            rays_d.data_ptr<float>(),
            positions.data_ptr<float>(),
            log_scales.data_ptr<float>(),
            quats.data_ptr<float>(),
            culled_prim_indices.data_ptr<int32_t>(),
            tile_offsets.data_ptr<int32_t>(),
            freq_bands.data_ptr<float>(),
            attenuation.data_ptr<float>(),
            (int)F,
            W1.data_ptr<float>(), b1.data_ptr<float>(),
            W2.data_ptr<float>(), b2.data_ptr<float>(),
            W3.data_ptr<float>(), b3.data_ptr<float>(),
            (int)N, (int)Hh, (int)IN,
            (int)Wimg, (int)Himg, (int)TW, (int)TH,
            (float)threshold,
            rgb.data_ptr<float>(), alpha.data_ptr<float>());
        AT_CUDA_CHECK(cudaGetLastError());

        return {rgb, alpha};
    }

    // EN: Placeholder autograd backward for presorted tile-center forward.
    // Gradient rendering is invoked explicitly from cumodel.py.
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs)
    {
        return {
            torch::Tensor(), // rays_o
            torch::Tensor(), // rays_d
            torch::Tensor(), // positions
            torch::Tensor(), // log_scales
            torch::Tensor(), // quats
            torch::Tensor(), // culled_prim_indices
            torch::Tensor(), // tile_offsets
            torch::Tensor(), // freq_bands
            torch::Tensor(), // attenuation
            torch::Tensor(), // W1
            torch::Tensor(), // b1
            torch::Tensor(), // W2
            torch::Tensor(), // b2
            torch::Tensor(), // W3
            torch::Tensor(), // b3
            torch::Tensor(), // Wimg
            torch::Tensor(), // Himg
            torch::Tensor(), // TW
            torch::Tensor(), // TH
            torch::Tensor()  // threshold
        };
    }
};

// EN: Python-exposed wrapper calling single-pass autograd apply.
std::tuple<torch::Tensor, torch::Tensor> render_tiles_autograd_wrapper(
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor positions,
    torch::Tensor log_scales,
    torch::Tensor quats,
    torch::Tensor culled_prim_indices,
    torch::Tensor tile_offsets,
    torch::Tensor freq_bands,
    torch::Tensor attenuation,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    torch::Tensor W3, torch::Tensor b3,
    int64_t Wimg, int64_t Himg,
    int64_t TW, int64_t TH,
    double threshold)
{
    auto result = RenderTilesCUDASinglePass::apply(
        rays_o, rays_d, positions, log_scales, quats, culled_prim_indices, tile_offsets,
        freq_bands, attenuation, W1, b1, W2, b2, W3, b3,
        Wimg, Himg, TW, TH, threshold);
    return {result[0], result[1]};
}

// EN: Python-exposed wrapper calling tile-center-order autograd apply.
std::tuple<torch::Tensor, torch::Tensor> render_tiles_autograd_tilecenter_wrapper(
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor positions,
    torch::Tensor log_scales,
    torch::Tensor quats,
    torch::Tensor culled_prim_indices,
    torch::Tensor tile_offsets,
    torch::Tensor freq_bands,
    torch::Tensor attenuation,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    torch::Tensor W3, torch::Tensor b3,
    int64_t Wimg, int64_t Himg,
    int64_t TW, int64_t TH,
    double threshold)
{
    auto result = RenderTilesCUDATileCenterOrder::apply(
        rays_o, rays_d, positions, log_scales, quats, culled_prim_indices, tile_offsets,
        freq_bands, attenuation, W1, b1, W2, b2, W3, b3,
        Wimg, Himg, TW, TH, threshold);
    return {result[0], result[1]};
}

// EN: Python-exposed wrapper calling presorted tile-center autograd apply.
std::tuple<torch::Tensor, torch::Tensor> render_tiles_autograd_tilecenter_presorted_wrapper(
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor positions,
    torch::Tensor log_scales,
    torch::Tensor quats,
    torch::Tensor culled_prim_indices,
    torch::Tensor tile_offsets,
    torch::Tensor freq_bands,
    torch::Tensor attenuation,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    torch::Tensor W3, torch::Tensor b3,
    int64_t Wimg, int64_t Himg,
    int64_t TW, int64_t TH,
    double threshold)
{
    auto result = RenderTilesCUDATileCenterPresorted::apply(
        rays_o, rays_d, positions, log_scales, quats, culled_prim_indices, tile_offsets,
        freq_bands, attenuation, W1, b1, W2, b2, W3, b3,
        Wimg, Himg, TW, TH, threshold);
    return {result[0], result[1]};
}

// EN: Forward-only Python wrapper for per-pixel sorted renderer (no autograd).
std::tuple<torch::Tensor, torch::Tensor> render_tiles_forward_wrapper(
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor positions,
    torch::Tensor log_scales,
    torch::Tensor quats,
    torch::Tensor culled_prim_indices,
    torch::Tensor tile_offsets,
    torch::Tensor freq_bands,
    torch::Tensor attenuation,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    torch::Tensor W3, torch::Tensor b3,
    int64_t Wimg, int64_t Himg,
    int64_t TW, int64_t TH,
    double threshold)
{
    TORCH_CHECK(rays_o.is_cuda() && rays_o.dtype()==at::kFloat && rays_o.dim()==2 && rays_o.size(1)==3);
    TORCH_CHECK(rays_d.is_cuda() && rays_d.dtype()==at::kFloat && rays_d.dim()==2 && rays_d.size(1)==3);
    TORCH_CHECK(positions.is_cuda() && positions.dtype()==at::kFloat && positions.dim()==2 && positions.size(1)==3);
    TORCH_CHECK(log_scales.is_cuda() && log_scales.dtype()==at::kFloat && log_scales.dim()==2 && log_scales.size(1)==3);
    TORCH_CHECK(quats.is_cuda() && quats.dtype()==at::kFloat && quats.dim()==2 && quats.size(1)==4);
    TORCH_CHECK(culled_prim_indices.is_cuda() && culled_prim_indices.dtype()==at::kInt && culled_prim_indices.dim()==1);
    TORCH_CHECK(tile_offsets.is_cuda() && tile_offsets.dtype()==at::kInt && tile_offsets.dim()==1);
    TORCH_CHECK(freq_bands.is_cuda() && freq_bands.dtype()==at::kFloat && freq_bands.dim()==1);
    TORCH_CHECK(attenuation.is_cuda() && attenuation.dtype()==at::kFloat && attenuation.dim()==1);

    const int64_t B = rays_o.size(0);
    TORCH_CHECK(B == (Wimg * Himg), "B must equal W*H for tile renderer");
    validate_tile_candidate_capacity(
        culled_prim_indices, tile_offsets, Wimg, Himg, TW, TH, 0,
        "render_tiles_forward");

    const int64_t N = positions.size(0);
    const int64_t Hh = W2.size(1);
    const int64_t IN = W1.size(2);
    const int64_t F = freq_bands.size(0);
    TORCH_CHECK(IN == 6 * F * 2, "Encoded dim must be 6*F*2");
    TORCH_CHECK(Hh <= MLPSP_MAX_H, "render_tiles_backward_tilecenter_full currently supports hidden_dim <= MLPSP_MAX_H");

    auto opts_f = rays_o.options().dtype(at::kFloat);
    auto rgb = torch::zeros({B, 3}, opts_f);
    auto alpha = torch::zeros({B, 1}, opts_f);
    auto overflow = torch::zeros({1}, rays_o.options().dtype(at::kInt));

    dim3 block((unsigned)TW, (unsigned)TH, 1);
    dim3 grid((unsigned)((Wimg + TW - 1)/TW), (unsigned)((Himg + TH - 1)/TH), 1);
    render_tiles_kernel_with_sort<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        rays_o.data_ptr<float>(),
        rays_d.data_ptr<float>(),
        positions.data_ptr<float>(),
        log_scales.data_ptr<float>(),
        quats.data_ptr<float>(),
        culled_prim_indices.data_ptr<int32_t>(),
        tile_offsets.data_ptr<int32_t>(),
        freq_bands.data_ptr<float>(),
        attenuation.data_ptr<float>(),
        (int)F,
        W1.data_ptr<float>(), b1.data_ptr<float>(),
        W2.data_ptr<float>(), b2.data_ptr<float>(),
        W3.data_ptr<float>(), b3.data_ptr<float>(),
        (int)N, (int)Hh, (int)IN,
        (int)Wimg, (int)Himg, (int)TW, (int)TH,
        (float)threshold,
        rgb.data_ptr<float>(), alpha.data_ptr<float>(),
        overflow.data_ptr<int32_t>());
    AT_CUDA_CHECK(cudaGetLastError());
    check_per_pixel_intersection_overflow(overflow, "render_tiles_forward");
    return {rgb, alpha};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> render_tiles_forward_per_pixel_with_feat_wrapper(
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor positions,
    torch::Tensor log_scales,
    torch::Tensor quats,
    torch::Tensor culled_prim_indices,
    torch::Tensor tile_offsets,
    torch::Tensor freq_bands,
    torch::Tensor attenuation,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    torch::Tensor W3, torch::Tensor b3,
    torch::Tensor feat_attr,
    double gating_beta,
    double gating_temp,
    int64_t Wimg, int64_t Himg,
    int64_t TW, int64_t TH,
    double threshold)
{
    TORCH_CHECK(rays_o.is_cuda() && rays_o.dtype()==at::kFloat && rays_o.dim()==2 && rays_o.size(1)==3);
    TORCH_CHECK(rays_d.is_cuda() && rays_d.dtype()==at::kFloat && rays_d.dim()==2 && rays_d.size(1)==3);
    TORCH_CHECK(positions.is_cuda() && positions.dtype()==at::kFloat && positions.dim()==2 && positions.size(1)==3);
    TORCH_CHECK(log_scales.is_cuda() && log_scales.dtype()==at::kFloat && log_scales.dim()==2 && log_scales.size(1)==3);
    TORCH_CHECK(quats.is_cuda() && quats.dtype()==at::kFloat && quats.dim()==2 && quats.size(1)==4);
    TORCH_CHECK(culled_prim_indices.is_cuda() && culled_prim_indices.dtype()==at::kInt && culled_prim_indices.dim()==1);
    TORCH_CHECK(tile_offsets.is_cuda() && tile_offsets.dtype()==at::kInt && tile_offsets.dim()==1);
    TORCH_CHECK(freq_bands.is_cuda() && freq_bands.dtype()==at::kFloat && freq_bands.dim()==1);
    TORCH_CHECK(attenuation.is_cuda() && attenuation.dtype()==at::kFloat && attenuation.dim()==1);
    TORCH_CHECK(feat_attr.is_cuda() && feat_attr.dtype()==at::kFloat && (feat_attr.dim()==2 || feat_attr.dim()==3));

    const int64_t B = rays_o.size(0);
    TORCH_CHECK(B == (Wimg * Himg), "B must equal W*H for tile renderer");
    validate_tile_candidate_capacity(
        culled_prim_indices, tile_offsets, Wimg, Himg, TW, TH, 0,
        "render_tiles_forward_per_pixel_with_feat");

    const int64_t N = positions.size(0);
    const int64_t Hh = W2.size(1);
    const int64_t IN = W1.size(2);
    const int64_t F = freq_bands.size(0);
    TORCH_CHECK(IN == 6 * F * 2, "Encoded dim must be 6*F*2");
    TORCH_CHECK(feat_attr.size(0) == N, "feat_attr first dimension must match primitive count");

    int64_t feat_slots = 1;
    int64_t Cfeat = 0;
    if (feat_attr.dim() == 2) {
      Cfeat = feat_attr.size(1);
    } else {
      feat_slots = feat_attr.size(1);
      Cfeat = feat_attr.size(2);
      TORCH_CHECK(feat_slots == 4 || feat_slots == 5, "3D feat_attr expects slots in {4,5}");
    }
    TORCH_CHECK(Cfeat > 0, "feat_attr feature dimension must be positive");
    TORCH_CHECK(Cfeat <= 512, "feat_attr feature dimension must be <= 512");

    auto opts_f = rays_o.options().dtype(at::kFloat);
    auto rgb = torch::zeros({B, 3}, opts_f);
    auto alpha = torch::zeros({B, 1}, opts_f);
    auto feat = torch::zeros({B, Cfeat}, opts_f);
    auto overflow = torch::zeros({1}, rays_o.options().dtype(at::kInt));

    dim3 block((unsigned)TW, (unsigned)TH, 1);
    dim3 grid((unsigned)((Wimg + TW - 1)/TW), (unsigned)((Himg + TH - 1)/TH), 1);
    render_tiles_kernel_with_sort_and_feat<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        rays_o.data_ptr<float>(),
        rays_d.data_ptr<float>(),
        positions.data_ptr<float>(),
        log_scales.data_ptr<float>(),
        quats.data_ptr<float>(),
        culled_prim_indices.data_ptr<int32_t>(),
        tile_offsets.data_ptr<int32_t>(),
        freq_bands.data_ptr<float>(),
        attenuation.data_ptr<float>(),
        (int)F,
        W1.data_ptr<float>(), b1.data_ptr<float>(),
        W2.data_ptr<float>(), b2.data_ptr<float>(),
        W3.data_ptr<float>(), b3.data_ptr<float>(),
        feat_attr.data_ptr<float>(), (int)feat_slots, (int)Cfeat,
        (float)gating_beta, (float)gating_temp,
        (int)N, (int)Hh, (int)IN,
        (int)Wimg, (int)Himg, (int)TW, (int)TH,
        (float)threshold,
        rgb.data_ptr<float>(), alpha.data_ptr<float>(), feat.data_ptr<float>(),
        overflow.data_ptr<int32_t>());
    AT_CUDA_CHECK(cudaGetLastError());
    check_per_pixel_intersection_overflow(overflow, "render_tiles_forward_per_pixel_with_feat");
    return {rgb, alpha, feat};
}

// EN: Forward-only Python wrapper for tile-center ordered renderer (dtype-dispatched).
std::tuple<torch::Tensor, torch::Tensor> render_tiles_forward_tilecenter_wrapper(
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor positions,
    torch::Tensor log_scales,
    torch::Tensor quats,
    torch::Tensor culled_prim_indices,
    torch::Tensor tile_offsets,
    torch::Tensor freq_bands,
    torch::Tensor attenuation,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    torch::Tensor W3, torch::Tensor b3,
    int64_t Wimg, int64_t Himg,
    int64_t TW, int64_t TH,
    double threshold)
{
    TORCH_CHECK(((int)TW * (int)TH) > 0 && ((int)TW * (int)TH) <= 256, "Tile-center order kernel currently supports 0 < TW*TH <= 256");
    TORCH_CHECK(rays_o.is_cuda() && rays_o.dtype()==at::kFloat && rays_o.dim()==2 && rays_o.size(1)==3);
    TORCH_CHECK(rays_d.is_cuda() && rays_d.dtype()==at::kFloat && rays_d.dim()==2 && rays_d.size(1)==3);
    TORCH_CHECK(positions.is_cuda() && positions.dtype()==at::kFloat && positions.dim()==2 && positions.size(1)==3);
    TORCH_CHECK(log_scales.is_cuda() && log_scales.dtype()==at::kFloat && log_scales.dim()==2 && log_scales.size(1)==3);
    TORCH_CHECK(quats.is_cuda() && quats.dtype()==at::kFloat && quats.dim()==2 && quats.size(1)==4);
    TORCH_CHECK(culled_prim_indices.is_cuda() && culled_prim_indices.dtype()==at::kInt && culled_prim_indices.dim()==1);
    TORCH_CHECK(tile_offsets.is_cuda() && tile_offsets.dtype()==at::kInt && tile_offsets.dim()==1);
    TORCH_CHECK(freq_bands.is_cuda() && freq_bands.dtype()==at::kFloat && freq_bands.dim()==1);
    TORCH_CHECK(attenuation.is_cuda() && attenuation.dtype()==at::kFloat && attenuation.dim()==1);
    TORCH_CHECK(W1.is_cuda() && b1.is_cuda() && W2.is_cuda() && b2.is_cuda() && W3.is_cuda() && b3.is_cuda(),
                "MLP tensors must be CUDA tensors");
    TORCH_CHECK(W1.scalar_type() == b1.scalar_type() &&
                W1.scalar_type() == W2.scalar_type() &&
                W1.scalar_type() == b2.scalar_type() &&
                W1.scalar_type() == W3.scalar_type() &&
                W1.scalar_type() == b3.scalar_type(),
                "W/b tensors must share the same dtype");
    TORCH_CHECK(
        W1.scalar_type() == at::kFloat || W1.scalar_type() == at::kHalf || W1.scalar_type() == at::kBFloat16,
        "W/b tensors must be float32, float16, or bfloat16");

    const int64_t B = rays_o.size(0);
    TORCH_CHECK(B == (Wimg * Himg), "B must equal W*H for tile renderer");
    validate_tile_candidate_capacity(
        culled_prim_indices, tile_offsets, Wimg, Himg, TW, TH, MLPSP_MAX_TILE_PRIMS,
        "render_tiles_forward_tilecenter");

    const int64_t N = positions.size(0);
    const int64_t Hh = W2.size(1);
    const int64_t IN = W1.size(2);
    const int64_t F = freq_bands.size(0);
    TORCH_CHECK(IN == 6 * F * 2, "Encoded dim must be 6*F*2");

    auto opts_f = rays_o.options().dtype(at::kFloat);
    auto rgb = torch::zeros({B, 3}, opts_f);
    auto alpha = torch::zeros({B, 1}, opts_f);

    dim3 block(256, 1, 1);
    dim3 grid((unsigned)((Wimg + TW - 1)/TW), (unsigned)((Himg + TH - 1)/TH), 1);
    const int Hcap = (int)(Hh > MLPSP_MAX_H ? MLPSP_MAX_H : Hh);
    const int shared_floats = Hcap * (int)IN + Hcap + Hcap * Hcap + Hcap + 4 * Hcap + 4;
    const size_t shared_bytes = (size_t)shared_floats * sizeof(float);

    if (W1.scalar_type() == at::kFloat) {
        render_tiles_kernel_tilecenter_order<float><<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        rays_o.data_ptr<float>(),
        rays_d.data_ptr<float>(),
        positions.data_ptr<float>(),
        log_scales.data_ptr<float>(),
        quats.data_ptr<float>(),
        culled_prim_indices.data_ptr<int32_t>(),
        tile_offsets.data_ptr<int32_t>(),
        freq_bands.data_ptr<float>(),
        attenuation.data_ptr<float>(),
        (int)F,
        W1.data_ptr<float>(), b1.data_ptr<float>(),
        W2.data_ptr<float>(), b2.data_ptr<float>(),
        W3.data_ptr<float>(), b3.data_ptr<float>(),
        (int)N, (int)Hh, (int)IN,
        (int)Wimg, (int)Himg, (int)TW, (int)TH,
        (float)threshold,
        rgb.data_ptr<float>(), alpha.data_ptr<float>());
    } else if (W1.scalar_type() == at::kHalf) {
        render_tiles_kernel_tilecenter_order<__half><<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
          rays_o.data_ptr<float>(),
          rays_d.data_ptr<float>(),
          positions.data_ptr<float>(),
          log_scales.data_ptr<float>(),
          quats.data_ptr<float>(),
          culled_prim_indices.data_ptr<int32_t>(),
          tile_offsets.data_ptr<int32_t>(),
          freq_bands.data_ptr<float>(),
          attenuation.data_ptr<float>(),
          (int)F,
          reinterpret_cast<const __half*>(W1.data_ptr<at::Half>()), reinterpret_cast<const __half*>(b1.data_ptr<at::Half>()),
          reinterpret_cast<const __half*>(W2.data_ptr<at::Half>()), reinterpret_cast<const __half*>(b2.data_ptr<at::Half>()),
          reinterpret_cast<const __half*>(W3.data_ptr<at::Half>()), reinterpret_cast<const __half*>(b3.data_ptr<at::Half>()),
          (int)N, (int)Hh, (int)IN,
          (int)Wimg, (int)Himg, (int)TW, (int)TH,
          (float)threshold,
          rgb.data_ptr<float>(), alpha.data_ptr<float>());
    } else {
        render_tiles_kernel_tilecenter_order<__nv_bfloat16><<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
          rays_o.data_ptr<float>(),
          rays_d.data_ptr<float>(),
          positions.data_ptr<float>(),
          log_scales.data_ptr<float>(),
          quats.data_ptr<float>(),
          culled_prim_indices.data_ptr<int32_t>(),
          tile_offsets.data_ptr<int32_t>(),
          freq_bands.data_ptr<float>(),
          attenuation.data_ptr<float>(),
          (int)F,
          reinterpret_cast<const __nv_bfloat16*>(W1.data_ptr<at::BFloat16>()), reinterpret_cast<const __nv_bfloat16*>(b1.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16*>(W2.data_ptr<at::BFloat16>()), reinterpret_cast<const __nv_bfloat16*>(b2.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16*>(W3.data_ptr<at::BFloat16>()), reinterpret_cast<const __nv_bfloat16*>(b3.data_ptr<at::BFloat16>()),
          (int)N, (int)Hh, (int)IN,
          (int)Wimg, (int)Himg, (int)TW, (int)TH,
          (float)threshold,
          rgb.data_ptr<float>(), alpha.data_ptr<float>());
    }
    AT_CUDA_CHECK(cudaGetLastError());
    return {rgb, alpha};
}

std::tuple<torch::Tensor, torch::Tensor> render_tiles_forward_tilecenter_stats_wrapper(
    torch::Tensor rays_o, torch::Tensor rays_d,
    torch::Tensor positions, torch::Tensor log_scales, torch::Tensor quats,
    torch::Tensor culled_prim_indices, torch::Tensor tile_offsets,
    torch::Tensor freq_bands, torch::Tensor attenuation,
    int F,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    torch::Tensor W3, torch::Tensor b3,
    int N, int H, int IN,
    int Wimg, int Himg, int TW, int TH,
    float thr,
    torch::Tensor gt_rgb,                     // ✅ NEW: (Himg*Wimg, 3) float cuda
    torch::Tensor stat_sum_w, torch::Tensor stat_max_w, torch::Tensor stat_hits,
    torch::Tensor stat_sum_werr,              // ✅ NEW: (N) float cuda
    torch::Tensor stat_contrib_count          // ✅ NEW: (N) int32 cuda
){
    TORCH_CHECK(((int)TW * (int)TH) > 0 && ((int)TW * (int)TH) <= 256, "Tile-center order kernel currently supports 0 < TW*TH <= 256");
    TORCH_CHECK(rays_o.is_cuda() && rays_o.dtype()==at::kFloat && rays_o.dim()==2 && rays_o.size(1)==3);
    TORCH_CHECK(rays_d.is_cuda() && rays_d.dtype()==at::kFloat && rays_d.dim()==2 && rays_d.size(1)==3);
    TORCH_CHECK(positions.is_cuda() && positions.dtype()==at::kFloat && positions.dim()==2 && positions.size(1)==3);
    TORCH_CHECK(log_scales.is_cuda() && log_scales.dtype()==at::kFloat && log_scales.dim()==2 && log_scales.size(1)==3);
    TORCH_CHECK(quats.is_cuda() && quats.dtype()==at::kFloat && quats.dim()==2 && quats.size(1)==4);

    TORCH_CHECK(culled_prim_indices.is_cuda() && culled_prim_indices.dtype()==at::kInt && culled_prim_indices.dim()==1);
    TORCH_CHECK(tile_offsets.is_cuda() && tile_offsets.dtype()==at::kInt && tile_offsets.dim()==1);
    TORCH_CHECK(freq_bands.is_cuda() && freq_bands.dtype()==at::kFloat && freq_bands.dim()==1);
    TORCH_CHECK(attenuation.is_cuda() && attenuation.dtype()==at::kFloat && attenuation.dim()==1);

    TORCH_CHECK(W1.is_cuda() && W1.dtype()==at::kFloat);
    TORCH_CHECK(b1.is_cuda() && b1.dtype()==at::kFloat);
    TORCH_CHECK(W2.is_cuda() && W2.dtype()==at::kFloat);
    TORCH_CHECK(b2.is_cuda() && b2.dtype()==at::kFloat);
    TORCH_CHECK(W3.is_cuda() && W3.dtype()==at::kFloat);
    TORCH_CHECK(b3.is_cuda() && b3.dtype()==at::kFloat);

    TORCH_CHECK(stat_sum_w.is_cuda() && stat_sum_w.dtype()==at::kFloat && stat_sum_w.numel()==N);
    TORCH_CHECK(stat_max_w.is_cuda() && stat_max_w.dtype()==at::kFloat && stat_max_w.numel()==N);
    TORCH_CHECK(stat_hits.is_cuda() && stat_hits.dtype()==at::kInt && stat_hits.numel()==N);

    // ✅ NEW checks
    TORCH_CHECK(gt_rgb.is_cuda() && gt_rgb.dtype()==at::kFloat && gt_rgb.dim()==2 && gt_rgb.size(0)==Himg*Wimg && gt_rgb.size(1)==3);
    TORCH_CHECK(stat_sum_werr.is_cuda() && stat_sum_werr.dtype()==at::kFloat && stat_sum_werr.numel()==N);
    TORCH_CHECK(stat_contrib_count.is_cuda() && stat_contrib_count.dtype()==at::kInt && stat_contrib_count.numel()==N);

    validate_tile_candidate_capacity(
        culled_prim_indices, tile_offsets, Wimg, Himg, TW, TH, MLPSP_MAX_TILE_PRIMS,
        "render_tiles_forward_tilecenter_stats");

    auto out_rgb = torch::empty({Himg * Wimg, 3}, torch::TensorOptions().device(rays_o.device()).dtype(torch::kFloat));
    auto out_alpha = torch::empty({Himg * Wimg, 1}, torch::TensorOptions().device(rays_o.device()).dtype(torch::kFloat));

    dim3 grid((Wimg + TW - 1) / TW, (Himg + TH - 1) / TH);
    dim3 block(256);

    const int Hcap = (H > 64) ? 64 : H;
    const int total_floats = Hcap * IN + Hcap + Hcap * Hcap + Hcap + 4 * Hcap + 4;
    const size_t shmem = (size_t)total_floats * sizeof(float);

    render_tiles_kernel_tilecenter_order_stats<<<grid, block, shmem>>>(
        rays_o.data_ptr<float>(), rays_d.data_ptr<float>(),
        positions.data_ptr<float>(), log_scales.data_ptr<float>(), quats.data_ptr<float>(),
        culled_prim_indices.data_ptr<int32_t>(), tile_offsets.data_ptr<int32_t>(),
        freq_bands.data_ptr<float>(), attenuation.data_ptr<float>(),
        F,
        W1.data_ptr<float>(), b1.data_ptr<float>(),
        W2.data_ptr<float>(), b2.data_ptr<float>(),
        W3.data_ptr<float>(), b3.data_ptr<float>(),
        N, H, IN,
        Wimg, Himg, TW, TH,
        thr,
        gt_rgb.data_ptr<float>(),                 // ✅ NEW
        out_rgb.data_ptr<float>(), out_alpha.data_ptr<float>(),
        stat_sum_w.data_ptr<float>(), stat_max_w.data_ptr<float>(), stat_hits.data_ptr<int32_t>(),
        stat_sum_werr.data_ptr<float>(),          // ✅ NEW
        stat_contrib_count.data_ptr<int32_t>()    // ✅ NEW
    );

    return {out_rgb, out_alpha};
}



__global__ void render_tiles_kernel_tilecenter_order_with_t(
    const float* __restrict__ rays_o,
    const float* __restrict__ rays_d,
    const float* __restrict__ pos,
    const float* __restrict__ log_scales,
    const float* __restrict__ quats,
    const int32_t* __restrict__ culled_prim_indices,
    const int32_t* __restrict__ tile_offsets,
    const float* __restrict__ freq_bands,
    const float* __restrict__ attenuation,
    const int F,
    const float* __restrict__ W1, const float* __restrict__ b1,
    const float* __restrict__ W2, const float* __restrict__ b2,
    const float* __restrict__ W3, const float* __restrict__ b3,
    const int N, const int H, const int IN,
    const int Wimg, const int Himg, const int TW, const int TH,
    const float thr,
    float* __restrict__ out_rgb,
    float* __restrict__ out_alpha,
    float* __restrict__ out_t
){
  // NOTE: This kernel is intended to be *bitwise identical* to
  // `render_tiles_kernel_tilecenter_order` for RGB/alpha, except that it
  // additionally outputs a per-pixel expected depth `t`.

  const int tile_x = blockIdx.x;
  const int tile_y = blockIdx.y;
  // Use a 1D block of (TW*TH) threads so CUB block primitives can be used safely.
  const int tid = (int)threadIdx.x;
  const int tile_threads = TW * TH;
  const bool in_tile_thread = (tid < tile_threads);
  const int lx = in_tile_thread ? (tid % TW) : 0;
  const int ly = in_tile_thread ? (tid / TW) : 0;
  const int x = tile_x * TW + lx;
  const int y = tile_y * TH + ly;
  const bool valid_pix = in_tile_thread && (x < Wimg) && (y < Himg);
  const int pix = valid_pix ? (y * Wimg + x) : 0;

  const int num_tiles_x = (Wimg + TW - 1) / TW;
  const int tile_id = tile_y * num_tiles_x + tile_x;

  // per-pixel ray
  const float rox = valid_pix ? rays_o[pix*3+0] : 0.f;
  const float roy = valid_pix ? rays_o[pix*3+1] : 0.f;
  const float roz = valid_pix ? rays_o[pix*3+2] : 0.f;
  const float rdx = valid_pix ? rays_d[pix*3+0] : 0.f;
  const float rdy = valid_pix ? rays_d[pix*3+1] : 0.f;
  const float rdz = valid_pix ? rays_d[pix*3+2] : 0.f;

  // tile-center ray for ordering key
  const int cxp = min(tile_x * TW + (TW >> 1), Wimg - 1);
  const int cyp = min(tile_y * TH + (TH >> 1), Himg - 1);
  const int pix_c = cyp * Wimg + cxp;
  const float crox = rays_o[pix_c*3+0];
  const float croy = rays_o[pix_c*3+1];
  const float croz = rays_o[pix_c*3+2];
  const float crdx = rays_d[pix_c*3+0];
  const float crdy = rays_d[pix_c*3+1];
  const float crdz = rays_d[pix_c*3+2];

  const int32_t prim_start = tile_offsets[tile_id];
  const int32_t prim_end = tile_offsets[tile_id + 1];
  const int tile_len = (int)(prim_end - prim_start);

  float3 C = make_float3(0.f, 0.f, 0.f);
  float trans = 1.f;
  float w_acc = 0.f;
  float t_acc = 0.f;

  if (tile_len <= 0) {
    if (valid_pix) {
      out_rgb[pix*3+0] = 0.f;
      out_rgb[pix*3+1] = 0.f;
      out_rgb[pix*3+2] = 0.f;
      out_alpha[pix] = 0.f;
      if (out_t) out_t[pix] = 0.f;
    }
    return;
  }

  // Block-level radix sort uses a fixed 256-thread block; only the first TW*TH threads map to pixels
  // while the remaining threads participate in sort/staging work.
  constexpr int MAX_TILE_PRIMS = MLPSP_MAX_TILE_PRIMS;
  constexpr int BLOCK_THREADS = 256;
  constexpr int ITEMS_PER_THREAD = MAX_TILE_PRIMS / BLOCK_THREADS; // 3
  static_assert(MAX_TILE_PRIMS % BLOCK_THREADS == 0, "MAX_TILE_PRIMS must be divisible by BLOCK_THREADS");

  __shared__ int s_pid[MAX_TILE_PRIMS];
  using BlockRadixSortT = cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD, int>;
  __shared__ typename BlockRadixSortT::TempStorage sort_storage;

  const int num_threads = (int)blockDim.x;
  const int load_n = tile_len < MAX_TILE_PRIMS ? tile_len : MAX_TILE_PRIMS;

  // Load tile list + compute key on tile-center ray, then radix-sort by key (ascending).
  // We use CUB's row-major ("blocked") arrangement: thread 0 owns items [0,1], thread 1 owns [2,3], ...
  // Use float keys directly.
  float keys[ITEMS_PER_THREAD];
  int vals[ITEMS_PER_THREAD];

#pragma unroll
  for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
    const int i = tid * ITEMS_PER_THREAD + it;
    if (i < load_n) {
      const int p = (int)culled_prim_indices[prim_start + i];
      vals[it] = p;
      const float vx = pos[p*3+0] - crox;
      const float vy = pos[p*3+1] - croy;
      const float vz = pos[p*3+2] - croz;
      const float tkey = fmaf(vx, crdx, fmaf(vy, crdy, vz * crdz));
      keys[it] = (tkey > 0.f) ? tkey : 1e30f;
    } else {
      vals[it] = -1;
      keys[it] = 1e30f;
    }
  }

  // Radix sort across the block; output remains in row-major ("blocked") layout.
  BlockRadixSortT(sort_storage).Sort(keys, vals);
  __syncthreads();

#pragma unroll
  for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
    const int i = tid * ITEMS_PER_THREAD + it;
    s_pid[i] = vals[it];
  }
  __syncthreads();

  // dynamic shared memory for weights
  extern __shared__ float sW[];
  const int Hcap = H > 64 ? 64 : H;
  const int W1n = Hcap * IN;
  const int b1n = Hcap;
  const int W2n = Hcap * Hcap;
  const int b2n = Hcap;
  const int W3n = 4 * Hcap;
  const int b3n = 4;
  float* sW1 = sW;
  float* sb1 = sW1 + W1n;
  float* sW2 = sb1 + b1n;
  float* sb2 = sW2 + W2n;
  float* sW3 = sb2 + b2n;
  float* sb3 = sW3 + W3n;
  const int total_floats = W1n + b1n + W2n + b2n + W3n + b3n;

  auto load_weights = [&](int p) {
    const float* W1p = W1 + (size_t)p * H * IN;
    const float* b1p = b1 + (size_t)p * H;
    const float* W2p = W2 + (size_t)p * H * H;
    const float* b2p = b2 + (size_t)p * H;
    const float* W3p = W3 + (size_t)p * 4 * H;
    const float* b3p = b3 + (size_t)p * 4;

    for (int idx = tid; idx < total_floats; idx += num_threads) {
      int off = idx;
      if (off < W1n) { sW1[off] = W1p[off]; continue; }
      off -= W1n;
      if (off < b1n) { sb1[off] = b1p[off]; continue; }
      off -= b1n;
      if (off < W2n) { sW2[off] = W2p[off]; continue; }
      off -= W2n;
      if (off < b2n) { sb2[off] = b2p[off]; continue; }
      off -= b2n;
      if (off < W3n) { sW3[off] = W3p[off]; continue; }
      off -= W3n;
      if (off < b3n) { sb3[off] = b3p[off]; }
    }
    __syncthreads();
  };

  auto eval_and_blend = [&](int p) {
    if (valid_pix && (trans >= 1e-3f)) {
      float t, dx, dy, dz, decay;
      const float cx = pos[p*3+0];
      const float cy = pos[p*3+1];
      const float cz = pos[p*3+2];
      const float lgx = log_scales[p*3+0];
      const float lgy = log_scales[p*3+1];
      const float lgz = log_scales[p*3+2];
      const float qw  = quats[p*4+0];
      const float qx  = quats[p*4+1];
      const float qy  = quats[p*4+2];
      const float qz  = quats[p*4+3];
      float qlx, qly, qlz;
    if (hit_and_features_rot(rox, roy, roz, rdx, rdy, rdz,
                             cx, cy, cz,
                             lgx, lgy, lgz,
                             qw, qx, qy, qz,
                             thr,
                             t, dx, dy, dz,
                             qlx, qly, qlz,
                             decay)) {
        float v6[6] = {rdx, rdy, rdz, dx, dy, dz};

        float h1[32];
        for (int h_idx = 0; h_idx < Hcap; ++h_idx) h1[h_idx] = sb1[h_idx];

        const int cos_offset = IN / 2;
        for (int c = 0; c < 6; ++c) {
          for (int f = 0; f < F; ++f) {
            const float fb = freq_bands[f];
            const float att = attenuation[f];
            const float xproj = v6[c] * fb * att;
            const int j = c * F + f;
            const float s = __sinf(xproj);
            const float co = __cosf(xproj);
            for (int h_idx = 0; h_idx < Hcap; ++h_idx) {
              const float* Wi = sW1 + (size_t)h_idx * IN;
              h1[h_idx] = fmaf(Wi[j], s, h1[h_idx]);
              h1[h_idx] = fmaf(Wi[j + cos_offset], co, h1[h_idx]);
            }
          }
        }
        for (int h_idx = 0; h_idx < Hcap; ++h_idx) h1[h_idx] = h1[h_idx] > 0.f ? h1[h_idx] : 0.f;

        float h2[32];
        for (int o = 0; o < Hcap; ++o) {
          float acc = sb2[o];
          const float* W2row = sW2 + (size_t)o * Hcap;
          for (int h_idx = 0; h_idx < Hcap; ++h_idx) acc = fmaf(W2row[h_idx], h1[h_idx], acc);
          h2[o] = acc > 0.f ? acc : 0.f;
        }

        float o3 = sb3[0], o4 = sb3[1], o5 = sb3[2], o6 = sb3[3];
        for (int h_idx = 0; h_idx < Hcap; ++h_idx) {
          const float hi = h2[h_idx];
          o3 = fmaf(sW3[0*Hcap + h_idx], hi, o3);
          o4 = fmaf(sW3[1*Hcap + h_idx], hi, o4);
          o5 = fmaf(sW3[2*Hcap + h_idx], hi, o5);
          o6 = fmaf(sW3[3*Hcap + h_idx], hi, o6);
        }

        const float sigma_raw = fast_softplus(o6);
        const float sigma = sigma_raw * decay;
        const float alpha = 1.f - __expf(-sigma);
        const float w = trans * alpha;

        // expected-depth accumulation (only extra computation vs base kernel)
        w_acc = fmaf(1.f, w, w_acc);      // w_acc += w
        t_acc = fmaf(w, t, t_acc);        // t_acc += w * t

        const float R = fast_sigmoid(o3);
        const float G = fast_sigmoid(o4);
        const float Bc = fast_sigmoid(o5);
        C.x = fmaf(w, R, C.x);
        C.y = fmaf(w, G, C.y);
        C.z = fmaf(w, Bc, C.z);
        trans *= (1.f - alpha);
      }
    }
    __syncthreads();
  };

  // sorted prefix
  for (int i = 0; i < load_n; ++i) {
    const int p = s_pid[i];
    if (p < 0 || p >= N) continue;
    load_weights(p);
    eval_and_blend(p);
    const int active = (valid_pix && (trans >= 1e-3f)) ? 1 : 0;
    if (__syncthreads_or(active) == 0) break;
  }

  // unsorted tail (if any)
  if (tile_len > load_n) {
    for (int i = prim_start + load_n; i < prim_end; ++i) {
      const int p = (int)culled_prim_indices[i];
      if (p < 0 || p >= N) continue;
      load_weights(p);
      eval_and_blend(p);
      const int active = (valid_pix && (trans >= 1e-3f)) ? 1 : 0;
      if (__syncthreads_or(active) == 0) break;
    }
  }

  if (valid_pix) {
    out_rgb[pix*3+0] = C.x;
    out_rgb[pix*3+1] = C.y;
    out_rgb[pix*3+2] = C.z;
    out_alpha[pix]   = 1.f - trans;
    if (out_t) {
      const float denom = (w_acc > 1e-8f) ? w_acc : 1.f;
      out_t[pix] = t_acc / denom;
    }
  }
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> render_tiles_forward_tilecenter_with_t_wrapper(
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor positions,
    torch::Tensor log_scales,
    torch::Tensor quats,
    torch::Tensor culled_prim_indices,
    torch::Tensor tile_offsets,
    torch::Tensor freq_bands,
    torch::Tensor attenuation,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    torch::Tensor W3, torch::Tensor b3,
    int64_t Wimg, int64_t Himg,
    int64_t TW, int64_t TH,
    double threshold)
{
    TORCH_CHECK(((int)TW * (int)TH) > 0 && ((int)TW * (int)TH) <= 256, "Tile-center order kernel currently supports 0 < TW*TH <= 256");
    TORCH_CHECK(rays_o.is_cuda() && rays_o.dtype()==at::kFloat && rays_o.dim()==2 && rays_o.size(1)==3);
    TORCH_CHECK(rays_d.is_cuda() && rays_d.dtype()==at::kFloat && rays_d.dim()==2 && rays_d.size(1)==3);
    TORCH_CHECK(positions.is_cuda() && positions.dtype()==at::kFloat && positions.dim()==2 && positions.size(1)==3);
    TORCH_CHECK(log_scales.is_cuda() && log_scales.dtype()==at::kFloat && log_scales.dim()==2 && log_scales.size(1)==3);
    TORCH_CHECK(culled_prim_indices.is_cuda() && culled_prim_indices.dtype()==at::kInt && culled_prim_indices.dim()==1);
    TORCH_CHECK(tile_offsets.is_cuda() && tile_offsets.dtype()==at::kInt && tile_offsets.dim()==1);
    TORCH_CHECK(freq_bands.is_cuda() && freq_bands.dtype()==at::kFloat && freq_bands.dim()==1);
    TORCH_CHECK(attenuation.is_cuda() && attenuation.dtype()==at::kFloat && attenuation.dim()==1);

    const int64_t B = rays_o.size(0);
    TORCH_CHECK(B == (Wimg * Himg), "B must equal W*H for tile renderer");
    validate_tile_candidate_capacity(
        culled_prim_indices, tile_offsets, Wimg, Himg, TW, TH, MLPSP_MAX_TILE_PRIMS,
        "render_tiles_forward_tilecenter_with_t");

    const int64_t N = positions.size(0);
    const int64_t Hh = W2.size(1);
    const int64_t IN = W1.size(2);
    const int64_t F = freq_bands.size(0);
    TORCH_CHECK(IN == 6 * F * 2, "Encoded dim must be 6*F*2");

    auto opts_f = rays_o.options().dtype(at::kFloat);
    auto rgb   = torch::zeros({B, 3}, opts_f);
    auto alpha = torch::zeros({B, 1}, opts_f);
    auto t     = torch::zeros({B, 1}, opts_f);

    dim3 block(256, 1, 1);
    dim3 grid((unsigned)((Wimg + TW - 1)/TW), (unsigned)((Himg + TH - 1)/TH), 1);

    const int Hcap = (int)(Hh > 64 ? 64 : Hh);
    const int shared_floats = Hcap * (int)IN + Hcap + Hcap * Hcap + Hcap + 4 * Hcap + 4;
    const size_t shared_bytes = (size_t)shared_floats * sizeof(float);

    render_tiles_kernel_tilecenter_order_with_t<<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        rays_o.data_ptr<float>(),
        rays_d.data_ptr<float>(),
        positions.data_ptr<float>(),
        log_scales.data_ptr<float>(),
        quats.data_ptr<float>(),
        culled_prim_indices.data_ptr<int32_t>(),
        tile_offsets.data_ptr<int32_t>(),
        freq_bands.data_ptr<float>(),
        attenuation.data_ptr<float>(),
        (int)F,
        W1.data_ptr<float>(), b1.data_ptr<float>(),
        W2.data_ptr<float>(), b2.data_ptr<float>(),
        W3.data_ptr<float>(), b3.data_ptr<float>(),
        (int)N, (int)Hh, (int)IN,
        (int)Wimg, (int)Himg, (int)TW, (int)TH,
        (float)threshold,
        rgb.data_ptr<float>(), alpha.data_ptr<float>(), t.data_ptr<float>());

    AT_CUDA_CHECK(cudaGetLastError());
    return {rgb, alpha, t};
}

__global__ void render_tiles_kernel_tilecenter_order_with_feat(
    const float* __restrict__ rays_o,
    const float* __restrict__ rays_d,
    const float* __restrict__ pos,
    const float* __restrict__ logs,
    const float* __restrict__ quats,
    const int32_t* __restrict__ culled_prim_indices,
    const int32_t* __restrict__ tile_offsets,
    const float* __restrict__ freq_bands,
    const float* __restrict__ attenuation,
    const int F,
    const float* __restrict__ W1, const float* __restrict__ b1,
    const float* __restrict__ W2, const float* __restrict__ b2,
    const float* __restrict__ W3, const float* __restrict__ b3,
    const float* __restrict__ feat_attr,
    const int feat_slots,
    const int Cfeat,
    const float gating_beta,
    const float gating_temp,
    const int N, const int H, const int IN,
    const int Wimg, const int Himg, const int TW, const int TH,
    const float thr,
    float* __restrict__ out_rgb,
    float* __restrict__ out_alpha,
    float* __restrict__ out_feat)
{
  const int tile_x = blockIdx.x;
  const int tile_y = blockIdx.y;
  const int tid = (int)threadIdx.x;
  const int tile_threads = TW * TH;
  const bool in_tile_thread = (tid < tile_threads);
  const int lx = in_tile_thread ? (tid % TW) : 0;
  const int ly = in_tile_thread ? (tid / TW) : 0;
  const int x = tile_x * TW + lx;
  const int y = tile_y * TH + ly;
  const bool valid_pix = in_tile_thread && (x < Wimg) && (y < Himg);
  const int pix = valid_pix ? (y * Wimg + x) : 0;

  const int num_tiles_x = (Wimg + TW - 1) / TW;
  const int tile_id = tile_y * num_tiles_x + tile_x;

  const float rox = valid_pix ? rays_o[pix*3+0] : 0.f;
  const float roy = valid_pix ? rays_o[pix*3+1] : 0.f;
  const float roz = valid_pix ? rays_o[pix*3+2] : 0.f;
  const float rdx = valid_pix ? rays_d[pix*3+0] : 0.f;
  const float rdy = valid_pix ? rays_d[pix*3+1] : 0.f;
  const float rdz = valid_pix ? rays_d[pix*3+2] : 0.f;

  const int cxp = min(tile_x * TW + (TW >> 1), Wimg - 1);
  const int cyp = min(tile_y * TH + (TH >> 1), Himg - 1);
  const int pix_c = cyp * Wimg + cxp;
  const float crox = rays_o[pix_c*3+0];
  const float croy = rays_o[pix_c*3+1];
  const float croz = rays_o[pix_c*3+2];
  const float crdx = rays_d[pix_c*3+0];
  const float crdy = rays_d[pix_c*3+1];
  const float crdz = rays_d[pix_c*3+2];

  const int32_t prim_start = tile_offsets[tile_id];
  const int32_t prim_end = tile_offsets[tile_id + 1];
  const int tile_len = (int)(prim_end - prim_start);

  float3 C = make_float3(0.f, 0.f, 0.f);
  float trans = 1.f;

  if (tile_len <= 0) {
    if (valid_pix) {
      out_rgb[pix*3+0] = 0.f;
      out_rgb[pix*3+1] = 0.f;
      out_rgb[pix*3+2] = 0.f;
      out_alpha[pix] = 0.f;
      for (int c = 0; c < Cfeat; ++c) out_feat[pix*Cfeat + c] = 0.f;
    }
    return;
  }

  constexpr int MAX_TILE_PRIMS = MLPSP_MAX_TILE_PRIMS;
  constexpr int BLOCK_THREADS = 256;
  constexpr int ITEMS_PER_THREAD = MAX_TILE_PRIMS / BLOCK_THREADS;
  static_assert(MAX_TILE_PRIMS % BLOCK_THREADS == 0, "MAX_TILE_PRIMS must be divisible by BLOCK_THREADS");

  __shared__ int s_pid[MAX_TILE_PRIMS];
  using BlockRadixSortT = cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD, int>;
  __shared__ typename BlockRadixSortT::TempStorage sort_storage;

  const int num_threads = (int)blockDim.x;
  const int load_n = tile_len < MAX_TILE_PRIMS ? tile_len : MAX_TILE_PRIMS;

  float keys[ITEMS_PER_THREAD];
  int vals[ITEMS_PER_THREAD];
#pragma unroll
  for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
    const int i = tid * ITEMS_PER_THREAD + it;
    if (i < load_n) {
      const int p = (int)culled_prim_indices[prim_start + i];
      vals[it] = p;
      const float cx = pos[p*3+0];
      const float cy = pos[p*3+1];
      const float cz = pos[p*3+2];
      const float lgx = logs[p*3+0];
      const float lgy = logs[p*3+1];
      const float lgz = logs[p*3+2];
      const float qw  = quats[p*4+0];
      const float qx  = quats[p*4+1];
      const float qy  = quats[p*4+2];
      const float qz  = quats[p*4+3];
      const float tkey = tstar_key_tilecenter(
          crox, croy, croz, crdx, crdy, crdz,
          cx, cy, cz,
          lgx, lgy, lgz,
          qw, qx, qy, qz
      );
      keys[it] = (tkey > 0.f) ? tkey : 1e30f;
    } else {
      vals[it] = -1;
      keys[it] = 1e30f;
    }
  }
  BlockRadixSortT(sort_storage).Sort(keys, vals);
  __syncthreads();
#pragma unroll
  for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
    const int i = tid * ITEMS_PER_THREAD + it;
    s_pid[i] = vals[it];
  }
  __syncthreads();

  extern __shared__ float sW[];
  const int Hcap = H > 64 ? 64 : H;
  const int W1n = Hcap * IN;
  const int b1n = Hcap;
  const int W2n = Hcap * Hcap;
  const int b2n = Hcap;
  const int W3n = 4 * Hcap;
  const int b3n = 4;
  float* sW1 = sW;
  float* sb1 = sW1 + W1n;
  float* sW2 = sb1 + b1n;
  float* sb2 = sW2 + W2n;
  float* sW3 = sb2 + b2n;
  float* sb3 = sW3 + W3n;
  const int total_floats = W1n + b1n + W2n + b2n + W3n + b3n;

  auto load_weights = [&](int p) {
    const float* W1p = W1 + (size_t)p * H * IN;
    const float* b1p = b1 + (size_t)p * H;
    const float* W2p = W2 + (size_t)p * H * H;
    const float* b2p = b2 + (size_t)p * H;
    const float* W3p = W3 + (size_t)p * 4 * H;
    const float* b3p = b3 + (size_t)p * 4;

    for (int idx = tid; idx < total_floats; idx += num_threads) {
      int off = idx;
      if (off < W1n) { sW1[off] = W1p[off]; continue; }
      off -= W1n;
      if (off < b1n) { sb1[off] = b1p[off]; continue; }
      off -= b1n;
      if (off < W2n) { sW2[off] = W2p[off]; continue; }
      off -= W2n;
      if (off < b2n) { sb2[off] = b2p[off]; continue; }
      off -= b2n;
      if (off < W3n) { sW3[off] = W3p[off]; continue; }
      off -= W3n;
      if (off < b3n) { sb3[off] = b3p[off]; }
    }
    __syncthreads();
  };

  auto eval_and_blend = [&](int p) {
    if (valid_pix && (trans >= 1e-3f)) {
      float t, dx, dy, dz, decay;
      const float cx = pos[p*3+0];
      const float cy = pos[p*3+1];
      const float cz = pos[p*3+2];
      const float lgx = logs[p*3+0];
      const float lgy = logs[p*3+1];
      const float lgz = logs[p*3+2];
      const float qw  = quats[p*4+0];
      const float qx  = quats[p*4+1];
      const float qy  = quats[p*4+2];
      const float qz  = quats[p*4+3];
      float qlx, qly, qlz;
      if (hit_and_features_rot(rox, roy, roz, rdx, rdy, rdz,
                               cx, cy, cz,
                               lgx, lgy, lgz,
                               qw, qx, qy, qz,
                               thr,
                               t, dx, dy, dz,
                               qlx, qly, qlz,
                               decay)) {
        float v6[6] = {rdx, rdy, rdz, dx, dy, dz};

        float h1[64];
        for (int h_idx = 0; h_idx < Hcap; ++h_idx) h1[h_idx] = sb1[h_idx];

        const int cos_offset = IN / 2;
        for (int c = 0; c < 6; ++c) {
          for (int f = 0; f < F; ++f) {
            const float fb = freq_bands[f];
            const float att = attenuation[f];
            const float xproj = v6[c] * fb * att;
            const int j = c * F + f;
            const float s = __sinf(xproj);
            const float co = __cosf(xproj);
            for (int h_idx = 0; h_idx < Hcap; ++h_idx) {
              const float* Wi = sW1 + (size_t)h_idx * IN;
              h1[h_idx] = fmaf(Wi[j], s, h1[h_idx]);
              h1[h_idx] = fmaf(Wi[j + cos_offset], co, h1[h_idx]);
            }
          }
        }
        for (int h_idx = 0; h_idx < Hcap; ++h_idx) h1[h_idx] = h1[h_idx] > 0.f ? h1[h_idx] : 0.f;

        float h2[64];
        for (int o = 0; o < Hcap; ++o) {
          float acc = sb2[o];
          const float* W2row = sW2 + (size_t)o * Hcap;
          for (int h_idx = 0; h_idx < Hcap; ++h_idx) acc = fmaf(W2row[h_idx], h1[h_idx], acc);
          h2[o] = acc > 0.f ? acc : 0.f;
        }

        float o3 = sb3[0], o4 = sb3[1], o5 = sb3[2], o6 = sb3[3];
        for (int h_idx = 0; h_idx < Hcap; ++h_idx) {
          const float hi = h2[h_idx];
          o3 = fmaf(sW3[0*Hcap + h_idx], hi, o3);
          o4 = fmaf(sW3[1*Hcap + h_idx], hi, o4);
          o5 = fmaf(sW3[2*Hcap + h_idx], hi, o5);
          o6 = fmaf(sW3[3*Hcap + h_idx], hi, o6);
        }

        const float sigma_raw = fast_softplus(o6);
        const float sigma = sigma_raw * decay;
        const float alpha = 1.f - __expf(-sigma);
        const float w = trans * alpha;

        const float R = fast_sigmoid(o3);
        const float G = fast_sigmoid(o4);
        const float Bc = fast_sigmoid(o5);
        C.x = fmaf(w, R, C.x);
        C.y = fmaf(w, G, C.y);
        C.z = fmaf(w, Bc, C.z);

        float* dst_feat = out_feat + (size_t)pix * Cfeat;
        if (feat_slots <= 1) {
          const float* src_feat = feat_attr + (size_t)p * Cfeat;
          for (int c = 0; c < Cfeat; ++c) dst_feat[c] = fmaf(w, src_feat[c], dst_feat[c]);
        } else if (feat_slots == 5) {
          float g0, g1, g2, g3;
          semantic_tetra_gates(qlx, qly, qlz, lgx, lgy, lgz, gating_beta, gating_temp, g0, g1, g2, g3);
          const float* base = feat_attr + ((size_t)p * feat_slots + 0) * Cfeat;
          const float* d0 = feat_attr + ((size_t)p * feat_slots + 1) * Cfeat;
          const float* d1 = feat_attr + ((size_t)p * feat_slots + 2) * Cfeat;
          const float* d2 = feat_attr + ((size_t)p * feat_slots + 3) * Cfeat;
          const float* d3 = feat_attr + ((size_t)p * feat_slots + 4) * Cfeat;
          for (int c = 0; c < Cfeat; ++c) {
            float mix = base[c];
            mix = fmaf(g0, d0[c], mix);
            mix = fmaf(g1, d1[c], mix);
            mix = fmaf(g2, d2[c], mix);
            mix = fmaf(g3, d3[c], mix);
            dst_feat[c] = fmaf(w, mix, dst_feat[c]);
          }
        } else if (feat_slots == 4) {
          float g0, g1, g2, g3;
          semantic_tetra_gates(qlx, qly, qlz, lgx, lgy, lgz, gating_beta, gating_temp, g0, g1, g2, g3);
          const float* s0 = feat_attr + ((size_t)p * feat_slots + 0) * Cfeat;
          const float* s1 = feat_attr + ((size_t)p * feat_slots + 1) * Cfeat;
          const float* s2 = feat_attr + ((size_t)p * feat_slots + 2) * Cfeat;
          const float* s3 = feat_attr + ((size_t)p * feat_slots + 3) * Cfeat;
          for (int c = 0; c < Cfeat; ++c) {
            float mix = 0.f;
            mix = fmaf(g0, s0[c], mix);
            mix = fmaf(g1, s1[c], mix);
            mix = fmaf(g2, s2[c], mix);
            mix = fmaf(g3, s3[c], mix);
            dst_feat[c] = fmaf(w, mix, dst_feat[c]);
          }
        } else {
          // Fallback for unsupported slot counts: use the first slot.
          const float* src_feat = feat_attr + (size_t)p * feat_slots * Cfeat;
          for (int c = 0; c < Cfeat; ++c) dst_feat[c] = fmaf(w, src_feat[c], dst_feat[c]);
        }
        trans *= (1.f - alpha);
      }
    }
    __syncthreads();
  };

  if (valid_pix) {
    for (int c = 0; c < Cfeat; ++c) out_feat[pix*Cfeat + c] = 0.f;
  }
  __syncthreads();

  for (int i = 0; i < load_n; ++i) {
    const int p = s_pid[i];
    if (p < 0 || p >= N) continue;
    load_weights(p);
    eval_and_blend(p);
    const int active = (valid_pix && (trans >= 1e-3f)) ? 1 : 0;
    if (__syncthreads_or(active) == 0) break;
  }
  if (tile_len > load_n) {
    for (int i = prim_start + load_n; i < prim_end; ++i) {
      const int p = (int)culled_prim_indices[i];
      if (p < 0 || p >= N) continue;
      load_weights(p);
      eval_and_blend(p);
      const int active = (valid_pix && (trans >= 1e-3f)) ? 1 : 0;
      if (__syncthreads_or(active) == 0) break;
    }
  }

  if (valid_pix) {
    out_rgb[pix*3+0] = C.x;
    out_rgb[pix*3+1] = C.y;
    out_rgb[pix*3+2] = C.z;
    out_alpha[pix]   = 1.f - trans;
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> render_tiles_forward_tilecenter_with_feat_wrapper(
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor positions,
    torch::Tensor log_scales,
    torch::Tensor quats,
    torch::Tensor culled_prim_indices,
    torch::Tensor tile_offsets,
    torch::Tensor freq_bands,
    torch::Tensor attenuation,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    torch::Tensor W3, torch::Tensor b3,
    torch::Tensor feat_attr,
    double gating_beta,
    double gating_temp,
    int64_t Wimg, int64_t Himg,
    int64_t TW, int64_t TH,
    double threshold)
{
    TORCH_CHECK(((int)TW * (int)TH) > 0 && ((int)TW * (int)TH) <= 256, "Tile-center order kernel currently supports 0 < TW*TH <= 256");
    TORCH_CHECK(rays_o.is_cuda() && rays_o.dtype()==at::kFloat && rays_o.dim()==2 && rays_o.size(1)==3);
    TORCH_CHECK(rays_d.is_cuda() && rays_d.dtype()==at::kFloat && rays_d.dim()==2 && rays_d.size(1)==3);
    TORCH_CHECK(positions.is_cuda() && positions.dtype()==at::kFloat && positions.dim()==2 && positions.size(1)==3);
    TORCH_CHECK(log_scales.is_cuda() && log_scales.dtype()==at::kFloat && log_scales.dim()==2 && log_scales.size(1)==3);
    TORCH_CHECK(quats.is_cuda() && quats.dtype()==at::kFloat && quats.dim()==2 && quats.size(1)==4);
    TORCH_CHECK(culled_prim_indices.is_cuda() && culled_prim_indices.dtype()==at::kInt && culled_prim_indices.dim()==1);
    TORCH_CHECK(tile_offsets.is_cuda() && tile_offsets.dtype()==at::kInt && tile_offsets.dim()==1);
    TORCH_CHECK(freq_bands.is_cuda() && freq_bands.dtype()==at::kFloat && freq_bands.dim()==1);
    TORCH_CHECK(attenuation.is_cuda() && attenuation.dtype()==at::kFloat && attenuation.dim()==1);
    TORCH_CHECK(feat_attr.is_cuda() && feat_attr.dtype()==at::kFloat && (feat_attr.dim()==2 || feat_attr.dim()==3));

    const int64_t B = rays_o.size(0);
    TORCH_CHECK(B == (Wimg * Himg), "B must equal W*H for tile renderer");
    validate_tile_candidate_capacity(
        culled_prim_indices, tile_offsets, Wimg, Himg, TW, TH, MLPSP_MAX_TILE_PRIMS,
        "render_tiles_forward_tilecenter_with_feat");

    const int64_t N = positions.size(0);
    const int64_t Hh = W2.size(1);
    const int64_t IN = W1.size(2);
    const int64_t F = freq_bands.size(0);
    TORCH_CHECK(IN == 6 * F * 2, "Encoded dim must be 6*F*2");
    TORCH_CHECK(feat_attr.size(0) == N, "feat_attr first dimension must match primitive count");
    int64_t feat_slots = 1;
    int64_t Cfeat = 0;
    if (feat_attr.dim() == 2) {
      Cfeat = feat_attr.size(1);
    } else {
      feat_slots = feat_attr.size(1);
      Cfeat = feat_attr.size(2);
      TORCH_CHECK(feat_slots == 4 || feat_slots == 5, "3D feat_attr expects slots in {4,5}");
    }
    TORCH_CHECK(Cfeat > 0, "feat_attr feature dimension must be positive");
    TORCH_CHECK(Cfeat <= 512, "feat_attr feature dimension must be <= 512");

    auto opts_f = rays_o.options().dtype(at::kFloat);
    auto rgb   = torch::zeros({B, 3}, opts_f);
    auto alpha = torch::zeros({B, 1}, opts_f);
    auto feat  = torch::zeros({B, Cfeat}, opts_f);

    dim3 block(256, 1, 1);
    dim3 grid((unsigned)((Wimg + TW - 1)/TW), (unsigned)((Himg + TH - 1)/TH), 1);
    const int Hcap = (int)(Hh > 64 ? 64 : Hh);
    const int shared_floats = Hcap * (int)IN + Hcap + Hcap * Hcap + Hcap + 4 * Hcap + 4;
    const size_t shared_bytes = (size_t)shared_floats * sizeof(float);

    render_tiles_kernel_tilecenter_order_with_feat<<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        rays_o.data_ptr<float>(),
        rays_d.data_ptr<float>(),
        positions.data_ptr<float>(),
        log_scales.data_ptr<float>(),
        quats.data_ptr<float>(),
        culled_prim_indices.data_ptr<int32_t>(),
        tile_offsets.data_ptr<int32_t>(),
        freq_bands.data_ptr<float>(),
        attenuation.data_ptr<float>(),
        (int)F,
        W1.data_ptr<float>(), b1.data_ptr<float>(),
        W2.data_ptr<float>(), b2.data_ptr<float>(),
        W3.data_ptr<float>(), b3.data_ptr<float>(),
        feat_attr.data_ptr<float>(), (int)feat_slots, (int)Cfeat,
        (float)gating_beta, (float)gating_temp,
        (int)N, (int)Hh, (int)IN,
        (int)Wimg, (int)Himg, (int)TW, (int)TH,
        (float)threshold,
        rgb.data_ptr<float>(), alpha.data_ptr<float>(), feat.data_ptr<float>());
    AT_CUDA_CHECK(cudaGetLastError());
    return {rgb, alpha, feat};
}

// EN: Python wrapper for backward count prepass in per-pixel traversal.
torch::Tensor render_tiles_backward_count_per_pixel_wrapper(
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor positions,
    torch::Tensor log_scales,
    torch::Tensor quats,
    torch::Tensor culled_prim_indices,
    torch::Tensor tile_offsets,
    torch::Tensor freq_bands,
    torch::Tensor attenuation,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    torch::Tensor W3, torch::Tensor b3,
    int64_t Wimg, int64_t Himg,
    int64_t TW, int64_t TH,
    double threshold)
{
    TORCH_CHECK(rays_o.is_cuda() && rays_o.dtype()==at::kFloat && rays_o.dim()==2 && rays_o.size(1)==3);
    TORCH_CHECK(rays_d.is_cuda() && rays_d.dtype()==at::kFloat && rays_d.dim()==2 && rays_d.size(1)==3);
    TORCH_CHECK(positions.is_cuda() && positions.dtype()==at::kFloat && positions.dim()==2 && positions.size(1)==3);
    TORCH_CHECK(log_scales.is_cuda() && log_scales.dtype()==at::kFloat && log_scales.dim()==2 && log_scales.size(1)==3);
    TORCH_CHECK(quats.is_cuda() && quats.dtype()==at::kFloat && quats.dim()==2 && quats.size(1)==4);
    TORCH_CHECK(culled_prim_indices.is_cuda() && culled_prim_indices.dtype()==at::kInt && culled_prim_indices.dim()==1);
    TORCH_CHECK(tile_offsets.is_cuda() && tile_offsets.dtype()==at::kInt && tile_offsets.dim()==1);
    TORCH_CHECK(freq_bands.is_cuda() && freq_bands.dtype()==at::kFloat && freq_bands.dim()==1);
    TORCH_CHECK(attenuation.is_cuda() && attenuation.dtype()==at::kFloat && attenuation.dim()==1);

    const int64_t B = rays_o.size(0);
    TORCH_CHECK(B == (Wimg * Himg), "B must equal W*H for tile renderer");
    validate_tile_candidate_capacity(
        culled_prim_indices, tile_offsets, Wimg, Himg, TW, TH, 0,
        "render_tiles_backward_count_per_pixel");

    const int64_t N = positions.size(0);
    const int64_t Hh = W2.size(1);
    const int64_t IN = W1.size(2);
    const int64_t F = freq_bands.size(0);
    TORCH_CHECK(IN == 6 * F * 2, "Encoded dim must be 6*F*2");

    auto counts = torch::zeros({B}, rays_o.options().dtype(at::kInt));
    auto overflow = torch::zeros({1}, rays_o.options().dtype(at::kInt));
    dim3 block((unsigned)TW, (unsigned)TH, 1);
    dim3 grid((unsigned)((Wimg + TW - 1)/TW), (unsigned)((Himg + TH - 1)/TH), 1);
    render_tiles_backward_count_per_pixel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        rays_o.data_ptr<float>(),
        rays_d.data_ptr<float>(),
        positions.data_ptr<float>(),
        log_scales.data_ptr<float>(),
        quats.data_ptr<float>(),
        culled_prim_indices.data_ptr<int32_t>(),
        tile_offsets.data_ptr<int32_t>(),
        freq_bands.data_ptr<float>(),
        attenuation.data_ptr<float>(),
        (int)F,
        W1.data_ptr<float>(), b1.data_ptr<float>(),
        W2.data_ptr<float>(), b2.data_ptr<float>(),
        W3.data_ptr<float>(), b3.data_ptr<float>(),
        (int)N, (int)Hh, (int)IN,
        (int)Wimg, (int)Himg, (int)TW, (int)TH,
        (float)threshold,
        counts.data_ptr<int32_t>(),
        overflow.data_ptr<int32_t>());
    AT_CUDA_CHECK(cudaGetLastError());
    check_per_pixel_intersection_overflow(overflow, "render_tiles_backward_count_per_pixel");
    return counts;
}

// EN: Python wrapper for backward count prepass in tile-center traversal.
torch::Tensor render_tiles_backward_count_tilecenter_wrapper(
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor positions,
    torch::Tensor log_scales,
    torch::Tensor quats,
    torch::Tensor culled_prim_indices,
    torch::Tensor tile_offsets,
    torch::Tensor freq_bands,
    torch::Tensor attenuation,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    torch::Tensor W3, torch::Tensor b3,
    int64_t Wimg, int64_t Himg,
    int64_t TW, int64_t TH,
    double threshold)
{
    TORCH_CHECK(((int)TW * (int)TH) > 0 && ((int)TW * (int)TH) <= 256, "Tile-center order kernel currently supports 0 < TW*TH <= 256");
    TORCH_CHECK(rays_o.is_cuda() && rays_o.dtype()==at::kFloat && rays_o.dim()==2 && rays_o.size(1)==3);
    TORCH_CHECK(rays_d.is_cuda() && rays_d.dtype()==at::kFloat && rays_d.dim()==2 && rays_d.size(1)==3);
    TORCH_CHECK(positions.is_cuda() && positions.dtype()==at::kFloat && positions.dim()==2 && positions.size(1)==3);
    TORCH_CHECK(log_scales.is_cuda() && log_scales.dtype()==at::kFloat && log_scales.dim()==2 && log_scales.size(1)==3);
    TORCH_CHECK(culled_prim_indices.is_cuda() && culled_prim_indices.dtype()==at::kInt && culled_prim_indices.dim()==1);
    TORCH_CHECK(tile_offsets.is_cuda() && tile_offsets.dtype()==at::kInt && tile_offsets.dim()==1);
    TORCH_CHECK(freq_bands.is_cuda() && freq_bands.dtype()==at::kFloat && freq_bands.dim()==1);
    TORCH_CHECK(attenuation.is_cuda() && attenuation.dtype()==at::kFloat && attenuation.dim()==1);

    const int64_t B = rays_o.size(0);
    TORCH_CHECK(B == (Wimg * Himg), "B must equal W*H for tile renderer");
    // This legacy count path has a smaller per-pixel MAX_HITS than the current
    // full tilecenter backward, so its conservative candidate cap is 256.
    validate_tile_candidate_capacity(
        culled_prim_indices, tile_offsets, Wimg, Himg, TW, TH, 256,
        "render_tiles_backward_count_tilecenter");

    const int64_t N = positions.size(0);
    const int64_t Hh = W2.size(1);
    const int64_t IN = W1.size(2);
    const int64_t F = freq_bands.size(0);
    TORCH_CHECK(IN == 6 * F * 2, "Encoded dim must be 6*F*2");

    auto counts = torch::zeros({B}, rays_o.options().dtype(at::kInt));
    dim3 block(256, 1, 1);
    dim3 grid((unsigned)((Wimg + TW - 1)/TW), (unsigned)((Himg + TH - 1)/TH), 1);
    render_tiles_backward_count_tilecenter<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        rays_o.data_ptr<float>(),
        rays_d.data_ptr<float>(),
        positions.data_ptr<float>(),
        log_scales.data_ptr<float>(),
        quats.data_ptr<float>(),
        culled_prim_indices.data_ptr<int32_t>(),
        tile_offsets.data_ptr<int32_t>(),
        freq_bands.data_ptr<float>(),
        attenuation.data_ptr<float>(),
        (int)F,
        W1.data_ptr<float>(), b1.data_ptr<float>(),
        W2.data_ptr<float>(), b2.data_ptr<float>(),
        W3.data_ptr<float>(), b3.data_ptr<float>(),
        (int)N, (int)Hh, (int)IN,
        (int)Wimg, (int)Himg, (int)TW, (int)TH,
        (float)threshold,
        counts.data_ptr<int32_t>());
    AT_CUDA_CHECK(cudaGetLastError());
    return counts;
}

// EN: Python wrapper for backward emit pass in per-pixel traversal.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
render_tiles_backward_emit_per_pixel_wrapper(
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor positions,
    torch::Tensor log_scales,
    torch::Tensor quats,
    torch::Tensor culled_prim_indices,
    torch::Tensor tile_offsets,
    torch::Tensor freq_bands,
    torch::Tensor attenuation,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    torch::Tensor W3, torch::Tensor b3,
    torch::Tensor grad_out_rgb,
    torch::Tensor grad_T_end,
    torch::Tensor counts,
    torch::Tensor offsets,
    int64_t total_records,
    int64_t Wimg, int64_t Himg,
    int64_t TW, int64_t TH,
    double threshold)
{
    TORCH_CHECK(rays_o.is_cuda() && rays_o.dtype()==at::kFloat && rays_o.dim()==2 && rays_o.size(1)==3);
    TORCH_CHECK(rays_d.is_cuda() && rays_d.dtype()==at::kFloat && rays_d.dim()==2 && rays_d.size(1)==3);
    TORCH_CHECK(positions.is_cuda() && positions.dtype()==at::kFloat && positions.dim()==2 && positions.size(1)==3);
    TORCH_CHECK(log_scales.is_cuda() && log_scales.dtype()==at::kFloat && log_scales.dim()==2 && log_scales.size(1)==3);
    TORCH_CHECK(quats.is_cuda() && quats.dtype()==at::kFloat && quats.dim()==2 && quats.size(1)==4);
    TORCH_CHECK(culled_prim_indices.is_cuda() && culled_prim_indices.dtype()==at::kInt && culled_prim_indices.dim()==1);
    TORCH_CHECK(tile_offsets.is_cuda() && tile_offsets.dtype()==at::kInt && tile_offsets.dim()==1);
    TORCH_CHECK(freq_bands.is_cuda() && freq_bands.dtype()==at::kFloat && freq_bands.dim()==1);
    TORCH_CHECK(attenuation.is_cuda() && attenuation.dtype()==at::kFloat && attenuation.dim()==1);
    TORCH_CHECK(grad_out_rgb.is_cuda() && grad_out_rgb.dtype()==at::kFloat && grad_out_rgb.dim()==2 && grad_out_rgb.size(1)==3);
    TORCH_CHECK(grad_T_end.is_cuda() && grad_T_end.dtype()==at::kFloat && grad_T_end.dim()==1);
    TORCH_CHECK(counts.is_cuda() && counts.dtype()==at::kInt && counts.dim()==1);
    TORCH_CHECK(offsets.is_cuda() && offsets.dtype()==at::kInt && offsets.dim()==1);

    const int64_t B = rays_o.size(0);
    TORCH_CHECK(B == (Wimg * Himg), "B must equal W*H for tile renderer");
    validate_tile_candidate_capacity(
        culled_prim_indices, tile_offsets, Wimg, Himg, TW, TH, 0,
        "render_tiles_backward_emit_per_pixel");
    TORCH_CHECK(grad_T_end.numel() == B, "grad_T_end must have size B");
    TORCH_CHECK(counts.numel() == B, "counts must have size B");
    TORCH_CHECK(offsets.numel() == B, "offsets must have size B");

    const int64_t N = positions.size(0);
    const int64_t Hh = W2.size(1);
    const int64_t IN = W1.size(2);
    const int64_t F = freq_bands.size(0);
    TORCH_CHECK(IN == 6 * F * 2, "Encoded dim must be 6*F*2");

    auto opts_f = rays_o.options().dtype(at::kFloat);
    auto opts_i = rays_o.options().dtype(at::kInt);
    auto out_prim_id = torch::zeros({total_records}, opts_i);
    auto out_x6 = torch::zeros({total_records, 6}, opts_f);
    auto out_grad_rgb = torch::zeros({total_records, 3}, opts_f);
    auto out_grad_sigma_raw = torch::zeros({total_records, 1}, opts_f);
    auto grad_positions = torch::zeros({N, 3}, opts_f);
    auto grad_log_scales = torch::zeros({N, 3}, opts_f);
    auto grad_quats = torch::zeros({N, 4}, opts_f);
    auto overflow = torch::zeros({1}, opts_i);
    dim3 block((unsigned)TW, (unsigned)TH, 1);
    dim3 grid((unsigned)((Wimg + TW - 1)/TW), (unsigned)((Himg + TH - 1)/TH), 1);
    render_tiles_backward_emit_per_pixel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        rays_o.data_ptr<float>(),
        rays_d.data_ptr<float>(),
        positions.data_ptr<float>(),
        log_scales.data_ptr<float>(),
        quats.data_ptr<float>(),
        culled_prim_indices.data_ptr<int32_t>(),
        tile_offsets.data_ptr<int32_t>(),
        freq_bands.data_ptr<float>(),
        attenuation.data_ptr<float>(),
        (int)F,
        W1.data_ptr<float>(), b1.data_ptr<float>(),
        W2.data_ptr<float>(), b2.data_ptr<float>(),
        W3.data_ptr<float>(), b3.data_ptr<float>(),
        (int)N, (int)Hh, (int)IN,
        (int)Wimg, (int)Himg, (int)TW, (int)TH,
        (float)threshold,
        grad_out_rgb.data_ptr<float>(),
        grad_T_end.data_ptr<float>(),
        counts.data_ptr<int32_t>(),
        offsets.data_ptr<int32_t>(),
        out_prim_id.data_ptr<int32_t>(),
        out_x6.data_ptr<float>(),
        out_grad_rgb.data_ptr<float>(),
        out_grad_sigma_raw.data_ptr<float>(),
        grad_positions.data_ptr<float>(),
        grad_log_scales.data_ptr<float>(),
        grad_quats.data_ptr<float>(),
        overflow.data_ptr<int32_t>());
    AT_CUDA_CHECK(cudaGetLastError());
    check_per_pixel_intersection_overflow(overflow, "render_tiles_backward_emit_per_pixel");
    return {out_prim_id, out_x6, out_grad_rgb, out_grad_sigma_raw, grad_positions, grad_log_scales, grad_quats};
}

// EN: Python wrapper for backward emit pass in tile-center traversal.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
render_tiles_backward_emit_tilecenter_wrapper(
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor positions,
    torch::Tensor log_scales,
    torch::Tensor quats,
    torch::Tensor culled_prim_indices,
    torch::Tensor tile_offsets,
    torch::Tensor freq_bands,
    torch::Tensor attenuation,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    torch::Tensor W3, torch::Tensor b3,
    torch::Tensor grad_out_rgb,
    torch::Tensor grad_T_end,
    torch::Tensor counts,
    torch::Tensor offsets,
    int64_t total_records,
    int64_t Wimg, int64_t Himg,
    int64_t TW, int64_t TH,
    double threshold)
{
    TORCH_CHECK(((int)TW * (int)TH) > 0 && ((int)TW * (int)TH) <= 256, "Tile-center order kernel currently supports 0 < TW*TH <= 256");
    TORCH_CHECK(rays_o.is_cuda() && rays_o.dtype()==at::kFloat && rays_o.dim()==2 && rays_o.size(1)==3);
    TORCH_CHECK(rays_d.is_cuda() && rays_d.dtype()==at::kFloat && rays_d.dim()==2 && rays_d.size(1)==3);
    TORCH_CHECK(positions.is_cuda() && positions.dtype()==at::kFloat && positions.dim()==2 && positions.size(1)==3);
    TORCH_CHECK(log_scales.is_cuda() && log_scales.dtype()==at::kFloat && log_scales.dim()==2 && log_scales.size(1)==3);
    TORCH_CHECK(culled_prim_indices.is_cuda() && culled_prim_indices.dtype()==at::kInt && culled_prim_indices.dim()==1);
    TORCH_CHECK(tile_offsets.is_cuda() && tile_offsets.dtype()==at::kInt && tile_offsets.dim()==1);
    TORCH_CHECK(freq_bands.is_cuda() && freq_bands.dtype()==at::kFloat && freq_bands.dim()==1);
    TORCH_CHECK(attenuation.is_cuda() && attenuation.dtype()==at::kFloat && attenuation.dim()==1);
    TORCH_CHECK(grad_out_rgb.is_cuda() && grad_out_rgb.dtype()==at::kFloat && grad_out_rgb.dim()==2 && grad_out_rgb.size(1)==3);
    TORCH_CHECK(grad_T_end.is_cuda() && grad_T_end.dtype()==at::kFloat && grad_T_end.dim()==1);
    TORCH_CHECK(counts.is_cuda() && counts.dtype()==at::kInt && counts.dim()==1);
    TORCH_CHECK(offsets.is_cuda() && offsets.dtype()==at::kInt && offsets.dim()==1);

    const int64_t B = rays_o.size(0);
    TORCH_CHECK(B == (Wimg * Himg), "B must equal W*H for tile renderer");
    validate_tile_candidate_capacity(
        culled_prim_indices, tile_offsets, Wimg, Himg, TW, TH, MLPSP_MAX_TILE_PRIMS,
        "render_tiles_backward_emit_tilecenter");
    TORCH_CHECK(grad_T_end.numel() == B, "grad_T_end must have size B");
    TORCH_CHECK(counts.numel() == B, "counts must have size B");
    TORCH_CHECK(offsets.numel() == B, "offsets must have size B");

    const int64_t N = positions.size(0);
    const int64_t Hh = W2.size(1);
    const int64_t IN = W1.size(2);
    const int64_t F = freq_bands.size(0);
    TORCH_CHECK(IN == 6 * F * 2, "Encoded dim must be 6*F*2");

    auto opts_f = rays_o.options().dtype(at::kFloat);
    auto opts_i = rays_o.options().dtype(at::kInt);
    auto out_prim_id = torch::zeros({total_records}, opts_i);
    auto out_x6 = torch::zeros({total_records, 6}, opts_f);
    auto out_grad_rgb = torch::zeros({total_records, 3}, opts_f);
    auto out_grad_sigma_raw = torch::zeros({total_records, 1}, opts_f);
    auto grad_positions = torch::zeros({N, 3}, opts_f);
    auto grad_log_scales = torch::zeros({N, 3}, opts_f);

    dim3 block(256, 1, 1);
    dim3 grid((unsigned)((Wimg + TW - 1)/TW), (unsigned)((Himg + TH - 1)/TH), 1);
    render_tiles_backward_emit_tilecenter<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        rays_o.data_ptr<float>(),
        rays_d.data_ptr<float>(),
        positions.data_ptr<float>(),
        log_scales.data_ptr<float>(),
        quats.data_ptr<float>(),
        culled_prim_indices.data_ptr<int32_t>(),
        tile_offsets.data_ptr<int32_t>(),
        freq_bands.data_ptr<float>(),
        attenuation.data_ptr<float>(),
        (int)F,
        W1.data_ptr<float>(), b1.data_ptr<float>(),
        W2.data_ptr<float>(), b2.data_ptr<float>(),
        W3.data_ptr<float>(), b3.data_ptr<float>(),
        (int)N, (int)Hh, (int)IN,
        (int)Wimg, (int)Himg, (int)TW, (int)TH,
        (float)threshold,
        grad_out_rgb.data_ptr<float>(),
        grad_T_end.data_ptr<float>(),
        counts.data_ptr<int32_t>(),
        offsets.data_ptr<int32_t>(),
        out_prim_id.data_ptr<int32_t>(),
        out_x6.data_ptr<float>(),
        out_grad_rgb.data_ptr<float>(),
        out_grad_sigma_raw.data_ptr<float>(),
        grad_positions.data_ptr<float>(),
        grad_log_scales.data_ptr<float>());
    AT_CUDA_CHECK(cudaGetLastError());
    return {out_prim_id, out_x6, out_grad_rgb, out_grad_sigma_raw, grad_positions, grad_log_scales};
}

torch::Tensor render_tiles_backward_featattr_tilecenter_wrapper(
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor positions,
    torch::Tensor log_scales,
    torch::Tensor quats,
    torch::Tensor culled_prim_indices,
    torch::Tensor tile_offsets,
    torch::Tensor freq_bands,
    torch::Tensor attenuation,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    torch::Tensor W3, torch::Tensor b3,
    torch::Tensor feat_attr,
    torch::Tensor grad_out_feat,
    double gating_beta,
    double gating_temp,
    int64_t Wimg, int64_t Himg,
    int64_t TW, int64_t TH,
    double threshold)
{
    TORCH_CHECK(((int)TW * (int)TH) > 0 && ((int)TW * (int)TH) <= 256, "Tile-center order kernel currently supports 0 < TW*TH <= 256");
    TORCH_CHECK(rays_o.is_cuda() && rays_o.dtype()==at::kFloat && rays_o.dim()==2 && rays_o.size(1)==3);
    TORCH_CHECK(rays_d.is_cuda() && rays_d.dtype()==at::kFloat && rays_d.dim()==2 && rays_d.size(1)==3);
    TORCH_CHECK(positions.is_cuda() && positions.dtype()==at::kFloat && positions.dim()==2 && positions.size(1)==3);
    TORCH_CHECK(log_scales.is_cuda() && log_scales.dtype()==at::kFloat && log_scales.dim()==2 && log_scales.size(1)==3);
    TORCH_CHECK(quats.is_cuda() && quats.dtype()==at::kFloat && quats.dim()==2 && quats.size(1)==4);
    TORCH_CHECK(culled_prim_indices.is_cuda() && culled_prim_indices.dtype()==at::kInt && culled_prim_indices.dim()==1);
    TORCH_CHECK(tile_offsets.is_cuda() && tile_offsets.dtype()==at::kInt && tile_offsets.dim()==1);
    TORCH_CHECK(freq_bands.is_cuda() && freq_bands.dtype()==at::kFloat && freq_bands.dim()==1);
    TORCH_CHECK(attenuation.is_cuda() && attenuation.dtype()==at::kFloat && attenuation.dim()==1);
    TORCH_CHECK(feat_attr.is_cuda() && feat_attr.dtype()==at::kFloat && (feat_attr.dim()==2 || feat_attr.dim()==3));
    TORCH_CHECK(grad_out_feat.is_cuda() && grad_out_feat.dtype()==at::kFloat && grad_out_feat.dim()==2);

    const int64_t B = rays_o.size(0);
    TORCH_CHECK(B == (Wimg * Himg), "B must equal W*H for tile renderer");
    TORCH_CHECK(grad_out_feat.size(0) == B, "grad_out_feat must have shape (B, Cfeat)");
    validate_tile_candidate_capacity(
        culled_prim_indices, tile_offsets, Wimg, Himg, TW, TH, MLPSP_MAX_TILE_PRIMS,
        "render_tiles_backward_featattr_tilecenter");

    const int64_t N = positions.size(0);
    const int64_t Hh = W2.size(1);
    const int64_t IN = W1.size(2);
    const int64_t F = freq_bands.size(0);
    TORCH_CHECK(IN == 6 * F * 2, "Encoded dim must be 6*F*2");
    TORCH_CHECK(feat_attr.size(0) == N, "feat_attr first dimension must match primitive count");
    int64_t feat_slots = 1;
    int64_t Cfeat = 0;
    if (feat_attr.dim() == 2) {
      Cfeat = feat_attr.size(1);
    } else {
      feat_slots = feat_attr.size(1);
      Cfeat = feat_attr.size(2);
      TORCH_CHECK(feat_slots == 4 || feat_slots == 5, "3D feat_attr expects slots in {4,5}");
    }
    TORCH_CHECK(grad_out_feat.size(1) == Cfeat, "grad_out_feat feature dim must match feat_attr");
    TORCH_CHECK(Cfeat > 0, "grad_out_feat second dimension must be positive");
    TORCH_CHECK(Cfeat <= 512, "grad_out_feat second dimension must be <= 512");

    auto grad_feat_attr = (feat_slots == 1)
        ? torch::zeros({N, Cfeat}, rays_o.options().dtype(at::kFloat))
        : torch::zeros({N, feat_slots, Cfeat}, rays_o.options().dtype(at::kFloat));
    dim3 block(256, 1, 1);
    dim3 grid((unsigned)((Wimg + TW - 1)/TW), (unsigned)((Himg + TH - 1)/TH), 1);
    render_tiles_backward_featattr_tilecenter<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        rays_o.data_ptr<float>(),
        rays_d.data_ptr<float>(),
        positions.data_ptr<float>(),
        log_scales.data_ptr<float>(),
        quats.data_ptr<float>(),
        culled_prim_indices.data_ptr<int32_t>(),
        tile_offsets.data_ptr<int32_t>(),
        freq_bands.data_ptr<float>(),
        attenuation.data_ptr<float>(),
        (int)F,
        W1.data_ptr<float>(), b1.data_ptr<float>(),
        W2.data_ptr<float>(), b2.data_ptr<float>(),
        W3.data_ptr<float>(), b3.data_ptr<float>(),
        (int)N, (int)Hh, (int)IN,
        (int)Wimg, (int)Himg, (int)TW, (int)TH,
        (float)threshold,
        grad_out_feat.data_ptr<float>(),
        (int)feat_slots,
        (int)Cfeat,
        (float)gating_beta,
        (float)gating_temp,
        grad_feat_attr.data_ptr<float>());
    AT_CUDA_CHECK(cudaGetLastError());
    return grad_feat_attr;
}

torch::Tensor render_tiles_backward_featattr_per_pixel_wrapper(
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor positions,
    torch::Tensor log_scales,
    torch::Tensor quats,
    torch::Tensor culled_prim_indices,
    torch::Tensor tile_offsets,
    torch::Tensor freq_bands,
    torch::Tensor attenuation,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    torch::Tensor W3, torch::Tensor b3,
    torch::Tensor feat_attr,
    torch::Tensor grad_out_feat,
    double gating_beta,
    double gating_temp,
    int64_t Wimg, int64_t Himg,
    int64_t TW, int64_t TH,
    double threshold)
{
    TORCH_CHECK(rays_o.is_cuda() && rays_o.dtype()==at::kFloat && rays_o.dim()==2 && rays_o.size(1)==3);
    TORCH_CHECK(rays_d.is_cuda() && rays_d.dtype()==at::kFloat && rays_d.dim()==2 && rays_d.size(1)==3);
    TORCH_CHECK(positions.is_cuda() && positions.dtype()==at::kFloat && positions.dim()==2 && positions.size(1)==3);
    TORCH_CHECK(log_scales.is_cuda() && log_scales.dtype()==at::kFloat && log_scales.dim()==2 && log_scales.size(1)==3);
    TORCH_CHECK(quats.is_cuda() && quats.dtype()==at::kFloat && quats.dim()==2 && quats.size(1)==4);
    TORCH_CHECK(culled_prim_indices.is_cuda() && culled_prim_indices.dtype()==at::kInt && culled_prim_indices.dim()==1);
    TORCH_CHECK(tile_offsets.is_cuda() && tile_offsets.dtype()==at::kInt && tile_offsets.dim()==1);
    TORCH_CHECK(freq_bands.is_cuda() && freq_bands.dtype()==at::kFloat && freq_bands.dim()==1);
    TORCH_CHECK(attenuation.is_cuda() && attenuation.dtype()==at::kFloat && attenuation.dim()==1);
    TORCH_CHECK(feat_attr.is_cuda() && feat_attr.dtype()==at::kFloat && (feat_attr.dim()==2 || feat_attr.dim()==3));
    TORCH_CHECK(grad_out_feat.is_cuda() && grad_out_feat.dtype()==at::kFloat && grad_out_feat.dim()==2);

    const int64_t B = rays_o.size(0);
    TORCH_CHECK(B == (Wimg * Himg), "B must equal W*H for tile renderer");
    TORCH_CHECK(grad_out_feat.size(0) == B, "grad_out_feat must have shape (B, Cfeat)");
    validate_tile_candidate_capacity(
        culled_prim_indices, tile_offsets, Wimg, Himg, TW, TH, 0,
        "render_tiles_backward_featattr_per_pixel");

    const int64_t N = positions.size(0);
    const int64_t Hh = W2.size(1);
    const int64_t IN = W1.size(2);
    const int64_t F = freq_bands.size(0);
    TORCH_CHECK(IN == 6 * F * 2, "Encoded dim must be 6*F*2");
    TORCH_CHECK(feat_attr.size(0) == N, "feat_attr first dimension must match primitive count");
    int64_t feat_slots = 1;
    int64_t Cfeat = 0;
    if (feat_attr.dim() == 2) {
      Cfeat = feat_attr.size(1);
    } else {
      feat_slots = feat_attr.size(1);
      Cfeat = feat_attr.size(2);
      TORCH_CHECK(feat_slots == 4 || feat_slots == 5, "3D feat_attr expects slots in {4,5}");
    }
    TORCH_CHECK(grad_out_feat.size(1) == Cfeat, "grad_out_feat feature dim must match feat_attr");
    TORCH_CHECK(Cfeat > 0, "grad_out_feat second dimension must be positive");
    TORCH_CHECK(Cfeat <= 512, "grad_out_feat second dimension must be <= 512");

    auto grad_feat_attr = (feat_slots == 1)
        ? torch::zeros({N, Cfeat}, rays_o.options().dtype(at::kFloat))
        : torch::zeros({N, feat_slots, Cfeat}, rays_o.options().dtype(at::kFloat));
    auto overflow = torch::zeros({1}, rays_o.options().dtype(at::kInt));

    dim3 block((unsigned)TW, (unsigned)TH, 1);
    dim3 grid((unsigned)((Wimg + TW - 1)/TW), (unsigned)((Himg + TH - 1)/TH), 1);
    render_tiles_backward_featattr_per_pixel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        rays_o.data_ptr<float>(),
        rays_d.data_ptr<float>(),
        positions.data_ptr<float>(),
        log_scales.data_ptr<float>(),
        quats.data_ptr<float>(),
        culled_prim_indices.data_ptr<int32_t>(),
        tile_offsets.data_ptr<int32_t>(),
        freq_bands.data_ptr<float>(),
        attenuation.data_ptr<float>(),
        (int)F,
        W1.data_ptr<float>(), b1.data_ptr<float>(),
        W2.data_ptr<float>(), b2.data_ptr<float>(),
        W3.data_ptr<float>(), b3.data_ptr<float>(),
        (int)N, (int)Hh, (int)IN,
        (int)Wimg, (int)Himg, (int)TW, (int)TH,
        (float)threshold,
        grad_out_feat.data_ptr<float>(),
        (int)feat_slots,
        (int)Cfeat,
        (float)gating_beta,
        (float)gating_temp,
        grad_feat_attr.data_ptr<float>(),
        overflow.data_ptr<int32_t>());
    AT_CUDA_CHECK(cudaGetLastError());
    check_per_pixel_intersection_overflow(overflow, "render_tiles_backward_featattr_per_pixel");
    return grad_feat_attr;
}

// EN: Python wrapper for full tile-center backward where CUDA computes MLP grads directly.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
render_tiles_backward_tilecenter_full_wrapper(
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor positions,
    torch::Tensor log_scales,
    torch::Tensor quats,
    torch::Tensor culled_prim_indices,
    torch::Tensor tile_offsets,
    torch::Tensor freq_bands,
    torch::Tensor attenuation,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    torch::Tensor W3, torch::Tensor b3,
    torch::Tensor grad_out_rgb,
    torch::Tensor grad_T_end,
    int64_t Wimg, int64_t Himg,
    int64_t TW, int64_t TH,
    double threshold)
{
    TORCH_CHECK(((int)TW * (int)TH) > 0 && ((int)TW * (int)TH) <= 256, "Tile-center order kernel currently supports 0 < TW*TH <= 256");
    TORCH_CHECK(rays_o.is_cuda() && rays_o.dtype()==at::kFloat && rays_o.dim()==2 && rays_o.size(1)==3);
    TORCH_CHECK(rays_d.is_cuda() && rays_d.dtype()==at::kFloat && rays_d.dim()==2 && rays_d.size(1)==3);
    TORCH_CHECK(positions.is_cuda() && positions.dtype()==at::kFloat && positions.dim()==2 && positions.size(1)==3);
    TORCH_CHECK(log_scales.is_cuda() && log_scales.dtype()==at::kFloat && log_scales.dim()==2 && log_scales.size(1)==3);
    TORCH_CHECK(quats.is_cuda() && quats.dtype()==at::kFloat && quats.dim()==2 && quats.size(1)==4);
    TORCH_CHECK(culled_prim_indices.is_cuda() && culled_prim_indices.dtype()==at::kInt && culled_prim_indices.dim()==1);
    TORCH_CHECK(tile_offsets.is_cuda() && tile_offsets.dtype()==at::kInt && tile_offsets.dim()==1);
    TORCH_CHECK(freq_bands.is_cuda() && freq_bands.dtype()==at::kFloat && freq_bands.dim()==1);
    TORCH_CHECK(attenuation.is_cuda() && attenuation.dtype()==at::kFloat && attenuation.dim()==1);
    TORCH_CHECK(W1.is_cuda() && b1.is_cuda() && W2.is_cuda() && b2.is_cuda() && W3.is_cuda() && b3.is_cuda(),
                "MLP tensors must be CUDA tensors");
    TORCH_CHECK(W1.scalar_type() == b1.scalar_type() &&
                W1.scalar_type() == W2.scalar_type() &&
                W1.scalar_type() == b2.scalar_type() &&
                W1.scalar_type() == W3.scalar_type() &&
                W1.scalar_type() == b3.scalar_type(),
                "W/b tensors must share the same dtype");
    TORCH_CHECK(
        W1.scalar_type() == at::kFloat || W1.scalar_type() == at::kHalf || W1.scalar_type() == at::kBFloat16,
        "W/b tensors must be float32, float16, or bfloat16");
    TORCH_CHECK(grad_out_rgb.is_cuda() && grad_out_rgb.dtype()==at::kFloat && grad_out_rgb.dim()==2 && grad_out_rgb.size(1)==3);
    TORCH_CHECK(grad_T_end.is_cuda() && grad_T_end.dtype()==at::kFloat && grad_T_end.dim()==1);

    const int64_t B = rays_o.size(0);
    TORCH_CHECK(B == (Wimg * Himg), "B must equal W*H for tile renderer");
    TORCH_CHECK(grad_T_end.numel() == B, "grad_T_end must have size B");
    validate_tile_candidate_capacity(
        culled_prim_indices, tile_offsets, Wimg, Himg, TW, TH, MLPSP_MAX_TILE_PRIMS,
        "render_tiles_backward_tilecenter_full");

    const int64_t N = positions.size(0);
    const int64_t Hh = W2.size(1);
    const int64_t IN = W1.size(2);
    const int64_t F = freq_bands.size(0);
    TORCH_CHECK(IN == 6 * F * 2, "Encoded dim must be 6*F*2");

    auto opts_f = rays_o.options().dtype(at::kFloat);
    auto grad_W1 = torch::zeros(W1.sizes(), opts_f);
    auto grad_b1 = torch::zeros(b1.sizes(), opts_f);
    auto grad_W2 = torch::zeros(W2.sizes(), opts_f);
    auto grad_b2 = torch::zeros(b2.sizes(), opts_f);
    auto grad_W3 = torch::zeros(W3.sizes(), opts_f);
    auto grad_b3 = torch::zeros(b3.sizes(), opts_f);
    auto grad_positions = torch::zeros({N, 3}, opts_f);
    auto grad_log_scales = torch::zeros({N, 3}, opts_f);
    auto grad_quats = torch::zeros({N,4}, opts_f);

    dim3 block(256, 1, 1);
    dim3 grid((unsigned)((Wimg + TW - 1)/TW), (unsigned)((Himg + TH - 1)/TH), 1);
    if (W1.scalar_type() == at::kFloat) {
      render_tiles_backward_tilecenter_full<float><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        rays_o.data_ptr<float>(),
        rays_d.data_ptr<float>(),
        positions.data_ptr<float>(),
        log_scales.data_ptr<float>(),
        quats.data_ptr<float>(),
        culled_prim_indices.data_ptr<int32_t>(),
        tile_offsets.data_ptr<int32_t>(),
        freq_bands.data_ptr<float>(),
        attenuation.data_ptr<float>(),
        (int)F,
        W1.data_ptr<float>(), b1.data_ptr<float>(),
        W2.data_ptr<float>(), b2.data_ptr<float>(),
        W3.data_ptr<float>(), b3.data_ptr<float>(),
        (int)N, (int)Hh, (int)IN,
        (int)Wimg, (int)Himg, (int)TW, (int)TH,
        (float)threshold,
        grad_out_rgb.data_ptr<float>(),
        grad_T_end.data_ptr<float>(),
        grad_W1.data_ptr<float>(), grad_b1.data_ptr<float>(),
        grad_W2.data_ptr<float>(), grad_b2.data_ptr<float>(),
        grad_W3.data_ptr<float>(), grad_b3.data_ptr<float>(),
        grad_positions.data_ptr<float>(),
        grad_log_scales.data_ptr<float>(),
        grad_quats.data_ptr<float>());
    } else if (W1.scalar_type() == at::kHalf) {
      render_tiles_backward_tilecenter_full<__half><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        rays_o.data_ptr<float>(),
        rays_d.data_ptr<float>(),
        positions.data_ptr<float>(),
        log_scales.data_ptr<float>(),
        quats.data_ptr<float>(),
        culled_prim_indices.data_ptr<int32_t>(),
        tile_offsets.data_ptr<int32_t>(),
        freq_bands.data_ptr<float>(),
        attenuation.data_ptr<float>(),
        (int)F,
        reinterpret_cast<const __half*>(W1.data_ptr<at::Half>()), reinterpret_cast<const __half*>(b1.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(W2.data_ptr<at::Half>()), reinterpret_cast<const __half*>(b2.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(W3.data_ptr<at::Half>()), reinterpret_cast<const __half*>(b3.data_ptr<at::Half>()),
        (int)N, (int)Hh, (int)IN,
        (int)Wimg, (int)Himg, (int)TW, (int)TH,
        (float)threshold,
        grad_out_rgb.data_ptr<float>(),
        grad_T_end.data_ptr<float>(),
        grad_W1.data_ptr<float>(), grad_b1.data_ptr<float>(),
        grad_W2.data_ptr<float>(), grad_b2.data_ptr<float>(),
        grad_W3.data_ptr<float>(), grad_b3.data_ptr<float>(),
        grad_positions.data_ptr<float>(),
        grad_log_scales.data_ptr<float>(),
        grad_quats.data_ptr<float>());
    } else {
      render_tiles_backward_tilecenter_full<__nv_bfloat16><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        rays_o.data_ptr<float>(),
        rays_d.data_ptr<float>(),
        positions.data_ptr<float>(),
        log_scales.data_ptr<float>(),
        quats.data_ptr<float>(),
        culled_prim_indices.data_ptr<int32_t>(),
        tile_offsets.data_ptr<int32_t>(),
        freq_bands.data_ptr<float>(),
        attenuation.data_ptr<float>(),
        (int)F,
        reinterpret_cast<const __nv_bfloat16*>(W1.data_ptr<at::BFloat16>()), reinterpret_cast<const __nv_bfloat16*>(b1.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(W2.data_ptr<at::BFloat16>()), reinterpret_cast<const __nv_bfloat16*>(b2.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(W3.data_ptr<at::BFloat16>()), reinterpret_cast<const __nv_bfloat16*>(b3.data_ptr<at::BFloat16>()),
        (int)N, (int)Hh, (int)IN,
        (int)Wimg, (int)Himg, (int)TW, (int)TH,
        (float)threshold,
        grad_out_rgb.data_ptr<float>(),
        grad_T_end.data_ptr<float>(),
        grad_W1.data_ptr<float>(), grad_b1.data_ptr<float>(),
        grad_W2.data_ptr<float>(), grad_b2.data_ptr<float>(),
        grad_W3.data_ptr<float>(), grad_b3.data_ptr<float>(),
        grad_positions.data_ptr<float>(),
        grad_log_scales.data_ptr<float>(),
        grad_quats.data_ptr<float>());
    }
    AT_CUDA_CHECK(cudaGetLastError());
    return {grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3, grad_positions, grad_log_scales, grad_quats};
}

std::vector<torch::Tensor>
render_tiles_backward_tilecenter_traversal_wrapper(
    torch::Tensor rays_o, torch::Tensor rays_d,
    torch::Tensor pos, torch::Tensor log_scales,
    torch::Tensor quats,
    torch::Tensor culled_prim_indices, torch::Tensor tile_offsets,
    torch::Tensor freq_bands, torch::Tensor attenuation,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    torch::Tensor W3, torch::Tensor b3,
    int64_t Wimg, int64_t Himg, int64_t TW, int64_t TH,
    double thr)
{
  validate_tile_candidate_capacity(
      culled_prim_indices, tile_offsets, Wimg, Himg, TW, TH, MLPSP_MAX_TILE_PRIMS,
      "render_tiles_backward_tilecenter_traversal");
  const auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(rays_o.device());
  const auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(rays_o.device());
  auto out_used = torch::zeros({Himg * Wimg}, opts_i);
  auto out_trans = torch::zeros({Himg * Wimg}, opts_f);

  const int64_t N = pos.size(0);
  const int64_t Hh = W2.size(1);
  const int64_t IN = W1.size(2);
  const int64_t F = freq_bands.size(0);
  TORCH_CHECK(Hh <= MLPSP_MAX_H, "render_tiles_backward_tilecenter_backprop_noatomic currently supports hidden_dim <= MLPSP_MAX_H");

  const dim3 grid((unsigned)((Wimg + TW - 1) / TW), (unsigned)((Himg + TH - 1) / TH), 1);
  const dim3 block(256, 1, 1);
  render_tiles_backward_tilecenter_traversal<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      rays_o.data_ptr<float>(),
      rays_d.data_ptr<float>(),
      pos.data_ptr<float>(),
      log_scales.data_ptr<float>(),
      quats.data_ptr<float>(),
      culled_prim_indices.data_ptr<int32_t>(),
      tile_offsets.data_ptr<int32_t>(),
      freq_bands.data_ptr<float>(),
      attenuation.data_ptr<float>(),
      (int)F,
      W1.data_ptr<float>(), b1.data_ptr<float>(),
      W2.data_ptr<float>(), b2.data_ptr<float>(),
      W3.data_ptr<float>(), b3.data_ptr<float>(),
      (int)N, (int)Hh, (int)IN,
      (int)Wimg, (int)Himg, (int)TW, (int)TH,
      (float)thr,
      out_used.data_ptr<int32_t>(),
      out_trans.data_ptr<float>());
  return {out_used, out_trans};
}

// EN: Python wrapper for backprop-only tile-center profiling kernel (no atomics).
torch::Tensor
render_tiles_backward_tilecenter_backprop_noatomic_wrapper(
    torch::Tensor rays_o, torch::Tensor rays_d,
    torch::Tensor pos, torch::Tensor log_scales,
    torch::Tensor quats,
    torch::Tensor culled_prim_indices, torch::Tensor tile_offsets,
    torch::Tensor freq_bands, torch::Tensor attenuation,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    torch::Tensor W3, torch::Tensor b3,
    torch::Tensor grad_out_rgb, torch::Tensor grad_T_end,
    int64_t Wimg, int64_t Himg, int64_t TW, int64_t TH,
    double thr)
{
  validate_tile_candidate_capacity(
      culled_prim_indices, tile_offsets, Wimg, Himg, TW, TH, MLPSP_MAX_TILE_PRIMS,
      "render_tiles_backward_tilecenter_backprop_noatomic");
  auto opts_f = rays_o.options().dtype(at::kFloat);
  auto out_sink = torch::zeros({Himg * Wimg}, opts_f);

  const int64_t N = pos.size(0);
  const int64_t Hh = W2.size(1);
  const int64_t IN = W1.size(2);
  const int64_t F = freq_bands.size(0);

  const dim3 grid((unsigned)((Wimg + TW - 1) / TW), (unsigned)((Himg + TH - 1) / TH), 1);
  const dim3 block(256, 1, 1);
  render_tiles_backward_tilecenter_backprop_noatomic<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      rays_o.data_ptr<float>(),
      rays_d.data_ptr<float>(),
      pos.data_ptr<float>(),
      log_scales.data_ptr<float>(),
      quats.data_ptr<float>(),
      culled_prim_indices.data_ptr<int32_t>(),
      tile_offsets.data_ptr<int32_t>(),
      freq_bands.data_ptr<float>(),
      attenuation.data_ptr<float>(),
      (int)F,
      W1.data_ptr<float>(), b1.data_ptr<float>(),
      W2.data_ptr<float>(), b2.data_ptr<float>(),
      W3.data_ptr<float>(), b3.data_ptr<float>(),
      (int)N, (int)Hh, (int)IN,
      (int)Wimg, (int)Himg, (int)TW, (int)TH,
      (float)thr,
      grad_out_rgb.data_ptr<float>(),
      grad_T_end.data_ptr<float>(),
      out_sink.data_ptr<float>());
  return out_sink;
}

// EN: Binds CUDA wrapper functions into the Python extension module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("render_tiles_autograd", &render_tiles_autograd_wrapper, "Legacy tile forward; parameter gradients are not implemented");
  m.def("render_tiles_autograd_tilecenter", &render_tiles_autograd_tilecenter_wrapper, "Legacy tile-center forward; parameter gradients are not implemented");
  m.def("render_tiles_autograd_tilecenter_presorted", &render_tiles_autograd_tilecenter_presorted_wrapper, "Tile MLP splatting (prim-major, presorted per-tile list; intended for globaldepth ordering)");
  m.def("render_tiles_forward", &render_tiles_forward_wrapper, "Tile splatting forward (per-pixel sort, no autograd)");
  m.def("render_tiles_forward_per_pixel_with_feat", &render_tiles_forward_per_pixel_with_feat_wrapper, "Tile splatting forward (per-pixel sort, no autograd, with rendered feature map)");
  m.def("render_tiles_forward_tilecenter", &render_tiles_forward_tilecenter_wrapper, "Tile splatting forward (tile-center order, no autograd)");
  m.def("render_tiles_forward_tilecenter_stats", &render_tiles_forward_tilecenter_stats_wrapper, "Tilecenter forward (no-grad) with stats.");
  m.def("render_tiles_forward_tilecenter_with_t", &render_tiles_forward_tilecenter_with_t_wrapper, "Tile splatting forward (tile-center order, no autograd, with expected depth t)");
  m.def("render_tiles_forward_tilecenter_with_feat", &render_tiles_forward_tilecenter_with_feat_wrapper, "Tile splatting forward (tile-center order, no autograd, with rendered feature map)");
  m.def("render_tiles_backward_count_per_pixel", &render_tiles_backward_count_per_pixel_wrapper, "Tile splatting backward count (per-pixel)");
  m.def("render_tiles_backward_count_tilecenter", &render_tiles_backward_count_tilecenter_wrapper, "Tile splatting backward count (tilecenter)");
  m.def("render_tiles_backward_emit_per_pixel", &render_tiles_backward_emit_per_pixel_wrapper, "Tile splatting backward emit (per-pixel)");
  m.def("render_tiles_backward_emit_tilecenter", &render_tiles_backward_emit_tilecenter_wrapper, "Tile splatting backward emit (tilecenter)");
  m.def("render_tiles_backward_featattr_per_pixel", &render_tiles_backward_featattr_per_pixel_wrapper, "Tile splatting backward feat_attr-only (per-pixel)");
  m.def("render_tiles_backward_featattr_tilecenter", &render_tiles_backward_featattr_tilecenter_wrapper, "Tile splatting backward feat_attr-only (tilecenter)");
  m.def("render_tiles_backward_tilecenter_full", &render_tiles_backward_tilecenter_full_wrapper, "Tile splatting backward full (tilecenter, MLP grads in CUDA)");
  m.def("render_tiles_backward_tilecenter_traversal", &render_tiles_backward_tilecenter_traversal_wrapper, "Tile splatting backward traversal-only (tilecenter profiling)");
  m.def("render_tiles_backward_tilecenter_backprop_noatomic", &render_tiles_backward_tilecenter_backprop_noatomic_wrapper, "Tile splatting backward backprop-only (tilecenter profiling, no atomics)");
  m.def("_test_atomic_add_warp_agg", &test_atomic_add_warp_agg_wrapper, "Internal sparse-peer warp atomic test hook");
}
