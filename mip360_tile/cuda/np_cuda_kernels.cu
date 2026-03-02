// Copyright (c) 2025
//
// Draft CUDA kernels for NeuralPrimitiveScene candidate selection (Pass-1/2).
// These kernels mirror the Triton functionality in vmodel.py:
//  - Pass-1: per-ray count of primitives whose Gaussian score exceeds a threshold
//  - Pass-2: compact write of (ray_id, prim_id, t, dx, dy, dz) for all hits
//
// Notes:
//  - This is an initial functional version prioritizing clarity and correctness.
//  - It uses block-per-ray scheduling with intra-block work partition over primitives.
//  - For Pass-2 we do a two-phase per-block scan to compute per-thread write offsets.
//  - Future work: tile-based scheduling, cp.async double-buffering, Tensor Core usage.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

// Small epsilon to avoid degenerate divisions
#ifndef NPK_EPS
#define NPK_EPS 1e-12f
#endif

// Utility: check CUDA errors in debug builds
#ifndef NDEBUG
#define CUDA_CHECK(err) \
  do { \
    cudaError_t err__ = (err); \
    if (err__ != cudaSuccess) { \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); \
    } \
  } while (0)
#else
#define CUDA_CHECK(err) do { (void)(err); } while (0)
#endif

// ----------------------------
// Geometry helper per primitive
// ----------------------------
__device__ __forceinline__ bool primitive_hit(
    const float rox, const float roy, const float roz,
    const float rdx, const float rdy, const float rdz,
    const float cx,  const float cy,  const float cz,
    const float logs,
    const float thr,
    float &t, float &dx, float &dy, float &dz)
{
  // radius from log scale
  const float r  = __expf(logs);
  const float r2 = fmaf(r, r, NPK_EPS);

  // vector from ray origin to primitive center
  const float vx = cx - rox;
  const float vy = cy - roy;
  const float vz = cz - roz;

  // projection length along ray dir
  t = fmaf(vx, rdx, fmaf(vy, rdy, vz * rdz));
  if (t <= 0.0f) return false;

  // closest point on ray to center
  const float px = rox + t * rdx;
  const float py = roy + t * rdy;
  const float pz = roz + t * rdz;

  // perpendicular vector from center to ray
  dx = cx - px;
  dy = cy - py;
  dz = cz - pz;
  const float perp2 = fmaxf(dx*dx + dy*dy + dz*dz, 0.0f);

  // Gaussian score
  const float score = __expf(-0.5f * perp2 / r2);
  return score > thr;
}

// =====================
// Pass-1: Count hits
// =====================
template <int THREADS>
__global__ void count_kernel(
    const float* __restrict__ ray_o,   // (B,3)
    const float* __restrict__ ray_d,   // (B,3) (assumed normalized by caller)
    const float* __restrict__ pos,     // (N,3)
    const float* __restrict__ logs,    // (N,1)
    const int B,
    const int N,
    const float thr,
    int* __restrict__ out_counts)      // (B)
{
  const int ray = blockIdx.x;
  if (ray >= B) return;

  __shared__ int s_counts[THREADS];
  int local_count = 0;

  const float rox = ray_o[ray*3 + 0];
  const float roy = ray_o[ray*3 + 1];
  const float roz = ray_o[ray*3 + 2];
  const float rdx = ray_d[ray*3 + 0];
  const float rdy = ray_d[ray*3 + 1];
  const float rdz = ray_d[ray*3 + 2];

  for (int n = threadIdx.x; n < N; n += THREADS) {
    const float cx = pos[n*3 + 0];
    const float cy = pos[n*3 + 1];
    const float cz = pos[n*3 + 2];
    const float lg = logs[n];

    float t, dx, dy, dz;
    if (primitive_hit(rox, roy, roz, rdx, rdy, rdz, cx, cy, cz, lg, thr, t, dx, dy, dz)) {
      local_count += 1;
    }
  }

  s_counts[threadIdx.x] = local_count;
  __syncthreads();

  // parallel reduction over THREADS (power of two assumed)
  for (int stride = THREADS >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_counts[threadIdx.x] += s_counts[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    out_counts[ray] = s_counts[0];
  }
}

// =====================
// Pass-2: Compact write
// =====================
template <int THREADS>
__global__ void write_kernel(
    const float* __restrict__ ray_o,   // (B,3)
    const float* __restrict__ ray_d,   // (B,3)
    const float* __restrict__ pos,     // (N,3)
    const float* __restrict__ logs,    // (N,1)
    const int B,
    const int N,
    const float thr,
    const long* __restrict__ offsets,  // (B) int64
    long* __restrict__ out_ray,        // (M)
    long* __restrict__ out_prim,       // (M)
    float* __restrict__ out_t,         // (M)
    float* __restrict__ out_dx,        // (M)
    float* __restrict__ out_dy,        // (M)
    float* __restrict__ out_dz)        // (M)
{
  const int ray = blockIdx.x;
  if (ray >= B) return;

  __shared__ int s_counts[THREADS];
  __shared__ int s_prefix[THREADS];

  const float rox = ray_o[ray*3 + 0];
  const float roy = ray_o[ray*3 + 1];
  const float roz = ray_o[ray*3 + 2];
  const float rdx = ray_d[ray*3 + 0];
  const float rdy = ray_d[ray*3 + 1];
  const float rdz = ray_d[ray*3 + 2];

  // phase-1: per-thread counts
  int local_count = 0;
  for (int n = threadIdx.x; n < N; n += THREADS) {
    const float cx = pos[n*3 + 0];
    const float cy = pos[n*3 + 1];
    const float cz = pos[n*3 + 2];
    const float lg = logs[n];

    float t, dx, dy, dz;
    if (primitive_hit(rox, roy, roz, rdx, rdy, rdz, cx, cy, cz, lg, thr, t, dx, dy, dz)) {
      local_count += 1;
    }
  }
  s_counts[threadIdx.x] = local_count;
  __syncthreads();

  // exclusive scan over THREADS to get thread-level offsets within this ray segment
  // simple Hillis-Steele scan
  int val = s_counts[threadIdx.x];
  int acc = 0;
  for (int offset = 1; offset < THREADS; offset <<= 1) {
    int nbh = 0;
    if (threadIdx.x >= offset) {
      nbh = s_counts[threadIdx.x - offset];
    } else {
      nbh = 0;
    }
    __syncthreads();
    s_counts[threadIdx.x] = val + nbh;
    __syncthreads();
    val = s_counts[threadIdx.x];
  }
  if (threadIdx.x == 0) {
    s_prefix[0] = 0;
  }
  __syncthreads();
  if (threadIdx.x > 0) {
    s_prefix[threadIdx.x] = s_counts[threadIdx.x - 1];
  }
  __syncthreads();

  const long base = offsets[ray];
  const int thread_base = s_prefix[threadIdx.x];

  // phase-2: re-iterate and write
  int write_idx = 0; // within-thread ordinal
  for (int n = threadIdx.x; n < N; n += THREADS) {
    const float cx = pos[n*3 + 0];
    const float cy = pos[n*3 + 1];
    const float cz = pos[n*3 + 2];
    const float lg = logs[n];

    float t, dx, dy, dz;
    if (primitive_hit(rox, roy, roz, rdx, rdy, rdz, cx, cy, cz, lg, thr, t, dx, dy, dz)) {
      const long out_i = base + thread_base + write_idx;
      out_ray[out_i]  = static_cast<long>(ray);
      out_prim[out_i] = static_cast<long>(n);
      out_t[out_i]    = t;
      out_dx[out_i]   = dx;
      out_dy[out_i]   = dy;
      out_dz[out_i]   = dz;
      write_idx++;
    }
  }
}

// -----------------
// Host wrappers
// -----------------

static void check_inputs(const torch::Tensor& t, const std::string& name, at::ScalarType dtype, int dims, int last_dim) {
  TORCH_CHECK(t.is_cuda(), name, " must be CUDA tensor");
  TORCH_CHECK(t.dtype() == dtype, name, " dtype mismatch");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.dim() == dims, name, " must have ", dims, " dims");
  if (last_dim > 0) TORCH_CHECK(t.size(-1) == last_dim, name, " last dim must be ", last_dim);
}

torch::Tensor count_above_threshold(
    torch::Tensor ray_o,    // (B,3) float32
    torch::Tensor ray_d,    // (B,3) float32 (assumed normalized by caller)
    torch::Tensor pos,      // (N,3) float32
    torch::Tensor logs,     // (N,1) or (N) float32
    double threshold)
{
  check_inputs(ray_o, "ray_o", at::kFloat, 2, 3);
  check_inputs(ray_d, "ray_d", at::kFloat, 2, 3);
  TORCH_CHECK(pos.is_cuda() && pos.dtype() == at::kFloat && pos.is_contiguous() && pos.dim() == 2 && pos.size(1) == 3, "pos invalid");
  TORCH_CHECK(logs.is_cuda() && logs.dtype() == at::kFloat && logs.is_contiguous(), "logs invalid");

  const int64_t B = ray_o.size(0);
  const int64_t N = pos.size(0);

  auto counts = torch::empty({B}, ray_o.options().dtype(at::kInt));

  constexpr int THREADS = 256;
  dim3 grid(B);
  dim3 block(THREADS);
  count_kernel<THREADS><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      ray_o.data_ptr<float>(),
      ray_d.data_ptr<float>(),
      pos.data_ptr<float>(),
      logs.dim() == 2 ? logs.data_ptr<float>() : logs.data_ptr<float>(),
      (int)B,
      (int)N,
      (float)threshold,
      counts.data_ptr<int>());
  CUDA_CHECK(cudaGetLastError());
  return counts;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> write_above_threshold(
    torch::Tensor ray_o,    // (B,3) float32
    torch::Tensor ray_d,    // (B,3) float32
    torch::Tensor pos,      // (N,3) float32
    torch::Tensor logs,     // (N,1) or (N) float32
    torch::Tensor offsets,  // (B) int64 (exclusive prefix of counts)
    double threshold,
    int64_t M)
{
  check_inputs(ray_o, "ray_o", at::kFloat, 2, 3);
  check_inputs(ray_d, "ray_d", at::kFloat, 2, 3);
  TORCH_CHECK(pos.is_cuda() && pos.dtype() == at::kFloat && pos.is_contiguous() && pos.dim() == 2 && pos.size(1) == 3, "pos invalid");
  TORCH_CHECK(logs.is_cuda() && logs.dtype() == at::kFloat && logs.is_contiguous(), "logs invalid");
  TORCH_CHECK(offsets.is_cuda() && offsets.dtype() == at::kLong && offsets.is_contiguous() && offsets.dim() == 1, "offsets invalid");

  const int64_t B = ray_o.size(0);
  const int64_t N = pos.size(0);

  auto opts_f = ray_o.options().dtype(at::kFloat);
  auto opts_l = ray_o.options().dtype(at::kLong);

  auto out_ray  = torch::empty({M}, opts_l);
  auto out_prim = torch::empty({M}, opts_l);
  auto out_t    = torch::empty({M}, opts_f);
  auto out_dx   = torch::empty({M}, opts_f);
  auto out_dy   = torch::empty({M}, opts_f);
  auto out_dz   = torch::empty({M}, opts_f);

  constexpr int THREADS = 256;
  dim3 grid(B);
  dim3 block(THREADS);
  write_kernel<THREADS><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      ray_o.data_ptr<float>(),
      ray_d.data_ptr<float>(),
      pos.data_ptr<float>(),
      logs.dim() == 2 ? logs.data_ptr<float>() : logs.data_ptr<float>(),
      (int)B,
      (int)N,
      (float)threshold,
      offsets.data_ptr<long>(),
      out_ray.data_ptr<long>(),
      out_prim.data_ptr<long>(),
      out_t.data_ptr<float>(),
      out_dx.data_ptr<float>(),
      out_dy.data_ptr<float>(),
      out_dz.data_ptr<float>());
  CUDA_CHECK(cudaGetLastError());
  return {out_ray, out_prim, out_t, out_dx, out_dy, out_dz};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("count_above_threshold", &count_above_threshold, "Pass-1: per-ray hit count");
  m.def("write_above_threshold", &write_above_threshold, "Pass-2: compact write of hits");
}


