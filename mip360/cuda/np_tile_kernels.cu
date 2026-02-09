// Tile-based MLP splatting kernel with per-pixel sorting.
// One block = one tile (TW x TH threads). Each thread (pixel) gathers
// intersections from a per-tile primitive list, sorts them by depth,
// then evaluates the MLP and alpha-blends in the correct order.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

// Block radix sort (CUB)
#include <cub/block/block_radix_sort.cuh>

// Fast math helpers
__device__ __forceinline__ float fast_sigmoid(float x) {
  return 1.0f / (1.0f + __expf(-x));
}
__device__ __forceinline__ float fast_softplus(float x) {
  if (x > 20.f) return x;
  if (x < -20.f) return __expf(x);
  return log1pf(__expf(x));
}

// Compute closest approach info and Gaussian culling
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

// Kernel: render tiles with per-pixel sorting
__global__ void render_tiles_kernel_with_sort(
    // Rays and primitives
    const float* __restrict__ rays_o,
    const float* __restrict__ rays_d,
    const float* __restrict__ pos,
    const float* __restrict__ logs,
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
    float* __restrict__ out_alpha
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
  int intersection_count = 0;

  const int32_t prim_start = tile_offsets[tile_id];
  const int32_t prim_end = tile_offsets[tile_id + 1];

  for (int k = prim_start; k < prim_end; ++k) {
    const int p = culled_prim_indices[k];
    float t, dx, dy, dz, decay;
    const float cx = pos[p*3+0];
    const float cy = pos[p*3+1];
    const float cz = pos[p*3+2];
    const float lg = logs[p];

    if (hit_and_features(rox, roy, roz, rdx, rdy, rdz, cx, cy, cz, lg, thr, t, dx, dy, dz, decay)) {
      if (intersection_count < MAX_INTERSECTIONS) {
        intersection_t[intersection_count] = t;
        intersection_prim_id[intersection_count] = p;
        intersection_decay[intersection_count] = decay;
        intersection_dx[intersection_count] = dx;
        intersection_dy[intersection_count] = dy;
        intersection_dz[intersection_count] = dz;
        intersection_count++;
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

    float h1[64];
    int Hcap = H > 64 ? 64 : H;
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

    float h2[64];
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

// Kernel: tile-level approximate ordering (tile-center ray) + prim-major traversal.
// NOTE: Blending order becomes tile-shared (approximate). Per-pixel t is still used inside hit_and_features
// for distance/feature computation (not for ordering).
__global__ void render_tiles_kernel_tilecenter_order(
    const float* __restrict__ rays_o,
    const float* __restrict__ rays_d,
    const float* __restrict__ pos,
    const float* __restrict__ logs,
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
  // Use a 1D block of (TW*TH) threads so CUB block primitives can be used safely.
  const int tid = (int)threadIdx.x;
  const int lx = tid % TW;
  const int ly = tid / TW;
  const int x = tile_x * TW + lx;
  const int y = tile_y * TH + ly;
  const bool valid_pix = (x < Wimg) && (y < Himg);
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

  // Block-level radix sort assumes a fixed block size (16x16 => 256 threads).
  // `cumodel.py` currently uses tile_size=16, so this holds for normal use.
  constexpr int MAX_TILE_PRIMS = 512;
  constexpr int BLOCK_THREADS = 256;
  constexpr int ITEMS_PER_THREAD = MAX_TILE_PRIMS / BLOCK_THREADS; // 2
  static_assert(MAX_TILE_PRIMS % BLOCK_THREADS == 0, "MAX_TILE_PRIMS must be divisible by BLOCK_THREADS");

  __shared__ int s_pid[MAX_TILE_PRIMS];
  using BlockRadixSortT = cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD, int>;
  __shared__ typename BlockRadixSortT::TempStorage sort_storage;

  const int num_threads = TW * TH;
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
      const float lg = logs[p];
      if (hit_and_features(rox, roy, roz, rdx, rdy, rdz, cx, cy, cz, lg, thr, t, dx, dy, dz, decay)) {
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

// Kernel: prim-major traversal, but assumes per-tile primitive lists are ALREADY ORDERED (no in-kernel sort).
// Intended for "globaldepth" mode where the host sorts each tile's list by primitive center camera-space forward (view-space z).
__global__ void render_tiles_kernel_tilecenter_presorted(
    const float* __restrict__ rays_o,
    const float* __restrict__ rays_d,
    const float* __restrict__ pos,
    const float* __restrict__ logs,
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
  const int lx = tid % TW;
  const int ly = tid / TW;
  const int x = tile_x * TW + lx;
  const int y = tile_y * TH + ly;
  const bool valid_pix = (x < Wimg) && (y < Himg);
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

  constexpr int MAX_TILE_PRIMS = 512;
  __shared__ int s_pid[MAX_TILE_PRIMS];

  const int num_threads = TW * TH;
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
      const float lg = logs[p];
      if (hit_and_features(rox, roy, roz, rdx, rdy, rdz, cx, cy, cz, lg, thr, t, dx, dy, dz, decay)) {
        float v6[6] = {rdx, rdy, rdz, dx, dy, dz};

        float h1[64];
        for (int h_idx = 0; h_idx < Hcap; ++h_idx) h1[h_idx] = sb1[h_idx];

        const int cos_offset = IN / 2;
        for (int c = 0; c < 6; ++c) {
          for (int f = 0; f < F; ++f) {
            const float fb = freq_bands[f];
            const float a = attenuation[f];
            const float v = v6[c] * fb;
            const float sv = __sinf(v) * a;
            const float cv = __cosf(v) * a;

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

        float h2[64];
        for (int h_idx = 0; h_idx < Hcap; ++h_idx) h2[h_idx] = sb2[h_idx];
        for (int o = 0; o < Hcap; ++o) {
          const float xo = h1[o];
          const float* wrow = sW2 + (size_t)o * Hcap;
          for (int h_idx = 0; h_idx < Hcap; ++h_idx) {
            h2[h_idx] = fmaf(wrow[h_idx], xo, h2[h_idx]);
          }
        }
        for (int h_idx = 0; h_idx < Hcap; ++h_idx) h2[h_idx] = h2[h_idx] > 0.f ? h2[h_idx] : 0.f;

        float out4[4] = {sb3[0], sb3[1], sb3[2], sb3[3]};
        for (int o = 0; o < Hcap; ++o) {
          const float xo = h2[o];
          const float* wrow = sW3 + (size_t)o;
          out4[0] = fmaf(wrow[0 * Hcap], xo, out4[0]);
          out4[1] = fmaf(wrow[1 * Hcap], xo, out4[1]);
          out4[2] = fmaf(wrow[2 * Hcap], xo, out4[2]);
          out4[3] = fmaf(wrow[3 * Hcap], xo, out4[3]);
        }

        const float R = fast_sigmoid(out4[0]);
        const float G = fast_sigmoid(out4[1]);
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
    load_weights(p);
    eval_and_blend(p);
    const int all_done = __syncthreads_or((!valid_pix) || (trans < 1e-3f));
    if (all_done) break;
    __syncthreads();
  }

  if (valid_pix) {
    out_rgb[pix*3+0] = C.x;
    out_rgb[pix*3+1] = C.y;
    out_rgb[pix*3+2] = C.z;
    out_alpha[pix]   = 1.f - trans;
  }
}


class RenderTilesCUDASinglePass : public torch::autograd::Function<RenderTilesCUDASinglePass> {
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor rays_o,
        torch::Tensor rays_d,
        torch::Tensor positions,
        torch::Tensor log_scales,
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
        TORCH_CHECK(log_scales.is_cuda() && log_scales.dtype()==at::kFloat && log_scales.dim()==1);
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

        // Per-pixel kernel expects a 2D block layout (threadIdx.x/y map to pixel in tile).
        dim3 block((unsigned)TW, (unsigned)TH, 1);
        dim3 grid((unsigned)((Wimg + TW - 1)/TW), (unsigned)((Himg + TH - 1)/TH), 1);
        render_tiles_kernel_with_sort<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            rays_o.data_ptr<float>(),
            rays_d.data_ptr<float>(),
            positions.data_ptr<float>(),
            log_scales.data_ptr<float>(),
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
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor rays_o,
        torch::Tensor rays_d,
        torch::Tensor positions,
        torch::Tensor log_scales,
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
        // This kernel uses a CUB BlockRadixSort specialization that assumes 16x16 threads (256 threads).
        TORCH_CHECK((int)TW == 16 && (int)TH == 16, "Tile-center order kernel currently requires TW==16 and TH==16");

        TORCH_CHECK(rays_o.is_cuda() && rays_o.dtype()==at::kFloat && rays_o.dim()==2 && rays_o.size(1)==3);
        TORCH_CHECK(rays_d.is_cuda() && rays_d.dtype()==at::kFloat && rays_d.dim()==2 && rays_d.size(1)==3);
        TORCH_CHECK(positions.is_cuda() && positions.dtype()==at::kFloat && positions.dim()==2 && positions.size(1)==3);
        TORCH_CHECK(log_scales.is_cuda() && log_scales.dtype()==at::kFloat && log_scales.dim()==1);
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

        // Tile-center order kernel uses a 1D block (TW*TH threads) so CUB block primitives work correctly.
        dim3 block((unsigned)(TW * TH), 1, 1);
        dim3 grid((unsigned)((Wimg + TW - 1)/TW), (unsigned)((Himg + TH - 1)/TH), 1);

        const int Hcap = (int)(Hh > 64 ? 64 : Hh);
        const int shared_floats = Hcap * (int)IN + Hcap + Hcap * Hcap + Hcap + 4 * Hcap + 4;
        const size_t shared_bytes = (size_t)shared_floats * sizeof(float);

        render_tiles_kernel_tilecenter_order<<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
            rays_o.data_ptr<float>(),
            rays_d.data_ptr<float>(),
            positions.data_ptr<float>(),
            log_scales.data_ptr<float>(),
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

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs)
    {
        return {
            torch::Tensor(), // rays_o
            torch::Tensor(), // rays_d
            torch::Tensor(), // positions
            torch::Tensor(), // log_scales
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
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor rays_o,
        torch::Tensor rays_d,
        torch::Tensor positions,
        torch::Tensor log_scales,
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
        TORCH_CHECK((int)TW == 16 && (int)TH == 16, "Tile-center presorted kernel currently requires TW==16 and TH==16");

        TORCH_CHECK(rays_o.is_cuda() && rays_o.dtype()==at::kFloat && rays_o.dim()==2 && rays_o.size(1)==3);
        TORCH_CHECK(rays_d.is_cuda() && rays_d.dtype()==at::kFloat && rays_d.dim()==2 && rays_d.size(1)==3);
        TORCH_CHECK(positions.is_cuda() && positions.dtype()==at::kFloat && positions.dim()==2 && positions.size(1)==3);
        TORCH_CHECK(log_scales.is_cuda() && log_scales.dtype()==at::kFloat && log_scales.dim()==1);
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

        dim3 block((unsigned)(TW * TH), 1, 1);
        dim3 grid((unsigned)((Wimg + TW - 1)/TW), (unsigned)((Himg + TH - 1)/TH), 1);

        const int Hcap = (int)(Hh > 64 ? 64 : Hh);
        const int shared_floats = Hcap * (int)IN + Hcap + Hcap * Hcap + Hcap + 4 * Hcap + 4;
        const size_t shared_bytes = (size_t)shared_floats * sizeof(float);

        render_tiles_kernel_tilecenter_presorted<<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
            rays_o.data_ptr<float>(),
            rays_d.data_ptr<float>(),
            positions.data_ptr<float>(),
            log_scales.data_ptr<float>(),
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

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs)
    {
        return {
            torch::Tensor(), // rays_o
            torch::Tensor(), // rays_d
            torch::Tensor(), // positions
            torch::Tensor(), // log_scales
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

// Wrapper function to be exposed to Python
std::tuple<torch::Tensor, torch::Tensor> render_tiles_autograd_wrapper(
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor positions,
    torch::Tensor log_scales,
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
        rays_o, rays_d, positions, log_scales, culled_prim_indices, tile_offsets,
        freq_bands, attenuation, W1, b1, W2, b2, W3, b3,
        Wimg, Himg, TW, TH, threshold);
    return {result[0], result[1]};
}

std::tuple<torch::Tensor, torch::Tensor> render_tiles_autograd_tilecenter_wrapper(
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor positions,
    torch::Tensor log_scales,
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
        rays_o, rays_d, positions, log_scales, culled_prim_indices, tile_offsets,
        freq_bands, attenuation, W1, b1, W2, b2, W3, b3,
        Wimg, Himg, TW, TH, threshold);
    return {result[0], result[1]};
}

std::tuple<torch::Tensor, torch::Tensor> render_tiles_autograd_tilecenter_presorted_wrapper(
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor positions,
    torch::Tensor log_scales,
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
        rays_o, rays_d, positions, log_scales, culled_prim_indices, tile_offsets,
        freq_bands, attenuation, W1, b1, W2, b2, W3, b3,
        Wimg, Himg, TW, TH, threshold);
    return {result[0], result[1]};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("render_tiles_autograd", &render_tiles_autograd_wrapper, "Tile single-pass MLP splatting (autograd ready)");
  m.def("render_tiles_autograd_tilecenter", &render_tiles_autograd_tilecenter_wrapper, "Tile MLP splatting (tile-center order, prim-major, shared weight staging)");
  m.def("render_tiles_autograd_tilecenter_presorted", &render_tiles_autograd_tilecenter_presorted_wrapper, "Tile MLP splatting (prim-major, presorted per-tile list; intended for globaldepth ordering)");
}