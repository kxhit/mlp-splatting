// Tile-based MLP splatting kernel with per-pixel sorting.
// One block = one tile (TW x TH threads). Each thread (pixel) gathers
// intersections from a per-tile primitive list, sorts them by depth,
// then evaluates the MLP and alpha-blends in the correct order.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

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

// Host binding
std::tuple<torch::Tensor, torch::Tensor> render_tiles_single_pass(
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("render_tiles_single_pass", &render_tiles_single_pass, "Tile single-pass MLP splatting");
}