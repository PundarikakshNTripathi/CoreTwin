#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>

namespace coretwin {
namespace kernels {

// Preprocessing kernel:
// 1. Crop to person bounding box (implicitly handled by taking cropped offset in grid)
// 2. Resize to static shape (e.g., 256x256)
// 3. Normalize (subtract mean, divide by std)
// 4. NHWC to NCHW layout
// 5. Cast to FP16
__global__ void preprocess_kernel(
    const uint8_t* __restrict__ input_img,  // HWC (typically BGR or RGB)
    half* __restrict__ output_tensor,       // CHW, FP16
    int input_w, int input_h, int input_stride,
    int crop_x, int crop_y, int crop_w, int crop_h,
    int target_w, int target_h,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= target_w || y >= target_h) return;

    // Nearest neighbor or bilinear. Using simple nearest neighbor for brevity here,
    // though bilinear is preferable in production.
    float src_x = x * (float)crop_w / target_w;
    float src_y = y * (float)crop_h / target_h;

    int ix = min((int)src_x, crop_w - 1) + crop_x;
    int iy = min((int)src_y, crop_h - 1) + crop_y;

    // Clamp to image bounds
    ix = max(0, min(ix, input_w - 1));
    iy = max(0, min(iy, input_h - 1));

    int src_idx = iy * input_stride + ix * 3;
    
    // Read RGB (assuming input is RGB)
    float r = input_img[src_idx + 0];
    float g = input_img[src_idx + 1];
    float b = input_img[src_idx + 2];

    // Normalize
    r = (r / 255.0f - mean_r) / std_r;
    g = (g / 255.0f - mean_g) / std_g;
    b = (b / 255.0f - mean_b) / std_b;

    // Output indices for NCHW
    int out_idx_r = 0 * (target_w * target_h) + y * target_w + x;
    int out_idx_g = 1 * (target_w * target_h) + y * target_w + x;
    int out_idx_b = 2 * (target_w * target_h) + y * target_w + x;

    // Cast to FP16 and write
    output_tensor[out_idx_r] = __float2half(r);
    output_tensor[out_idx_g] = __float2half(g);
    output_tensor[out_idx_b] = __float2half(b);
}

void launch_preprocess_kernel(
    const uint8_t* d_input, half* d_output,
    int input_w, int input_h, int input_stride,
    int crop_x, int crop_y, int crop_w, int crop_h,
    int target_w, int target_h,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((target_w + block.x - 1) / block.x, (target_h + block.y - 1) / block.y);

    // Standard ImageNet means and stds
    float mean_r = 0.485f, mean_g = 0.456f, mean_b = 0.406f;
    float std_r = 0.229f, std_g = 0.224f, std_b = 0.225f;

    preprocess_kernel<<<grid, block, 0, stream>>>(
        d_input, d_output,
        input_w, input_h, input_stride,
        crop_x, crop_y, crop_w, crop_h,
        target_w, target_h,
        mean_r, mean_g, mean_b,
        std_r, std_g, std_b
    );
}

} // namespace kernels
} // namespace coretwin
