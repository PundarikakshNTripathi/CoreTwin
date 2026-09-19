#include "lbs_plugin.h"
#include <vector>

namespace coretwin {
namespace kernels {

// Basic 4x4 matrix multiply on host
mat4 mul(const mat4& A, const mat4& B) {
    mat4 C;
    for (int i=0; i<4; ++i) {
        for (int j=0; j<4; ++j) {
            float sum = 0;
            for (int k=0; k<4; ++k) {
                sum += A.m[i*4 + k] * B.m[k*4 + j];
            }
            C.m[i*4 + j] = sum;
        }
    }
    return C;
}

// CPU reference implementation
void cpu_forward_kinematics_and_lbs(
    int num_joints,
    int num_vertices,
    const int* parents,           
    const mat4* local_transforms, 
    const float* templates,       
    const float* weights,         
    const int* joint_indices,     
    float* out_vertices           
) {
    // 1. Forward kinematics
    std::vector<mat4> global_transforms(num_joints);
    for (int i = 0; i < num_joints; ++i) {
        if (parents[i] == -1) {
            global_transforms[i] = local_transforms[i];
        } else {
            global_transforms[i] = mul(global_transforms[parents[i]], local_transforms[i]);
        }
    }

    // 2. Linear Blend Skinning
    for (int i = 0; i < num_vertices; ++i) {
        float vx = templates[i*3 + 0];
        float vy = templates[i*3 + 1];
        float vz = templates[i*3 + 2];
        float vw = 1.0f; // Homogeneous coordinate

        float ox = 0, oy = 0, oz = 0;

        for (int j = 0; j < 4; ++j) { // Max 4 influences
            float w = weights[i*4 + j];
            if (w > 0.0f) {
                int j_idx = joint_indices[i*4 + j];
                const mat4& T = global_transforms[j_idx];
                
                float tx = T.m[0]*vx + T.m[1]*vy + T.m[2]*vz + T.m[3]*vw;
                float ty = T.m[4]*vx + T.m[5]*vy + T.m[6]*vz + T.m[7]*vw;
                float tz = T.m[8]*vx + T.m[9]*vy + T.m[10]*vz + T.m[11]*vw;

                ox += w * tx;
                oy += w * ty;
                oz += w * tz;
            }
        }
        
        out_vertices[i*3 + 0] = ox;
        out_vertices[i*3 + 1] = oy;
        out_vertices[i*3 + 2] = oz;
    }
}

// --- CUDA GPU Kernels ---

// Kernel 1: Forward Kinematics. Since num_joints is small (~24), 
// we do it in a single thread block to easily synchronize.
__global__ void fk_kernel(int num_joints, const int* parents, const mat4* local_transforms, mat4* global_transforms) {
    // We launch exactly 1 block with num_joints threads.
    int tid = threadIdx.x;
    if (tid >= num_joints) return;

    // Because kinematic tree has dependencies, we need a sequential pass or topological sort.
    // SMPL joints are strictly topologically sorted (parent index < child index).
    // Thread 0 processes joint 0, sync. Thread 1 processes joint 1, sync, etc.
    // For extreme low latency on 24 joints, one thread doing a loop is often faster than block syncs.
    if (tid == 0) {
        for (int i = 0; i < num_joints; ++i) {
            if (parents[i] == -1) {
                global_transforms[i] = local_transforms[i];
            } else {
                const mat4& A = global_transforms[parents[i]];
                const mat4& B = local_transforms[i];
                mat4 C;
                #pragma unroll
                for (int r=0; r<4; ++r) {
                    #pragma unroll
                    for (int c=0; c<4; ++c) {
                        C.m[r*4 + c] = A.m[r*4 + 0]*B.m[0*4 + c] +
                                       A.m[r*4 + 1]*B.m[1*4 + c] +
                                       A.m[r*4 + 2]*B.m[2*4 + c] +
                                       A.m[r*4 + 3]*B.m[3*4 + c];
                    }
                }
                global_transforms[i] = C;
            }
        }
    }
}

__global__ void lbs_kernel(
    int num_vertices,
    const mat4* __restrict__ global_transforms,
    const float* __restrict__ templates,
    const float* __restrict__ weights,
    const int* __restrict__ joint_indices,
    float* __restrict__ out_vertices
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_vertices) return;

    float vx = templates[i*3 + 0];
    float vy = templates[i*3 + 1];
    float vz = templates[i*3 + 2];
    
    float ox = 0, oy = 0, oz = 0;

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float w = weights[i*4 + j];
        if (w > 0.0f) {
            int j_idx = joint_indices[i*4 + j];
            // Shared memory could be used here for global_transforms, but texture/L1 cache is usually sufficient for 24 matrices (24 * 64 bytes = 1.5 KB).
            mat4 T = global_transforms[j_idx];
            
            ox += w * (T.m[0]*vx + T.m[1]*vy + T.m[2]*vz + T.m[3]);
            oy += w * (T.m[4]*vx + T.m[5]*vy + T.m[6]*vz + T.m[7]);
            oz += w * (T.m[8]*vx + T.m[9]*vy + T.m[10]*vz + T.m[11]);
        }
    }
    
    out_vertices[i*3 + 0] = ox;
    out_vertices[i*3 + 1] = oy;
    out_vertices[i*3 + 2] = oz;
}

int LBSPlugin::enqueue(
    int num_joints,
    int num_vertices,
    const int* d_parents,
    const mat4* d_local_transforms,
    const float* d_templates,
    const float* d_weights,
    const int* d_joint_indices,
    float* d_out_vertices,
    cudaStream_t stream
) {
    // We allocate a temporary buffer for global transforms. 
    // In a real TRT plugin, this would be requested via getWorkspaceSize().
    mat4* d_global_transforms = nullptr;
    cudaMallocAsync(&d_global_transforms, num_joints * sizeof(mat4), stream);

    // Launch approach: Separate small kernel for FK, then massive parallel kernel for LBS.
    // Why: The kinematic tree (FK) is strictly serial and tiny (24 nodes = ~1.5KB). 
    // Attempting to fuse FK into the LBS kernel requires grid-wide synchronization or redundant computation per-block.
    // Two kernels launched on the same stream back-to-back have practically zero CPU-side latency overhead, 
    // and correctly isolate the serial bottleneck from the massively parallel skinning loop.
    
    fk_kernel<<<1, 32, 0, stream>>>(num_joints, d_parents, d_local_transforms, d_global_transforms);

    int blockSize = 256;
    int numBlocks = (num_vertices + blockSize - 1) / blockSize;
    lbs_kernel<<<numBlocks, blockSize, 0, stream>>>(
        num_vertices, d_global_transforms, d_templates, d_weights, d_joint_indices, d_out_vertices
    );

    cudaFreeAsync(d_global_transforms, stream);
    return 0; // Success
}

} // namespace kernels
} // namespace coretwin
