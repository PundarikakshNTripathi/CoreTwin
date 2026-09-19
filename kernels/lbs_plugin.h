#pragma once
#include <cuda_runtime.h>
#include <iostream>

namespace coretwin {
namespace kernels {

// Simple matrix structure for joint transforms
struct mat4 {
    float m[16];
};

// Host CPU reference for forward kinematics and LBS
void cpu_forward_kinematics_and_lbs(
    int num_joints,
    int num_vertices,
    const int* parents,           // [num_joints]
    const mat4* local_transforms, // [num_joints]
    const float* templates,       // [num_vertices * 3]
    const float* weights,         // [num_vertices * 4] (assuming max 4 influences)
    const int* joint_indices,     // [num_vertices * 4]
    float* out_vertices           // [num_vertices * 3]
);

// Plugin interface placeholder for TensorRT IPluginV2DynamicExt
class LBSPlugin {
public:
    // This represents the enqueue call of IPluginV2DynamicExt
    int enqueue(
        int num_joints,
        int num_vertices,
        const int* d_parents,
        const mat4* d_local_transforms,
        const float* d_templates,
        const float* d_weights,
        const int* d_joint_indices,
        float* d_out_vertices,
        cudaStream_t stream
    );
};

} // namespace kernels
} // namespace coretwin
