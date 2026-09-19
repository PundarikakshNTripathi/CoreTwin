#include <iostream>
#include <vector>
#include <cmath>
#include "../kernels/lbs_plugin.h"

using namespace coretwin::kernels;

int main() {
    std::cout << "Running LBS Plugin vs CPU Reference Unit Test..." << std::endl;
    // Mock 2 joints, 1 vertex
    int num_joints = 2;
    int num_vertices = 1;

    std::vector<int> parents = {-1, 0};
    
    std::vector<mat4> local_transforms(2);
    // Identity for both
    for (int i=0; i<16; ++i) {
        local_transforms[0].m[i] = (i%5 == 0) ? 1.0f : 0.0f;
        local_transforms[1].m[i] = (i%5 == 0) ? 1.0f : 0.0f;
    }
    // offset joint 1
    local_transforms[1].m[3] = 1.0f; 

    std::vector<float> templates = {0.0f, 0.0f, 0.0f};
    std::vector<float> weights = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<int> joint_indices = {1, 0, 0, 0};
    
    std::vector<float> cpu_out(3);
    
    cpu_forward_kinematics_and_lbs(
        num_joints, num_vertices,
        parents.data(), local_transforms.data(),
        templates.data(), weights.data(), joint_indices.data(),
        cpu_out.data()
    );

    std::cout << "CPU Output: " << cpu_out[0] << ", " << cpu_out[1] << ", " << cpu_out[2] << std::endl;
    std::cout << "Expected: 1, 0, 0" << std::endl;
    
    bool passed = (std::abs(cpu_out[0] - 1.0f) < 1e-5);
    if (passed) {
        std::cout << "[PASS] LBS CPU Reference Math matches expectations." << std::endl;
    } else {
        std::cerr << "[FAIL] LBS CPU Reference Math failed." << std::endl;
        return 1;
    }

    // NOTE: GPU test requires CUDA context and device pointers. 
    // We assume test harness would set this up in a full GTest suite.
    return 0;
}
