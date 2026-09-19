#include <iostream>
#include <vector>
#include <cmath>
#include "../filter/ik_solver.h"

using namespace coretwin::filter;

int main() {
    std::cout << "Running Kinematic Constraint Solver Unit Test..." << std::endl;
    
    // Skeleton: root(0) -> child(1) -> grandchild(2)
    std::vector<int> parents = {-1, 0, 1};
    std::vector<float> bone_lengths = {0.0f, 10.0f, 5.0f}; // Canonical lengths

    IKSolver solver(parents, bone_lengths);

    std::vector<Joint> joints(3);
    // Setup a deliberately violating pose
    joints[0].pos = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
    joints[1].pos = Eigen::Vector3f(0.0f, 15.0f, 0.0f); // Length 15, expected 10
    joints[2].pos = Eigen::Vector3f(10.0f, 15.0f, 0.0f); // Length 10, expected 5

    float pre_err1 = (joints[1].pos - joints[0].pos).norm() - bone_lengths[1];
    float pre_err2 = (joints[2].pos - joints[1].pos).norm() - bone_lengths[2];
    std::cout << "Pre-correction deviation: Bone 1 = " << pre_err1 << ", Bone 2 = " << pre_err2 << std::endl;

    solver.enforce_bone_lengths(joints);

    float post_len1 = (joints[1].pos - joints[0].pos).norm();
    float post_len2 = (joints[2].pos - joints[1].pos).norm();

    std::cout << "Post-correction length: Bone 1 = " << post_len1 << ", Bone 2 = " << post_len2 << std::endl;

    bool passed = (std::abs(post_len1 - bone_lengths[1]) < 1e-4f) && 
                  (std::abs(post_len2 - bone_lengths[2]) < 1e-4f);
    
    if (passed) {
        std::cout << "[PASS] Bone lengths successfully enforced." << std::endl;
        return 0;
    } else {
        std::cerr << "[FAIL] Bone lengths violate constraints!" << std::endl;
        return 1;
    }
}
