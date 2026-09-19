#pragma once
#include <vector>
#include <Eigen/Dense>

namespace coretwin {
namespace filter {

// Simple 3D Joint
struct Joint {
    Eigen::Vector3f pos;
};

class IKSolver {
public:
    IKSolver(const std::vector<int>& parents, const std::vector<float>& canonical_bone_lengths);

    // Bone-length lock constraint
    void enforce_bone_lengths(std::vector<Joint>& joints);

    // CCD IK correction for a specific limb chain targeting an end-effector position
    void solve_ccd_ik(std::vector<Joint>& joints, const std::vector<int>& chain_indices, 
                      const Eigen::Vector3f& target, int max_iterations = 3);

    // Joint-angle clamping (mock limits)
    void clamp_joint_angles(std::vector<Joint>& joints);

private:
    std::vector<int> parents_;
    std::vector<float> bone_lengths_;
};

} // namespace filter
} // namespace coretwin
