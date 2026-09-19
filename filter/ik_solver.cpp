#include "ik_solver.h"
#include <iostream>

namespace coretwin {
namespace filter {

IKSolver::IKSolver(const std::vector<int>& parents, const std::vector<float>& canonical_bone_lengths)
    : parents_(parents), bone_lengths_(canonical_bone_lengths) {}

void IKSolver::enforce_bone_lengths(std::vector<Joint>& joints) {
    for (size_t i = 1; i < joints.size(); ++i) { // Skip root
        int p = parents_[i];
        if (p == -1) continue;

        Eigen::Vector3f dir = joints[i].pos - joints[p].pos;
        float current_len = dir.norm();
        if (current_len > 1e-5f) {
            dir /= current_len;
            // Rescale child position to match canonical bone length
            joints[i].pos = joints[p].pos + dir * bone_lengths_[i];
        }
    }
}

void IKSolver::solve_ccd_ik(std::vector<Joint>& joints, const std::vector<int>& chain_indices, 
                            const Eigen::Vector3f& target, int max_iterations) {
    if (chain_indices.empty()) return;
    
    int end_effector = chain_indices.front(); // first element is leaf

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Iterate from end effector up to the root of the chain
        for (size_t i = 1; i < chain_indices.size(); ++i) {
            int current_joint = chain_indices[i];
            
            Eigen::Vector3f curr_pos = joints[current_joint].pos;
            Eigen::Vector3f eff_pos = joints[end_effector].pos;
            
            Eigen::Vector3f to_eff = (eff_pos - curr_pos).normalized();
            Eigen::Vector3f to_target = (target - curr_pos).normalized();
            
            // Compute rotation from to_eff -> to_target
            float cos_theta = to_eff.dot(to_target);
            if (cos_theta < 0.9999f) {
                Eigen::Vector3f axis = to_eff.cross(to_target).normalized();
                float angle = std::acos(std::clamp(cos_theta, -1.0f, 1.0f));
                
                Eigen::AngleAxisf rot(angle, axis);
                
                // Apply rotation to all child joints in the chain
                for (size_t j = 0; j < i; ++j) {
                    int child = chain_indices[j];
                    joints[child].pos = curr_pos + rot * (joints[child].pos - curr_pos);
                }
            }
        }
        
        // Early convergence check
        if ((joints[end_effector].pos - target).norm() < 1e-3f) {
            break;
        }
    }
}

void IKSolver::clamp_joint_angles(std::vector<Joint>& joints) {
    // Placeholder: anatomically plausible ranges (e.g., knee shouldn't hyperextend).
    // In a real system, you'd extract Euler angles, clamp, and rebuild positions/quats.
}

} // namespace filter
} // namespace coretwin
