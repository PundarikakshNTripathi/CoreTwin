#pragma once
#include <vector>
#include <cmath>
#include <iostream>

namespace coretwin {
namespace inference {

// Mock structure for 3D Keypoints from Maxine / RTMPose
struct Keypoint3D {
    float x, y, z, confidence;
};

class KeypointPrior {
public:
    KeypointPrior() {
        std::cout << "Keypoint Prior Model (Maxine/RTMPose) initialized." << std::endl;
    }

    // Task 1a: Use keypoints as a crop/bbox stabilization prior
    void stabilize_crop(float& bbox_x, float& bbox_y, float& bbox_w, float& bbox_h, 
                        const std::vector<Keypoint3D>& current_keypoints,
                        const std::vector<Keypoint3D>& prev_keypoints) {
        if (current_keypoints.empty() || prev_keypoints.empty()) return;
        
        // Mock stabilization logic: blend bbox based on keypoint movement
        // In a real system, we'd compute the center of mass of keypoints and smooth it.
        float alpha = 0.7f; // Exponential moving average factor
        // (Just a stub to demonstrate the architectural hook)
        // bbox_x = alpha * bbox_x + (1 - alpha) * ...
    }

    // Task 1b: Consistency check flagging low-confidence frames
    bool check_consistency(const std::vector<Keypoint3D>& prior_keypoints,
                           const float* smplx_joints, int num_joints, float threshold_mm = 50.0f) {
        if (prior_keypoints.size() < num_joints) return false;

        float max_error = 0.0f;
        for (int i = 0; i < num_joints; ++i) {
            float dx = prior_keypoints[i].x - smplx_joints[i*3 + 0];
            float dy = prior_keypoints[i].y - smplx_joints[i*3 + 1];
            float dz = prior_keypoints[i].z - smplx_joints[i*3 + 2];
            float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (dist > max_error) max_error = dist;
        }

        if (max_error > threshold_mm) {
            std::cerr << "[WARNING] Consistency check failed. SMPL-X regression diverges from prior by " 
                      << max_error << " mm (Threshold: " << threshold_mm << " mm)." << std::endl;
            return false; // Low confidence
        }
        return true;
    }
};

} // namespace inference
} // namespace coretwin
