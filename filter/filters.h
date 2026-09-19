#pragma once
#include <vector>
#include <cmath>
#include <Eigen/Dense>

namespace coretwin {
namespace filter {

// 4.1 Kalman Filter (Constant Velocity State-Space)
class KalmanFilter {
public:
    KalmanFilter(float dt = 1.0f/60.0f, float process_noise = 1e-2f, float measurement_noise = 1e-1f);
    
    // x: [pos_x, pos_y, pos_z]
    void update(float& x, float& y, float& z);

private:
    float dt_;
    // We run 3 independent 1D filters for X, Y, Z to simplify as a defensible assumption (PRD 4.1)
    Eigen::Matrix2f F_; // State transition
    Eigen::Matrix<float, 1, 2> H_; // Measurement model
    Eigen::Matrix2f Q_; // Process noise covariance
    float R_;           // Measurement noise covariance

    struct State1D {
        Eigen::Vector2f x_hat; // [position, velocity]^T
        Eigen::Matrix2f P;     // Error covariance
        State1D() {
            x_hat.setZero();
            P.setIdentity();
        }
    };

    State1D state_x_, state_y_, state_z_;

    float filter_1d(State1D& state, float measurement);
};

// 4.2 One-Euro Filter (Adaptive Low-Pass)
class OneEuroFilter {
public:
    OneEuroFilter(float freq = 60.0f, float mincutoff = 1.0f, float beta = 0.0f, float dcutoff = 1.0f);
    
    void update(float& x, float& y, float& z);

private:
    float freq_, mincutoff_, beta_, dcutoff_;
    
    struct State1D {
        float x_prev = 0.0f;
        float dx_prev = 0.0f;
        bool first_time = true;
    };

    State1D state_x_, state_y_, state_z_;
    
    float alpha(float cutoff);
    float filter_1d(State1D& state, float measurement);
};

} // namespace filter
} // namespace coretwin
