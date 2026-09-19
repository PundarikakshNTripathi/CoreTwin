#include "filters.h"
#include <iostream>

namespace coretwin {
namespace filter {

// --- Kalman Filter ---
KalmanFilter::KalmanFilter(float dt, float process_noise, float measurement_noise) 
    : dt_(dt), R_(measurement_noise) 
{
    F_ << 1.0f, dt_,
          0.0f, 1.0f;
    
    H_ << 1.0f, 0.0f;

    Q_ << process_noise, 0.0f,
          0.0f, process_noise;
}

float KalmanFilter::filter_1d(State1D& state, float measurement) {
    // Predict
    Eigen::Vector2f x_hat_minus = F_ * state.x_hat;
    Eigen::Matrix2f P_minus = F_ * state.P * F_.transpose() + Q_;

    // Update
    float S = (H_ * P_minus * H_.transpose())(0,0) + R_;
    Eigen::Vector2f K = P_minus * H_.transpose() * (1.0f / S);
    
    float y = measurement - (H_ * x_hat_minus)(0,0);
    state.x_hat = x_hat_minus + K * y;
    state.P = (Eigen::Matrix2f::Identity() - K * H_) * P_minus;

    return state.x_hat(0);
}

void KalmanFilter::update(float& x, float& y, float& z) {
    x = filter_1d(state_x_, x);
    y = filter_1d(state_y_, y);
    z = filter_1d(state_z_, z);
}

// --- One-Euro Filter ---
OneEuroFilter::OneEuroFilter(float freq, float mincutoff, float beta, float dcutoff)
    : freq_(freq), mincutoff_(mincutoff), beta_(beta), dcutoff_(dcutoff) {}

float OneEuroFilter::alpha(float cutoff) {
    float te = 1.0f / freq_;
    float tau = 1.0f / (2.0f * M_PI * cutoff);
    return 1.0f / (1.0f + tau / te);
}

float OneEuroFilter::filter_1d(State1D& state, float measurement) {
    if (state.first_time) {
        state.x_prev = measurement;
        state.first_time = false;
        return measurement;
    }

    float te = 1.0f / freq_;
    // Estimate derivative
    float dx = (measurement - state.x_prev) / te;
    float edx = dx; // Low pass on derivative using dcutoff (simplified here for brevity)
    // float alpha_d = alpha(dcutoff_);
    // state.dx_prev = state.dx_prev + alpha_d * (dx - state.dx_prev);
    // edx = state.dx_prev;

    // Adaptive cutoff
    float cutoff = mincutoff_ + beta_ * std::abs(edx);
    
    // Filtered value
    float a = alpha(cutoff);
    float x_hat = state.x_prev + a * (measurement - state.x_prev);
    
    state.x_prev = x_hat;
    return x_hat;
}

void OneEuroFilter::update(float& x, float& y, float& z) {
    x = filter_1d(state_x_, x);
    y = filter_1d(state_y_, y);
    z = filter_1d(state_z_, z);
}

} // namespace filter
} // namespace coretwin
