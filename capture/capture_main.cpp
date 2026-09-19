#include <iostream>
#include <vector>
#include <stdexcept>

#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#endif

// We use OpenCV as a fallback
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

namespace coretwin {
namespace capture {

bool check_dma_buf_support(const std::string& device_path) {
#ifdef __linux__
    int fd = open(device_path.c_str(), O_RDWR | O_NONBLOCK, 0);
    if (fd < 0) {
        std::cerr << "Failed to open device: " << device_path << std::endl;
        return false;
    }

    struct v4l2_capability cap;
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) < 0) {
        std::cerr << "Failed to query capabilities" << std::endl;
        close(fd);
        return false;
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        std::cerr << "Device does not support video capture" << std::endl;
        close(fd);
        return false;
    }

    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
        std::cerr << "Device does not support streaming I/O" << std::endl;
        close(fd);
        return false;
    }

    struct v4l2_requestbuffers req = {0};
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_DMABUF;

    if (ioctl(fd, VIDIOC_REQBUFS, &req) < 0) {
        std::cerr << "V4L2_MEMORY_DMABUF not supported by this device/driver." << std::endl;
        close(fd);
        return false;
    }

    std::cout << "V4L2_MEMORY_DMABUF is SUPPORTED!" << std::endl;
    close(fd);
    return true;
#else
    std::cerr << "V4L2 DMA-BUF is only supported on Linux. On Windows, falling back to pinned-memory." << std::endl;
    return false;
#endif
}

class CameraCapture {
public:
    CameraCapture(int device_id = 0) {
        cap_.open(device_id, cv::CAP_ANY);
        if (!cap_.isOpened()) {
            throw std::runtime_error("Failed to open camera with OpenCV fallback.");
        }
    }

    // Pinned memory fallback
    bool get_frame_pinned(cv::cuda::GpuMat& d_frame) {
        cv::Mat frame;
        cap_ >> frame;
        if (frame.empty()) return false;
        
        // Use OpenCV CUDA upload (uses page-locked host memory if allocated via CudaMem)
        // Here we just do a simple upload to demonstrate the fallback
        d_frame.upload(frame);
        return true;
    }

private:
    cv::VideoCapture cap_;
};

} // namespace capture
} // namespace coretwin

int main() {
    std::cout << "Testing DMA-BUF support..." << std::endl;
    bool dma_supported = coretwin::capture::check_dma_buf_support("/dev/video0");
    if (!dma_supported) {
        std::cout << "Using Pinned-Memory Fallback." << std::endl;
        try {
            coretwin::capture::CameraCapture cam(0);
            std::cout << "Pinned-Memory Fallback initialized successfully." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error initializing fallback: " << e.what() << std::endl;
        }
    }
    return 0;
}
