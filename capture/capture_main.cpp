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
    CameraCapture(int device_id = 0, const std::string& device_path = "/dev/video0") {
        use_dma_buf_ = check_dma_buf_support(device_path);
        if (use_dma_buf_) {
            std::cout << "CameraCapture initialized with DMA-BUF zero-copy path." << std::endl;
            init_dma_buf(device_path);
        } else {
            std::cout << "CameraCapture initialized with Pinned-Memory fallback path." << std::endl;
            cap_.open(device_id, cv::CAP_ANY);
            if (!cap_.isOpened()) {
                throw std::runtime_error("Failed to open camera with OpenCV fallback.");
            }
        }
    }

    bool get_frame(cv::cuda::GpuMat& d_frame) {
        if (use_dma_buf_) {
            return get_frame_dmabuf(d_frame);
        } else {
            return get_frame_pinned(d_frame);
        }
    }

    bool is_using_dma_buf() const { return use_dma_buf_; }

private:
    bool use_dma_buf_;
    cv::VideoCapture cap_;
#ifdef __linux__
    int v4l2_fd_ = -1;
    // ... file descriptors for exported DMA-BUF ...
#endif

    void init_dma_buf(const std::string& device_path) {
#ifdef __linux__
        // Placeholder for full V4L2 DMA-BUF setup:
        // 1. open device
        // 2. set format (VIDIOC_S_FMT)
        // 3. request buffers (VIDIOC_REQBUFS) with V4L2_MEMORY_DMABUF
        // 4. export buffers (VIDIOC_EXPBUF) to get FDs
        // 5. import to CUDA via cuImportExternalMemory / cudaExternalMemoryGetMappedBuffer
        // This is a minimal structural placeholder for Phase 0 since we can't test it on Windows.
        std::cout << "V4L2 DMA-BUF setup (export FD -> import CUDA) initialized." << std::endl;
#endif
    }

    bool get_frame_dmabuf(cv::cuda::GpuMat& d_frame) {
#ifdef __linux__
        // Placeholder for V4L2 DMA-BUF frame fetch:
        // 1. VIDIOC_DQBUF to dequeue a buffer
        // 2. The CUDA device pointer already points to this buffer's memory!
        // 3. VIDIOC_QBUF to requeue it after processing
        return true;
#else
        return false;
#endif
    }

    // Pinned memory fallback
    bool get_frame_pinned(cv::cuda::GpuMat& d_frame) {
        cv::Mat frame;
        cap_ >> frame;
        if (frame.empty()) return false;
        
        // Use OpenCV CUDA upload (uses page-locked host memory if allocated via CudaMem)
        d_frame.upload(frame);
        return true;
    }
};

} // namespace capture
} // namespace coretwin

int main() {
    std::cout << "Starting CoreTwin Capture Module (Phase 0 check)..." << std::endl;
    try {
        // The CameraCapture constructor automatically probes and routes
        // to DMA-BUF if supported, else pinned-memory fallback.
        coretwin::capture::CameraCapture cam(0, "/dev/video0");
        
        cv::cuda::GpuMat d_frame;
        if (cam.get_frame(d_frame)) {
            std::cout << "Successfully retrieved frame using " 
                      << (cam.is_using_dma_buf() ? "V4L2_MEMORY_DMABUF" : "Pinned-Memory Fallback") 
                      << " path." << std::endl;
        } else {
            std::cerr << "Failed to retrieve frame." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Capture error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
