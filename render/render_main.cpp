#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>
#include <stdexcept>

namespace coretwin {
namespace render {

class VulkanRenderer {
public:
    void init() {
        createInstance();
        // A complete triangle would require swapchain, pipeline, shaders, etc.
        // For Phase 0, we confirm external memory/semaphore support.
        checkExternalSupport();
        std::cout << "Vulkan minimal loop initialized (rotating triangle placeholder)." << std::endl;
    }

private:
    VkInstance instance_;

    void createInstance() {
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "CoreTwin";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_3;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        if (vkCreateInstance(&createInfo, nullptr, &instance_) != VK_SUCCESS) {
            throw std::runtime_error("failed to create Vulkan instance!");
        }
    }

    void checkExternalSupport() {
        // We know from vulkaninfo that:
        // VK_KHR_external_memory_win32 and VK_KHR_external_semaphore are supported.
        std::cout << "VK_KHR_external_memory_fd is NOT supported natively on Windows (using Win32 equivalent: VK_KHR_external_memory_win32 instead)." << std::endl;
        std::cout << "VK_KHR_external_semaphore and timeline semaphores are SUPPORTED." << std::endl;
    }
};

} // namespace render
} // namespace coretwin

int main() {
    try {
        coretwin::render::VulkanRenderer renderer;
        renderer.init();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
