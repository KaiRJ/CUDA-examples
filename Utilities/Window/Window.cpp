#include "Window.hpp"

#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include <stdexcept>

static void initialise_glfw();
static void load_opengl_pointers();
static void framebuffer_size_callback(GLFWwindow* window, int width, int height);

Window::Window(int width, int height)
{
    initialise_glfw();
    createWindow(width, height);
    load_opengl_pointers();
}

void Window::createWindow(int width, int height)
{
    m_window = glfwCreateWindow(width, height, "CUDA Examples", nullptr, nullptr);
    if (m_window == nullptr)
    {
        throw std::runtime_error("Failed to create GLFW window");
    }
    glfwMakeContextCurrent(m_window);
    glfwSetFramebufferSizeCallback(m_window, framebuffer_size_callback);
}

bool Window::wasKeyPressed(int key) const
{
    return glfwGetKey(m_window, key) == GLFW_PRESS;
}

void Window::processInput() const
{
    if (wasKeyPressed(GLFW_KEY_ESCAPE))
    {
        glfwSetWindowShouldClose(m_window, true);
    }
}

bool Window::shouldClose() const { return glfwWindowShouldClose(m_window); }

void Window::swapBuffers() const { glfwSwapBuffers(m_window); }

// initialize and configure glfw
static void initialise_glfw()
{
    if (!glfwInit())
    {
        throw std::runtime_error("Failed to start GLFW");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); // OpenGL version
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true); // slow!
}

// load all OpenGL function pointers from glad
static void load_opengl_pointers()
{
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        throw std::runtime_error("Failed to initialize GLAD");
    }
}

// whenever the window size changed this callback function executes
static void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}
