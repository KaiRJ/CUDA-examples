#ifndef WINDOW_H
#define WINDOW_H

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

class Window
{
  public:
    Window(int width, int height);

    bool WasKeyPressed(int key) const;
    void ProcessInput() const;
    bool ShouldClose() const;
    void SwapBuffers() const;

  private:
    GLFWwindow* m_window {nullptr};

    void CreateWindow(int width, int height);
};

#endif
