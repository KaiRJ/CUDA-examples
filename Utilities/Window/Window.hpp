#ifndef WINDOW_H
#define WINDOW_H

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

class Window
{
  public:
    Window(int width, int height);

    bool wasKeyPressed(int key) const;
    void processInput() const;
    bool shouldClose() const;
    void swapBuffers() const;

  private:
    GLFWwindow* m_window {nullptr};

    void createWindow(int width, int height);
};

#endif
