#ifndef __CPU_BITMAP_H__
#define __CPU_BITMAP_H__
#define GLFW_INCLUDE_NONE

#include "Window/Window.hpp"
#include <GLFW/glfw3.h>

class CPUBitmap
{
  public:
    CPUBitmap(int width, int height);
    ~CPUBitmap();

    long image_size() const { return m_width * m_height * 4; }
    unsigned char* get_pixels() { return m_pixels; }

    void Draw() const;

  private:
    unsigned char* m_pixels {};
    int m_width {0};
    int m_height {0};
};

#endif
