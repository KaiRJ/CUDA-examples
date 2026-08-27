#include "CPUBitmap/CPUBitmap.hpp"

#include "glad/glad.h"

CPUBitmap::CPUBitmap(int width, int height) : m_width {width}, m_height {height}
{
    m_pixels = new unsigned char[image_size()] {};
}

CPUBitmap::~CPUBitmap() { delete[] m_pixels; }

void CPUBitmap::Draw() const
{
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glRasterPos2f(-1.0f, -1.0f);
    glDrawPixels(m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, m_pixels);
    glPixelZoom(1.0f, 1.0f);
    glFlush();
}
