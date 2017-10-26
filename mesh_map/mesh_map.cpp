#include <string>
#include <iostream>
#include <memory>
#include <set>
#include <vector>

#ifdef WIN32
    #define WIN32_LEAN_AND_MEAN 1
    #define NOMINMAX 1
    #include <windows.h>
#endif
#if defined(_WIN64)
    #include <windows.h>
#endif

#if defined(__APPLE__)
    #include <OpenGL/gl.h>
    #include <OpenGL/glu.h>
#else
    #include <GL/gl.h>
    #include <GL/glu.h>
#endif

#include "opencv2/core.hpp"
#include "opencv2/core/opengl.hpp"
//#include "opencv2/core/cuda.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/calib3d.hpp>
#include "opencv2/photo.hpp"

#include "TriMesh.h"
#include "XForm.h"

// More about gFalgs- https://gflags.github.io/gflags/
#include <gflags/gflags.h>
// For old version of gflags
#ifndef GFLAGS_NAMESPACE
    #define GFLAGS_NAMESPACE google
#endif

//
// gFlags - Configuration
//

DEFINE_int32(grid, 1, "Number of parts per axis");

//
// Macros
//

const float PI = 3.14159265358979323846264f;
#define RAD_TO_DEG(x) ((x) * 180 / PI)
#define DEG_TO_RAD(x) ((x) * PI / 180)


int dbgCount = 0;
// #define DEBUG
#define DBG(params) \
    std::cout << dbgCount++ << ") " << __FUNCTION__ << ": " << params << std::endl

#ifdef DEBUG
    #define DBG_T(params) DBG(params)
#else
    #define DBG_T(params)
#endif

#define MAIN_WINDOW_NAME "OpenGL"
#define DEPTH_WINDOW_NAME "Depth"

//
// Consts
//
#define CV_COLOR_BGR(name,b,g,r) \
    const cv::Scalar name((b), (g), (r))
#define GL_COLOR_RGB(name,r,g,b) \
    const GLfloat name##GL[3] = { (r) / 255.0f, (g) / 255.0f, (b) / 255.0f }

#define CV_COLOR(name,b,g,r) \
    CV_COLOR_BGR(name,(b),(g),(r)); \
    GL_COLOR_RGB(name,(r),(g),(b))

CV_COLOR(green,   0, 255, 0);
CV_COLOR(red,     0, 0, 255);
CV_COLOR(blue,    255, 0, 0);
CV_COLOR(white,   255, 255, 255);
CV_COLOR(black,   0, 0, 0);
CV_COLOR(yellow,  0, 255, 255);
CV_COLOR(light_yellow,  242, 255, 255);
CV_COLOR(magenta, 255, 0, 255);
CV_COLOR(orange,  0, 165, 255);
CV_COLOR(azul,    255, 255, 0);
CV_COLOR(brown,   19, 69, 139);
CV_COLOR(gray,    128, 128, 128);
CV_COLOR(pink,    203, 192, 255);
CV_COLOR(purple,  128, 0, 128);
CV_COLOR(voilet,  255, 0, 127);

std::vector<const GLfloat *> glColorsVec = {
    greenGL,
    redGL,
    blueGL,
    blackGL,
    yellowGL,
    magentaGL,
    orangeGL,
    azulGL,
    brownGL,
    grayGL,
    pinkGL,
    purpleGL,
    voiletGL
};

//
// Globals
//
GLfloat const *backgroundColorGL = blackGL;
GLfloat const *facesColorGL = whiteGL;

std::string gModelFilePath;
int gWinWidth = 800;
int gWinHeight = 600;
float fov = 0.7f;
bool gReset = true;

std::unique_ptr<trimesh::TriMesh> themesh;
trimesh::xform xf; // Current location and orientation

//cv::Mat gDepthImg(cv::Size(gWinWidth, gWinHeight), CV_8UC1);

cv::Mat gFullMap;
cv::Mat gFullGroundLevelEstimationMap;
bool gAutoNav = false;
bool gUpperView = true;

struct OrthoProjection {
    float left;
    float right;
    float bottom;
    float top;
    float neardist;
    float fardist;

    OrthoProjection() : left(0.0f), right(0.0f), bottom(0.0f), top(0.0f), neardist(0.0f), fardist(0.0f) {}
    OrthoProjection(float left, float right, float bottom, float top, float neardist, float fardist) : left(left),
        right(right), bottom(bottom), top(top), neardist(neardist), fardist(fardist) {}

    void set(float left, float right, float bottom, float top, float neardist, float fardist)
    {
        left = left;
        right = right;
        bottom = bottom;
        top = top;
        neardist = neardist;
        fardist = fardist;
    }

    std::string print(unsigned precision = 17) const
    {
        std::stringstream ss;
        ss << std::setprecision(precision) << "(left,right,bottom,top,neardist,fardist)=(" << left << ", " << right <<
            ", " << bottom << ", " << top << ", " << neardist << ", " << fardist << ")";
        return ss.str();
    }

    bool write(const std::string &filePath)
    {
        std::ofstream f(filePath);
        f << print();
        f.close();
        return f.good();
    }
};
std::vector<OrthoProjection> gOrthoViews;
unsigned gViewIdx = 0;

//
// Utilities
//
static inline std::ostream& operator<<(std::ostream &out, const OrthoProjection &data)
{
    out << data.print(6);

    return out;
}

template <typename T>
inline std::ostream& operator<<(std::ostream &out, const std::vector<T> &vector)
{
    for(auto i : vector)
        out << i << ", ";

    if (vector.size())
        out << "\b\b";

    return out;
}

// E.g. input "C:\Dir\File.bat" -> "File.bat".
// E.g. input "File.bat" -> "File.bat".
std::string getFileName(const std::string &strPath)
{
    size_t found = strPath.find_last_of("/\\");
    return strPath.substr(found + 1);
}

std::string getFileNameNoExt(const std::string &strPath)
{
    std::string fileName = getFileName(strPath);

    size_t pos = fileName.find_last_of(".");
    if (pos != std::string::npos)
        fileName = fileName.substr(0, pos);

    return fileName;
}

std::string getFilePathNoExt(const std::string &filePath)
{
    size_t pos = filePath.find_last_of(".");
    if (pos == std::string::npos) // No file extension
        return filePath;

    return filePath.substr(0, pos);
}

// Fixes a bug that sometime the initial resize does not stick and we get back to the default win size
void verifySize(const std::string &winName)
{
    cv::setOpenGlContext(winName);

    GLint V[4];
    glGetIntegerv(GL_VIEWPORT, V);
    GLint width = V[2], height = V[3];

    if (width == 100 && height == 30) // Default window size
    {
        DBG("Window BUG [" << winName << "] has default size. Resizing");
        cv::resizeWindow(winName, gWinWidth, gWinHeight);
    }
}

void printViewPort()
{
    GLint V[4];
    glGetIntegerv(GL_VIEWPORT, V);
    GLint width = V[2], height = V[3];

    DBG("v0: " << V[0] << ", v1: " << V[1] << ", width: " << width << ", height: " << height);
}

void printProjectionMatrix()
{
    GLdouble projection[16];
    glGetDoublev(GL_PROJECTION_MATRIX, projection);
    cv::Mat glProjMat(cv::Size(4, 4), CV_64FC1, projection);

    DBG("GL projection matrix:\n" << glProjMat.t());
}

void printModelViewMatrix()
{
    GLdouble modelview[16];
    glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
    cv::Mat glModelViewMat(cv::Size(4, 4), CV_64FC1, modelview);

    DBG("GL model view matrix:\n" << glModelViewMat.t());
}

void printCurrentMatrixMode()
{
    GLint mode;
    glGetIntegerv(GL_MATRIX_MODE, &mode);

    std::string modeStr;
    switch (mode)
    {
    case GL_MODELVIEW:
        modeStr = "GL_MODELVIEW";
        break;
    case GL_PROJECTION:
        modeStr = "GL_PROJECTION";
        break;
    case GL_TEXTURE:
        modeStr = "GL_TEXTURE";
        break;
    default:
        modeStr = "Unknown matrix mode";
        break;
    }

    DBG("Current matrix mode is: " << modeStr);
}

std::string getOpenCVmatType(const cv::Mat &mat)
{
    int number = mat.type();

    // Find type
    int imgTypeInt = number % 8;
    std::string imgTypeString;

    switch (imgTypeInt)
    {
        case 0:
            imgTypeString = "8U";
            break;
        case 1:
            imgTypeString = "8S";
            break;
        case 2:
            imgTypeString = "16U";
            break;
        case 3:
            imgTypeString = "16S";
            break;
        case 4:
            imgTypeString = "32S";
            break;
        case 5:
            imgTypeString = "32F";
            break;
        case 6:
            imgTypeString = "64F";
            break;
        default:
            break;
    }

    // find channel
    int channel = (number/8) + 1;

    std::stringstream type;
    type<<"CV_"<<imgTypeString<<"C"<<channel;

    return type.str();
}

cv::Mat getCvMatFromScreen(const std::string &winName)
{
    cv::setOpenGlContext(winName); // Sets the specified window as current OpenGL context

    // Read pixels
    GLint V[4];
    glGetIntegerv(GL_VIEWPORT, V);
    GLint width = V[2], height = V[3];
    std::unique_ptr<char> buf = std::unique_ptr<char>(new char[width * height * 3]);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    GLenum format = GL_BGR;
    glReadPixels(V[0], V[1], width, height, format, GL_UNSIGNED_BYTE, buf.get());

    // Flip top-to-bottom
    for (int i = 0; i < height / 2; i++)
    {
        char *row1 = buf.get() + 3 * width * i;
        char *row2 = buf.get() + 3 * width * (height - 1 - i);
        for (int j = 0; j < 3 * width; j++)
            std::swap(row1[j], row2[j]);
    }

    cv::Mat img(cv::Size(width, height), CV_8UC3, buf.get(), cv::Mat::AUTO_STEP);

    return img.clone();
}

cv::Mat getDepthCvMatFromScreen(const std::string &winName)
{
    cv::setOpenGlContext(winName); // Sets the specified window as current OpenGL context

    // Read pixels
    GLint V[4];
    glGetIntegerv(GL_VIEWPORT, V);
    GLint width = V[2], height = V[3];
    std::unique_ptr<float> buf = std::unique_ptr<float>(new float[width * height]);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(V[0], V[1], width, height, GL_DEPTH_COMPONENT, GL_FLOAT, buf.get());

    // Flip top-to-bottom
    for (int i = 0; i < height / 2; i++)
    {
        float *row1 = buf.get() + width * i;
        float *row2 = buf.get() + width * (height - 1 - i);
        for (int j = 0; j < width; j++)
            std::swap(row1[j], row2[j]);
    }

    cv::Mat img = cv::Mat(cv::Size(width, height), CV_32FC1, buf.get(), cv::Mat::AUTO_STEP).clone();

    return img;
}

//
// Implementation
//

// Draw triangle strips.  They are stored as length followed by values.
void drawTstrips()
{
    const int *t = &themesh->tstrips[0];
    const int *end = t + themesh->tstrips.size();
    while (likely(t < end))
    {
        int striplen = *t++;
        glDrawElements(GL_TRIANGLE_STRIP, striplen, GL_UNSIGNED_INT, t);
        t += striplen;
    }
}

void drawFaces()
{
    glDisable(GL_CULL_FACE); // Order of vertexes should not determine faces normals (I.e. Draw both sides of face)

    // Enable the vertex array
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, &themesh->vertices[0][0]);

    glColor3fv(facesColorGL);

    // Draw the mesh, possibly with color and/or lighting
    glDepthFunc(GL_LESS);
    glEnable(GL_DEPTH_TEST);
    glPolygonOffset(5.0f, 30.0f);
    glEnable(GL_POLYGON_OFFSET_FILL);

    // Draw the geometry - using display list
    if (!glIsList(1))
    {
        glNewList(1, GL_COMPILE);
        drawTstrips();
        glEndList();
    }
    glCallList(1);
    glDisableClientState(GL_VERTEX_ARRAY);

    //drawGround();
}

void getRgbFromInt(int rgbColor, unsigned &r, unsigned &g, unsigned &b)
{
    r = (rgbColor & 0xFF0000) >> 16;
    g = (rgbColor & 0xFF00) >> 8;
    b = rgbColor & 0xFF;
}

void drawEdge(int vertex1, int vertex2, int rgbColor)
{
#if 1
    unsigned r, g, b;
    getRgbFromInt(rgbColor, r, g, b);
    GLfloat color[3] = { (float)r / 255.0f, (float)g / 255.0f, (float)b / 255.0f };
#else
    GLfloat const * color = glColorsVec[rgbColor % glColorsVec.size()];
#endif
    glColor3fv(color);

    // Draw the edge
    glVertex3fv(themesh->vertices[vertex1]);
    glVertex3fv(themesh->vertices[vertex2]);
}

static inline GLint myGluUnProject(GLdouble winX, GLdouble winY, GLdouble winZ,
	const GLdouble *model, const GLdouble *proj, const GLint *view,
	GLdouble* objX, GLdouble* objY, GLdouble* objZ)
{
    trimesh::xform xfMat = inv(trimesh::xform(proj) * trimesh::xform(model));
    trimesh::Vec<3,double> v = xfMat * trimesh::Vec<3,double>(
        (winX - view[0]) / view[2] * 2.0 - 1.0,
        (winY - view[1]) / view[3] * 2.0 - 1.0,
        winZ * 2.0 - 1.0);
    *objX = v[0];
    *objY = v[1];
    *objZ = v[2];
    return GL_TRUE;
}

// Taken from Trimesh2 library - GLCamera
// Read back the framebuffer at the given pixel, and determine
// the 3D point there.  If there's nothing there, reads back a
// number of pixels farther and farther away.
bool read_depth(int x, int y, trimesh::point &p)
{
    GLdouble M[16], P[16];
    GLint V[4];
    glGetDoublev(GL_MODELVIEW_MATRIX, M);
    glGetDoublev(GL_PROJECTION_MATRIX, P);
    glGetIntegerv(GL_VIEWPORT, V);

    static const float dx[] = { 0, 1,-1,-1, 1, 3,-3, 0, 0, 6,-6,-6, 6, 25,-25,  0,  0 };
    static const float dy[] = { 0, 1, 1,-1,-1, 0, 0, 3,-3, 6, 6,-6,-6,  0,  0, 25,-25 };
    const float scale = 0.01f;
    const int displacements = sizeof(dx) / sizeof(float);

    int xmin = V[0], xmax = V[0] + V[2] - 1, ymin = V[1], ymax = V[1] + V[3] - 1;

    for (int i = 0 ; i < displacements; i++)
    {
        int xx = std::min(std::max(x + int(dx[i] * scale * V[2]), xmin), xmax);
        int yy = std::min(std::max(y + int(dy[i] * scale * V[3]), ymin), ymax);
        float d;
        glReadPixels(xx, yy, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &d);

        static float maxd = 0.0f;
        if (!maxd)
        {
            glScissor(xx, yy, 1, 1);
            glEnable(GL_SCISSOR_TEST);
            glClearDepth(1);
            glClear(GL_DEPTH_BUFFER_BIT);
            glReadPixels(xx, yy, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &maxd);
            if (maxd)
            {
                glClearDepth(d / maxd);
                glClear(GL_DEPTH_BUFFER_BIT);
            }
            glDisable(GL_SCISSOR_TEST);
            glClearDepth(1);
            if (!maxd)
                return false;
        }

        d /= maxd;
        if (d > 0.0001f && d < 0.9999f)
        {
            GLdouble X, Y, Z;
            myGluUnProject(xx, yy, d, M, P, V, &X, &Y, &Z);
            p = trimesh::point((float)X, (float)Y, (float)Z);
            return true;
        }
    }

    return false;
}

OrthoProjection getFullViewParams()
{
    DBG("Entered");

    // Taken from upperView
    trimesh::xform xfMat = trimesh::xform::trans(0, 0, -3.5f / fov * themesh->bsphere.r) *
        trimesh::xform::trans(-themesh->bsphere.center);

    trimesh::point scene_center = xfMat * themesh->bsphere.center;
    //DBG("scene_center:" << scene_center);
    float scene_size = themesh->bsphere.r;

    GLint V[4];
    glGetIntegerv(GL_VIEWPORT, V);
    int width = V[2], height = V[3];

    float surface_depth = 0.0f;
    trimesh::point surface_point;
    if (read_depth(width/2, height/2, surface_point))
        surface_depth = -surface_point[2];

#define DOF 10.0f
#define MAXDOF 10000.0f
    float fardist  = std::max(-(scene_center[2] - scene_size), scene_size / DOF);
    float neardist = std::max(-(scene_center[2] + scene_size), scene_size / MAXDOF);
    surface_depth = std::min(surface_depth, fardist);
    surface_depth = std::max(surface_depth, neardist);
    surface_depth = std::max(surface_depth, fardist / MAXDOF);
    neardist = std::max(neardist, surface_depth / DOF);

    float diag = sqrt(float(trimesh::sqr(width) + trimesh::sqr(height)));
    float top = (float) height/diag * 0.5f * fov * neardist;
    float bottom = -top;
    float right = (float) width/diag * 0.5f * fov * neardist;
    float left = -right;

    OrthoProjection fullViewProj(left, right, bottom, top, neardist, fardist);
    DBG("Done. Full view projection [" << fullViewProj << "]");
    return fullViewProj;
}

/*
#if 1
    //glMatrixMode(GL_PROJECTION);
    //glLoadIdentity();
    //glFrustum(left, right, bottom, top, neardist, fardist);
    //glOrtho(left, right, bottom, top, neardist, fardist);

    DBG("glOrtho/glFrustum values- left [" << left << "], right [" << right << "], bottom [" << bottom << "], top [" <<
        top << "], neardist [" << neardist << "], fardist [" << fardist << "]");

    //glMatrixMode(GL_MODELVIEW);
    //glLoadIdentity();
#else
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    //glOrtho(-2.5, 2.5, -2.5, 2.5, 1, 18);

    trimesh::box &bbox = themesh->bbox;
    float left =themesh->bbox.min[0];
    float right =themesh->bbox.max[0];
    float bottom =themesh->bbox.min[1];
    float top = themesh->bbox.max[1];

    float cameraZ = -xf[14];
    //float bboxHeight = themesh->bbox.max[2] - themesh->bbox.min[2];
    float neardist = cameraZ - bbox.max[2];
    float fardist = cameraZ - bbox.min[2];

    glOrtho(left, right, bottom, top, neardist, fardist+3);

    DBG("glOrtho: left [" << left << "], right [" << right << "], bottom [" << bottom << "], top [" << top <<
        "], neardist [" << neardist << "], fardist [" << fardist << "]");

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
#endif
}*/

std::vector<OrthoProjection> splitViews(unsigned partsPerAxis)
{
    DBG("partsPerAxis [" << partsPerAxis << "]");

    const OrthoProjection orthoFullView = getFullViewParams();

    float widthPart = (orthoFullView.right - orthoFullView.left) / float(partsPerAxis);
    float heightPart = (orthoFullView.top - orthoFullView.bottom) / float(partsPerAxis);

    DBG("widthPart [" << widthPart << "], heightPart [" << heightPart << "]");

    if (widthPart <= 0 || heightPart <= 0) // Sanity
        throw std::runtime_error("Width/Height cannot be <= 0");

    // XXX: If the stitches are not good enough it is possible I should use the end of one part as the start of the
    // other

    std::vector<OrthoProjection> orthoViews;
    unsigned count = 0;
    for (unsigned x = 0; x < partsPerAxis; x++)
    {
        for (unsigned y = 0; y < partsPerAxis; y++)
        {
            orthoViews.push_back(OrthoProjection(
                orthoFullView.left + x * widthPart,
                orthoFullView.left + (x + 1) * widthPart,
                orthoFullView.bottom + y * heightPart,
                orthoFullView.bottom + (y + 1) * heightPart,
                orthoFullView.neardist,
                orthoFullView.fardist));

            DBG("Index " << count << ", x,y " << x << ", " << y << ", is: " << orthoViews[count]);
            count++;
        }
    }

    return orthoViews;
}

void setViewParams(const OrthoProjection &orthoProjData)
{
    DBG_T("orthoProjData: " << orthoProjData);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
//moti
   // glMultMatrixd((double *)xf);
    glOrtho(orthoProjData.left, orthoProjData.right, orthoProjData.bottom, orthoProjData.top, orthoProjData.neardist,
        orthoProjData.fardist);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

// Angles are in radians
trimesh::XForm<double> getCamRotMatRad(float yaw, float pitch, float roll)
{
    float a = roll, b = yaw, c = pitch;

    //http://msl.cs.uiuc.edu/planning/node102.html
    trimesh::XForm<double> rotMat = trimesh::XForm<double>
        (std::cos(a)*std::cos(b), std::cos(a)*std::sin(b)*std::sin(c)-std::sin(a)*std::cos(c),
            std::cos(a)*std::sin(b)*std::cos(c)+std::sin(a)*std::sin(c), 0,

         std::sin(a)*std::cos(b), std::sin(a)*std::sin(b)*std::sin(c)+std::cos(a)*std::cos(c),
         std::sin(a)*std::sin(b)*std::cos(c)-std::cos(a)*std::sin(c), 0,

         -std::sin(b), std::cos(b)*std::sin(c), std::cos(b)*std::cos(c), 0,

         0, 0, 0, 1);

    trimesh::transpose(rotMat); // Since XForm is column-major

    //DBG("Rotation-\n" << rotMat);
    return rotMat;
}

trimesh::XForm<double> getCamRotMatDeg(float yaw, float pitch, float roll)
{
    return getCamRotMatRad(DEG_TO_RAD(yaw), DEG_TO_RAD(pitch), DEG_TO_RAD(roll));
}

trimesh::xform gLastUsedXf;

void drawModel(trimesh::xform &xf)
{
    //DBG("---BEFORE---");
    //DBG("xf:\n" << xf);

    if (gReset)
    {
        DBG("Reseting...");
        gOrthoViews = splitViews(FLAGS_grid);
        gFullMap = cv::Mat(gWinHeight*FLAGS_grid, gWinWidth*FLAGS_grid, CV_8UC3);
        gFullGroundLevelEstimationMap = cv::Mat(gWinHeight*FLAGS_grid, gWinWidth*FLAGS_grid, CV_8UC3);
        gViewIdx = 0;

        gReset = false;
    }

    //setViewParams(gOrthoViews[gViewIdx]);

    //printCurrentMatrixMode();
    //printModelViewMatrix();
    //printProjectionMatrix();
    //printViewPort();

    // Transform and draw
    glPushMatrix();

    trimesh::xform xfOrtho = trimesh::xform::ortho(gOrthoViews[gViewIdx].left,gOrthoViews[gViewIdx].right,
            gOrthoViews[gViewIdx].bottom, gOrthoViews[gViewIdx].top,
            gOrthoViews[gViewIdx].neardist, gOrthoViews[gViewIdx].fardist) * xf;

    if (!gUpperView)
    {
        xfOrtho[14] *= -1;
        xfOrtho = getCamRotMatDeg(0, 180 , 0) * xfOrtho;
    }

    gLastUsedXf = xfOrtho;
    glMultMatrixd((double *)xfOrtho);


    //DBG("---AFTER---");
    //DBG("xf:\n" << xf);
    //printModelViewMatrix();
    //printProjectionMatrix();

    drawFaces();

    glPopMatrix(); // Don't forget to pop the Matrix
}

void cls()
{
    // Specify clear values for the color buffers
    glClearColor(backgroundColorGL[0], backgroundColorGL[1], backgroundColorGL[2],  1);

    // Specifies the depth value used when the depth buffer is cleared. The initial value is 1.
    glClearDepth(1);

    // Clear buffers to preset values
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void redraw(void *userData)
{
    DBG_T("Entered");

    //DrawData *data = static_cast<DrawData *>(userData);

    verifySize(MAIN_WINDOW_NAME);

    cls();
    drawModel(xf);

    DBG_T("Done");
}

bool exportCurrentViewAndOrthoData(const std::string &modelFile)
{
    cv::Mat img = getCvMatFromScreen(MAIN_WINDOW_NAME);
    if (img.empty())
    {
        std::cout << "Export failed. img is empty" << std::endl;
        return false;
    }

    std::string prefix = getFileNameNoExt(modelFile);
    std::string mapFileName = prefix + "_cur_map.png";
    if (!cv::imwrite(mapFileName, img))
    {
        std::cout << "Failed saving map " << mapFileName << std::cout;
        return false;
    }

    std::string orthoDataFileName = prefix + "_cur_map_ortho_data.txt";
    if (!gOrthoViews[gViewIdx].write(orthoDataFileName))
    {
        std::cout << "Failed saving ortho data " << orthoDataFileName << std::cout;
        return false;
    }

    DBG("Done exporting map image and orthographic projection data");
    return true;
}

// OpenCV Mouse - cv::setMouseCallback()
//   EVENT_MOUSEMOVE 	indicates that the mouse pointer has moved over the window.
//   EVENT_LBUTTONDOWN 	indicates that the left mouse button is pressed.
//   EVENT_RBUTTONDOWN 	indicates that the right mouse button is pressed.
//   EVENT_MBUTTONDOWN 	indicates that the middle mouse button is pressed.
//   EVENT_LBUTTONUP 	indicates that left mouse button is released.
//   EVENT_RBUTTONUP 	indicates that right mouse button is released.
//   EVENT_MBUTTONUP 	indicates that middle mouse button is released.
//   EVENT_LBUTTONDBLCLK 	indicates that left mouse button is double clicked.
//   EVENT_RBUTTONDBLCLK 	indicates that right mouse button is double clicked.
//   EVENT_MBUTTONDBLCLK 	indicates that middle mouse button is double clicked.
//   EVENT_MOUSEWHEEL 	positive and negative values mean forward and backward scrolling, respectively.
//   EVENT_MOUSEHWHEEL 	positive and negative values mean right and left scrolling, respectively.
void mouseCallbackFunc(int event, int x, int y, int flags, void *userdata)
{
    std::stringstream msg;

    switch (event)
    {
    case cv::EVENT_MOUSEMOVE:
        msg << "Mouse move";

        if (flags == cv::EVENT_FLAG_LBUTTON)
            msg << " with left button";
        else if (flags == cv::EVENT_FLAG_MBUTTON)
            msg << " with middle button";
        else if (flags == cv::EVENT_FLAG_RBUTTON)
            msg << " with right button";
        break;

    case cv::EVENT_LBUTTONDOWN:
        msg << "Mouse left press";
        break;

    case cv::EVENT_RBUTTONDOWN:
        msg << "Mouse right press";
        break;

    case cv::EVENT_MBUTTONDOWN:
        msg << "Mouse middle press";
        break;

    case cv::EVENT_LBUTTONUP:
        msg << "Mouse left release";
        break;

    case cv::EVENT_RBUTTONUP:
        msg << "Mouse right release";
        break;

    case cv::EVENT_MBUTTONUP:
        msg << "Mouse middle release";
        break;

    case cv::EVENT_MOUSEWHEEL:
        msg << "Wheel";

        if (flags > 0)
            msg << " up";
        else
            msg << " down";
        break;

    default:
        msg << "Other " << event << ", flags " << flags;
    }

    /*

    DBG(msg.str() << ", " << cv::Point(x, y) << ", flags: " << flags);

    cv::Mat img = getCvMatFromScreen(MAIN_WINDOW_NAME);
    //cv::Mat gray;
    //cvtColor(img, gray, CV_BGR2GRAY); // Perform gray scale conversion
    cv::Mat bw = img;//gray.setTo(255, gray > 0);

    std::string tempWinName = "Temp";
    cv::namedWindow(tempWinName, cv::WINDOW_AUTOSIZE);
    cv::moveWindow(tempWinName, 50, 50);

    cv::circle(bw, cv::Point(x, y), 5, white, -1);
    //cv::putText(bw, "", cv::Point(x, y - 10), cv::FONT_HERSHEY_SIMPLEX, 1.0, white);
    cv::imshow(tempWinName, bw);
    //cv::waitKey(0);

    DBG("Image size: " << bw.size());

    OrthoProjection &orthoProj = gOrthoViews[gViewIdx];
    float realX = orthoProj.left + x/float(bw.cols) * (orthoProj.right - orthoProj.left);
    float realY = orthoProj.bottom + y/float(bw.rows) * (orthoProj.top - orthoProj.bottom);

    DBG("Real world coordinates: (X, Y): (" << realX << ", " << realY << ")");
    */
}

static inline GLint xfUnProject(GLdouble winX, GLdouble winY, GLdouble winZ,
	const trimesh::xform &curXf, const GLint *view,
	GLdouble *objX, GLdouble *objY, GLdouble *objZ)
{
    trimesh::xform xfMat = inv(curXf);
    trimesh::Vec<3,double> v = xfMat * trimesh::Vec<3,double>(
        (winX - view[0]) / view[2] * 2.0 - 1.0,
        (winY - view[1]) / view[3] * 2.0 - 1.0,
        winZ * 2.0 - 1.0);
    *objX = v[0];
    *objY = v[1];
    *objZ = v[2];
    return GL_TRUE;
}

cv::Mat getWorldDepth()
{
    cv::Mat depthImg = getDepthCvMatFromScreen(MAIN_WINDOW_NAME);
    cv::Mat worldDepth(depthImg.size(), CV_8UC1, cv::Scalar());

    GLint V[4];
    glGetIntegerv(GL_VIEWPORT, V);
    GLdouble X, Y, Z;

    for (int y = 0; y < depthImg.rows; y++)
    {
        for (int x = 0; x < depthImg.cols; x++)
        {
            char pWorldDepth;

            if (depthImg.at<float>(y, x) < 1)
            {
                xfUnProject(x, y, depthImg.at<float>(y, x), gLastUsedXf, V, &X, &Y, &Z);

                // Sanity checks
                if (Z < 0)
                    DBG("Negative depth [" << Z << "]");
                if (Z > 254.0)
                    DBG("Point value too High (Over 254) [" << Z << "]");

                pWorldDepth = std::round(Z);
            }
            else
            {
                // Ground - Unknown value
                pWorldDepth = 255;
            }

            worldDepth.at<char>(y, x) = pWorldDepth;
        }
    }

    for (int y = 0; y < worldDepth.rows / 2; y++)
    {
        uchar *row1 = worldDepth.data + worldDepth.cols * y;
        uchar *row2 = worldDepth.data + worldDepth.cols * (worldDepth.rows - 1 - y);
        for (int x = 0; x < worldDepth.cols; x++)
            std::swap(row1[x], row2[x]);
    }

    return worldDepth;
}

cv::Point gMouseDepthPointPress;
std::string gMouseDepthStr1;
std::string gMouseDepthStr2;
void mouseCallbackDepthFunc(int event, int x, int y, int flags, void *userdata)
{
    if (event != cv::EVENT_LBUTTONDOWN)
        return;

    //printProjectionMatrix();
    //printModelViewMatrix();
    //DBG("xf\n" << xf);
    //DBG("gLastUsedXf\n" << gLastUsedXf);

    cv::Mat curDepthImg = getDepthCvMatFromScreen(MAIN_WINDOW_NAME);

    cv::Point p(x, y);

    GLdouble M[16], P[16];
    GLint V[4];
    glGetDoublev(GL_MODELVIEW_MATRIX, M);
    glGetDoublev(GL_PROJECTION_MATRIX, P);
    glGetIntegerv(GL_VIEWPORT, V);
    GLdouble X, Y, Z;

    xfUnProject(x, y, curDepthImg.at<float>(y, x), gLastUsedXf, V, &X, &Y, &Z);

    std::stringstream ss;
    ss << "p " << p << ", depth [" << curDepthImg.at<float>(y, x) << "]";
    std::stringstream ss1;
    ss1 << "Unproject coordinates (X, Y, Z) [" << X << ", " << Y << ", " << Z << "]";

    cv::Mat depthImg4show;
    cvtColor(curDepthImg, depthImg4show, CV_GRAY2BGR);
    //DBG("depthImg4show type " << getOpenCVmatType(depthImg4show));

    // Write to globals to print on next updateDepth()
    gMouseDepthPointPress = p;
    gMouseDepthStr1 = ss.str();
    gMouseDepthStr2 = ss1.str();
}

cv::Mat convertFloatMatToUchar(const cv::Mat &mat)
{
    double min, max;
    cv::minMaxIdx(mat, &min, &max);
    cv::Mat adjMap;
    cv::convertScaleAbs(mat, adjMap, 255 / max);
    DBG("adjMap type " << getOpenCVmatType(adjMap));
    return adjMap;
}

void updateDepth()
{
    cv::Mat curDepthImg = getDepthCvMatFromScreen(MAIN_WINDOW_NAME);

    if (gMouseDepthPointPress != cv::Point())
    {
        cv::circle(curDepthImg, gMouseDepthPointPress, 5, green, -1);
        cv::putText(curDepthImg, gMouseDepthStr1, cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, red, 1);
        cv::putText(curDepthImg, gMouseDepthStr2, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, red, 1);
    }

    cv::imshow(DEPTH_WINDOW_NAME, curDepthImg);
}

void handleKeyboard(int key)
{
    if ((char)key == -1)
        return;

#if __linux__
    key &= 0xff;
#endif

    //DBG("key [" << key << "], char [" << (char)key << "]");

    switch (key)
    {
    case 'a':
        gAutoNav = !gAutoNav;
        DBG("gAutoNav update to: " << (gAutoNav ? "True" : "False"));

        if (gAutoNav)
        {
            gViewIdx = 0;
            gUpperView = true;
        }
        break;

    case 's':
        exportCurrentViewAndOrthoData(gModelFilePath);
        break;

    case 'r':
        gReset = true;
        DBG("Refresh");
        break;

    case 'n':
        if (gViewIdx < gOrthoViews.size() - 1)
        {
            gViewIdx++;
            DBG("gViewIdx update to: " << gViewIdx);
        }
        break;

    case 'p':
        if (gViewIdx > 0)
        {
            gViewIdx--;
            DBG("gViewIdx update to: " << gViewIdx);
        }
        break;

    case 'h':
        gUpperView = true;
        break;
    case 'l':
        gUpperView = false;
        break;
    }
}

void initWindow(const std::string &winName, int winWidth, int winHeight, void (*redrawFunc)(void *))
{
    cv::namedWindow(winName, cv::WINDOW_OPENGL | cv::WINDOW_NORMAL);
    cv::resizeWindow(winName, winWidth, winHeight);

    cv::setOpenGlDrawCallback(winName, redrawFunc, 0); // Set OpenGL render handler for the specified window
}

// Set the view to look at the middle of the mesh, from reasonably far away
void upperView()
{
    DBG_T("Entered");

    DBG("Reset view to look at the middle of the mesh, from reasonably far away");
    xf = trimesh::xform::trans(0, 0, -3.5f / fov * themesh->bsphere.r) *
        trimesh::xform::trans(-themesh->bsphere.center);

    //DBG("xf:\n" << xf);
}

void loadModel(const std::string &fileName)
{
    themesh = std::unique_ptr<trimesh::TriMesh>(trimesh::TriMesh::read(fileName));
    if (!themesh)
    {
        std::cerr << "Failed reading model file: " << fileName << std::endl;
        exit(1);
    }

    themesh->need_tstrips();
    themesh->need_bsphere();
    themesh->need_bbox();
    DBG("bbox center: " << themesh->bbox.center() << ", bbox radius: " << themesh->bbox.radius());
    themesh->need_normals();
    themesh->need_curvatures();
    themesh->need_dcurv();
    themesh->need_faces();
    themesh->need_across_edge();

    DBG("Mesh file [" << fileName << "] loaded");
    DBG("Vertices num [" << themesh->vertices.size() << "], faces num [" << themesh->faces.size() <<
        "], tstrips num [" << themesh->tstrips.size() << "], normals num [" << themesh->normals.size() << "]");

    upperView();
}

cv::Mat gInpaintImg;
void mouseCallbackInpaintFunc(int event, int x, int y, int flags, void *userdata)
{
    if (event != cv::EVENT_LBUTTONDOWN)
        return;

    cv::Point p(x, y);
    DBG("p " << p << ", value [" << int(gInpaintImg.at<char>(y, x)) << "]");
}

void autoNavigation()
{
    unsigned gridRow = gViewIdx % FLAGS_grid;
    unsigned gridCol = gViewIdx / FLAGS_grid;
    cv::Point offset(gridCol*gWinWidth, gFullMap.rows - (gridRow + 1) * gWinHeight);
    cv::Rect gridRect(offset, cv::Size(gWinWidth, gWinHeight));

    if (gUpperView)
    {
        cv::Mat img = getCvMatFromScreen(MAIN_WINDOW_NAME);
        if (img.empty())
            throw std::runtime_error("Image from screen is empty");

        if (img.rows != gWinHeight || img.cols != gWinWidth) // Sanity
            throw std::runtime_error("Image has different size than what is requested");

        //DBG("gFullMap size " << gFullMap.size() << ", img size " << img.size() << ", gridRect " << gridRect);
        //cv::namedWindow(std::to_string(gViewIdx));
        //cv::imshow(std::to_string(gViewIdx), img);

        img.copyTo(gFullMap(gridRect));
        gUpperView = false;
    }
    else
    {
        cv::Mat worldDepthImg = getWorldDepth();

        // Inpainting mask, 8-bit 1-channel image. Non-zero pixels indicate the area that needs to be inpainted.
        cv::Mat inpaintMask;
        cv::cvtColor(gFullMap(gridRect), inpaintMask, CV_BGR2GRAY); // Perform gray scale conversion
        inpaintMask = 255 - inpaintMask;         //cv::imshow("inpaintMask", inpaintMask);

        // Adding border to solve some artifact bug on image borders
        cv::copyMakeBorder(inpaintMask, inpaintMask, 5, 5, 5, 5, cv::BORDER_CONSTANT, 255);
        cv::copyMakeBorder(worldDepthImg, worldDepthImg, 5, 5, 5, 5, cv::BORDER_CONSTANT, 255);

        cv::Mat inpaintImg;
        cv::inpaint(worldDepthImg, inpaintMask, inpaintImg, 5, cv::INPAINT_NS); //INPAINT_NS or INPAINT_TELEA
        // Removing the border
        inpaintImg = inpaintImg(cv::Rect(cv::Point(5, 5), cv::Size(gWinWidth, gWinHeight)));

        /* if (0)
        {
            // Inpaint debug section
            cv::imshow("worldDepthImg", worldDepthImg);
            cv::imshow("inpaintMask", inpaintMask);
            gInpaintImg = inpaintImg;
            cv::namedWindow("inpaintImg", cv::WINDOW_AUTOSIZE);
            cv::setMouseCallback("inpaintImg", mouseCallbackInpaintFunc, NULL);
            cv::imshow("inpaintImg", inpaintImg);
            cv::waitKey(0);
        } */

        cv::cvtColor(inpaintImg, inpaintImg, CV_GRAY2BGR);
        inpaintImg.copyTo(gFullGroundLevelEstimationMap(gridRect));

        gUpperView = true;
    }

    if (gUpperView == false)
    {
        // Wait another iteration (Getting underground look)
    }
    else if (gViewIdx < gOrthoViews.size() - 1)
    {
        gViewIdx++;
    }
    else
    {
        DBG("Done navigation");
        gAutoNav = false;

        std::string prefix = getFileNameNoExt(gModelFilePath);

        std::string mapFileName = prefix + "_map.png";
        if (!cv::imwrite(mapFileName, gFullMap))
        {
            std::cout << "Failed saving map " << mapFileName << std::cout;
        }

        std::string groundLevelEstimationFileName = prefix + "_ground_level_map.png";
        if (!cv::imwrite(groundLevelEstimationFileName, gFullGroundLevelEstimationMap))
        {
            std::cout << "Failed saving map " << groundLevelEstimationFileName << std::cout;
        }

        std::string orthoDataFileName = prefix + "_map_ortho_data.txt";
        if (!getFullViewParams().write(orthoDataFileName))
        {
            std::cout << "Failed saving ortho data " << orthoDataFileName << std::cout;
        }

        DBG("Map, ground-level images and ortho data saved to [" << mapFileName << "], [" <<
            groundLevelEstimationFileName << "] and [" << orthoDataFileName << "]");

        cv::Mat fullResized;
        cv::resize(gFullMap, fullResized, cv::Size(800, 600));
        cv::imshow("fullMap-resized", fullResized);

        //cv::imshow("fullMap", gFullMap);

        cv::Mat fullGroundResized;
        cv::resize(gFullGroundLevelEstimationMap, fullGroundResized, cv::Size(800, 600));
        cv::imshow("full-ground-level-resized", fullGroundResized);
    }
}

int main(int argc, char* argv[])
{
    std::string usage = std::string(argv[0]) + " [model_file] [flags]";

    GFLAGS_NAMESPACE::SetUsageMessage(usage);
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    if (argc < 2)
    {
        std::cout << "Usage: " << usage << std::endl;

        gModelFilePath = "../cube_layout.obj";
        std::cout << "Running with default model: " << gModelFilePath << std::endl;
    }
    else
    {
        gModelFilePath = argv[1];

        for (int i = 1; i < argc; i++)
        {
            if (argv[i][0] == '+')
                sscanf(argv[i] + 1, "%d,%d", &gWinWidth, &gWinHeight);
        }
    }

    loadModel(gModelFilePath);

    initWindow(MAIN_WINDOW_NAME, gWinWidth, gWinHeight, redraw);
    cv::setMouseCallback(MAIN_WINDOW_NAME, mouseCallbackFunc, NULL);

    cv::namedWindow(DEPTH_WINDOW_NAME, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(DEPTH_WINDOW_NAME, mouseCallbackDepthFunc, NULL);
    cv::moveWindow(DEPTH_WINDOW_NAME, 50, 50);

    for (;;)
    {
        cv::updateWindow(MAIN_WINDOW_NAME);
        updateDepth();

        int key = cv::waitKey(100);
        if (key == 27) // ESC key
            break;

        handleKeyboard(key);

        if (gAutoNav)
            autoNavigation();
    }

    cv::setOpenGlDrawCallback(MAIN_WINDOW_NAME, 0, 0);
    cv::destroyAllWindows();

    return 0;
}

