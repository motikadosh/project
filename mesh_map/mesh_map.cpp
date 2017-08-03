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

#include "TriMesh.h"
#include "XForm.h"

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
const unsigned screenDivider = 2;
std::array<std::array<cv::Mat, screenDivider>, screenDivider> mapImages;
unsigned gCurrentMap = 0; // Must init to full map to get full model dimensions

std::unique_ptr<trimesh::TriMesh> themesh;
trimesh::xform xf; // Current location and orientation

std::vector<std::pair<int, int> > edges;

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

    std::string print() const
    {
        std::stringstream ss;
        ss << std::setprecision(17) << "(left,right,bottom,top,neardist,fardist)=(" << left << ", " << right << ", " <<
            bottom << ", " << top << ", " << neardist << ", " << fardist << ")";
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
OrthoProjection gOrthoProjData(0, 0, 0, 0,0, 0);

//
// Utilities
//
static inline std::ostream& operator<<(std::ostream &out, const OrthoProjection &data)
{
    out << data.print();

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

cv::Mat getCvMatFromScreen()
{
    cv::setOpenGlContext(MAIN_WINDOW_NAME); // Sets the specified window as current OpenGL context

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

void drawEdges()
{
    glLineWidth(1);

    glBegin(GL_LINES);
    // Mapping of vector index to color. I.e. first edge will be black(0, 0, 0)
    for (unsigned i = 0; i < edges.size(); i++)
        drawEdge(edges[i].first, edges[i].second, i);
    glEnd();
}

static inline GLint myGluUnProject(GLdouble winX, GLdouble winY, GLdouble winZ,
	const GLdouble *model, const GLdouble *proj, const GLint *view,
	GLdouble* objX, GLdouble* objY, GLdouble* objZ)
{
    trimesh::xform xf = inv(trimesh::xform(proj) * trimesh::xform(model));
    trimesh::Vec<3,double> v = xf * trimesh::Vec<3,double>(
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
    trimesh::point scene_center = xf * themesh->bsphere.center;
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

    return OrthoProjection(left, right, bottom, top, neardist, fardist);
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

void setViewParams(const OrthoProjection &orthoProjData)
{
    DBG("orthoProjData: " << orthoProjData);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(orthoProjData.left, orthoProjData.right, orthoProjData.bottom, orthoProjData.top, orthoProjData.neardist,
        orthoProjData.fardist);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void disectViews(const OrthoProjection &orthoFullView, unsigned partsPerAxis, OrthoProjection orthoViews[])
{
    float widthPart = (orthoFullView.right - orthoFullView.left) / partsPerAxis;
    float heightPart = (orthoFullView.top - orthoFullView.bottom) / partsPerAxis;

    // XXX: If the stitches are not good enough it is possible I should use the end of one part as the start of the
    // other

    for (unsigned x = 0; x < partsPerAxis; x++)
    {
        for (unsigned y = 0; y < partsPerAxis; y++)
        {
            unsigned i = x + y*partsPerAxis;

            orthoViews[i].set(
                orthoFullView.left + x*widthPart,
                orthoFullView.left + x*(widthPart + 1),
                orthoFullView.bottom + y*heightPart,
                orthoFullView.bottom + y*(heightPart + 1),
                orthoFullView.neardist,
                orthoFullView.fardist);

            DBG("part " << i << ", x,y " << x << ", " << y << ", is:" << orthoViews[i]);
        }
    }
}

void drawModel(trimesh::xform &xf)
{
    //DBG("---BEFORE---");
    //DBG("xf:\n" << xf);

    // TODO: Consider using std array
    static OrthoProjection orthoViews[screenDivider*screenDivider];
    if (gCurrentMap == 0)
    {
        gOrthoProjData = getFullViewParams();
        DBG("gOrthoProjData: " << gOrthoProjData);

        disectViews(gOrthoProjData, screenDivider, orthoViews);

        gCurrentMap = 1;
    }

    setViewParams(orthoViews[gCurrentMap]);

    //printCurrentMatrixMode();
    //printModelViewMatrix();
    //printProjectionMatrix();
    //printViewPort();

    // Transform and draw
    glPushMatrix();
    glMultMatrixd((double *)xf);

    //DBG("---AFTER---");
    //printModelViewMatrix();
    //printProjectionMatrix();

    drawFaces();
    //drawEdges();

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

    cls();
    drawModel(xf);

    DBG_T("Done");
}

bool exportMapAndOrthoData(const std::string &modelFile)
{
    cv::Mat img = getCvMatFromScreen();
    //cv::Mat gray;
    //cvtColor(img, gray, CV_BGR2GRAY); // Perform gray scale conversion
    //cv::Mat bw = gray.setTo(255, gray > 0);

    if (img.empty())
    {
        std::cout << "Export failed. img is empty" << std::endl;
        return false;
    }

    std::string prefix = getFileNameNoExt(modelFile);
    std::string mapFileName =  prefix + "_map.png";
    if (!cv::imwrite(mapFileName, img))
    {
        std::cout << "Failed saving map " << mapFileName << std::cout;
        return false;
    }

    std::string orthoDataFileName = prefix + "_map_ortho_data.txt";
    if (!gOrthoProjData.write(orthoDataFileName))
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
        exportMapAndOrthoData(gModelFilePath);
        msg << "Mouse right press";
        break;

    case cv::EVENT_MBUTTONDOWN:
        gCurrentMap++;
        DBG("gCurrentMap update to: " << gCurrentMap);

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

    DBG(msg.str() << ", " << cv::Point(x, y) << ", flags: " << flags);

    cv::Mat img = getCvMatFromScreen();
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

    float realX = gOrthoProjData.left + x/float(bw.cols) * (gOrthoProjData.right - gOrthoProjData.left);
    float realY = gOrthoProjData.bottom + y/float(bw.rows) * (gOrthoProjData.top - gOrthoProjData.bottom);

    DBG("Real world coordinates: (X, Y): (" << realX << ", " << realY << ")");
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
}

void populateModelEdges(float maxDihedralAngle = 135)
{
    std::set<std::pair<int, int> > edgesSet;
    for (unsigned i = 0; i < themesh->faces.size(); i++)
    {
        for (unsigned j = 0; j < 3; j++)
        {

            // Calculate angle between 2 faces normals, if there is no adjacent face (Boundary), returned angle is 0
            float angle = RAD_TO_DEG(themesh->dihedral(i, j));
            //DBG("Angle between face " << i << " to across edge number " << j << " is " << angle);

            // Skip faces that almost overlap (Probably curvatures)
            if (angle > maxDihedralAngle)
                continue;

            int v1 = themesh->faces[i][(j + 1) % 3];
            int v2 = themesh->faces[i][(j + 2) % 3];

            // To make sure only unique edges enter the set I make sure first element is always bigger than second
            std::pair<int, int> edge;
            if (v1 > v2)
                edge = std::make_pair(v1, v2);
            else
                edge = std::make_pair(v2, v1);

            edgesSet.insert(edge);
        }
    }

    // Now the set of edges is ready and without any duplicates, I convert it to a vector to have easier access and
    // support already written code.

    edges.clear();
    for (auto edge : edgesSet)
        edges.push_back(edge);

    DBG("Edges vector size: " << edges.size());
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

    populateModelEdges(135);

    DBG("Mesh file [" << fileName << "] loaded");
    DBG("Vertices num [" << themesh->vertices.size() << "], faces num [" << themesh->faces.size() <<
        "], tstrips num [" << themesh->tstrips.size() << "], normals num [" << themesh->normals.size() << "]");

    upperView();
}

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " [model_file] [WinSize +#,+#]" << std::endl;

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

    for (;;)
    {
        cv::updateWindow(MAIN_WINDOW_NAME);

        int key = cv::waitKey(0);
        if (key == 27) // ESC key
            break;

        //handleKeyboard(key);
    }

    cv::setOpenGlDrawCallback(MAIN_WINDOW_NAME, 0, 0);
    cv::destroyAllWindows();

    return 0;
}

