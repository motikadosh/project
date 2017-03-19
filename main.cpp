// http://docs.opencv.org/trunk/d2/d3c/group__core__opengl.html

#include <iostream>
#include <memory>
#include <limits>
#include <set>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>

#define FS_DELIMITER_LINUX "/"
#define FS_DELIMITER_WINDOWS "\\"

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
#include "GLCamera.h"

//
// Macros
//

const float PI = 3.14159265358979323846264f;
#define RAD_TO_DEG(x) ((x) * 180 / PI)
#define DEG_TO_RAD(x) ((x) * PI / 180)

#define MAIN_WINDOW_NAME "OpenGL"
#define VERTEX_WINDOW_NAME "vertex_win"
#define CV_WINDOW_NAME "cv_win"

#define DBG(params) \
    std::cout << dbgCount++ << ") " << __FUNCTION__ << ": " << params << std::endl

// #define DEBUG
#ifdef DEBUG
    #define DBG_T(params) DBG(params)
#else
    #define DBG_T(params)
#endif

// Keyboard keys
// Keys list: https://pythonhosted.org/pyglet/api/pyglet.window.key-module.html
#define ESC_KEY         27
#define SPACE_KEY       32
#define KEY_TAB         9

#define KEY_LEFT        'Q'
#define KEY_RIGHT       'S'
#define KEY_UP          'R'
#define KEY_DOWN        'T'

#define KEY_ENTER       13

/*
#if defined(_MSC_VER) || defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(WIN64) || \
    defined(_WIN64) || defined(__WIN64__)

    #define KEY_ENTER       13
    #define KEY_DELETE      3014656
    #define KEY_BACKSPACE   8
    #define KEY_HOME        2359296
    #define KEY_LEFT        2424832
    #define KEY_UP          2490368
    #define KEY_RIGHT       2555904
    #define KEY_DOWN        2621440
    #define KEY_PAGEUP      2162688
    #define KEY_PAGEDOWN    2228224
    #define KEY_END         2293760

    #define FS_DELIMITER FS_DELIMITER_WINDOWS
    #define TO_DEV_NULL " 2>nul"
#else // Linux
    #define KEY_ENTER       10
    #define KEY_DELETE      65535
    #define KEY_BACKSPACE   65288
    #define KEY_HOME        65360
    #define KEY_LEFT        81
    #define KEY_UP          82
    #define KEY_RIGHT       83
    #define KEY_DOWN        84
    //#define KEY_LEFT        65361
    //#define KEY_UP          65362
    //#define KEY_RIGHT       65363
    //#define KEY_DOWN        65364
    #define KEY_PAGEUP      65365
    #define KEY_PAGEDOWN    65366
    #define KEY_END         65367

    #define FS_DELIMITER FS_DELIMITER_LINUX
    #define TO_DEV_NULL " &> /dev/null"
#endif*/

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

//
// Globals
//
// TODO: Use GL colors instead of CV
cv::Scalar backgroundColor = white;
cv::Scalar facesColor = backgroundColor;
const GLfloat *groundColorGL = whiteGL; // whiteGL;

trimesh::GLCamera camera;
trimesh::xform xf; // Current location and orientation
std::string xfFileName;
bool menuOn = false;
bool taggingOn = false;
int dbgCount = 0;

int winWidth = 800;
int winHeight = 600;
float fov = 0.7f;
float rotStep = M_PI / 36.0; // 10 degrees
int xyStep = 10;
int zStep = 100;

std::unique_ptr<trimesh::TriMesh> themesh;
std::vector<std::pair<int, int> > edges;
std::vector<int> vertexesIdxInView;
cv::Mat cvMainImg;
cv::Mat cvVertexImg;

cv::Mat cvMap;
trimesh::GLCamera cameraMap;
trimesh::xform xfMap; // Map location and orientation

cv::Mat cvPhoto;
bool showMap = false;
cv::Mat cvShowPhoto;

std::string markedPtsFile;
std::vector<trimesh::point> marked3Dpoints;
std::vector<cv::Point> marked2Dpoints;

std::vector<cv::Point2f> projectedMarks;

cv::Mat projMat;

int marksRadius = 4;


//
// Pre-declarations
//
void updateWindows();

//
// Utilities
//

template <typename T>
inline std::ostream& operator<<(std::ostream &out, const std::vector<T> &vector)
{
    for(auto i : vector)
        out << i << ", ";

    if (vector.size())
        out << "\b\b";

    return out;
}

template <typename T>
inline std::ostream& operator<<(std::ostream &out, const std::set<T> &set)
{
    for(auto i : set)
        out << i << ", ";

    if (set.size())
        out << "\b\b";

    return out;
}

void imshowAndWait(const cv::Mat &img, unsigned waitTime = 0, const std::string &winName = "temp")
{
    if (img.empty())
        return;

    //cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);
    cv::imshow(winName, img);
    cv::waitKey(waitTime);
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

    DBG("GL proj mat:\n" << glProjMat);
    // Sanity check
    //for (auto i : projection)
      //  DBG(i);
}

void printModelViewMatrix()
{
    GLdouble modelview[16];
    glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
    cv::Mat glModelViewMat(cv::Size(4, 4), CV_64FC1, modelview);

    DBG("GL model view mat:\n" << glModelViewMat);

    // Sanity check
    for (auto i : modelview)
        DBG(i);
}

#define FONT_SCALE 2
template <typename T>
inline void showPoints(cv::Mat &frame, const std::vector<cv::Point_<T>> &marks, const cv::Scalar &color,
    bool showNumbers=true, float fontScale=FONT_SCALE)
{
    for (unsigned i = 0; i < marks.size(); i++)
    {
        if (showNumbers)
        {
            cv::putText(frame, std::to_string(i), marks[i], cv::FONT_HERSHEY_COMPLEX_SMALL, fontScale, color, 1,
                CV_AA);
        }
        else
        {
            //cv::Point target(marks[i].x + 2, marks[i].y);
            //cv::line(frame, marks[i], target, color, 3);
            cv::circle(frame, marks[i], 10, color, CV_FILLED);
        }
    }
}

// Fixes a bug that sometime the initial resize does not stick and we get back to the default win size
void verifySize(const std::string &winName)
{
    GLint V[4];
    glGetIntegerv(GL_VIEWPORT, V);
    GLint width = V[2], height = V[3];

    if (width == 100 && height == 30) // Default window size
        cv::resizeWindow(winName, winWidth, winHeight);
}

unsigned int getColorInPoint(const cv::Mat &img, int x, int y)
{
    if (x > img.cols || y > img.rows)
        return 0xFFFFFF;

    cv::Vec3b bgrVec = img.at<cv::Vec3b>(y, x);
    unsigned int rgbColor = bgrVec[2] << 16 | bgrVec[1] << 8 | bgrVec[0];
    return rgbColor;
}

cv::Mat getGlWindowAsCvMat(const std::string &winName)
{
    DBG_T("Entered");

    // Switch OpenGL context to chosen window - So we take the data from the right window
    cv::setOpenGlContext(winName);

    // Read pixels
    GLint V[4];
    glGetIntegerv(GL_VIEWPORT, V);
    GLint width = V[2], height = V[3];

    std::unique_ptr<char> buf = std::unique_ptr<char>(new char[width * height * 3]);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(V[0], V[1], width, height, GL_BGR, GL_UNSIGNED_BYTE, buf.get());

    // Flip top-to-bottom
    for (int i = 0; i < height / 2; i++)
    {
        char *row1 = buf.get() + 3 * width * i;
        char *row2 = buf.get() + 3 * width * (height - 1 - i);
        for (int j = 0; j < 3 * width; j++)
            std::swap(row1[j], row2[j]);
    }

    cv::Mat img(cv::Size(width, height), CV_8UC3, buf.get(), cv::Mat::AUTO_STEP);

    DBG_T("Done");
    return img.clone();
}

//
// Pose estimation
//

cv::Matx31d tvec, rvec;
cv::Matx33d intrinsicMatrix;

std::vector<cv::Point3f> conv3dPointTrimeshToOpenCV(std::vector<trimesh::point> pts)
{
    std::vector<cv::Point3f> cvPts;
    for (auto p : pts)
        cvPts.push_back(cv::Point3f(p[0], p[1], p[2]));

    return cvPts;
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

void convertMatToVector(const cv::Mat &mat, std::vector<float> &array)
{
    if (mat.isContinuous())
    {
        array.assign((float *)mat.datastart, (float *)mat.dataend);
    }
    else
    {
        for (int i = 0; i < mat.rows; ++i)
            array.insert(array.end(), (float *)mat.ptr<uchar>(i), (float *)mat.ptr<uchar>(i) + mat.cols);
    }
}

// http://stackoverflow.com/questions/130829/3d-to-2d-projection-matrix
// http://www.cs.cmu.edu/~16385/lectures/Lecture17.pdf
// https://ags.cs.uni-kl.de/fileadmin/inf_ags/3dcv-ws13-14/3DCV_lec05_parameterEstimation.pdf
// http://docs.opencv.org/3.1.0/dc/d2c/tutorial_real_time_pose.html
void findProjectionMatrix(std::vector<cv::Point3f> &p3d, std::vector<cv::Point2f> &p2d)
{
    DBG("Entered");

    if (p3d.size() != p2d.size())
    {
        DBG("Error- marked points vector have different sizes");
        return;
    }

    std::vector<cv::Point3f> homoP2d;
    cv::convertPointsToHomogeneous(p2d, homoP2d);

    //DBG("homoP2d:\n" << homoP2d);
    cv::Mat mat2d = cv::Mat::zeros(cv::Size(3, p3d.size()), CV_32FC1);
    for (unsigned i = 0; i < p3d.size(); i++)
    {
        mat2d.at<float>(i, 0) = homoP2d[i].x;
        mat2d.at<float>(i, 1) = homoP2d[i].y;
        mat2d.at<float>(i, 2) = homoP2d[i].z;
    }
    DBG("mat2d:\n" << mat2d);

    //DBG("p3d:\n" << p3d);
    cv::Mat mat3d = cv::Mat::zeros(cv::Size(4, p3d.size()), CV_32FC1) + 1;
    for (unsigned i = 0; i < p3d.size(); i++)
    {
        mat3d.at<float>(i, 0) = p3d[i].x;
        mat3d.at<float>(i, 1) = p3d[i].y;
        mat3d.at<float>(i, 2) = p3d[i].z;
    }
    DBG("mat3d:\n" << mat3d);

    // Creating DLT matrix (See- http://www.cs.cmu.edu/~16385/lectures/Lecture17.pdf)
    cv::Mat dlt;
    for (unsigned i = 0; i < p2d.size(); i++)
    {
        cv::Mat dltMat = cv::Mat::zeros(cv::Size(12, 2), CV_32FC1);
        mat3d(cv::Rect(0, i, 4, 1)).copyTo(dltMat(cv::Rect(0, 0, 4, 1)));
        mat3d(cv::Rect(0, i, 4, 1)).copyTo(dltMat(cv::Rect(8, 0, 4, 1)));
        dltMat(cv::Rect(8, 0, 4, 1)) *= -p2d[i].x;
        mat3d(cv::Rect(0, i, 4, 1)).copyTo(dltMat(cv::Rect(4, 1, 4, 1)));
        mat3d(cv::Rect(0, i, 4, 1)).copyTo(dltMat(cv::Rect(8, 1, 4, 1)));
        dltMat(cv::Rect(8, 1, 4, 1)) *= -p2d[i].y;

        //DBG("dltMat:\n" << dltMat);
        dlt.push_back(dltMat);
    }
    DBG("dlt:\n" << dlt);

    cv::SVD::solveZ(dlt, projMat);
    projMat = projMat.reshape(0, 3);
    DBG("Projection matrix:\n" << projMat);

    // Check reuslty by projecting the 3D points and see where the fall
    for (unsigned i = 0; i < p2d.size(); i++)
    {
        cv::Mat vec = mat3d(cv::Rect(0, i, 4, 1)).t();
        //DBG("vec:\n" << vec);

        cv::Mat resHomo = projMat * vec;
        //DBG("resHomo:\n" << resHomo << "\nSize: " << resHomo.size() << ", channels: " << resHomo.channels());

        std::vector<float> array;
        convertMatToVector(resHomo, array);

        cv::Mat res;
        cv::convertPointsFromHomogeneous(array, res);
        DBG(i << ") res:\n" << res << ", Res should be " << p2d[i]);
    }
}

void calcPoseEstimation()
{
    DBG_T("Entered");

    if (marked2Dpoints.empty() || marked3Dpoints.empty())
    {
        DBG("No marked points. Ignoring calcPoseEstimation request");
        return;
    }

    std::vector<cv::Point3f> points3d = conv3dPointTrimeshToOpenCV(marked3Dpoints);

    std::vector<cv::Point2f> points2d;
    for (auto p : marked2Dpoints)
        points2d.push_back(cv::Point2f(p.x, p.y));

    findProjectionMatrix(points3d, points2d);
    //updateWindows();

    cv::Size imageSize = cvPhoto.size();

    bool useRansac = true;
    bool horizontalMirror = false;
    int cameraFov = 70;

    int tX = imageSize.width / 2;
    int tY = imageSize.height / 2;
    // Focal length for radial lens should be same for x and y axis
    // TODO: Notice there are 2 options for the next line. Which is better?
    //float focalLength = false ? tY / tan(DEG_TO_RAD(cameraFov / 2)) : tY * tan(DEG_TO_RAD(cameraFov));

    // Calculate focal length f and the field of view (FOV) of a rectilinear lens-
    // http://www.bdimitrov.de/kmp/technology/fov.html
    float filmDiagonal = sqrt(pow(imageSize.width, 2) + pow(imageSize.height, 2));

    //float focalLength = filmDiagonal / (2 * tan(DEG_TO_RAD(cameraFov / 2)));
    // XXX: For some reason the result is a lot better when I do not convert to Radians
    float focalLength = filmDiagonal / (2 * tan(cameraFov / 2));

    intrinsicMatrix = cv::Matx33d(
        focalLength, 0,            tX,
        0,           focalLength,  tY,
        0,           0,            1);

    if (horizontalMirror)
        intrinsicMatrix = cv::Matx33d(-1, 0, 0, 0, 1, 0, 0, 0, 1) * intrinsicMatrix;

    int pnpFlag[5] = {
        cv::SOLVEPNP_ITERATIVE,
        cv::SOLVEPNP_P3P,
        cv::SOLVEPNP_EPNP,
        cv::SOLVEPNP_DLS,
        cv::SOLVEPNP_UPNP
    };
#define SOLVE_PNP_MODE_IDX 0

    if (pnpFlag[SOLVE_PNP_MODE_IDX] == cv::SOLVEPNP_P3P)
    {
        points3d = std::vector<cv::Point3f>(points3d.begin(), points3d.begin() + 4);
        points2d = std::vector<cv::Point2f>(points2d.begin(), points2d.begin() + 4);
    }

    if (useRansac)
    {
        //Trying using  solvePnPRansac, no better !
        cv::solvePnPRansac(points3d, points2d, intrinsicMatrix, cv::noArray(), rvec, tvec,
            false, 1000, 0.8/*err thresh to treat as outlier*/, 0.99, cv::noArray(), pnpFlag[SOLVE_PNP_MODE_IDX]);
    }
    else
    {
        cv::solvePnP(points3d, points2d, intrinsicMatrix, cv::noArray(), rvec, tvec, false,
            pnpFlag[SOLVE_PNP_MODE_IDX]);
    }

    //cv::Ptr<cv::Formatter> formatMat = cv::Formatter::get(cv::Formatter::FMT_DEFAULT);
    //formatMat->set64fPrecision(3);
    //DBG_V << "rvec " << formatMat->format(rvec.t()) << ", tvec " << formatMat->format(tvec.t());

    DBG("rvec\n" << rvec << "\ntvec\n" << tvec);

    // Get rotation matrix
    cv::Matx33d rotation;
    cv::Rodrigues(rvec, rotation);

    // https://courses.cs.washington.edu/courses/cse455/09wi/Lects/lect5.pdf
    // look for "From world coordinates to pixel coordinates"
    //cv::Matx44d mExt = {
    cv::Matx34d mExt = {
        rotation(0, 0), rotation(0, 1), rotation(0, 2), tvec(0),
        rotation(1, 0), rotation(1, 1), rotation(1, 2), tvec(1),
        rotation(2, 0), rotation(2, 1), rotation(2, 2), tvec(2)
    };
    //    0,              0,              0,              1
    //};
    DBG("mExt:\n" << mExt);

    cv::Matx34d cameraCalib = intrinsicMatrix * mExt;
    DBG("cameraCalib:\n" << cameraCalib);

    //cv::Matx44d invCheck = headRotTransHomMatFromCam.inv();
    //DBG("invCheck: " << invCheck);

    //cv::Matx31d vvvv = cameraCalib * cv::Matx41d(points3d[0].x, points3d[0].y, points3d[0].z, 1);
    //DBG("vertex 0:\n" << points3d[0] << "\nvvvv:\n" << vvvv);
    //DBG("vvvvvvvv: " << vvvv(0, 0) << ", " << vvvv(1, 0) << ", " << vvvv(2, 0));
    //DBG("vvvvvvvv: " << vvvv(0, 0) / vvvv(2, 0) << ", " << vvvv(1, 0)/ vvvv(2, 0));

    cv::projectPoints(points3d, rvec, tvec, intrinsicMatrix, cv::noArray(), projectedMarks);

    DBG("xf\n" << xf);

    trimesh::xform xfView(rotation(0, 0), rotation(0, 1), rotation(0, 2), tvec(0),
        rotation(1, 0), rotation(1, 1), rotation(1, 2), tvec(1),
        rotation(2, 0), rotation(2, 1), rotation(2, 2), tvec(2),
        0,              0,              0,              1);

    /*
    double axis[3] = {1, 0, 0};
    xfView = trimesh::xform::rot(PI, axis) * trimesh::transp(xfView);

    DBG("xfView\n" << xfView);

    xf = xfView;
    */

    updateWindows();

    DBG_T("Done");
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

void drawGround()
{
    float minZ = themesh->stat(trimesh::TriMesh::STAT_MIN, trimesh::TriMesh::STAT_Z);
    //DBG("minZ: " << minZ);
    trimesh::point c = themesh->bsphere.center;

    // Draw center point
    //glColor3fv(redGL);
    //glPointSize(10);
    //glBegin(GL_POINTS);
    //glVertex3fv(c);
    //glEnd();

    glColor3fv(groundColorGL);
    glBegin(GL_QUADS);
    c[2] = 0;
    float r = themesh->bsphere.r;

    glVertex3fv(c + trimesh::point(-r, -r, minZ));
    glVertex3fv(c + trimesh::point(r, -r, minZ));
    glVertex3fv(c + trimesh::point(r, r, minZ));
    glVertex3fv(c + trimesh::point(-r, r, minZ));
    glEnd();
}

void drawFaces()
{
    glDisable(GL_CULL_FACE); // Order of vertexes should not determine faces normals (I.e. Draw both sides of face)

    // Enable the vertex array
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, &themesh->vertices[0][0]);

    glColor3f(facesColor[2] / 255.0f, facesColor[1] / 255.0f, facesColor[0] / 255.0f);

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

    drawGround();
}

void drawBoundaries()
{
    edges.clear();

    glLineWidth(1);

    glBegin(GL_LINES);
    for (unsigned i = 0; i < themesh->faces.size(); i++)
    {
        for (unsigned j = 0; j < 3; j++)
        {
            if (themesh->across_edge[i][j] >= 0)
                continue;

            // Mapping of vector index to color. I.e. first edge will be black(0, 0, 0)
            unsigned rgbColor = edges.size();
            unsigned r, g, b;
            r = (rgbColor & 0xFF0000) >> 16;
            g = (rgbColor & 0xFF00) >> 8;
            b = rgbColor & 0xFF;
            glColor3f((float)r / 255.0f, (float)g / 255.0f, (float)b / 255.0f);

            int v1 = themesh->faces[i][(j + 1) % 3];
            int v2 = themesh->faces[i][(j + 2) % 3];

            edges.push_back(std::make_pair(v1, v2)); // Add the edge to edges vector

            // Draw the edge
            glVertex3fv(themesh->vertices[v1]);
            glVertex3fv(themesh->vertices[v2]);
        }
    }
    glEnd();

    //DBG("Edges vector size: " << edges.size());
}

void getRgbFromInt(int num, unsigned &r, unsigned &g, unsigned &b)
{
    r = (num & 0xFF0000) >> 16;
    g = (num & 0xFF00) >> 8;
    b = num & 0xFF;
}

void drawVertexes()
{
    glPointSize(2);

    glBegin(GL_POINTS);
    for (unsigned i = 0; i < themesh->faces.size(); i++)
    {
        for (unsigned j = 0; j < 3; j++)
        {
            if (themesh->across_edge[i][j] >= 0)
                continue;

            unsigned r, g, b;

            // Mapping of vertex index to color
            int v1 = themesh->faces[i][(j + 1) % 3];
            getRgbFromInt(v1, r, g, b);
            glColor3f((float)r / 255.0f, (float)g / 255.0f, (float)b / 255.0f);
            glVertex3fv(themesh->vertices[v1]);

            int v2 = themesh->faces[i][(j + 2) % 3];
            getRgbFromInt(v2, r, g, b);
            glColor3f((float)r / 255.0f, (float)g / 255.0f, (float)b / 255.0f);
            glVertex3fv(themesh->vertices[v2]);
        }
    }
    glEnd();

    glPointSize(1);

    //DBG("Edges vector size: " << edges.size());
}

void cls()
{
    // TODO: Try removing the Alpha argument
    glClearColor(backgroundColor[2] / 255.0f, backgroundColor[1] / 255.0f, backgroundColor[0] / 255.0f,  1);
    glClearDepth(1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void drawMarkedPoints()
{
    glPointSize(4);
    glColor3f((float)red[2] / 255.0f, (float)red[1] / 255.0f, (float)red[0] / 255.0f);

    glBegin(GL_POINTS);
    for (auto p : marked3Dpoints)
        glVertex3fv(p);
    glEnd();

    glPointSize(1);
}

void findAllVertexesInView()
{
    vertexesIdxInView.clear();

    for (int y = 0; y < cvVertexImg.rows; y++)
    {
        for (int x = 0; x < cvVertexImg.cols; x++)
        {
            unsigned int rgbColor = getColorInPoint(cvVertexImg, x, y);
            if (rgbColor == 0xFFFFFF) // Skip Background - White
                continue;

            // Sanity - Sometime usually at startup there is garbage data
            if (rgbColor >= themesh->vertices.size())
            {
                DBG("Garbage data. Ignoring mat");
                vertexesIdxInView.clear();
                return;
            }

            vertexesIdxInView.push_back(rgbColor);
        }
    }

    //DBG("Vertex total: " << themesh->vertices.size() << ", Vertex in view: " << vertexesIdxInView.size());
}

void redrawVertex(void *userData)
{
    DBG_T("Entered");
    //DrawData *data = static_cast<DrawData *>(userData);

    // glViewport(0, 0, winWidth, winHeight);
    //printViewPort();
    verifySize(VERTEX_WINDOW_NAME);

    camera.setupGL(xf * themesh->bsphere.center, themesh->bsphere.r);
    cls();

    // Transform and draw
    glPushMatrix();
    glMultMatrixd((double *)xf);

    drawFaces();
    drawVertexes();

    glPopMatrix(); // Don't forget to pop the Matrix

    //cvMainImg = getGlWindowAsCvMat(MAIN_WINDOW_NAME);
    cvVertexImg = getGlWindowAsCvMat(VERTEX_WINDOW_NAME);

    findAllVertexesInView();

    //imshowAndWait(cvVertexImg, 30, CV_WINDOW_NAME);

    DBG_T("Done");
}

void drawModel(trimesh::xform &xf, trimesh::GLCamera &camera)
{
    camera.setupGL(xf * themesh->bsphere.center, themesh->bsphere.r);

    // Transform and draw
    glPushMatrix();
    glMultMatrixd((double *)xf);

    drawFaces();
    drawBoundaries();
    drawMarkedPoints();

    glPopMatrix(); // Don't forget to pop the Matrix
}

void redraw(void *userData)
{
    DBG_T("Entered");

    //DrawData *data = static_cast<DrawData *>(userData);

    cls();
    drawModel(xf, camera);

    //glutSwapBuffers();
    // void glRotated(GLdouble angle, GLdouble x, GLdouble y, GLdouble z);
    // glRotate produces a rotation of angle degrees around the vector x y z.
    //glRotated(0.6, 0, 1, 0);

    DBG_T("Done");
}

void redrawPhoto(void *userData)
{
    DBG_T("Entered");

    cv::Mat textureCv = cvShowPhoto;

    //cameraMap.setupGL(xfMap * themesh->bsphere.center, themesh->bsphere.r);

    glEnable(GL_TEXTURE_2D);

    // Create Texture
    GLuint texture[1];
    glGenTextures(1, &texture[0]);
    glBindTexture(GL_TEXTURE_2D, texture[0]); // 2d texture (x and y size)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); // scale linearly when image bigger than texture
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // scale linearly when image smalled than texture

    // 2d texture, level of detail 0 (normal), 3 components (red, green, blue), x size from image, y size from image,
    // border 0 (normal), rgb color data, unsigned byte data, and finally the data itself.
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, textureCv.cols, textureCv.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, textureCv.data);

    glColor3f(1, 1, 1); // set global color to white, otherwise this color will be (somehow) added to the texture

    //float x = 1;
    //float y = 1;
    float x = 1;
    float y = 1;

    xfMap = trimesh::xform::trans(0, 0, -3.5f / fov * themesh->bsphere.r) *
        trimesh::xform::trans(-themesh->bsphere.center);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-x, -y);
    glTexCoord2f(0.0f, 1.0f); glVertex2f( x, -y);
    glTexCoord2f(1.0f, 1.0f); glVertex2f( x, y);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(-x, y);

    //glTexCoord2f(0.0f, 0.0f); glVertex3f(-x, -y, 0.0f);
    //glTexCoord2f(0.0f, 1.0f); glVertex3f( x, -y, 0.0f);
    //glTexCoord2f(1.0f, 1.0f); glVertex3f( x, y, 0.0f);
    //glTexCoord2f(1.0f, 0.0f); glVertex3f(-x, y, 0.0f);

    glEnd();

    glDeleteTextures(1,  &texture[0]);
    glDisable(GL_TEXTURE_2D);

    DBG_T("Done");
}

// Set the view to look at the middle of the mesh, from reasonably far away or use xfFile if available
void resetView()
{
    if (xf.read(xfFileName))
    {
        DBG("Reset view from " << xfFileName);
    }
    else
    {
        DBG("Reset view to look at the middle of the mesh, from reasonably far away");
        xf = trimesh::xform::trans(0, 0, -3.5f / fov * themesh->bsphere.r) *
            trimesh::xform::trans(-themesh->bsphere.center);
    }
}

// Set the view to look at the middle of the mesh, from reasonably far away
void upperView()
{
    DBG("Reset view to look at the middle of the mesh, from reasonably far away");
    xf = trimesh::xform::trans(0, 0, -3.5f / fov * themesh->bsphere.r) *
        trimesh::xform::trans(-themesh->bsphere.center);
}

// Angles are in radians
trimesh::XForm<double> getRotMat(float yaw = 0.0, float pitch = 0.0, float roll = 0.0)
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

#define IMAGE_FORMAT_EXT "png"

// Save the current screen to PNG/PPM file
void takeScreenshot()
{
    DBG_T("Entered");

    // Find first non-used filename
    const char fileNamePattern[] = "sample_images/img%d." IMAGE_FORMAT_EXT;
    int imgNum = 0;
    FILE *f;
    char fileName[1024];

    while (1)
    {
        sprintf(fileName, fileNamePattern, imgNum++);
        f = fopen(fileName, "rb");
        if (!f)
        {
            std::cout << "Saving image " << fileName << std::endl;
            break;
        }
        fclose(f);
    }

    // Read pixels
    GLint V[4];
    glGetIntegerv(GL_VIEWPORT, V);
    GLint width = V[2], height = V[3];
    std::unique_ptr<char> buf = std::unique_ptr<char>(new char[width * height * 3]);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    GLenum format = std::string(IMAGE_FORMAT_EXT) == "ppm" ? GL_RGB : GL_BGR;
    glReadPixels(V[0], V[1], width, height, format, GL_UNSIGNED_BYTE, buf.get());

    // Flip top-to-bottom
    for (int i = 0; i < height / 2; i++)
    {
        char *row1 = buf.get() + 3 * width * i;
        char *row2 = buf.get() + 3 * width * (height - 1 - i);
        for (int j = 0; j < 3 * width; j++)
            std::swap(row1[j], row2[j]);
    }

    if (std::string(IMAGE_FORMAT_EXT) == "ppm")
    {
        // Write out PPM file
        f = fopen(fileName, "wb");
        if (!f)
        {
            std::cout << "Failed saving file " << fileName << std::cout;
            return;
        }

        fprintf(f, "P6\n%d %d\n255\n", width, height);
        fwrite(buf.get(), width * height * 3, 1, f);
        fclose(f);
    }
    else
    {
        // The following link discusses this conversion-
        // http://stackoverflow.com/questions/9097756/converting-data-from-glreadpixels-to-opencvmat
        cv::Mat img(cv::Size(width, height), CV_8UC3, buf.get(), cv::Mat::AUTO_STEP);
        if (!cv::imwrite(fileName, img))
            std::cout << "Failed saving file " << fileName << std::cout;

        // Show image
        //std::string cvWindowName = "screenshot";
        //cv::namedWindow(cvWindowName, cv::WINDOW_AUTOSIZE);
        //cv::imshow(cvWindowName, img);
        //cv::waitKey(0);
    }

    DBG_T("Done");
}

void printMatDetailed(const std::string &name, const cv::Mat &mat)
{
    DBG(name << ", size: " << mat.size() << ", channels: " << mat.channels() << ", type: " << getOpenCVmatType(mat) <<
        "\n" << mat);
}

std::vector<cv::Point> projectCvPoints(std::vector<cv::Point3f> pts, const cv::Mat &projectionMatrix)
{
    cv::Mat vectors3dHomo;
    convertPointsToHomogeneous(pts, vectors3dHomo);
    vectors3dHomo = vectors3dHomo.reshape(1, pts.size()).t();
    //printMatDetailed("vectors3dHomo", vectors3dHomo);
    //printMatDetailed("projectionMatrix", projectionMatrix);

    cv::Mat resHomo = projectionMatrix * vectors3dHomo;
    //printMatDetailed("resHomo", resHomo);

    std::vector<cv::Point> p2d;
    for (int col = 0; col < resHomo.cols; col++)
    {
        cv::Mat pMat;
        cv::convertPointsFromHomogeneous(resHomo(cv::Rect(col, 0, 1, 3)).t(), pMat);
        p2d.push_back(cv::Point(std::round(pMat.at<float>(0, 0)), std::round(pMat.at<float>(0, 1))));
    }

    return p2d;
}

std::vector<int> findEdgesInView(bool bothVertexesInView)
{
    std::vector<int> edgesInView;

    for (unsigned i = 0; i < edges.size(); i++)
    {
        std::pair<int, int> edge = edges[i];

        if (bothVertexesInView)
        {
            if (std::find(vertexesIdxInView.begin(), vertexesIdxInView.end(), edge.first) != vertexesIdxInView.end() &&
                std::find(vertexesIdxInView.begin(), vertexesIdxInView.end(), edge.second) != vertexesIdxInView.end())
            {
                edgesInView.push_back(i);
            }
        }
        else
        {
            if (std::find(vertexesIdxInView.begin(), vertexesIdxInView.end(), edge.first) != vertexesIdxInView.end() ||
                std::find(vertexesIdxInView.begin(), vertexesIdxInView.end(), edge.second) != vertexesIdxInView.end())
            {
                edgesInView.push_back(i);
            }
        }
    }

/*
    for (int v : vertexesIdxInView)
    {
        for (unsigned i = 0; i < edges.size(); i++)
        {
            std::pair<int, int> edge = edges[i];
            if (v == edge.first || v == edge.second)
                edgesInView.push_back(i);
        }
    }
    //moti
*/
    DBG("Edges in view size: " << edgesInView.size());
    return edgesInView;
}

void drawEdgesInViewAfterProjection()
{
    DBG("Entered");

    std::vector<int> edgesInView = findEdgesInView(true);
    for (int i : edgesInView)
    {
        std::pair<int, int> edge = edges[i];
        std::vector<trimesh::point> edgeVertices = { themesh->vertices[edge.first], themesh->vertices[edge.second] };

        std::vector<cv::Point> p2d = projectCvPoints(conv3dPointTrimeshToOpenCV(edgeVertices), projMat);
        cv::line(cvShowPhoto, p2d[0], p2d[1], blue);
    }

    DBG("Done");
}

void drawVertexesInViewAfterProjection()
{
    DBG("Entered");

    std::vector<trimesh::point> vertexesInView;
    for (auto idx : vertexesIdxInView)
        vertexesInView.push_back(themesh->vertices[idx]);
    std::vector<cv::Point3f> cvVertexesInView = conv3dPointTrimeshToOpenCV(vertexesInView);

    std::vector<cv::Point> p2d = projectCvPoints(cvVertexesInView, projMat);

    for (auto p : p2d)
        cv::circle(cvShowPhoto, p, marksRadius, blue, -1);

    DBG("Done");
}

void drawVertexesInViewAfterProjectionWithIntrinsicMatrix()
{
    std::vector<trimesh::point> vertexesInView;
    for (auto idx : vertexesIdxInView)
        vertexesInView.push_back(themesh->vertices[idx]);
    std::vector<cv::Point3f> cvVertexesInView = conv3dPointTrimeshToOpenCV(vertexesInView);

    std::vector<cv::Point2f> projectedVertexesInView;
    cv::projectPoints(cvVertexesInView, rvec, tvec, intrinsicMatrix, cv::noArray(), projectedVertexesInView);

    for (auto p : projectedVertexesInView)
        cv::circle(cvShowPhoto, p, marksRadius, green, -1);
}

void drawEdgesInViewAfterProjectionWithIntrinsicMatrix()
{
    DBG("Entered");

    std::vector<int> edgesInView = findEdgesInView(true);
    for (int i : edgesInView)
    {
        std::pair<int, int> edge = edges[i];
        std::vector<trimesh::point> edgeVertices = { themesh->vertices[edge.first], themesh->vertices[edge.second] };

        std::vector<cv::Point2f> projectedLine;
        cv::projectPoints(conv3dPointTrimeshToOpenCV(edgeVertices), rvec, tvec, intrinsicMatrix, cv::noArray(),
            projectedLine);
        cv::line(cvShowPhoto, projectedLine[0], projectedLine[1], green);
    }

    DBG("Done");
}

void updateCvWindow()
{
    if (showMap)
    {
        cvShowPhoto = cvMap.clone();
    }
    else
    {
        cvShowPhoto = cvPhoto.clone();
        for (auto p : marked2Dpoints)
            cv::circle(cvShowPhoto, p, marksRadius, red, -1);

        //moti
        //for (auto p : projectedVertexesInView)
          //  cv::circle(cvShowPhoto, p, marksRadius, blue, -1);

        if (!projMat.empty())
        {
            drawVertexesInViewAfterProjection();
            drawEdgesInViewAfterProjection();
        }

        if (!projectedMarks.empty())
        {
            drawVertexesInViewAfterProjectionWithIntrinsicMatrix();
            drawEdgesInViewAfterProjectionWithIntrinsicMatrix();
        }

        for (auto p : projectedMarks)
            cv::circle(cvShowPhoto, p, marksRadius, yellow, -1);
    }

    cv::transpose(cvShowPhoto, cvShowPhoto);
    cv::flip(cvShowPhoto, cvShowPhoto, 1);

    cv::updateWindow(CV_WINDOW_NAME);
}

// Force windows to redraw its context and call draw callback
void updateWindows()
{
    DBG_T("Entered");

    cv::updateWindow(MAIN_WINDOW_NAME);
    cv::updateWindow(VERTEX_WINDOW_NAME);
    updateCvWindow();

    DBG_T("Done");
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
void mouseNavCallbackFunc(int event, int x, int y, int flags, void *userdata)
{
    std::stringstream msg;
    trimesh::Mouse::button b = trimesh::Mouse::NONE;

    if (event == cv::EVENT_MOUSEMOVE)
    {
        msg << "Move";

        if (flags == cv::EVENT_FLAG_LBUTTON)
            b = trimesh::Mouse::ROTATE;
        else if (flags == cv::EVENT_FLAG_MBUTTON)
            b = trimesh::Mouse::MOVEXY;
        else if (flags == cv::EVENT_FLAG_RBUTTON)
            b = trimesh::Mouse::MOVEZ;
    }
    else if (event == cv::EVENT_LBUTTONDOWN)
    {
        msg << "Left";
        b = trimesh::Mouse::ROTATE;
    }
    else if (event == cv::EVENT_RBUTTONDOWN)
    {
        msg << "Right";
        b = trimesh::Mouse::MOVEZ;
    }
    else if (event == cv::EVENT_MBUTTONDOWN)
    {
        msg << "Middle";
        b = trimesh::Mouse::MOVEXY;
    }
    else if (event == cv::EVENT_MOUSEWHEEL)
    {
        msg << "Wheel";

        if (flags > 0)
            b = trimesh::Mouse::WHEELUP;
        else
            b = trimesh::Mouse::WHEELDOWN;
    }
    else
    {
        msg << "Other " << event;
        DBG("Mouse event: " << msg.str() << ", " << cv::Point(x, y) << ", flags: " << flags);
    }

    if (b != trimesh::Mouse::NONE)
    {
        //DBG(xf);
        camera.mouse(x, y, b, xf * themesh->bsphere.center, themesh->bsphere.r, xf);
        //DBG(xf);

        updateWindows();
    }
}

void findNearestVertex(const cv::Point &p)
{
    int minDist = std::numeric_limits<int>::max(), minRgbColor = 0xFFFFFF;
    for (int y = 0; y < cvVertexImg.rows; y++)
    {
        for (int x = 0; x < cvVertexImg.cols; x++)
        {
            unsigned int rgbColor = getColorInPoint(cvVertexImg, x, y);
            if (rgbColor == 0xFFFFFF) // Skip Background - White
                continue;

            int dist = std::abs(p.x - x) + std::abs(p.y - y);
            if (dist < minDist)
            {
                minDist = dist;
                minRgbColor = rgbColor;

                //DBG("Point: " << cv::Point(x, y) << ", minDist: " << minDist << ", minRgbColor: " << minRgbColor);
            }
        }
    }

    //DBG("minDist: " << minDist << ", minRgbColor: " << minRgbColor << ", vertex: " << themesh->vertices[minRgbColor]);
    //DBG("Marked 3D points: " << marked3Dpoints);

    trimesh::point p3d = themesh->vertices[minRgbColor];
    DBG("2D point: " << p << ", 3D point: " << p3d);
    marked3Dpoints.push_back(p3d);
    updateWindows();
}

void mouseTagCallbackFunc2D(int event, int x, int y, int flags, void *userdata)
{
    std::stringstream msg;
    cv::Point p(x, y);

    // Calculate viewport-photo factor
    GLint V[4];
    glGetIntegerv(GL_VIEWPORT, V);
    GLint width = V[2], height = V[3];
    float widthFactor = width / (float)cvPhoto.cols;
    float heightFactor = height / (float)cvPhoto.rows;
    //DBG("Factors: " << widthFactor << ", " << heightFactor);

    if (event == cv::EVENT_LBUTTONDOWN)
    {
        msg << "Left";
        DBG("Mouse event: " << msg.str() << ", " << p << ", flags: " << flags);

        p.x = p.x / widthFactor;
        p.y = p.y / heightFactor;

        marksRadius = std::max(4, (int)(4 / (float)std::min(widthFactor, heightFactor)));

        marked2Dpoints.push_back(p);

        updateCvWindow();
    }
}

void mouseTagCallbackFunc3D(int event, int x, int y, int flags, void *userdata)
{
    std::stringstream msg;
    cv::Point p(x, y);

    if (event == cv::EVENT_LBUTTONDOWN)
    {
        msg << "Left";
        //DBG("Mouse event: " << msg.str() << ", " << p << ", flags: " << flags);

        findNearestVertex(p);
    }
}

void initWindow(const std::string &winName, int winWidth, int winHeight, void (*redrawFunc)(void *))
{
    cv::namedWindow(winName, cv::WINDOW_OPENGL | cv::WINDOW_NORMAL);
    cv::resizeWindow(winName, winWidth, winHeight);

    cv::setOpenGlDrawCallback(winName, redrawFunc, 0); // Set OpenGL render handler for the specified window
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
    themesh->need_normals();
    themesh->need_curvatures();
    themesh->need_dcurv();
    themesh->need_faces();
    themesh->need_across_edge();

    DBG("Mesh file [" << fileName << "] loaded");
    DBG("Vertices num [" << themesh->vertices.size() << "], faces num [" << themesh->faces.size() <<
        "], tstrips num [" << themesh->tstrips.size() << "], normals num [" << themesh->normals.size() << "]");

    // Generate xf file name from fileName
    xfFileName = trimesh::xfname(fileName);
    DBG("xfFileName: " << xfFileName);

    resetView();
}

void handleMenuKeyboard(int key)
{
    switch (key)
    {
    case 's':
        showMap = !showMap;
        DBG("Toggle show map is now: " << showMap);
        break;

    case 'c':
        std::cout << "Toggle faces-color (black/white)" << std::endl;
        facesColor = facesColor == white ? black : white;
        break;

    case 'i':
        takeScreenshot();
        break;

    case 'p':
        std::cout << xf << std::endl;
        break;

    case 'x':
        std::cout << "Saving current view to xfFile: " << xfFileName << std::endl;
        xf.write(xfFileName);
        break;

    case ')':
        fov /= 1.1f; camera.set_fov(fov);
        std::cout << "New fov value: " << fov << std::endl;
        break;
    case '(':
        fov *= 1.1f; camera.set_fov(fov);
        std::cout << "New fov value: " << fov << std::endl;
        break;

    default:
        //DBG("Unhandled key: [" << key << "]");
        break;
    }
}

void handleNavKeyboard(int key)
{
    if (key >= '0' && key <= '9')
    {
        std::string xfFile = trimesh::replace_ext(xfFileName, "") + std::string((const char *)&key) + ".xf";
        if (xf.read(xfFile))
        {
            DBG("Loaded view from " << xfFile);
        }
        else
        {
            if (xf.write(xfFile))
                DBG("Saved view to " << xfFile);
            else
                DBG("Failed saving view to " << xfFile);
        }
        return;
    }

    switch (key)
    {
    case ' ':
        resetView();
        break;
    case 'u':
        upperView();
        break;

    case '+':
        xyStep += 10;
        DBG("Increased xyStep to " << xyStep);
        break;
    case '-':
        xyStep -= 10;
        DBG("Decreased xyStep to " << xyStep);
        break;

    case '.':
        zStep += 10;
        DBG("Increased zStep to " << zStep);
        break;
    case ',':
        zStep -= 10;
        DBG("Decreased zStep to " << zStep);
        break;

        // Camera Position - XY
    case KEY_LEFT:
        xf = trimesh::xform::trans(xyStep, 0, 0) * xf;
        break;
    case KEY_RIGHT:
        xf = trimesh::xform::trans(-xyStep, 0, 0) * xf;
        break;

    case KEY_UP:
        xf = trimesh::xform::trans(0, -xyStep, 0) * xf;
        break;
    case KEY_DOWN:
        xf = trimesh::xform::trans(0, xyStep, 0) * xf;
        break;

        // Camera Position - Z - Zoom
    case ']':
        xf = trimesh::xform::trans(0, 0, zStep) * xf;
        break;
    case '[':
        xf = trimesh::xform::trans(0, 0, -zStep) * xf;
        break;

        // Camera Angle - Roll
    case 'e':
        xf = getRotMat(0.0, 0.0, rotStep) * xf;
        break;
    case 'q':
        xf = getRotMat(0.0, 0.0, -rotStep) * xf;
        break;

        // Camera Angle - Yaw
    case 'a':
        xf = getRotMat(-rotStep, 0.0, 0.0) * xf;
        break;
    case 'd':
        xf = getRotMat(rotStep, 0.0, 0.0) * xf;
        break;

        // Camera Angle - Pitch
    case 'w':
        xf = getRotMat(0.0, -rotStep, 0.0) * xf;
        break;
    case 's':
        xf = getRotMat(0.0, rotStep, 0.0) * xf;
        break;

    default:
        //DBG("Unhandled key: [" << key << "]");
        break;
    }
}

void saveMarkedPoints()
{
    std::ofstream outputFile(markedPtsFile);
    if(!outputFile)
    {
        DBG("Failed opening " << markedPtsFile);
        return;
    }

    outputFile << marked2Dpoints.size() << "\n";
    std::ostream_iterator<cv::Point> outputIterator2D(outputFile, "\n");
    std::copy(marked2Dpoints.begin(), marked2Dpoints.end(), outputIterator2D);

    outputFile << marked3Dpoints.size() << "\n";
    for (auto p : marked3Dpoints)
        outputFile << std::setprecision(17) << "(" << p[0] << ", " << p[1] << ", " << p[2] << ")\n";

    DBG("Saved marked points to " << markedPtsFile);
}

void loadMarkedPoints()
{
    FILE *pFile;
    pFile = fopen(markedPtsFile.c_str(), "r");

    int marked2dNum = 0;
    fscanf(pFile, "%d\n", &marked2dNum);
    //DBG("marked2dNum: " << marked2dNum);

    marked2Dpoints.clear();
    for (int i = 0; i < marked2dNum; i++)
    {
        cv::Point p;
        fscanf(pFile, "[%d, %d]\n", &p.x, &p.y);
        marked2Dpoints.push_back(p);
    }
    //DBG("marked2Dpoints:\n" << marked2Dpoints);

    int marked3dNum = 0;
    fscanf(pFile, "%d\n", &marked3dNum);
    //DBG("marked3dNum: " << marked3dNum);

    marked3Dpoints.clear();
    for (int i = 0; i < marked3dNum; i++)
    {
        trimesh::point p;
        fscanf(pFile, "(%f, %f, %f)\n", &p[0], &p[1], &p[2]);
        marked3Dpoints.push_back(p);
    }
    //DBG("marked3Dpoints:\n" << marked3Dpoints);

    DBG("Loaded marked points from " << markedPtsFile);
}

void handleTagKeyboard(int key)
{
    switch (key)
    {
    case ' ':
        resetView();
        break;

    case KEY_ENTER:
        calcPoseEstimation();
        break;

    case 's':
        saveMarkedPoints();
        break;

    case 'l':
        loadMarkedPoints();
        break;

    case 'r':
        DBG("Removing all marked points");
        marked2Dpoints.clear();
        marked3Dpoints.clear();
        break;

    case 'd':
        DBG("Removing last marked points");
        if (marked2Dpoints.size() > marked3Dpoints.size())
        {
            marked2Dpoints.pop_back();
        }
        else if (marked3Dpoints.size() > marked2Dpoints.size())
        {
            marked3Dpoints.pop_back();
        }
        else
        {
            if (!marked3Dpoints.empty())
            {
                marked2Dpoints.pop_back();
                marked3Dpoints.pop_back();
            }
            else
            {
                DBG("Already empty");
            }
        }
        break;

    default:
        //DBG("Unhandled key: [" << key << "]");
        break;
    }
}

void handleKeyboard(int key)
{
    //DBG("key [" << key << "], char [" << (char)key << "]");

    if (key == 't')
    {
        taggingOn = !taggingOn;
        if (taggingOn)
        {
            menuOn = true;
            cv::setMouseCallback(MAIN_WINDOW_NAME, mouseTagCallbackFunc3D, NULL);
            DBG("Tagging mode ON");
        }
        else
        {
            menuOn = false;
            cv::setMouseCallback(MAIN_WINDOW_NAME, mouseNavCallbackFunc, NULL);
            DBG("Tagging mode OFF");
        }
        return;
    }

    if (taggingOn)
    {
        handleTagKeyboard(key);
        return;
    }

    if (key == 'm')
    {
        menuOn = !menuOn; // toggle
        DBG("Menu state changed: " << menuOn);
        return;
    }

    if (menuOn)
    {
        handleMenuKeyboard(key);
        menuOn = false;
    }
    else
    {
        handleNavKeyboard(key);
    }
}

// XXX: Remember-
//      cv::setOpenGlContext(const string& winname); // Sets the specified window as current OpenGL context
int main(int argc, char* argv[])
{
    std::string modelFile, imageFile, mapFile;
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " [model_file] [image_file] [WinSize +#,+#]" << std::endl;
        //modelFile = "../samples/cube.obj";
        modelFile = "../berlin/berlin.obj";
        //imageFile = "../samples/cube_photo.png";
        imageFile = "data/inter_03.png";
        mapFile = "data/berlin_google_map.png";
        markedPtsFile = "data/markedPoints.txt";
    }
    else
    {
        modelFile = argv[1];

        if (argc > 2)
            imageFile = argv[2];

        for (int i = 1; i < argc; i++)
        {
            if (argv[i][0] == '+')
                sscanf(argv[i] + 1, "%d,%d", &winWidth, &winHeight);
        }
    }

    loadModel(modelFile);

    initWindow(MAIN_WINDOW_NAME, winWidth, winHeight, redraw);
    cv::setMouseCallback(MAIN_WINDOW_NAME, mouseNavCallbackFunc, NULL);
    initWindow(VERTEX_WINDOW_NAME, winWidth, winHeight, redrawVertex);
    cv::setMouseCallback(VERTEX_WINDOW_NAME, mouseNavCallbackFunc, NULL);

    // Init cvPhoto window
    cvPhoto = cv::imread(imageFile);
    cvMap = cv::imread(mapFile);
    xfMap = trimesh::xform::trans(0, 0, -3.5f / fov * themesh->bsphere.r) *
        trimesh::xform::trans(-themesh->bsphere.center);
    initWindow(CV_WINDOW_NAME, winWidth, winHeight, redrawPhoto);
    cv::setMouseCallback(CV_WINDOW_NAME, mouseTagCallbackFunc2D, NULL);

    for (;;)
    {
        updateWindows();
        int key = cv::waitKey(0);
        if (key == 27) // ESC key
            break;

        handleKeyboard(key);
    }

    cv::setOpenGlDrawCallback(MAIN_WINDOW_NAME, 0, 0);
    cv::setOpenGlDrawCallback(VERTEX_WINDOW_NAME, 0, 0);
    cv::setOpenGlDrawCallback(CV_WINDOW_NAME, 0, 0);
    cv::destroyAllWindows();

    return 0;
}

