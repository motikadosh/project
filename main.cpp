// http://docs.opencv.org/trunk/d2/d3c/group__core__opengl.html

#include <string>
#include <vector>
#include <set>
#include <iostream>
#include <sstream>
#include <fstream>
#include <memory>
#include <algorithm>
#include <limits>
#include <random>
#include <cstdio>
#include <iomanip>
#include <sys/stat.h>
#include <ctime>
#include <map>

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

// More about gFalgs- https://gflags.github.io/gflags/
#include <gflags/gflags.h>
// For old version of gflags
#ifndef GFLAGS_NAMESPACE
    #define GFLAGS_NAMESPACE google
#endif

#include <glog/logging.h>

//
// gFlags - Configuration
//
DEFINE_string(model, "../berlin/berlin.obj", "Path to model file");
DEFINE_string(pose, "", "Pose string to use as start-up. Format: \"x y z yaw pitch roll\"");
//DEFINE_string(xf, "", "Path to XF transformation to use at start-up");
DEFINE_string(output_dir, "", "Output directory name. In case of empty string, auto directory is created");
DEFINE_bool(resume_export, false, "Use check_point.txt to resume samples export");

DEFINE_bool(random_sampling, false, "Use random sampling or not");
DEFINE_int32(samples_num, 300, "Number of (x, y) pts to sample in auto navigation");
DEFINE_double(test_percent, 0.2, "Test percent");
DEFINE_int32(grid, 20, "Number of (x, y) pts to sample on each grid row in auto navigation, column is induced");
DEFINE_double(grid_test_offset, 0.5, "Offset in grid step from main grid to secondary grid");
DEFINE_double(camera_height, 2, "Camera height added to ground-level. I.e. Z of samples");

DEFINE_double(crop_upper_part, 0.33333, "Part of image to crop from top. 0- ignore");
DEFINE_int32(min_edge_pixels, 30, "Minimum number of pixels required for each edge");
DEFINE_int32(min_edges_threshold, 4, "Minimum number of edges for sample exporting");
DEFINE_double(min_data_threshold, 0 /*0.003*/, "Minimum data (pixels) percentage for sample exporting");
DEFINE_int32(kernel_size, 15, "Kernel size to apply on image before masking with faces");

DEFINE_int32(win_width, 800, "Width of the main window");
DEFINE_int32(win_height, 600, "Height of the main window");

DEFINE_bool(debug, false, "Debug mode");

//
// Macros
//

const float PI = 3.14159265358979323846264f;
#define RAD_TO_DEG(x) ((x) * 180 / PI)
#define DEG_TO_RAD(x) ((x) * PI / 180)

#define MAIN_WINDOW_NAME "OpenGL"
#define VERTEX_WINDOW_NAME "vertex_win"
#define FACES_WINDOW_NAME "faces_win"
#define CV_WINDOW_NAME "cv_win"
#define UPPER_MAP_WINDOW_NAME "upper_map"

#define DBG(params) \
    do { \
        std::cout << dbgCount++ << ") " << __FUNCTION__ << ": " << params << std::endl; \
        LOG(INFO) << __FUNCTION__ << ": " << params; \
    } while (0)

#define DBG_T(params) \
    if (FLAGS_debug) \
        DBG(params)

// Keyboard keys
// Keys list: https://pythonhosted.org/pyglet/api/pyglet.window.key-module.html
#define ESC_KEY         27
#define SPACE_KEY       32
#define KEY_TAB         9

#define KEY_ENTER       13
#define KEY_DELETE      255
#define KEY_BACKSPACE   8
#define KEY_HOME        80
#define KEY_LEFT        81
#define KEY_UP          82
#define KEY_RIGHT       83
#define KEY_DOWN        84
#define KEY_PAGEUP      85
#define KEY_PAGEDOWN    86
#define KEY_END         87

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
GLfloat const *backgroundColorGL = whiteGL;
GLfloat const *foregroundColorGL = blackGL;
const GLfloat *groundColorGL = whiteGL;

trimesh::GLCamera camera;
trimesh::xform xf; // Current location and orientation
std::string xfFileName;
int dbgCount = 0;

enum KeysGroup {
    KEYS_GROUP_NAV = 1,
    KEYS_GROUP_MENU,
    KEYS_GROUP_TAG
};
KeysGroup gKeysGroup = KEYS_GROUP_NAV;

float fov = 0.7f;
float rotStep = DEG_TO_RAD(10);
float xyzStep = 1;

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

std::vector<std::string> split(const std::string &s, char delim = ' ')
{
    std::stringstream ss(s);
    std::string item;
    std::vector<std::string> elems;
    while (std::getline(ss, item, delim))
    {
        elems.push_back(std::move(item));
    }

    return elems;
}

struct SamplePose {
    SamplePose(float x, float y, float z, float yaw, float pitch, float roll) :
        x(x), y(y), z(z), yaw(yaw), pitch(pitch), roll(roll) { }
    SamplePose(const std::string &poseStr)
    {
        std::vector<std::string> pose = split(poseStr);
        if (pose.size() != 6)
        {
            DBG("Invalid -pose (x y z yaw pitch roll) [" << poseStr << "]");
            throw std::invalid_argument("Invalid pose flag");
        }

        x = std::stof(pose[0]);
        y = std::stof(pose[1]);
        z = std::stof(pose[2]);
        yaw = std::stof(pose[3]);
        pitch = std::stof(pose[4]);
        roll = std::stof(pose[5]);
    }

    std::string print() const
    {
        std::stringstream ss;
        ss << std::setprecision(17) << "(x,y,z,yaw,pitch,roll)=(" << x << ", " << y << ", " << z << ", "
            << yaw << ", " << pitch << ", " << roll << ")";
        return ss.str();
    }

    bool write(const std::string &filePath)
    {
        std::ofstream f(filePath);
        f << print();
        f.close();
        return f.good();
    }

    // Members
    float x;
    float y;
    float z;

    // Angles should be in degrees
    float yaw;
    float pitch;
    float roll;
};

static inline std::ostream& operator<<(std::ostream &out, const SamplePose &pose)
{
    out << pose.print();
    return out;
}

struct DataSet {
    std::vector<cv::Point3f> samplePoints;
    std::vector<trimesh::xform> xfSamples;
    std::vector<SamplePose> samplesData;
};
std::map<std::string, DataSet> gDataSet;

bool gAutoNavigation = false;
unsigned gAutoNavigationIdx = 0;

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
        ss << std::setprecision(17) << "(left,right,bottom,top,neardist,fardist)" << SEP <<
            "(" << left << ", " << right << ", " << bottom << ", " << top << ", " << neardist << ", " << fardist << ")";
        return ss.str();
    }

    bool write(const std::string &filePath)
    {
        std::ofstream f(filePath);
        f << print();
        f.close();
        return f.good();
    }

    bool read(const std::string &filePath)
    {
        std::ifstream f(filePath);
        if (!f.is_open())
        {
            std::cout << "Unable to open file " << filePath << std::endl;
            return false;
        }

        std::string line;
        std::getline(f, line);

        std::size_t pos = line.find("=");
        line = line.substr(pos + 1);

        sscanf(line.c_str(), "(%f, %f, %f, %f, %f, %f)", &left, &right, &bottom, &top, &neardist, &fardist);

        f.close();
        return true;
    }

    cv::Point convertWorldPointToMap(const cv::Point3f &p, const cv::Size &mapSize)
    {
        int mapX = (p.x - left) / (right - left) * mapSize.width;
        int mapY = (p.y - bottom) / (top - bottom) * mapSize.height;

        return cv::Point(mapX, mapY);
    }

private:
    const std::string SEP = "=";
};
OrthoProjection gOrthoProjData(0, 0, 0, 0, 0, 0);

cv::Mat gModelMap;
cv::Mat gGroundLevelMap;

cv::Mat gSamplesMap;
cv::Mat gUpperMapShow;

//
// Pre-declarations
//
void updateWindows();

//
// Utilities
//

static inline std::ostream& operator<<(std::ostream &out, const OrthoProjection &data)
{
    out << data.print();

    return out;
}

template <typename T1, typename T2>
inline std::ostream& operator<<(std::ostream &out, const std::pair<T1, T2> &pair)
{
    out << "(" << pair.first << ", " << pair.second << ")";

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

template <typename T>
inline std::ostream& operator<<(std::ostream &out, const std::set<T> &set)
{
    for(auto i : set)
        out << i << ", ";

    if (set.size())
        out << "\b\b";

    return out;
}

// Linux only. Creates a directory if it is not already exists
// returns true if directory was created
//         false means a failure or directory already exists
bool makeDir(const std::string &dirName)
{
    struct stat sb;
    if (!(stat(dirName.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)))
    {
        if (mkdir(dirName.c_str(), ACCESSPERMS))
            throw std::runtime_error("Failed creating directory " + dirName);
        return true;
    }

    return false;
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

// E.g. input "C:\Dir\File.bat" -> "C:\Dir".
// E.g. input "File.bat" -> "".
std::string getDirName(const std::string &strPath)
{
    std::size_t found = strPath.find_last_of("/\\");
    return found == std::string::npos ? "" : strPath.substr(0, found);
}

std::string getRunningExeFilePath()
{
#define PRC_EXE_PATH_LEN 1024
    char pBuf[PRC_EXE_PATH_LEN];
    int len = PRC_EXE_PATH_LEN;
    bool isSuccess = false;

#if __linux__
    char szTmp[128];
    sprintf(szTmp, "/proc/%d/exe", getpid());
    int bytes = MIN(readlink(szTmp, pBuf, len), len - 1);
    if (bytes >= 0) // Success
        isSuccess = true;

    pBuf[bytes] = '\0';
#elif WIN32
    int bytes = GetModuleFileName(NULL, pBuf, len);
    if (bytes) // Success
        isSuccess = true;
#endif

    if (!isSuccess)
    {
        std::cerr << "Failed getting current running directory";
        throw std::runtime_error("Failed getting current running directory");
        return std::string();
    }

    return pBuf;
}

static size_t findLastWhiteSpace(const std::string &str)
{
    for (std::string::const_iterator it = str.end(); it != str.begin(); --it)
    {
        if (std::isspace(*it))
            return it - str.begin();
    }

    return std::string::npos;
}

void myPutText(cv::Mat &img, std::stringstream &ss, cv::Point org, bool lineWrap = true, bool autoScale = true,
    int fontFace = cv::FONT_HERSHEY_COMPLEX_SMALL, double fontScale = 0.8, cv::Scalar color = red,
    int thickness = 1, int lineType = 8, bool bottomLeftOrigin = false)
{
    if (autoScale)
    {
        // TODO: Get real window size
        float sizeFactor = std::min(img.rows, img.cols) / 480.0f;

        fontScale = fontScale * sizeFactor;
        thickness = int (thickness * sizeFactor);

        // Make sure the the new height does not exceed screen top
        int baseLine;
        cv::Size textSize = cv::getTextSize("A", fontFace, fontScale, thickness, &baseLine);
        if (org.y < textSize.height)
            org.y = int(textSize.height + std::min(img.rows, img.cols) * 0.01f);
    }

    unsigned offset = 0;
    std::string line;
    do
    {
        std::getline(ss, line);

        do
        {
            int baseLine;
            cv::Size textSize = cv::getTextSize(line, fontFace, fontScale, thickness, &baseLine);
            size_t splitPos = line.length();

            // Handle line wrapping
            while (lineWrap && org.x + textSize.width > img.cols)
            {
                size_t pos = findLastWhiteSpace(line.substr(0, splitPos));
                if (pos == std::string::npos)
                    break;
                else
                    splitPos = pos;

                textSize = cv::getTextSize(line.substr(0, splitPos), fontFace, fontScale, thickness, &baseLine);
            }

            cv::putText(img, line.substr(0, splitPos), org + cv::Point(0, offset), fontFace, fontScale,
                color, thickness, lineType, bottomLeftOrigin);

            line = line.substr(splitPos < line.length() ? splitPos + 1 : splitPos);

            offset += textSize.height + baseLine;
        }
        while (!line.empty());

    } while (!ss.eof());
}

void imshowAndWait(const cv::Mat &img, unsigned waitTime = 0, const std::string &winName = "temp")
{
    if (img.empty())
        return;

    //cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);
    cv::imshow(winName, img);
    cv::waitKey(waitTime);

    cv::destroyWindow(winName);
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
    //for (auto i : modelview)
      //  DBG(i);
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
    cv::setOpenGlContext(winName);

    GLint V[4];
    glGetIntegerv(GL_VIEWPORT, V);
    GLint width = V[2], height = V[3];

    if (width == 100 && height == 30) // Default window size
    {
        DBG("Window BUG [" << winName << "] has default size. Resizing");
        cv::resizeWindow(winName, FLAGS_win_width, FLAGS_win_height);
    }
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
        //points2d.push_back(cv::Point2f((p.x/(float)cvPhoto.cols - 0.5)*2, p.y/((float)cvPhoto.rows - 0.5)*2));

    findProjectionMatrix(points3d, points2d);
    updateWindows();

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

void drawFaces(GLfloat const *facesColor)
{
    glDisable(GL_CULL_FACE); // Order of vertexes should not determine faces normals (I.e. Draw both sides of face)

    // Enable the vertex array
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, &themesh->vertices[0][0]);

    glColor3fv(facesColor);

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

// Currently not used. Kept for reference.
// If used, notice current edges color is by faces indexes and not edges, i.e. not unique
void drawBoundaries()
{
    glLineWidth(2);

    glBegin(GL_LINES);
    for (unsigned i = 0; i < themesh->faces.size(); i++)
    {
        for (unsigned j = 0; j < 3; j++)
        {
            if (themesh->across_edge[i][j] >= 0)
                continue;

            int v1 = themesh->faces[i][(j + 1) % 3];
            int v2 = themesh->faces[i][(j + 2) % 3];
            drawEdge(v1, v2, i);
        }
    }
    glEnd();
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
    // Specify clear values for the color buffers
    glClearColor(backgroundColorGL[0], backgroundColorGL[1], backgroundColorGL[2],  1);

    // Specifies the depth value used when the depth buffer is cleared. The initial value is 1.
    glClearDepth(1);

    // Clear buffers to preset values
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void drawMarkedPoints()
{
    glPointSize(4);
    glColor3fv(redGL);

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

    camera.setupGL(xf * themesh->bsphere.center, themesh->bsphere.r);
    cls();

    // Transform and draw
    glPushMatrix();
    glMultMatrixd((double *)xf);

    drawFaces(backgroundColorGL);
    drawVertexes();

    glPopMatrix(); // Don't forget to pop the Matrix

    //cvMainImg = getGlWindowAsCvMat(MAIN_WINDOW_NAME);
    cvVertexImg = getGlWindowAsCvMat(VERTEX_WINDOW_NAME);

    findAllVertexesInView();

    //imshowAndWait(cvVertexImg, 30, CV_WINDOW_NAME);

    DBG_T("Done");
}

void redrawFaces(void *userData)
{
    DBG_T("Entered");
    //DrawData *data = static_cast<DrawData *>(userData);

    verifySize(FACES_WINDOW_NAME);

    cls();
    camera.setupGL(xf * themesh->bsphere.center, themesh->bsphere.r);

    // Transform and draw
    glPushMatrix();
    glMultMatrixd((double *)xf);

    drawFaces(foregroundColorGL);

    glPopMatrix(); // Don't forget to pop the Matrix

    DBG_T("Done");
}

void drawModel(trimesh::xform &xf, trimesh::GLCamera &camera)
{
    //DBG(edges);
    //DBG("\n" << xf);
    camera.setupGL(xf * themesh->bsphere.center, themesh->bsphere.r);

    // Transform and draw
    glPushMatrix();
    glMultMatrixd((double *)xf);

    drawFaces(backgroundColorGL);
    //drawBoundaries();
    drawEdges();
    drawMarkedPoints();

    glPopMatrix(); // Don't forget to pop the Matrix
}

void redraw(void *userData)
{
    DBG_T("Entered");

    //DrawData *data = static_cast<DrawData *>(userData);

    verifySize(MAIN_WINDOW_NAME);

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
    DBG_T("Entered");

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
    DBG_T("Entered");

    DBG("Reset view to look at the middle of the mesh, from reasonably far away");
    xf = trimesh::xform::trans(0, 0, -3.5f / fov * themesh->bsphere.r) *
        trimesh::xform::trans(-themesh->bsphere.center);
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

std::string getTimeStamp()
{
    time_t t = time(0);
    struct tm *tm = localtime(&t);

    char timeStamp[32];
    strftime(timeStamp, sizeof(timeStamp), "%Y_%m_%d-%H_%M_%S", tm);

    return timeStamp;
}

#define IMAGE_FORMAT_EXT "png"

// Save the current screen to PNG/PPM file
std::string takeScreenshot(const std::string &fileName = std::string())
{
    DBG_T("Entered");

    cv::setOpenGlContext(MAIN_WINDOW_NAME); // Sets the specified window as current OpenGL context

    FILE *f;
    std::string filePath;
    if (fileName.empty())
    {
        std::string fileNamePattern =
            FLAGS_output_dir + "img_" + getTimeStamp() + "_%d." IMAGE_FORMAT_EXT;

        // Find first non-used filePath
        char checkedFilePath[1024];
        int imgNum = 0;
        while (1)
        {
            sprintf(checkedFilePath, fileNamePattern.c_str(), imgNum++);
            f = fopen(checkedFilePath, "rb");
            if (!f) // Found non used file name
            {
                filePath = checkedFilePath;
                break;
            }

            fclose(f);
        }
    }
    else
    {
        filePath = FLAGS_output_dir + fileName + "." IMAGE_FORMAT_EXT;
    }
    DBG_T("Saving image " << filePath);

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
        f = fopen(filePath.c_str(), "wb");
        if (!f)
        {
            std::cout << "Failed saving file " << filePath << std::cout;
            return std::string();
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
        if (!cv::imwrite(filePath, img))
        {
            std::cout << "Failed saving file " << filePath << std::cout;
            return std::string();
        }

        // Show image
        //std::string cvWindowName = "screenshot";
        //cv::namedWindow(cvWindowName, cv::WINDOW_AUTOSIZE);
        //cv::imshow(cvWindowName, img);
        //cv::waitKey(0);
    }

    DBG_T("Done. filePath: " << filePath);
    return filePath;
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

std::vector<int> findEdgesInView(const cv::Mat &img, unsigned minEdgePixels)
{
    std::map<int, int> edgesCountInView;

    for (int y = 0; y < img.rows; y++)
    {
        for (int x = 0; x < img.cols; x++)
        {
            unsigned int rgbColor = getColorInPoint(img, x, y);
            if (rgbColor == 0xFFFFFF) // Skip Background - White
                continue;

            // Sanity - Sometime usually at startup there is garbage data
            if (rgbColor >= edges.size())
            {
                DBG("Garbage data. Ignoring mat");
                return std::vector<int>();
            }

            std::map<int, int>::iterator it = edgesCountInView.find(rgbColor);
            if (it == edgesCountInView.end())
                edgesCountInView[rgbColor] = 1;
            else
                it->second++;
        }
    }

    std::vector<int> edgesInView;
    for (auto edge : edgesCountInView)
    {
        if (edge.second > (int)minEdgePixels)
            edgesInView.push_back(edge.first);
    }

    DBG_T("Number of edges in view [" << edgesInView.size() << "]");
    return edgesInView;
}

std::vector<int> findEdgesInView(const std::vector<int> &vertexesIdxInView, bool bothVertexesInView)
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
*/

    DBG_T("Edges in view size: " << edgesInView.size());
    return edgesInView;
}

void drawEdgesInViewAfterProjection()
{
    DBG_T("Entered");

    std::vector<int> edgesInView = findEdgesInView(vertexesIdxInView, true);
    for (int i : edgesInView)
    {
        std::pair<int, int> edge = edges[i];
        std::vector<trimesh::point> edgeVertices = { themesh->vertices[edge.first], themesh->vertices[edge.second] };

        std::vector<cv::Point> p2d = projectCvPoints(conv3dPointTrimeshToOpenCV(edgeVertices), projMat);
        cv::line(cvShowPhoto, p2d[0], p2d[1], blue, 2);
    }

    DBG_T("Done");
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
    DBG_T("Entered");

    std::vector<int> edgesInView = findEdgesInView(vertexesIdxInView, true);
    for (int i : edgesInView)
    {
        std::pair<int, int> edge = edges[i];
        std::vector<trimesh::point> edgeVertices = { themesh->vertices[edge.first], themesh->vertices[edge.second] };

        std::vector<cv::Point2f> projectedLine;
        cv::projectPoints(conv3dPointTrimeshToOpenCV(edgeVertices), rvec, tvec, intrinsicMatrix, cv::noArray(),
            projectedLine);
        cv::line(cvShowPhoto, projectedLine[0], projectedLine[1], green, 2);
    }

    DBG_T("Done");
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
    cv::updateWindow(FACES_WINDOW_NAME);
    //cv::updateWindow(VERTEX_WINDOW_NAME);
    //updateCvWindow();

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

// Angles should be in degrees (Negative pitch is UP)
trimesh::xform getXfFromPose(const SamplePose &ps)
{
    if (ps.roll)
        throw std::invalid_argument("Z and Roll are not supported");

    // Determine the x,y position
    trimesh::xform xyzXf = trimesh::xform::trans(-ps.x, ps.y, -ps.z) *
        trimesh::xform::trans(-themesh->bsphere.center[0], -themesh->bsphere.center[1], 0);
    //DBG("xyzXf:\n" << xyzXf);

    // Base rotation so we will be looking horizontally
    xyzXf = getCamRotMatDeg(0.0, -90, 0.0) * xyzXf;
    //DBG("base rot:\n" << xyzXf);

    trimesh::xform yawXf = getCamRotMatDeg(ps.yaw, 0.0, 0.0);
    trimesh::xform pitchXf = getCamRotMatDeg(0.0, ps.pitch, 0.0);
    //DBG("yaw and pitch:\n" << yawXf << ", pitchXf:\n" << pitchXf);

    trimesh::xform fullXf = pitchXf * yawXf * xyzXf;

    //DBG("SamplePose " << ps);
    //DBG("fullXf:\n" << fullXf);

    return fullXf;
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
    DBG("bbox center: " << themesh->bbox.center() << ", bbox radius: " << themesh->bbox.radius() <<
        ", min: " << themesh->bbox.min << ", max: " << themesh->bbox.max);
    themesh->need_normals();
    themesh->need_curvatures();
    themesh->need_dcurv();
    themesh->need_faces();
    themesh->need_across_edge();

    populateModelEdges(135);

    DBG("Mesh file [" << fileName << "] loaded");
    DBG("Vertices num [" << themesh->vertices.size() << "], faces num [" << themesh->faces.size() <<
        "], tstrips num [" << themesh->tstrips.size() << "], normals num [" << themesh->normals.size() << "]");


    if (FLAGS_pose.empty())
    {
        // Generate xf file name from fileName
        xfFileName = trimesh::xfname(fileName);
        DBG("xfFileName: " << xfFileName);
        resetView();
    }
    else
    {
        xf = getXfFromPose(SamplePose(FLAGS_pose));
    }
}

void loadModelMap(const std::string &modelFile)
{
    std::string prefix = getFileNameNoExt(modelFile);

    std::string mapFileName =  prefix + "_map.png";
    gModelMap = cv::imread(mapFileName);
    if (gModelMap.empty())
    {
        throw std::runtime_error("Failed loading map");
        std::cout << "Failed loading map " << mapFileName << std::cout;
        return;
    }

    gSamplesMap = gModelMap.clone();
    cvtColor(gModelMap, gModelMap, CV_BGR2GRAY); // Perform gray scale conversion
    gModelMap = gModelMap.setTo(255, gModelMap > 0);
    DBG("Model map loaded. size " << gModelMap.size() << ", channels [" << gModelMap.channels() << "]");
    //imshowAndWait(gModelMap, 0);

    std::string groundLevelFileName = prefix + "_ground_level_map.png";
    gGroundLevelMap = cv::imread(groundLevelFileName);
    if (gGroundLevelMap.empty())
    {
        std::cout << "Failed loading groud-level map " << groundLevelFileName << std::cout;
        return;
    }

    cvtColor(gGroundLevelMap, gGroundLevelMap, CV_BGR2GRAY); // Perform gray scale conversion
    DBG("Ground-level map loaded. size " << gGroundLevelMap.size() << ", channels [" << gGroundLevelMap.channels() <<
        "]");
    //imshowAndWait(gGroundLevelMap, 0);

    if (gModelMap.size() != gGroundLevelMap.size()) // Sanity
        throw std::runtime_error("gModelMap.size() != gGroundLevelMap.size()");

    std::string orthoDataFileName = prefix + "_map_ortho_data.txt";
    gOrthoProjData.read(orthoDataFileName);
    DBG("Model Ortho Projection data " << gOrthoProjData);
}

void handleMenuKeyboard(int key)
{
    //DBG("key [" << key << "], char [" << (char)key << "]");

    switch (key)
    {
    case 's':
        showMap = !showMap;
        DBG("Toggle show map is now: " << showMap);
        break;

    //case 'c':
    //    std::cout << "Toggle faces-color (black/white)" << std::endl;
    //    facesColorGL = facesColorGL == whiteGL ? blackGL : whiteGL;
    //    break;

    case 'S':
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

void setAutoNavState(bool newState)
{
    gAutoNavigation = newState;
    DBG("gAutoNavigation: " << gAutoNavigation);
}

void handleNavKeyboard(int key)
{
    if (gAutoNavigation && key != 'g')
    {
        DBG("Nav commands not allowd in autoNavigation mode. Press 'g' to exit this mode");
        return;
    }

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

    case 'g':
        setAutoNavState(!gAutoNavigation);
        break;

    case 'i':
        gAutoNavigationIdx++;
        DBG("Up gAutoNavigationIdx [" << gAutoNavigationIdx << "]");
        break;
    case 'I':
        gAutoNavigationIdx--;
        DBG("Down gAutoNavigationIdx [" << gAutoNavigationIdx << "]");
        break;
    case 'o':
        {
            xf = gDataSet["train"].xfSamples[gAutoNavigationIdx];

            DBG("Jumping to train sample [#" << gAutoNavigationIdx << "], pose [" <<
                gDataSet["train"].samplesData[gAutoNavigationIdx] << "], xf\n" << xf);
        }
        break;

    case 'y':
        {
            std::cout << "Enter Yaw in degrees: ";
            float yaw = 0;
            std::cin >> yaw;
            xf = getCamRotMatDeg(yaw, 0.0, 0.0) * xf;
        }
        break;
    case 'p':
        {
            std::cout << "Enter Pitch in degrees: ";
            float pitch = 0;
            std::cin >> pitch;
            xf = getCamRotMatDeg(0.0, pitch, 0.0) * xf;
        }
        break;

    case '+':
    case '=':
        xyzStep *= 2;
        DBG("Increased xyzStep to " << xyzStep);
        break;
    case '-':
        xyzStep /= 2;
        DBG("Decreased xyzStep to " << xyzStep);
        break;
        // Camera Position - XY
    case KEY_LEFT:
        xf = trimesh::xform::trans(xyzStep, 0, 0) * xf;
        break;
    case KEY_RIGHT:
        xf = trimesh::xform::trans(-xyzStep, 0, 0) * xf;
        break;

    case KEY_UP:
        xf = trimesh::xform::trans(0, -xyzStep, 0) * xf;
        break;
    case KEY_DOWN:
        xf = trimesh::xform::trans(0, xyzStep, 0) * xf;
        break;

        // Camera Position - Z - Zoom
    case ']':
        xf = trimesh::xform::trans(0, 0, xyzStep) * xf;
        break;
    case '[':
        xf = trimesh::xform::trans(0, 0, -xyzStep) * xf;
        break;

        // Camera Angle - Roll
    case 'e':
        xf = getCamRotMatRad(0.0, 0.0, rotStep) * xf;
        break;
    case 'q':
        xf = getCamRotMatRad(0.0, 0.0, -rotStep) * xf;
        break;

        // Camera Angle - Yaw
    case 'a':
        xf = getCamRotMatRad(-rotStep, 0.0, 0.0) * xf;
        break;
    case 'd':
        xf = getCamRotMatRad(rotStep, 0.0, 0.0) * xf;
        break;

        // Camera Angle - Pitch
    case 'w':
        xf = getCamRotMatRad(0.0, -rotStep, 0.0) * xf;
        break;
    case 's':
        xf = getCamRotMatRad(0.0, rotStep, 0.0) * xf;
        break;

    default:
        //DBG("Unhandled key: [" << key << "]");
        break;
    }
}

void saveMarkedPoints()
{
    std::ofstream outputFile(markedPtsFile);
    if (!outputFile)
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
    if ((char)key == -1)
        return;

#if __linux__
    key &= 0xff;
#endif

    //DBG("key [" << key << "], char [" << (char)key << "]");

    switch (key)
    {
    case 'n':
        gKeysGroup = KEYS_GROUP_NAV;
        cv::setMouseCallback(MAIN_WINDOW_NAME, mouseNavCallbackFunc, NULL);
        DBG("Navigation mode");
        return;

    case 'm':
        gKeysGroup = KEYS_GROUP_MENU;
        DBG("Menu mode");
        return;

    case 't':
        gKeysGroup = KEYS_GROUP_TAG;
        cv::setMouseCallback(MAIN_WINDOW_NAME, mouseTagCallbackFunc3D, NULL);
        DBG("Tagging mode");
        return;
    }

    switch (gKeysGroup)
    {
    case KEYS_GROUP_NAV:
        handleNavKeyboard(key);
        break;

    case KEYS_GROUP_MENU:
        handleMenuKeyboard(key);
        break;

    case KEYS_GROUP_TAG:
        handleTagKeyboard(key);
        break;
    }
}

void checkPointAndSetZ(const cv::Point3f &p, std::vector<cv::Point3f> &samplePoints, cv::Mat &colorMap,
    const cv::Scalar &goodColor)
{
    cv::Point mapPoint = gOrthoProjData.convertWorldPointToMap(p, gModelMap.size());

    if (mapPoint.x < 0 || mapPoint.x >= gModelMap.cols || mapPoint.y < 0 || mapPoint.y >= gModelMap.rows)
    {
        DBG_T("P is out of map area. Skipping");
        return;
    }

    cv::Scalar color;
    if (gModelMap.at<char>(mapPoint.y, mapPoint.x))
    {
        DBG_T("Position- model(x,y), image(x,y): " << p << ", " << mapPoint << " is not free");
        color = red;
    }
    else if (gGroundLevelMap.at<char>(mapPoint.y, mapPoint.x) == 255)
    {
        DBG_T("Position- model(x,y), image(x,y): " << p << ", " << mapPoint << " unknown ground level");
        color = orange;
    }
    else
    {
        cv::Point3f pWithZ(p.x, p.y, gGroundLevelMap.at<char>(mapPoint.y, mapPoint.x) + FLAGS_camera_height);
        samplePoints.push_back(pWithZ);
        color = goodColor;
    }

    cv::circle(colorMap, mapPoint, 3, color, CV_FILLED);
}

void fillEachPointPoses(DataSet &dataSet)
{
    float rollDeg = 0; // Currently no roll is added
#if 1
    for (const cv::Point3f &p : dataSet.samplePoints)
    {
        int anglesPerXY = 0;
        for (float yawDeg = 0; yawDeg < 360; yawDeg += 5)
        {
            // Negative pitch values are UP
            for (float pitchDeg = 0;  pitchDeg >= -30; pitchDeg -= 5)
            {
                SamplePose pose(p.x, p.y, p.z, yawDeg, pitchDeg, rollDeg);

                trimesh::xform fullXf = getXfFromPose(pose);

                dataSet.xfSamples.push_back(fullXf);
                dataSet.samplesData.push_back(pose);

                if (dataSet.samplesData.size() % 10000 == 0)
                    DBG("Pose [#" << dataSet.samplesData.size() - 1 << "] example [" << pose << "], xf\n" << xf);
                anglesPerXY++;
            }
        }

        //DBG("anglesPerXY [" << anglesPerXY << "]");
    }
#else // No angles - good for debugging
    for (const cv::Point3f &p : dataSet.samplePoints)
    {
        SamplePose pose(p.x, p.y, p.z, 0, 0, rollDeg);

        trimesh::xform fullXf = getXfFromPose(pose);

        dataSet.xfSamples.push_back(fullXf);
        dataSet.samplesData.push_back(pose);
    }
#endif
}

void populateXfVector()
{
    gDataSet["train"] = DataSet();
    gDataSet["test"] = DataSet();

//#define ROUND_SAMPLING
#ifdef ROUND_SAMPLING
    float rad = themesh->bsphere.r;
    //trimesh::vec center = themesh->bsphere.center;

    // Make the radius a bit bigger so we would catch the model from also "outside"
    // This should probably be a function of the max height and the FOV or a const.
    // Should try (rad = rad + -3.5f / fov * maxOrAvgBuildingHeight)
    rad *= 0.8;

    float xMin = -rad;
    float xMax = rad;
    float yMin = -rad;
    float yMax = rad;
#else

    float width = themesh->bbox.max[0] - themesh->bbox.min[0];
    float height = themesh->bbox.max[1] - themesh->bbox.min[1];

    float xMin = -width/2;
    float xMax = width/2;
    float yMin = -height/2;
    float yMax = height/2;

    // Sampling area correction - probably due to model symmetry around
    if (getFileName(FLAGS_model) == "berlin.obj")
    {
        yMin += height / 8;
        yMax += height / 8;
    }
    else if (getFileName(FLAGS_model) == "cube_layout.obj")
    {
        xMax += height / 20;
    }

#endif
    DBG("xMin [" << xMin << "], xMax [" << xMax << "], yMin [" << yMin << "], yMax [" << yMax << "]");

    float groundZ = 0; // TODO: Take the value from the model min Z or calc min for the neighborhood

    if (FLAGS_random_sampling)
    {
        std::random_device rd;  // Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()

        // Use dis to transform the random unsigned int generated by gen into a double in [1, 2)
        // Each call to dis(gen) generates a new random double
        std::uniform_real_distribution<> disX(xMin, xMax);
        std::uniform_real_distribution<> disY(yMin, yMax);

        // Generate test random points
        while (gDataSet["test"].samplePoints.size() < FLAGS_samples_num * FLAGS_test_percent)
        {
            cv::Point3f newPoint(disX(gen), disY(gen), groundZ);
            checkPointAndSetZ(newPoint, gDataSet["test"].samplePoints, gSamplesMap, blue);
        }

        // Generate train random points
        while (gDataSet["train"].samplePoints.size() < FLAGS_samples_num * (1 - FLAGS_test_percent))
        {
            cv::Point3f newPoint(disX(gen), disY(gen), groundZ);
            checkPointAndSetZ(newPoint, gDataSet["train"].samplePoints, gSamplesMap, green);
        }
    }
    else
    {

#ifdef ROUND_SAMPLING
        float stepX = rad * 2 / FLAGS_grid;
        float stepY = rad * 2 / FLAGS_grid;
#else
        float stepX = width / FLAGS_grid;
        float stepY = height / FLAGS_grid;
#endif
        DBG("stepX [" << stepX << "], stepY [" << stepY << "]");

        for (float x = xMin; x < xMax; x += stepX)
            for (float y = yMin; y < yMax; y += stepY)
                checkPointAndSetZ(cv::Point3f(x, y, groundZ), gDataSet["train"].samplePoints, gSamplesMap, green);

        float offsetX = stepX * FLAGS_grid_test_offset;
        float offsetY = stepY * FLAGS_grid_test_offset;
        for (float x = xMin + offsetX; x < xMax - offsetX; x += stepX)
            for (float y = yMin + offsetY; y < yMax - offsetY; y += stepY)
                checkPointAndSetZ(cv::Point3f(x, y, groundZ), gDataSet["test"].samplePoints, gSamplesMap, blue);
    }

    std::string samplesPositionsMapFilePath = FLAGS_output_dir + "gSamplesMap.png";
    cv::imwrite("lastSamplesPositionsMap.png", gSamplesMap);
    if (!cv::imwrite(samplesPositionsMapFilePath, gSamplesMap))
        std::cout << "Failed saving gSamplesMap to [" << samplesPositionsMapFilePath << "]" << std::endl;

    DBG("training points [" << gDataSet["train"].samplePoints.size() << "], testing points [" <<
        gDataSet["test"].samplePoints.size() << "]");

    fillEachPointPoses(gDataSet["train"]);
    fillEachPointPoses(gDataSet["test"]);

    DBG("Training-set size [" << gDataSet["train"].xfSamples.size() << "], Testing-set size [" <<
        gDataSet["test"].xfSamples.size() << "]");
}

std::string projectionMatToStr(const trimesh::xform &xf)
{
    DBG(xf);
    std::stringstream ss;

    ss << std::setprecision(17);
    for (auto i : xf)
        ss << "_" << i;

    DBG(ss.str());
    return ss.str();
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

cv::Mat convertImgToBinary(const cv::Mat &img)
{
    cv::Mat binaryImg;
    cvtColor(img, binaryImg, CV_BGR2GRAY);
    cv::bitwise_not(binaryImg, binaryImg);
    binaryImg = binaryImg > 0;
    return binaryImg;
}

// Returns false if image does not have enough data/edges or imwrite fails
bool checkAndSaveCurrentSceen(const std::string &filePath)
{
    DBG_T("Entered");

    // Sanity
    // Notice this should not be taken too lightly, since we color each edge in its index number.
    // I.e. 1st one is BLACK. Therefore using BLACK background means the 1st edge will be lost.
    if (backgroundColorGL != whiteGL)
        throw std::runtime_error("This function is written for white background");

    cv::Mat img = getCvMatFromScreen(MAIN_WINDOW_NAME);
    cv::Mat facesImg = getCvMatFromScreen(FACES_WINDOW_NAME);

    // Sanity
    cv::Size winSize(FLAGS_win_width, FLAGS_win_height);
    if (img.size() != winSize || facesImg.size() != winSize)
    {
        DBG("img Size != winSize [" << img.size() << ", " << winSize << "] OR facesImg Size != winSize [" <<
            facesImg.size() << ", " << winSize << "]");
        throw std::runtime_error("Unexpected image size");
    }

    if (FLAGS_crop_upper_part)
    {
        cv::Rect upperPart(0, 0, img.cols, std::round(img.rows * FLAGS_crop_upper_part));
        img = img(upperPart);
        facesImg = facesImg(upperPart);
    }

    std::vector<int> edgesInView = findEdgesInView(img, FLAGS_min_edge_pixels);
    if ((int)edgesInView.size() < FLAGS_min_edges_threshold)
    {
        DBG_T("Skipping [" << filePath << "]. Edges [" << edgesInView.size() << "/" << FLAGS_min_edges_threshold<< "]");

        //imshowAndWait(img);
        return false;
    }

    if (FLAGS_min_data_threshold) // This condition is to save the countNonZero() when min_data_threshold is not used
    {
        cv::Mat invImg;
        cvtColor(img, invImg, CV_BGR2GRAY); // Perform gray scale conversion
        cv::bitwise_not(invImg, invImg);

        int nonZeroPixelsCount = cv::countNonZero(invImg);
        float imageContentPrecentage = nonZeroPixelsCount / float(img.cols * img.rows);
        if (imageContentPrecentage < FLAGS_min_data_threshold)
        {
            DBG_T("Skipping [" << filePath << "]. Pixel count [" << nonZeroPixelsCount << "], Data part [" <<
                imageContentPrecentage << "/" << FLAGS_min_data_threshold  << "]");
            return false;
        }
    }

    if (!cv::imwrite(filePath + "_edges.png", img))
    {
        std::cout << "Failed saving file " << filePath << std::cout;
        return false;
    }

    if (!cv::imwrite(filePath + "_faces.png", facesImg))
    {
        std::cout << "Failed saving file " << filePath << std::cout;
        return false;
    }

#if 0
    cv::Mat blurImg = convertImgToBinary(img);
    cv::GaussianBlur(blurImg, blurImg, cv::Size(FLAGS_kernel_size, FLAGS_kernel_size), 0);

    cv::Mat facesImgBinary = convertImgToBinary(facesImg);
    cv::bitwise_and(blurImg, facesImgBinary, blurImg);
    if (!cv::imwrite(filePath + "_blur.png", blurImg))
    {
        std::cout << "Failed saving file " << filePath << std::cout;
        return false;
    }
#endif

    DBG_T("Done");
    return true;
}

void saveCheckPoint(const std::string &modelFileName, const std::string &outDir, bool isTrain,
    unsigned autoNavigationIdx, int exportsNum, int skipsNum)
{
    std::string checkPointFile("check_point.txt", std::ofstream::trunc);
    std::ofstream outputFile(checkPointFile);
    if (!outputFile.is_open())
    {
        DBG("Failed opening " << checkPointFile);
        return;
    }

    outputFile << modelFileName << "\n";
    outputFile << outDir << "\n";
    outputFile << (isTrain ? "train" : "test") << "\n";
    outputFile << autoNavigationIdx << ", " << exportsNum << ", " << skipsNum << "\n";

    DBG("Done. checkPointFile [" << checkPointFile << "]");
}

void loadCheckPoint(const std::string &model_file_name, std::string &outDir, bool &isTrain,
    unsigned &autoNavigationIdx, int &exportsNum, int &skipsNum)
{
    DBG("Loading check point");

    std::string checkPointFile("check_point.txt", std::ofstream::trunc);
    std::ifstream inputFile(checkPointFile);
    if (!inputFile.is_open())
    {
        DBG("Failed opening " << checkPointFile);
        throw std::runtime_error("Failed loading check point file");
    }

    std::string line;

    std::getline(inputFile, line);
    if (line != model_file_name)
        throw std::runtime_error("Trying to resume export of different model " + line + " Vs. " + model_file_name);

    std::getline(inputFile, line);
    outDir = line;

    std::getline(inputFile, line);
    if (line == "train")
        isTrain = true;
    else if (line == "test")
        isTrain = false;
    else
        throw std::runtime_error("Resume export exception. Unknown data-set [" + line + "]");

    std::getline(inputFile, line);
    sscanf(line.c_str(), "%u, %d, %d", &autoNavigationIdx, &exportsNum, &skipsNum);

    DBG("Done loading check point data. set [" << (isTrain ? "Train" : "Test") << "], autoNavigationIdx [" <<
        autoNavigationIdx << "], exportsNum [" << exportsNum << "], skipsNum [" << skipsNum << "]");
}

std::string loadOutputDirFromCheckPoint()
{
    bool dummyBool;
    unsigned dummyUnsigned;
    int dummyInt;
    std::string outDir;
    loadCheckPoint(getFileName(FLAGS_model), outDir, dummyBool, dummyUnsigned, dummyInt, dummyInt);
    DBG("outDir [" << outDir << "]");
    return outDir;
}

void autoNavigate()
{
    static bool sIsTrain = true;
    static int sExportsNum = 0;
    static int sSkipsNum = 0;

    if (FLAGS_resume_export)
    {
        std::string dummy;
        loadCheckPoint(getFileName(FLAGS_model), dummy, sIsTrain, gAutoNavigationIdx, sExportsNum, sSkipsNum);
        FLAGS_resume_export = false;

        // Pass sanity chech - few lines further...
        std::vector<trimesh::xform> &xfSamples = sIsTrain ? gDataSet["train"].xfSamples : gDataSet["test"].xfSamples;
        xf = xfSamples[gAutoNavigationIdx - 1];
        return; // Render this xf to screen
    }

    DBG_T("Entered");

    std::vector<trimesh::xform> &xfSamples = sIsTrain ? gDataSet["train"].xfSamples : gDataSet["test"].xfSamples;
    std::vector<SamplePose> &samplesData = sIsTrain ? gDataSet["train"].samplesData : gDataSet["test"].samplesData;

    if (!gAutoNavigationIdx)
    {
        makeDir(FLAGS_output_dir + FS_DELIMITER_LINUX + (sIsTrain ? "train" : "test"));
    }
    else
    {
        // Skip 1st entry here since it is the pose before starting autoNavigation
        // FIXME: The last image will not be taken. Probably not worth fixing.

        // Take screen-shot of current pose and save the XF file in identical fileName
        std::stringstream ssDirName;
        ssDirName << FLAGS_output_dir << (sIsTrain ? "train" : "test") << FS_DELIMITER_LINUX <<
            samplesData[gAutoNavigationIdx - 1].x << "_" << samplesData[gAutoNavigationIdx - 1].y << FS_DELIMITER_LINUX;
        makeDir(ssDirName.str());

        std::stringstream ssNum;
        ssNum << std::setfill('0') << std::setw(6) << gAutoNavigationIdx - 1;
        std::string sampleFilePathNoExt = ssDirName.str() + "pose_" + ssNum.str();

        if (checkAndSaveCurrentSceen(sampleFilePathNoExt))
        {
            // Image is OK and was saved. Write also image data
            samplesData[gAutoNavigationIdx - 1].write(sampleFilePathNoExt + ".txt");

            xf.write(sampleFilePathNoExt + ".xf");

            sExportsNum++;
        }
        else
        {
            sSkipsNum++;
        }

        // Sanity
        if (xf != xfSamples[gAutoNavigationIdx - 1])
        {
            throw std::runtime_error("Somehow the XF was modified before we returned here. This might suggest the "
                "rendered image is not we think");
        }
    }

    xf = xfSamples[gAutoNavigationIdx];
    //DBG("xf:\n" << xf);

    if (FLAGS_debug || gAutoNavigationIdx % 1000 == 0)
    {
        DBG("sample [#" << gAutoNavigationIdx << "/" << samplesData.size() << "], dataSet [" <<
            (sIsTrain ? "Train" : "Test") << "], sExportsNum [" << sExportsNum << "], sSkipsNum [" << sSkipsNum <<
            "], samplesData [" << samplesData[gAutoNavigationIdx] << "]");

        saveCheckPoint(getFileName(FLAGS_model), FLAGS_output_dir, sIsTrain, gAutoNavigationIdx, sExportsNum,
            sSkipsNum);
    }

    gAutoNavigationIdx++;
    if (gAutoNavigationIdx == xfSamples.size())
    {
        DBG("Done auto navigating over [" << (sIsTrain ? "train" : "test") << "] set. Projections num [" <<
            xfSamples.size() << "], sExportsNum [" << sExportsNum << "], sSkipsNum [" << sSkipsNum << "]");

        if (sIsTrain)
        {
            sIsTrain = false;
        }
        else // test set is also done
        {
            setAutoNavState(false);
        }

        gAutoNavigationIdx = 0;
        sExportsNum = 0;
        sSkipsNum = 0;
    }

    DBG_T("Done");
}

// All outputs are in "sessions_outputs" directory
void handleOutputDir()
{
    std::string exeFilePath = getRunningExeFilePath();

    if (FLAGS_output_dir.empty())
        FLAGS_output_dir = getFileName(exeFilePath) + "_" + getTimeStamp();

    std::string sessionsOutputDir = getDirName(exeFilePath) + FS_DELIMITER_LINUX + "sessions_outputs" +
        FS_DELIMITER_LINUX;

    FLAGS_output_dir = sessionsOutputDir + FLAGS_output_dir + FS_DELIMITER_LINUX;

    std::cout << "exeFilePath ["<< exeFilePath << "], sessionsOutputDir [" << sessionsOutputDir <<
        "], FLAGS_output_dir [" << FLAGS_output_dir << "]\n";

    makeDir(sessionsOutputDir);
    makeDir(FLAGS_output_dir);
}

// Initialize Google's logging library
void initGoogleLogs(const std::string &appName)
{
    google::SetLogDestination(google::INFO, (FLAGS_output_dir + "log_").c_str());
    google::SetLogDestination(google::WARNING, "");
    google::SetLogDestination(google::ERROR, "");
    google::SetLogDestination(google::FATAL, "");

    google::InitGoogleLogging(appName.c_str());
}

void updateUpperMapShow()
{
    gUpperMapShow = gSamplesMap.clone();

    // https://www.opengl.org/discussion_boards/showthread.php/178484-Extracting-camera-position-from-a-ModelView-Matrix
    trimesh::xform xfInv = inv(xf);
    cv::Point3f curXFxyz(xfInv[12] - themesh->bsphere.center[0], themesh->bsphere.center[1] - xfInv[13], xfInv[14]);
    //DBG("xf\n" << xf);
    //DBG("xfInv\n" << xfInv);

    cv::circle(gUpperMapShow,
        gOrthoProjData.convertWorldPointToMap(curXFxyz, gModelMap.size()),
        (gModelMap.cols/ FLAGS_win_width) * 8, azul, CV_FILLED);

    cv::resize(gUpperMapShow, gUpperMapShow, cv::Size(FLAGS_win_width, FLAGS_win_height));

    std::stringstream ss;
    ss << "curXFxyz " << curXFxyz << ",\nxfInv:\n" << xfInv << "\nCenter " << themesh->bsphere.center;
    myPutText(gUpperMapShow, ss, cv::Point(5, 10));

    cv::imshow(UPPER_MAP_WINDOW_NAME, gUpperMapShow);
}

int main(int argc, char* argv[])
{
    std::string usage = std::string(argv[0]) + " [model_file] [flags]";

    GFLAGS_NAMESPACE::SetUsageMessage(usage);
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    handleOutputDir();
    if (FLAGS_resume_export)
        FLAGS_output_dir = loadOutputDirFromCheckPoint();
    initGoogleLogs(argv[0]);
    if (argc == 2)
    {
        FLAGS_model = argv[1];
    }
    else if (argc > 2)
    {
        std::cerr << "Extra arguments" << std::endl;
        std::cerr << "Usage: " << argv[0] << usage << std::endl;
        exit(1);
    }

    DBG_T("Entered. exe [" << argv[0] << "], current gFlags [" << GFLAGS_NAMESPACE::CommandlineFlagsIntoString() <<
        "]");

    //std::string imageFile, mapFile;
    //imageFile = "../samples/cube_photo.png";
    //imageFile = "data/inter_03.png";
    //imageFile = "data/model_image4.png";
    //mapFile = "data/berlin_google_map.png";
    //markedPtsFile = "data/markedPoints.txt";

    loadModel(FLAGS_model);
    loadModelMap(FLAGS_model);

    initWindow(MAIN_WINDOW_NAME, FLAGS_win_width, FLAGS_win_height, redraw);
    cv::setMouseCallback(MAIN_WINDOW_NAME, mouseNavCallbackFunc, NULL);
    initWindow(FACES_WINDOW_NAME, FLAGS_win_width, FLAGS_win_height, redrawFaces);
    //initWindow(VERTEX_WINDOW_NAME, FLAGS_win_width, FLAGS_win_height, redrawVertex);
    //cv::setMouseCallback(VERTEX_WINDOW_NAME, mouseNavCallbackFunc, NULL);

    // Init cvPhoto window
    //cvPhoto = cv::imread(imageFile);
    //cvMap = cv::imread(mapFile);
    //xfMap = trimesh::xform::trans(0, 0, -3.5f / fov * themesh->bsphere.r) *
      //  trimesh::xform::trans(-themesh->bsphere.center);
    //initWindow(CV_WINDOW_NAME, FLAGS_win_width, FLAGS_win_height, redrawPhoto);
    //cv::setMouseCallback(CV_WINDOW_NAME, mouseTagCallbackFunc2D, NULL);

    if (FLAGS_pose.empty())
        populateXfVector();

    updateWindows();
    verifySize(MAIN_WINDOW_NAME);
    verifySize(FACES_WINDOW_NAME);
    //verifySize(VERTEX_WINDOW_NAME);

    for (;;)
    {
        updateUpperMapShow();
        updateWindows();

        int key = cv::waitKey(gAutoNavigation ? 5 : 0); //33);
        if (key == 27) // ESC key
            break;

        handleKeyboard(key);

        if (gAutoNavigation)
            autoNavigate();
    }

    cv::setOpenGlDrawCallback(MAIN_WINDOW_NAME, 0, 0);
    cv::setOpenGlDrawCallback(FACES_WINDOW_NAME, 0, 0);
    //cv::setOpenGlDrawCallback(VERTEX_WINDOW_NAME, 0, 0);
    //cv::setOpenGlDrawCallback(CV_WINDOW_NAME, 0, 0);
    cv::destroyAllWindows();

    return 0;
}

