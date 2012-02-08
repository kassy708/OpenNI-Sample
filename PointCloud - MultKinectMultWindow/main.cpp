/**
	author @kassy708
	マルチキネクトとOpenGLによるポイントクラウド

*/
#include <iostream>
#include <stdexcept>
#include <map>
#include <sstream>

#include <GL/glut.h>
#include <opencv2/opencv.hpp>
#ifdef _DEBUG
    //Debugモードの場合
    #pragma comment(lib,"C:\\OpenCV2.2\\lib\\opencv_core220d.lib")            // opencv_core
    #pragma comment(lib,"C:\\OpenCV2.2\\lib\\opencv_imgproc220d.lib")        // opencv_imgproc
    #pragma comment(lib,"C:\\OpenCV2.2\\lib\\opencv_highgui220d.lib")        // opencv_highgui
#else
    //Releaseモードの場合
    #pragma comment(lib,"C:\\OpenCV2.2\\lib\\opencv_core220.lib")            // opencv_core
    #pragma comment(lib,"C:\\OpenCV2.2\\lib\\opencv_imgproc220.lib")        // opencv_imgproc
    #pragma comment(lib,"C:\\OpenCV2.2\\lib\\opencv_highgui220.lib")        // opencv_highgui
#endif

#include <XnCppWrapper.h>
#pragma comment(lib,"C:/Program files/OpenNI/Lib/openNI.lib")

#define SAMPLE_XML_PATH "C:/Program Files/OpenNI/Data/SamplesConfig.xml"
using namespace cv;
using namespace xn;

//openNIのための宣言・定義
//マクロ定義
#define KINECT_IMAGE_WIDTH 640
#define KINECT_IMAGE_HEGIHT 480
#define KINECT_DEPTH_WIDTH 640
#define KINECT_DEPTH_HEGIHT 480

Mat pointCloud_XYZ( KINECT_DEPTH_HEGIHT, KINECT_DEPTH_WIDTH, CV_32FC3);


void retrievePointCloudMap(Mat &depth,Mat &pointCloud_XYZ,DepthGenerator depthGenerator);    //3次元ポイントクラウドのための座標変換
void drawPointCloud(Mat &rgbImage,Mat &pointCloud_XYZ);        //ポイントクラウド描画

//openGLのための宣言・定義
//---変数宣言---
int FormWidth = 640;
int FormHeight = 480;
int mButton;
float twist, elevation, azimuth;
float cameraDistance = 0,cameraX = 0,cameraY = 0;
int xBegin, yBegin;
//---マクロ定義---
#define glFovy 45        //視角度
#define glZNear 1.0        //near面の距離
#define glZFar 150.0    //far面の距離
void polarview();        //視点変更


const XnMapOutputMode OUTPUT_MODE = { 640, 480, 30 };

class WindowData{
public:
	int windowID;
	int formWidth;
	int formHeight;
	int mButton;
	float twist, elevation, azimuth;
	float cameraDistance,cameraX,cameraY;
	int xBegin, yBegin;
	WindowData(){
		formWidth = 640;
		formHeight = 480;
		cameraDistance = 0,cameraX = 0,cameraY = 0;
	}
	//視点変更
	void polarview(){
		glTranslatef( cameraX, cameraY, cameraDistance);
		glRotatef( -twist, 0.0, 0.0, 1.0);
		glRotatef( -elevation, 1.0, 0.0, 0.0);
		glRotatef( -azimuth, 0.0, 1.0, 0.0);
	}
};
// Kinectごとの表示情報
struct Kinect{
	ImageGenerator  imageGenerator;
	DepthGenerator  depthGenerator;
	WindowData windowData;
};
std::map<int, Kinect> kinect;
NodeInfoList nodeList;


Context context;
ImageMetaData imageMD;
DepthMetaData depthMD;

// 検出されたデバイスを列挙する
void EnumerateProductionTrees(xn::Context& context, XnProductionNodeType type){
	xn::NodeInfoList nodes;
	XnStatus rc = context.EnumerateProductionTrees(type, NULL, nodes);
	if (rc != XN_STATUS_OK) {
		throw std::runtime_error(::xnGetStatusString(rc));
	}
	else if (nodes.Begin() == nodes.End()) {
		throw std::runtime_error("No devices found.");
	}
  
	for (xn::NodeInfoList::Iterator it = nodes.Begin (); it != nodes.End (); ++it) {
		std::cout <<
		::xnProductionNodeTypeToString( (*it).GetDescription().Type ) <<
		", " <<
		(*it).GetCreationInfo() << ", " <<
		(*it).GetInstanceName() << ", " <<
		(*it).GetDescription().strName << ", " <<
		(*it).GetDescription().strVendor << ", " <<
		std::endl;
    
		xn::NodeInfo info = *it;
		context.CreateProductionTree(info);
	}
}

// ジェネレータを作成する
template<typename T>
T CreateGenerator(const xn::NodeInfo& node)
{
	T g;
	XnStatus rc = node.GetInstance(g);
	if (rc != XN_STATUS_OK) {
	throw std::runtime_error(xnGetStatusString(rc));
	}
  
	g.SetMapOutputMode(OUTPUT_MODE);
  
	return g;
}

void showOepnNIVersion()
{
	XnVersion version = { 0 };
	XnStatus rc = xnGetVersion( &version );
	if (rc != XN_STATUS_OK) {
		throw std::runtime_error(::xnGetStatusString(rc));
	}

	std::cout << "OpenNI Version is " << 
		(XnUInt32)version.nMajor << "." << 
		(XnUInt32)version.nMinor << "." << 
		(XnUInt32)version.nMaintenance << "." << 
		(XnUInt32)version.nBuild << std::endl;
}


//初期化
void init(){
	try {
		showOepnNIVersion();

		XnStatus rc;
    
		rc = context.Init();
		if (rc != XN_STATUS_OK) {
			throw std::runtime_error(::xnGetStatusString(rc));
		}

		// 検出されたデバイスを利用可能として登録する
		EnumerateProductionTrees(context, XN_NODE_TYPE_DEVICE);
		EnumerateProductionTrees(context, XN_NODE_TYPE_IMAGE);
		EnumerateProductionTrees(context, XN_NODE_TYPE_DEPTH);
    
		// 登録されたデバイスを取得する
		std::cout << "xn::Context::EnumerateExistingNodes ... ";
		rc = context.EnumerateExistingNodes( nodeList );
		if ( rc != XN_STATUS_OK ) {
			throw std::runtime_error( ::xnGetStatusString( rc ) );
		}
		std::cout << "Success" << std::endl;

		// 登録されたデバイスからジェネレータを生成する
		for ( xn::NodeInfoList::Iterator it = nodeList.Begin();it != nodeList.End(); ++it ) {
      
			// インスタンス名の最後が番号になっている
			std::string name = (*it).GetInstanceName();
			int no = *name.rbegin() - '1';
      
			if ((*it).GetDescription().Type == XN_NODE_TYPE_IMAGE) {
				kinect[no].imageGenerator = CreateGenerator<xn::ImageGenerator>(*it);
			}
			else if ((*it).GetDescription().Type == XN_NODE_TYPE_DEPTH) {
				kinect[no].depthGenerator = CreateGenerator<xn::DepthGenerator>(*it);
			}
		}

		// ジェネレートを開始する
		context.StartGeneratingAll();

		// ビューポイントの設定や、カメラ領域を作成する
		for (std::map<int, Kinect>::iterator it = kinect.begin(); it != kinect.end(); ++it) {
			int no = it->first;
			Kinect& k = it->second;
			k.depthGenerator.GetAlternativeViewPointCap().SetViewPoint(k.imageGenerator);
		}
	}
	catch (std::exception& ex) {
		std::cout << ex.what() << std::endl;
	}
}

//描画
void display(){
	
    for (std::map<int, Kinect>::iterator it = kinect.begin(); it != kinect.end(); ++it) {
        Kinect& k = it->second;
		
		glutSetWindow(k.windowData.windowID);

		// clear screen and depth buffer
		glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT ); 
		// Reset the coordinate system before modifying
		glLoadIdentity();   
		glEnable(GL_DEPTH_TEST); //「Zバッファ」を有効
		gluLookAt(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0);   //視点の向き設定
		//wait and error processing
		context.WaitAnyUpdateAll();

		k.imageGenerator.GetMetaData(imageMD);
		k.depthGenerator.GetMetaData(depthMD);
		k.depthGenerator.GetAlternativeViewPointCap().SetViewPoint(k.imageGenerator);//ズレを補正

		Mat image(480,640,CV_8UC3,(unsigned char*)imageMD.WritableData());
		Mat depth(480,640,CV_16UC1,(unsigned char*)depthMD.WritableData());
         
		memcpy(image.data,imageMD.Data(),image.step * image.rows);    //イメージデータを格納
		memcpy(depth.data,depthMD.Data(),depth.step * depth.rows);    //深度データを格納
      
		//3次元ポイントクラウドのための座標変換
		retrievePointCloudMap(depth,pointCloud_XYZ,k.depthGenerator);

		//視点の変更
		polarview();  
		//ポイントクラウド
		drawPointCloud(image,pointCloud_XYZ);
             
		//convert color space RGB2BGR
		cvtColor(image,image,CV_RGB2BGR);     
     
		imshow(k.imageGenerator.GetName(),image);
		imshow(k.depthGenerator.GetName(),depth);

		
		glFlush();
		glutSwapBuffers();
	}
  
}
// アイドル時のコールバック
void idle(){
    //再描画要求
    glutPostRedisplay();
}
//ウィンドウのサイズ変更
void reshape (int width, int height){
    FormWidth = width;
    FormHeight = height;
    glViewport (0, 0, (GLsizei)width, (GLsizei)height);
    glMatrixMode (GL_PROJECTION);
    glLoadIdentity ();
    //射影変換行列の指定
    gluPerspective (glFovy, (GLfloat)width / (GLfloat)height,glZNear,glZFar); 
    glMatrixMode (GL_MODELVIEW);
}
//マウスの動き
void motion(int x, int y){
    int xDisp, yDisp;  
    xDisp = x - xBegin;
    yDisp = y - yBegin;
    switch (mButton) {
    case GLUT_LEFT_BUTTON:
        azimuth += (float) xDisp/2.0;
        elevation -= (float) yDisp/2.0;
        break;
    case GLUT_MIDDLE_BUTTON:
        cameraX -= (float) xDisp/40.0;
        cameraY += (float) yDisp/40.0;
        break;
    case GLUT_RIGHT_BUTTON:
		cameraDistance += (float) xDisp/40.0;
        break;
    }
    xBegin = x;
    yBegin = y;
}
//マウスの操作
void mouse(int button, int state, int x, int y){
    if (state == GLUT_DOWN) {
        switch(button) {
        case GLUT_RIGHT_BUTTON:
        case GLUT_MIDDLE_BUTTON:
        case GLUT_LEFT_BUTTON:
            mButton = button;
            break;
        }
        xBegin = x;
        yBegin = y;
    }
}
//視点変更
void polarview(){
    glTranslatef( cameraX, cameraY, cameraDistance);
    glRotatef( -twist, 0.0, 0.0, 1.0);
    glRotatef( -elevation, 1.0, 0.0, 0.0);
    glRotatef( -azimuth, 0.0, 1.0, 0.0);
}
//メイン
int main(int argc, char *argv[]){
    init();

    glutInit(&argc, argv);
	
    // 検出したすべてのKinectの画像を表示する
    for (std::map<int, Kinect>::iterator it = kinect.begin(); it != kinect.end(); ++it) {
		glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);
		glutInitWindowSize(FormWidth, FormHeight);
		it->second.windowData.windowID = glutCreateWindow(argv[0]);
		//コールバック
		if(it == kinect.begin())//最初だけ登録
			glutIdleFunc(idle);
		glutReshapeFunc (reshape);
		glutDisplayFunc(display);
		glutMouseFunc(mouse);
		glutMotionFunc(motion);
	}

    glutMainLoop();
    context.Shutdown();
    return 0;
}

//ポイントクラウド描画
void drawPointCloud(Mat &rgbImage,Mat &pointCloud_XYZ){
	static int x,y;
	glPointSize(2);
	glBegin(GL_POINTS);
	uchar *p = rgbImage.data;
	Point3f *point = (Point3f*)pointCloud_XYZ.data;
	for(y = 0;y < KINECT_DEPTH_HEGIHT;y++){
		for(x = 0;x < KINECT_DEPTH_WIDTH;x++,p += 3,point++){ 
			if(point->z == 0)
				continue;
			glColor3ubv(p);
			glVertex3f(point->x,point->y,point->z);
		}
	}
	glEnd();
}
//3次元ポイントクラウドのための座標変換
void retrievePointCloudMap(Mat &depth,Mat &pointCloud_XYZ,DepthGenerator depthGenerator){
	static const int size = KINECT_DEPTH_HEGIHT * KINECT_DEPTH_WIDTH;
	static XnPoint3D proj[size] = {0};
	static int x,y;
	XnPoint3D *p = proj;
	unsigned short* dp = (unsigned short*)depth.data;
	for(y = 0; y < KINECT_DEPTH_HEGIHT; y++ ){
		for(x = 0; x < KINECT_DEPTH_WIDTH; x++, p++, dp++){	
			p->X = x;
			p->Y = y;
			p->Z = *dp * 0.001f; // from mm to meters
		}
	}
	//現実座標に変換
	depthGenerator.ConvertProjectiveToRealWorld(size, proj, (XnPoint3D*)pointCloud_XYZ.data);
}   
