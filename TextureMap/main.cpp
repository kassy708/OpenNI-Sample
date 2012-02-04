/**
	author @kassy708
	OpenGLによるテクスチャマップ
	glBegin(GL_TRIANGLE_STRIP);を用いた
*/

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

DepthGenerator depthGenerator;// depth context
ImageGenerator imageGenerator;//image context
DepthMetaData depthMD;
ImageMetaData imageMD;
Context context;

Mat image(480,640,CV_8UC3);
Mat depth(480,640,CV_16UC1);  
//ポイントクラウドの座標
Mat pointCloud_XYZ(480,640,CV_32FC3,cv::Scalar::all(0));

void retrievePointCloudMap(Mat &depth,Mat &pointCloud_XYZ);    //3次元ポイントクラウドのための座標変換
void drawTextureMaps(Mat &rgbImage,Mat &pointCloud_XYZ);        //テクスチャマップ描画

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


//テクスチャを貼る閾値
#define THRESHOLD 0.1

//描画
void display(){  
    // clear screen and depth buffer
    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT ); 
    // Reset the coordinate system before modifying
    glLoadIdentity();   
    glEnable(GL_DEPTH_TEST); //「Zバッファ」を有効
    gluLookAt(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0);   //視点の向き設定
    //wait and error processing
    context.WaitAnyUpdateAll();

    imageGenerator.GetMetaData(imageMD);
    depthGenerator.GetMetaData(depthMD);
    depthGenerator.GetAlternativeViewPointCap().SetViewPoint(imageGenerator);//ズレを補正
         
    memcpy(image.data,imageMD.Data(),image.step * image.rows);    //イメージデータを格納
    memcpy(depth.data,depthMD.Data(),depth.step * depth.rows);    //深度データを格納
      
    //3次元ポイントクラウドのための座標変換
    retrievePointCloudMap(depth,pointCloud_XYZ);

    //視点の変更
    polarview();  
    //テクスチャマップ
    drawTextureMaps(image,pointCloud_XYZ);
             
    //convert color space RGB2BGR
    cvtColor(image,image,CV_RGB2BGR);     
     
    imshow("image",image);
    imshow("depth",depth);
  
    glFlush();
    glutSwapBuffers();
}
//初期化
void init(){
    context.InitFromXmlFile(SAMPLE_XML_PATH); 
    context.FindExistingNode(XN_NODE_TYPE_DEPTH, depthGenerator); 
    context.FindExistingNode(XN_NODE_TYPE_IMAGE, imageGenerator);
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
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(FormWidth, FormHeight);
    glutCreateWindow(argv[0]);
    //コールバック
    glutReshapeFunc (reshape);
    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    init();
    glutMainLoop();
    context.Shutdown();
    return 0;
}

//テクスチャマップ描画
void drawTextureMaps(Mat &rgbImage,Mat &pointCloud_XYZ){
	static int x,y;
	static uchar *p[2];
	static Point3f *point[2];
	int channel = rgbImage.channels();
	p[0] = rgbImage.data;										//上の色
	p[1] = rgbImage.data + rgbImage.step;						//下の色
	point[0] = (Point3f*)pointCloud_XYZ.data;					//上の座標
	point[1] = &((Point3f*)pointCloud_XYZ.data)[KINECT_DEPTH_WIDTH];	//下の座標
	for(y = 0;y < KINECT_DEPTH_HEGIHT - 1;y++){
		for(x = 0;x < KINECT_DEPTH_WIDTH - 1;x++,p[1] += channel,point[1]++,p[0] += channel,point[0]++){ 
			//奥行きが取得できてなかったら何もしない
			if(point[0]->z == 0)	
				continue;
			//対角の奥行きが遠ければテクスチャを貼らない
			if(abs(point[0]->z - (point[1] + 1)->z) > THRESHOLD || abs((point[0] + 1)->z - point[1]->z) > THRESHOLD)	
				continue;		

			//テクスチャを貼る
			glBegin(GL_TRIANGLE_STRIP);
			//左上
			glTexCoord2f(0, 0);
			glColor3ubv(p[0]);
			glVertex3f(point[0]->x,point[0]->y,point[0]->z);
			//左下
			glTexCoord2f(1, 0);
			glColor3ubv(p[1]);
			glVertex3f(point[1]->x,point[1]->y,point[1]->z);
			//右上
			glTexCoord2f(0, 1);
			glColor3ubv(p[0]+channel);
			glVertex3f((point[0] + 1)->x,(point[0] + 1)->y,(point[0] + 1)->z);
			//右下
			glTexCoord2f(1, 1);
			glColor3ubv(p[1]+channel);
			glVertex3f((point[1] + 1)->x,(point[1] + 1)->y,(point[1] + 1)->z);			

			glEnd();
		}
		p[0] += channel,point[0]++;
		p[1] += channel,point[1]++;
	}
}
//3次元ポイントクラウドのための座標変換
void retrievePointCloudMap(Mat &depth,Mat &pointCloud_XYZ){
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
