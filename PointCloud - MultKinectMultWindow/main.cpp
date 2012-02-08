/**
	author @kassy708
	�}���`�L�l�N�g��OpenGL�ɂ��|�C���g�N���E�h

*/
#include <iostream>
#include <stdexcept>
#include <map>
#include <sstream>

#include <GL/glut.h>
#include <opencv2/opencv.hpp>
#ifdef _DEBUG
    //Debug���[�h�̏ꍇ
    #pragma comment(lib,"C:\\OpenCV2.2\\lib\\opencv_core220d.lib")            // opencv_core
    #pragma comment(lib,"C:\\OpenCV2.2\\lib\\opencv_imgproc220d.lib")        // opencv_imgproc
    #pragma comment(lib,"C:\\OpenCV2.2\\lib\\opencv_highgui220d.lib")        // opencv_highgui
#else
    //Release���[�h�̏ꍇ
    #pragma comment(lib,"C:\\OpenCV2.2\\lib\\opencv_core220.lib")            // opencv_core
    #pragma comment(lib,"C:\\OpenCV2.2\\lib\\opencv_imgproc220.lib")        // opencv_imgproc
    #pragma comment(lib,"C:\\OpenCV2.2\\lib\\opencv_highgui220.lib")        // opencv_highgui
#endif

#include <XnCppWrapper.h>
#pragma comment(lib,"C:/Program files/OpenNI/Lib/openNI.lib")

#define SAMPLE_XML_PATH "C:/Program Files/OpenNI/Data/SamplesConfig.xml"
using namespace cv;
using namespace xn;

//openNI�̂��߂̐錾�E��`
//�}�N����`
#define KINECT_IMAGE_WIDTH 640
#define KINECT_IMAGE_HEGIHT 480
#define KINECT_DEPTH_WIDTH 640
#define KINECT_DEPTH_HEGIHT 480

Mat pointCloud_XYZ( KINECT_DEPTH_HEGIHT, KINECT_DEPTH_WIDTH, CV_32FC3);


void retrievePointCloudMap(Mat &depth,Mat &pointCloud_XYZ,DepthGenerator depthGenerator);    //3�����|�C���g�N���E�h�̂��߂̍��W�ϊ�
void drawPointCloud(Mat &rgbImage,Mat &pointCloud_XYZ);        //�|�C���g�N���E�h�`��

//openGL�̂��߂̐錾�E��`
//---�ϐ��錾---
int FormWidth = 640;
int FormHeight = 480;
int mButton;
float twist, elevation, azimuth;
float cameraDistance = 0,cameraX = 0,cameraY = 0;
int xBegin, yBegin;
//---�}�N����`---
#define glFovy 45        //���p�x
#define glZNear 1.0        //near�ʂ̋���
#define glZFar 150.0    //far�ʂ̋���
void polarview();        //���_�ύX


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
	//���_�ύX
	void polarview(){
		glTranslatef( cameraX, cameraY, cameraDistance);
		glRotatef( -twist, 0.0, 0.0, 1.0);
		glRotatef( -elevation, 1.0, 0.0, 0.0);
		glRotatef( -azimuth, 0.0, 1.0, 0.0);
	}
};
// Kinect���Ƃ̕\�����
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

// ���o���ꂽ�f�o�C�X��񋓂���
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

// �W�F�l���[�^���쐬����
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


//������
void init(){
	try {
		showOepnNIVersion();

		XnStatus rc;
    
		rc = context.Init();
		if (rc != XN_STATUS_OK) {
			throw std::runtime_error(::xnGetStatusString(rc));
		}

		// ���o���ꂽ�f�o�C�X�𗘗p�\�Ƃ��ēo�^����
		EnumerateProductionTrees(context, XN_NODE_TYPE_DEVICE);
		EnumerateProductionTrees(context, XN_NODE_TYPE_IMAGE);
		EnumerateProductionTrees(context, XN_NODE_TYPE_DEPTH);
    
		// �o�^���ꂽ�f�o�C�X���擾����
		std::cout << "xn::Context::EnumerateExistingNodes ... ";
		rc = context.EnumerateExistingNodes( nodeList );
		if ( rc != XN_STATUS_OK ) {
			throw std::runtime_error( ::xnGetStatusString( rc ) );
		}
		std::cout << "Success" << std::endl;

		// �o�^���ꂽ�f�o�C�X����W�F�l���[�^�𐶐�����
		for ( xn::NodeInfoList::Iterator it = nodeList.Begin();it != nodeList.End(); ++it ) {
      
			// �C���X�^���X���̍Ōオ�ԍ��ɂȂ��Ă���
			std::string name = (*it).GetInstanceName();
			int no = *name.rbegin() - '1';
      
			if ((*it).GetDescription().Type == XN_NODE_TYPE_IMAGE) {
				kinect[no].imageGenerator = CreateGenerator<xn::ImageGenerator>(*it);
			}
			else if ((*it).GetDescription().Type == XN_NODE_TYPE_DEPTH) {
				kinect[no].depthGenerator = CreateGenerator<xn::DepthGenerator>(*it);
			}
		}

		// �W�F�l���[�g���J�n����
		context.StartGeneratingAll();

		// �r���[�|�C���g�̐ݒ��A�J�����̈���쐬����
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

//�`��
void display(){
	
    for (std::map<int, Kinect>::iterator it = kinect.begin(); it != kinect.end(); ++it) {
        Kinect& k = it->second;
		
		glutSetWindow(k.windowData.windowID);

		// clear screen and depth buffer
		glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT ); 
		// Reset the coordinate system before modifying
		glLoadIdentity();   
		glEnable(GL_DEPTH_TEST); //�uZ�o�b�t�@�v��L��
		gluLookAt(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0);   //���_�̌����ݒ�
		//wait and error processing
		context.WaitAnyUpdateAll();

		k.imageGenerator.GetMetaData(imageMD);
		k.depthGenerator.GetMetaData(depthMD);
		k.depthGenerator.GetAlternativeViewPointCap().SetViewPoint(k.imageGenerator);//�Y����␳

		Mat image(480,640,CV_8UC3,(unsigned char*)imageMD.WritableData());
		Mat depth(480,640,CV_16UC1,(unsigned char*)depthMD.WritableData());
         
		memcpy(image.data,imageMD.Data(),image.step * image.rows);    //�C���[�W�f�[�^���i�[
		memcpy(depth.data,depthMD.Data(),depth.step * depth.rows);    //�[�x�f�[�^���i�[
      
		//3�����|�C���g�N���E�h�̂��߂̍��W�ϊ�
		retrievePointCloudMap(depth,pointCloud_XYZ,k.depthGenerator);

		//���_�̕ύX
		polarview();  
		//�|�C���g�N���E�h
		drawPointCloud(image,pointCloud_XYZ);
             
		//convert color space RGB2BGR
		cvtColor(image,image,CV_RGB2BGR);     
     
		imshow(k.imageGenerator.GetName(),image);
		imshow(k.depthGenerator.GetName(),depth);

		
		glFlush();
		glutSwapBuffers();
	}
  
}
// �A�C�h�����̃R�[���o�b�N
void idle(){
    //�ĕ`��v��
    glutPostRedisplay();
}
//�E�B���h�E�̃T�C�Y�ύX
void reshape (int width, int height){
    FormWidth = width;
    FormHeight = height;
    glViewport (0, 0, (GLsizei)width, (GLsizei)height);
    glMatrixMode (GL_PROJECTION);
    glLoadIdentity ();
    //�ˉe�ϊ��s��̎w��
    gluPerspective (glFovy, (GLfloat)width / (GLfloat)height,glZNear,glZFar); 
    glMatrixMode (GL_MODELVIEW);
}
//�}�E�X�̓���
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
//�}�E�X�̑���
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
//���_�ύX
void polarview(){
    glTranslatef( cameraX, cameraY, cameraDistance);
    glRotatef( -twist, 0.0, 0.0, 1.0);
    glRotatef( -elevation, 1.0, 0.0, 0.0);
    glRotatef( -azimuth, 0.0, 1.0, 0.0);
}
//���C��
int main(int argc, char *argv[]){
    init();

    glutInit(&argc, argv);
	
    // ���o�������ׂĂ�Kinect�̉摜��\������
    for (std::map<int, Kinect>::iterator it = kinect.begin(); it != kinect.end(); ++it) {
		glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);
		glutInitWindowSize(FormWidth, FormHeight);
		it->second.windowData.windowID = glutCreateWindow(argv[0]);
		//�R�[���o�b�N
		if(it == kinect.begin())//�ŏ������o�^
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

//�|�C���g�N���E�h�`��
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
//3�����|�C���g�N���E�h�̂��߂̍��W�ϊ�
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
	//�������W�ɕϊ�
	depthGenerator.ConvertProjectiveToRealWorld(size, proj, (XnPoint3D*)pointCloud_XYZ.data);
}   
