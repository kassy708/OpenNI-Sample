/**
	author @kassy708
	OpenGL�ɂ��e�N�X�`���}�b�v
	glBegin(GL_TRIANGLE_STRIP);��p����
*/

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

DepthGenerator depthGenerator;// depth context
ImageGenerator imageGenerator;//image context
DepthMetaData depthMD;
ImageMetaData imageMD;
Context context;

Mat image(480,640,CV_8UC3);
Mat depth(480,640,CV_16UC1);  
//�|�C���g�N���E�h�̍��W
Mat pointCloud_XYZ(480,640,CV_32FC3,cv::Scalar::all(0));

void retrievePointCloudMap(Mat &depth,Mat &pointCloud_XYZ);    //3�����|�C���g�N���E�h�̂��߂̍��W�ϊ�
void drawTextureMaps(Mat &rgbImage,Mat &pointCloud_XYZ);        //�e�N�X�`���}�b�v�`��

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


//�e�N�X�`����\��臒l
#define THRESHOLD 0.1

//�`��
void display(){  
    // clear screen and depth buffer
    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT ); 
    // Reset the coordinate system before modifying
    glLoadIdentity();   
    glEnable(GL_DEPTH_TEST); //�uZ�o�b�t�@�v��L��
    gluLookAt(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0);   //���_�̌����ݒ�
    //wait and error processing
    context.WaitAnyUpdateAll();

    imageGenerator.GetMetaData(imageMD);
    depthGenerator.GetMetaData(depthMD);
    depthGenerator.GetAlternativeViewPointCap().SetViewPoint(imageGenerator);//�Y����␳
         
    memcpy(image.data,imageMD.Data(),image.step * image.rows);    //�C���[�W�f�[�^���i�[
    memcpy(depth.data,depthMD.Data(),depth.step * depth.rows);    //�[�x�f�[�^���i�[
      
    //3�����|�C���g�N���E�h�̂��߂̍��W�ϊ�
    retrievePointCloudMap(depth,pointCloud_XYZ);

    //���_�̕ύX
    polarview();  
    //�e�N�X�`���}�b�v
    drawTextureMaps(image,pointCloud_XYZ);
             
    //convert color space RGB2BGR
    cvtColor(image,image,CV_RGB2BGR);     
     
    imshow("image",image);
    imshow("depth",depth);
  
    glFlush();
    glutSwapBuffers();
}
//������
void init(){
    context.InitFromXmlFile(SAMPLE_XML_PATH); 
    context.FindExistingNode(XN_NODE_TYPE_DEPTH, depthGenerator); 
    context.FindExistingNode(XN_NODE_TYPE_IMAGE, imageGenerator);
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
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(FormWidth, FormHeight);
    glutCreateWindow(argv[0]);
    //�R�[���o�b�N
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

//�e�N�X�`���}�b�v�`��
void drawTextureMaps(Mat &rgbImage,Mat &pointCloud_XYZ){
	static int x,y;
	static uchar *p[2];
	static Point3f *point[2];
	int channel = rgbImage.channels();
	p[0] = rgbImage.data;										//��̐F
	p[1] = rgbImage.data + rgbImage.step;						//���̐F
	point[0] = (Point3f*)pointCloud_XYZ.data;					//��̍��W
	point[1] = &((Point3f*)pointCloud_XYZ.data)[KINECT_DEPTH_WIDTH];	//���̍��W
	for(y = 0;y < KINECT_DEPTH_HEGIHT - 1;y++){
		for(x = 0;x < KINECT_DEPTH_WIDTH - 1;x++,p[1] += channel,point[1]++,p[0] += channel,point[0]++){ 
			//���s�����擾�ł��ĂȂ������牽�����Ȃ�
			if(point[0]->z == 0)	
				continue;
			//�Ίp�̉��s����������΃e�N�X�`����\��Ȃ�
			if(abs(point[0]->z - (point[1] + 1)->z) > THRESHOLD || abs((point[0] + 1)->z - point[1]->z) > THRESHOLD)	
				continue;		

			//�e�N�X�`����\��
			glBegin(GL_TRIANGLE_STRIP);
			//����
			glTexCoord2f(0, 0);
			glColor3ubv(p[0]);
			glVertex3f(point[0]->x,point[0]->y,point[0]->z);
			//����
			glTexCoord2f(1, 0);
			glColor3ubv(p[1]);
			glVertex3f(point[1]->x,point[1]->y,point[1]->z);
			//�E��
			glTexCoord2f(0, 1);
			glColor3ubv(p[0]+channel);
			glVertex3f((point[0] + 1)->x,(point[0] + 1)->y,(point[0] + 1)->z);
			//�E��
			glTexCoord2f(1, 1);
			glColor3ubv(p[1]+channel);
			glVertex3f((point[1] + 1)->x,(point[1] + 1)->y,(point[1] + 1)->z);			

			glEnd();
		}
		p[0] += channel,point[0]++;
		p[1] += channel,point[1]++;
	}
}
//3�����|�C���g�N���E�h�̂��߂̍��W�ϊ�
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
	//�������W�ɕϊ�
	depthGenerator.ConvertProjectiveToRealWorld(size, proj, (XnPoint3D*)pointCloud_XYZ.data);
}   
