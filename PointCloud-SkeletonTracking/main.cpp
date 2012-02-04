/**
	author @kassy708
	OpenGL�ɂ��|�C���g�N���E�h
	���[�U�g���b�L���O�Ɗ֐߂̕`��
	SamplesConfig.xml��<Node type="User"/>��ǉ����Ă�������
*/

#include <GL/glut.h>
#include <opencv2/opencv.hpp>
#ifdef _DEBUG
    //Debug���[�h�̏ꍇ
    #pragma comment(lib,"C:\\OpenCV2.2\\lib\\opencv_core220d.lib")           // opencv_core
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
UserGenerator userGenerator;
DepthMetaData depthMD;
ImageMetaData imageMD;
Context context;
//�g���b�L���O�̍ۂ̃R�[���o�b�N
void XN_CALLBACK_TYPE UserDetected( xn::UserGenerator& generator, XnUserID nId, void* pCookie );
void XN_CALLBACK_TYPE CalibrationEnd(xn::SkeletonCapability& capability, XnUserID nId, XnBool bSuccess, void* pCookie);


Mat image(480,640,CV_8UC3);
Mat depth(480,640,CV_16UC1);  
//�|�C���g�N���E�h�̍��W
Mat pointCloud_XYZ(480,640,CV_32FC3,cv::Scalar::all(0));

void retrievePointCloudMap(Mat &depth,Mat &pointCloud_XYZ);	//3�����|�C���g�N���E�h�̂��߂̍��W�ϊ�
void drawPointCloud(Mat &rgbImage,Mat &pointCloud_XYZ);		//�|�C���g�N���E�h�`��
void drawUserJoint();										//���[�U�̊֐߂�`��
void DrawLimb(XnUserID player, XnSkeletonJoint eJoint1, XnSkeletonJoint eJoint2);
void DrawJoint(XnUserID player, XnSkeletonJoint eJoint1);

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
    //�|�C���g�N���E�h
    drawPointCloud(image,pointCloud_XYZ);

	//�֐߂̕`��
	drawUserJoint();
             
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
	context.FindExistingNode(XN_NODE_TYPE_USER, userGenerator);   
    if (!userGenerator.IsCapabilitySupported(XN_CAPABILITY_SKELETON)) {   // �X�P���g���E�g���b�L���O���T�|�[�g���Ă��邩�m�F
		printf("���[�U�[���o���T�|�[�g���Ă܂���\n");
    }

    // �L�����u���[�V�����Ƀ|�[�Y���K�v
    xn::SkeletonCapability skeleton = userGenerator.GetSkeletonCap();
    if ( skeleton.NeedPoseForCalibration() ) {
		printf("����OpenNI�̃o�[�W�����̓|�[�Y�����ɂ͑Ή����Ă��܂���\n");
    }

    // ���[�U�[�F���̃R�[���o�b�N��o�^
    // �L�����u���[�V�����̃R�[���o�b�N��o�^
    XnCallbackHandle userCallbacks, calibrationCallbacks;
    userGenerator.RegisterUserCallbacks(&::UserDetected, 0, 0, userCallbacks);
    skeleton.RegisterCalibrationCallbacks( 0, &::CalibrationEnd, 0, calibrationCallbacks );

    skeleton.SetSkeletonProfile(XN_SKEL_PROFILE_ALL);
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

//�l���̍��i��`��
void drawUserJoint(){
	XnBool g_bDrawSkeleton = TRUE;
	XnBool g_bPrintID = TRUE;
	XnBool g_bPrintState = TRUE;
	char strLabel[50] = "";
	XnUserID aUsers[15];
	XnUInt16 nUsers = 15;
	userGenerator.GetUsers(aUsers, nUsers);
	for (int i = 0; i < nUsers; ++i){
		if (userGenerator.GetSkeletonCap().IsTracking(aUsers[i])){
			//�e�֐߂ɓ_��ł�
			glPointSize(5);
			glBegin(GL_POINTS);		
				glColor4f(0,255,0, 1);	
				for(int Joint = XN_SKEL_HEAD;Joint < XN_SKEL_RIGHT_FOOT + 1;Joint++)
					DrawJoint(aUsers[i],(XnSkeletonJoint)Joint);
			glEnd();

			//�e���ɐ�������
			glBegin(GL_LINES);
				glColor4f(255,0,0, 1);
				DrawLimb(aUsers[i], XN_SKEL_HEAD, XN_SKEL_NECK);

				DrawLimb(aUsers[i], XN_SKEL_NECK, XN_SKEL_LEFT_SHOULDER);
				DrawLimb(aUsers[i], XN_SKEL_LEFT_SHOULDER, XN_SKEL_LEFT_ELBOW);
				DrawLimb(aUsers[i], XN_SKEL_LEFT_ELBOW, XN_SKEL_LEFT_HAND);

				DrawLimb(aUsers[i], XN_SKEL_NECK, XN_SKEL_RIGHT_SHOULDER);
				DrawLimb(aUsers[i], XN_SKEL_RIGHT_SHOULDER, XN_SKEL_RIGHT_ELBOW);
				DrawLimb(aUsers[i], XN_SKEL_RIGHT_ELBOW, XN_SKEL_RIGHT_HAND);

				DrawLimb(aUsers[i], XN_SKEL_LEFT_SHOULDER, XN_SKEL_TORSO);
				DrawLimb(aUsers[i], XN_SKEL_RIGHT_SHOULDER, XN_SKEL_TORSO);

				DrawLimb(aUsers[i], XN_SKEL_TORSO, XN_SKEL_LEFT_HIP);
				DrawLimb(aUsers[i], XN_SKEL_LEFT_HIP, XN_SKEL_LEFT_KNEE);
				DrawLimb(aUsers[i], XN_SKEL_LEFT_KNEE, XN_SKEL_LEFT_FOOT);

				DrawLimb(aUsers[i], XN_SKEL_TORSO, XN_SKEL_RIGHT_HIP);
				DrawLimb(aUsers[i], XN_SKEL_RIGHT_HIP, XN_SKEL_RIGHT_KNEE);
				DrawLimb(aUsers[i], XN_SKEL_RIGHT_KNEE, XN_SKEL_RIGHT_FOOT);

				DrawLimb(aUsers[i], XN_SKEL_LEFT_HIP, XN_SKEL_RIGHT_HIP);

			glEnd();
		}
	}
}
void DrawJoint(XnUserID player, XnSkeletonJoint eJoint){
	if (!userGenerator.GetSkeletonCap().IsTracking(player)){
		printf("not tracked!\n");
		return;
	}
	XnSkeletonJointPosition joint;
	userGenerator.GetSkeletonCap().GetSkeletonJointPosition(player, eJoint, joint);

	//�M�����̒Ⴂ�ꍇ�`�悵�Ȃ�
	if (joint.fConfidence < 0.5)
		return;

	glVertex3f(joint.position.X* 0.001f, joint.position.Y* 0.001f, joint.position.Z* 0.001f);
}
void DrawLimb(XnUserID player, XnSkeletonJoint eJoint1, XnSkeletonJoint eJoint2){
	if (!userGenerator.GetSkeletonCap().IsTracking(player)){
		printf("not tracked!\n");
		return;
	}

	XnSkeletonJointPosition joint1, joint2;
	userGenerator.GetSkeletonCap().GetSkeletonJointPosition(player, eJoint1, joint1);
	userGenerator.GetSkeletonCap().GetSkeletonJointPosition(player, eJoint2, joint2);

	//�M�����̒Ⴂ�ꍇ�`�悵�Ȃ�
	if (joint1.fConfidence < 0.5 || joint2.fConfidence < 0.5)
		return;

	glVertex3f(joint1.position.X* 0.001f, joint1.position.Y* 0.001f, joint1.position.Z* 0.001f);
	glVertex3f(joint2.position.X* 0.001f, joint2.position.Y* 0.001f, joint2.position.Z* 0.001f);
}	
// ���[�U�[���o
void XN_CALLBACK_TYPE UserDetected( xn::UserGenerator& generator, XnUserID nId, void* pCookie )
{
	printf("���[�UID%d ���o  %d�l��\n",nId,generator.GetNumberOfUsers());
    generator.GetSkeletonCap().RequestCalibration(nId, TRUE);
}
// �L�����u���[�V�����̏I��
void XN_CALLBACK_TYPE CalibrationEnd(xn::SkeletonCapability& capability, XnUserID nId, XnBool bSuccess, void* pCookie)
{
    if ( bSuccess ) {
		printf("���[�UID%d �L�����u���[�V��������\n",nId);
        capability.StartTracking(nId);
    }
    else {
		printf("���[�UID%d �L�����u���[�V�������s\n",nId);
    }
}