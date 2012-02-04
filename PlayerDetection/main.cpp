/**
	author @kassy708
	�l�����o
*/

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

int main()
{
	//OpenNI
	DepthGenerator depthGenerator;
	ImageGenerator imageGenerator;
	UserGenerator userGenerator;
	DepthMetaData depthMD;
	ImageMetaData imageMD;
	SceneMetaData sceneMD;	
	Context context;

	//OpenCV
	Mat image(480,640,CV_8UC3);			//RGB�摜
	Mat depth(480,640,CV_16UC1);		//�[�x�摜
	Mat mask(480,640,CV_16UC1);			//�v���C���[�}�X�N�摜 
	Mat player(480,640,CV_8UC3);		//�l�ԉ摜
	Mat playerDepth(480,640,CV_16UC1);	//�l�ԉ摜
	Mat background(480,640,CV_8UC3);	//�w�i�摜
	Mat useMask(480,640,CV_16UC1);		//�g�p����}�X�N
	
	int maskSize = mask.step * mask.rows;	//�}�X�N�摜�̔z��
	
	//OpenNI�̏�����
    context.InitFromXmlFile(SAMPLE_XML_PATH); 
    context.FindExistingNode(XN_NODE_TYPE_DEPTH, depthGenerator); 
    context.FindExistingNode(XN_NODE_TYPE_IMAGE, imageGenerator);
	context.FindExistingNode(XN_NODE_TYPE_USER, userGenerator);
	
	//RGB�摜�ƐU���摜�̃Y����␳
	depthGenerator.GetAlternativeViewPointCap().SetViewPoint(imageGenerator);
	
	//�E�B���h�E�̏���
	cvNamedWindow("image");
	cvNamedWindow("depth");
	cvNamedWindow("player");
	cvNamedWindow("playerDepth");
	cvNamedWindow("background");

	int key = 0;
	while (key!='q'){
		context.WaitAndUpdateAll();
		
		imageGenerator.GetMetaData(imageMD);
		depthGenerator.GetMetaData(depthMD);
		userGenerator.GetUserPixels(0,sceneMD);	//���[�U�s�N�Z���擾
		
		memcpy(image.data,imageMD.Data(),image.step * image.rows);    //�C���[�W�f�[�^���i�[
		memcpy(depth.data,depthMD.Data(),depth.step * depth.rows);    //�[�x�f�[�^���i�[
		
		//BGR��RGB��
		cvtColor(image,image,CV_RGB2BGR);

        player = 0;			//������
        playerDepth = 0;	//������       
        memcpy(mask.data,sceneMD.Data(),maskSize);	//�}�X�N�f�[�^���R�s�[       
        mask.convertTo(useMask,CV_8UC1);			//�}�X�N�̕ϊ�
        image.copyTo(player,useMask);				//�}�X�N�𗘗p�����l�����o
        depth.copyTo(playerDepth,useMask);			//�}�X�N�𗘗p�����l�����s�����o
        background = image - player;				//�w�i�̂ݎ擾

		//��ʂɕ\��
		imshow("image",image);
		imshow("depth",depth);
		imshow("player",player);
		imshow("playerDepth",playerDepth);
		imshow("background",background);

		key = waitKey(33);
	}
	context.Shutdown();
	return 0;
}

