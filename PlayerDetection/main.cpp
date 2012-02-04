/**
	author @kassy708
	人物抽出
*/

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
	Mat image(480,640,CV_8UC3);			//RGB画像
	Mat depth(480,640,CV_16UC1);		//深度画像
	Mat mask(480,640,CV_16UC1);			//プレイヤーマスク画像 
	Mat player(480,640,CV_8UC3);		//人間画像
	Mat playerDepth(480,640,CV_16UC1);	//人間画像
	Mat background(480,640,CV_8UC3);	//背景画像
	Mat useMask(480,640,CV_16UC1);		//使用するマスク
	
	int maskSize = mask.step * mask.rows;	//マスク画像の配列数
	
	//OpenNIの初期化
    context.InitFromXmlFile(SAMPLE_XML_PATH); 
    context.FindExistingNode(XN_NODE_TYPE_DEPTH, depthGenerator); 
    context.FindExistingNode(XN_NODE_TYPE_IMAGE, imageGenerator);
	context.FindExistingNode(XN_NODE_TYPE_USER, userGenerator);
	
	//RGB画像と振動画像のズレを補正
	depthGenerator.GetAlternativeViewPointCap().SetViewPoint(imageGenerator);
	
	//ウィンドウの準備
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
		userGenerator.GetUserPixels(0,sceneMD);	//ユーザピクセル取得
		
		memcpy(image.data,imageMD.Data(),image.step * image.rows);    //イメージデータを格納
		memcpy(depth.data,depthMD.Data(),depth.step * depth.rows);    //深度データを格納
		
		//BGRをRGBへ
		cvtColor(image,image,CV_RGB2BGR);

        player = 0;			//初期化
        playerDepth = 0;	//初期化       
        memcpy(mask.data,sceneMD.Data(),maskSize);	//マスクデータをコピー       
        mask.convertTo(useMask,CV_8UC1);			//マスクの変換
        image.copyTo(player,useMask);				//マスクを利用した人物抽出
        depth.copyTo(playerDepth,useMask);			//マスクを利用した人物奥行き抽出
        background = image - player;				//背景のみ取得

		//画面に表示
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

