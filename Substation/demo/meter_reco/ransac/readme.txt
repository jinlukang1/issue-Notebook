运行方式：输入命令python ransac.py
其中,ransac程序中需要进行改动的变量有：
在config.py文件中：
NUM_CLASSES是关键点数量
TEST_PATH是待测试图片的路径；
ZERO_FULL_PATH与ZERO_FULL分别是正视图的图像存放的位置与名称；
RECTIFIED_PATH是将对齐后的图片输出的路径；
MODEL_PATH是训练好的模型存放的路径；
CUDA_DEVICE为使用的gpu编号

TEST_PATH存放待对齐的图片，其尺寸为：宽257，高353
RECTIFIED_PATH输出对齐后的图片，其尺寸与ZERO_FULL_PATH中存放的标准图片一致，宽467，高469。
输出图片的尺寸可以进行更改：在ransac.py的第70行，warpAffine函数后两个参数即为输出图片的尺寸。

运行程序前先保存并清理一下/data3/home_huxiaoming/rectified/ 这个文件夹，这里面的文件是上一次的运行结果。
