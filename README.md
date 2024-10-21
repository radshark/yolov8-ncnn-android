# 1.使用版本信息
- Yolov8.1.47

- Android Studio: 2022.2.1 gradle:8.0 

- ncnn:ncnn-20231027-android-vulkan 
地址 https://github.com/Tencent/ncnn/releases/download/20231027/ncnn-20231027-android-vulkan.zip 

- cv:opencv-mobile-3.4.20-android 
https://github.com/opencv/opencv/releases/download/4.10.0/opencv-3.4.20-android-sdk.zip 

- onnx转ncnn模型:ncnn-20231027-windows-vs2019 
地址https://github.com/Tencent/ncnn/releases/download/20231027/ncnn-20231027-windows-vs2019.zip

# 2.模型部分
## 2.1模型训练参考 yolov8 https://docs.ultralytics.com/zh/models/yolov8/
## 2.2 pt转onnx
(1)修改ultralytics/ultralytics/nn/modules/block.py中的class C2f(nn.Module)如下：

```
 def forward(self, x):
         """Forward pass through C2f layer."""
         x = self.cv1(x)
         x = [x, x[:, self.c:, ...]]
         x.extend(m(x[-1]) for m in self.m)
         x.pop(1)
         return self.cv2(torch.cat(x, 1))
```
(2)修改ultralytics/ultralytics/nn/modules/head.py中的class Detect(nn.Module)改动如下：
```
 def forward(self, x):
         """Concatenates and returns predicted bounding boxes and class probabilities."""
         shape = x[0].shape  # BCHW
         for i in range(self.nl):
             x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
         if self.training:
             return x
         elif self.dynamic or self.shape != shape:
             self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
             self.shape = shape
         # 中间部分注释掉，return语句替换为
         return torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).permute(0, 2, 1)
```
(3)安装onnx包，在终端运行下面命令
`pip install onnx coremltools onnx-simplifier`

(4)根据官方文档，创建demo.py，将pt权重文件转换成onnx格式，代码如下
```
from ultralytics import YOLO

#加载训练模型
model = YOLO("runs/detect/train4/weights/best.pt")
#opset的参数（11、12、13皆可），使用其他会造成手机上运行出现很多框重叠错误识别
#将模型导出为 ONNX 格式
success = model.export(format="onnx", simplify=True, opset=11)
```
(5)对onnx文件进行压缩，进入到onnx文件所在目录，运行下面命令
`python -m onnxsim best.onnx best-sim.onnx`

(6)将onnx文件转换成param、bin文件
下载ncnn-20231027-windows-vs2019解压后，打开文件夹，并将best-sim.onnx复制到该文件夹的对应位置，地址栏输入cmd后，在打开的命令行窗口输入`onnx2ncnn.exe best-sim.onnx best.param best.bin`，原文件夹生成了需要的bin文件和param文件

(7)模型转换完成后，恢复block.py、head.py代码。

# Android项目
(1)项目结构，参考https://github.com/FeiGeChuanShu/ncnn-android-yolov8/tree/main/ncnn-android-yolov8
![image](https://github.com/user-attachments/assets/41a97bb3-9afe-4449-9cf9-6e3431710b9b)

(2)下载的ncnn-20231027-android-vulkan、opencv-mobile-3.4.20-android 解压到jni目录下

(3)输出的bin和param放到项目assert文件下

![image](https://github.com/user-attachments/assets/846b15d5-995a-4044-889c-3a950afa63c2)

(4)修改CMakeList

![image](https://github.com/user-attachments/assets/088d3b50-21bf-40ec-991e-1377e43a680d)

(5)修改yolo.cpp

a.类型数量

![image](https://github.com/user-attachments/assets/aaa7998d-0f7a-4225-82e0-40df4e5bf4b8)

b.类型名称

![image](https://github.com/user-attachments/assets/ee5376ad-e340-4c7b-a72c-a45c8d4a5235)

c.模型名称

![image](https://github.com/user-attachments/assets/09f11f41-9d40-429b-9521-d183c627ae83)

(6)修改strings.xml文件

![image](https://github.com/user-attachments/assets/4a3e3b9d-c366-4d82-a127-f4406fb976cc)

(7)编译运行

