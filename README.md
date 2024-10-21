# 1.使用版本信息
Yolov8.1.47

Android Studio: 2022.2.1 gradle:8.0 

ncnn:ncnn-20231027-android-vulkan 
地址 https://github.com/Tencent/ncnn/releases/download/20231027/ncnn-20231027-android-vulkan.zip 

cv:opencv-mobile-3.4.20-android 
https://github.com/opencv/opencv/releases/download/4.10.0/opencv-3.4.20-android-sdk.zip 

onnx转ncnn模型:ncnn-20231027-windows-vs2019 
地址https://github.com/Tencent/ncnn/releases/download/20231027/ncnn-20231027-windows-vs2019.zip

# 2.关键流程
## 2.1模型训练参考 yolov8 https://docs.ultralytics.com/zh/models/yolov8/
## 2.2 pt转onnx
(1)修改ultralytics/ultralytics/nn/modules/block.py中的class C2f(nn.Module)如下：

 def forward(self, x):
        """Forward pass through C2f layer."""
        x = self.cv1(x)
        x = [x, x[:, self.c:, ...]]
        x.extend(m(x[-1]) for m in self.m)
        x.pop(1)
        return self.cv2(torch.cat(x, 1))
        
(2)修改ultralytics/ultralytics/nn/modules/head.py中的class Detect(nn.Module)改动如下：

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




