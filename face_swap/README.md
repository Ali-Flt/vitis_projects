# Face Swap on Zynq MPSoC

Hackster project: 

Target device: ZCU104 evaluation board

-Download shape predictor model for face landmark:
```
wget https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2
```

-Download dlib:

```
git clone https://github.com/davisking/dlib.git
```

-Command to compile code on x86:

```
g++ -std=c++17 -O3 -I../../../../../dlib/ dlib_facedetect_and_landmark.cpp source.cpp `pkg-config --cflags --libs opencv` -lpthread -lX11 -msse -msse2 -msse4.2 -mavx -o dlib_facedetect_and_landmark.o
```

-Vitis C++ Build Settings:
1. Includes -> add $workspace/dlib to include paths
2. set Optimization -> Optimization Level to O3
3. Miscellaneous -> add -std=c++17 to other flags
4. Libraries -> add :

xilinxopencl
opencv_photo
opencv_xphoto
opencv_highgui
vitis_ai_library-dpu_task
X11
vitis_ai_library-xnnpp
vitis_ai_library-model_config
vitis_ai_library-math
vart-util
xir
opencv_highgui
vitis_ai_library-facedetect
json-c
glog
opencv_core
opencv_videoio
opencv_imgproc
opencv_imgcodecs
pthread
rt
dl
crypt
stdc++

-To run face swap:
```
./face_swap video_file_name.webm
```


