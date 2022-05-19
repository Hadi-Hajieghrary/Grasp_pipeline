# Grasp_pipeline
<a name="Install the GPD python binding"></a>
This pipeline requires a GPU to be run. Detectron2 needs to be installed. It can be tricky to install it. Firstly make sure that nvcc -V display that the Cuda version is 11.3. If Cuda 11.3 is not installed, intall it. During the installation of Cuda one's needs to take care that the latest version of Cuda is not installing. For that, it's necessary to add "-11-3" to the last instruction to install cuda. Then, install Pytorch for Cuda 11.3. Then, install Detectron2. Then install ZED_SDK, it's important to install ZED SDK after Detectron2, because it uses Cuda 11.5. 
## 1) Install the GPD python binding
```
git clone https://github.com/patricknaughton01/gpd  
cd gpd
git clone https://github.com/pybind/pybind11
mkdir build
cd build
cmake .. -DBUILD_PYTHON=ON
make -j
```
<a name="Install AprilTag"></a>
## 1) Install AprilTag
```
git clone https://github.com/Tinker-Twins/AprilTag.git
cd ~/AprilTag
./install.sh
```
