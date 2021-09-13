# Darknet installation for windows
***Honestly, probably easier and faster to install linux and then install from there***
<p>
I used [this guide](https://medium.com/analytics-vidhya/installing-darknet-on-windows-462d84840e5a) and this readme is just a condensed version of it.
<p>
Reinstall OpenCV, ticking the option to add it to system path<br>
Reinstall Anaconda, ticking the option to add it to system path<br>
Install CUDA<br>
Download CuDNN (Make an nVidia developer account if you don't have one)<br>
Put CuDNN in your Cuda folder (If you are unsure where that is, an error message later will inform you)<br>
Download [vcpkg](https://github.com/Microsoft/vcpkg) and extract it somewhere memorable<br>
Open up a Powershell:<br>
cd to vcpkg's directory<br>
```powershell
bootstrap-vcpkg.bat
vcpkg integrate install
```
Open up your [Windows Environment Variables](https://www.alphr.com/environment-variables-windows-10/)<br>
Create one called: VCPKG_ROOT with a value of where you extracted vcpkg<br>
Create another one called: VCPKG_DEFAULT_TRIPLET with a value of x64-windows<br>
Powershell again:<br>
```powershell
cd $env:VCPKG_ROOT
.\vcpkg install pthreads opencv[ffmpeg]
```
Download darknet put it wherever you like<br>
Powershell again (ADMIN MODE):
```
Set-ExecutionPolicy -ExecutionPolicy Unrestricted
(Select yes to all)
(cd to your darknet folder)
.\build.ps1
(If you want to test it)
darknet.exe detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights -thresh 0.25
```

This took me a about a day's work to get working with the download times and troubleshooting involved, I hope it helps someone out in the future.



