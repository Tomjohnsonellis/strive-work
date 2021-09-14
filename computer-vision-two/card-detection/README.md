# Darknet installation for windows
***Honestly, probably easier and faster to install linux and then install from there***
<br>
  I used [this guide](https://medium.com/analytics-vidhya/installing-darknet-on-windows-462d84840e5a) and this readme is just a condensed version of it.<br>
  <br>
STEPS:<br>
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

# Day 2 - Data Collection
Today felt much more like progress, a pack of cards I ordered for this project arrived so now it was time to get some data from them.<br>
We decided on 4 images of each card against a green background (I used some green card from my local hobby store)<br>
For each card I took: A portrait close-up; a sideways shot; a zoomed out picture; and a skewed angle shot.<br>
I then transferred the images to my desktop, organised into 52 folders (1 for each unique card) and shared out with the team members.<br>



