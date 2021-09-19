# POST PROJECT CLARITY:
So, this project was not my finest work, but looking back our team has definitely learnt some hard lessons for future projects.
I've even done a presentation on it! [HERE](https://www.beautiful.ai/player/-MjyaaqRl2zX0PmSdC38)<br>
This readme will stay up as I believe it's important to admit mistakes and learn from them.


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
### Evening work
I've used  [Roboflow's annotator tool](https://roboflow.com/annotate) to create the bounding boxes, seems to be a pretty helpful tool!<br>
The bounding boxes cover the car's corner value and symbol (8♥, A♧), with 4 corners on each card, that'll be 16 per card to train on, it's definitely a small sample size and a scalable approach should definitely be considered for the future.<br>
An example of which is using one very clear image of a card, isolating the corner info's bounding box, then generating images from that with the help of image augmentation libraries. Different backgrounds, lighting, textures etc. which will give a much more robust model.

# Day 3 
Finished annotations about midday, exported them then fiddled about for a very long time trying to get yolov5 to run. After which point I remembered my office computer is a 10 year old dell so has no chance of training on the dataset within the week. Had to go through the setup again on my slightly more capable machine which also took considerable time. This project has really made me aware of how much time can be wasted on just system operation.<br>
I have an odd setup at the moment, an old dell running ubuntu, a thinkpad laptop running windows, and a remote virtual machine with a graphics card running windows and has the linux subsystem for windows installed where I train the models. I spend most of my time on the desktop as the laptop only has VGA outputs which my monitors do not accept and the laptop screen hurts my eyes after any significant use time. In order to transfer files from machine to machine I use a Synology NAS drive which I honestly find to be a pain. Linux subsystem for windows is headless so I have had to get to grips with sftp in the terminal today which took even more time. My setup used to be very simple and absolutely needs reorganising, it's more akin to a balancing act or carnival game than a working environment currently.


# Day 4
Model training. Now that all the needed data (annotated images) is on the machine capable of training a model, yolo is installed and it was time to start training! Unfortunately I am unable to use my graphics card for training as it doesn't seem to be available in the LinuxSubsystemForWindows, so the performance will not be what I had hoped. The model took about 6 hours to train, I used [wandb](wandb.ai) to check when it would be complete which was nice.<br>
After the model had finished training, I made a prediction where it would only show confidence values over 15%, the result was no labels being displayed at all.<br> "How odd" I thought, better try again with just any confidence levels and...<br>
![image](https://user-images.githubusercontent.com/10008476/133681769-f1088ceb-8c17-4775-89a6-320722ed6ad4.png)<br>
A total waste of time, error graphs appeard to be going down but my brain is friend from all the finickity commands I've done for the past two days so I honestly have no idea.<br>
I spent what was left of the day grabbing some already available data from roboflow, navigating the headless WSFL to get everything in the right places and at the time of writing have started training another model on it, in ~30 mins it is 5% through epoch 1/50 so I am going to bed.

