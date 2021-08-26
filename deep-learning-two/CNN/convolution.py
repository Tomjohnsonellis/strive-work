import torch
import torchvision
from torchvision.io.image import ImageReadMode
import matplotlib.pyplot as plt

"""
torch.nn.Conv2d
This is used to apply a convolution to some input

Typical usage:
m = nn.Conv2d(>Parameters go here<)
output = m(input)

It has quite a few parameters so I'll go through them here
"""



# How many channels does your input have? E.g. 3 for an RGB image
in_channels = 3
# How many channels will your network output?
out_channels = 3
# Size of the convolution window? E.g. 3x3
kernel_size = 3 # Or (3,6) for a 3x6
# (Optional) Stride size? (Space between convolution points)
stride = 3
# (Optional) How much padding? (Size of dummy data added around the input)
padding = 1
# (Optional) What will you pad the input with? (Defaults to 0s)
padding_mode = "zeros"
# (Optional) How much space between kernel elements?
dilation = 3 # Spreads out the "vision" of the kernel filter
# (Optional) How many groups of convolutions? 2 would split the conv layer in half, careful with this!
groups = 1
# (Optional) Do you want a bias? Defaults to True, probbaly leave this alone
bias = True
# (Optional) Any device changes?
device=None # "cuda" will run it on the GPU
# (Optional) Datatype?
dtype=torch.float

# That's all the possible parameters, let's use them!
m = torch.nn.Conv2d(
    in_channels=in_channels, 
    out_channels=out_channels,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding,
    padding_mode=padding_mode,
    dilation=dilation,
    groups=groups,
    bias=bias,
    device=device,
    dtype=dtype,
    )

# And let's try it out!
# Get some image
image = torchvision.io.read_image("deep-learning-two/CNN/fruits.jpg", ImageReadMode.RGB).float()
# Convolutions expect a batch of images, we're just doing one image so need add a dimension
image = image.unsqueeze(0)
convolved = m(image)

# And matplotlib expects just the image data
convolved = convolved.squeeze(0)
# In a certain format
convolved = convolved.permute(1,2,0)
convolved = convolved.detach().numpy()
plt.imshow(convolved)
plt.show()
