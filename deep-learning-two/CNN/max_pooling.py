import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision.io.image import ImageReadMode

# import os
# print(os.getcwd())


def max_pool_image(image_tensor) -> None:
    image_tensor = image_tensor.float()
    # m = torch.nn.AvgPool2d(1,stride=1)
    # pooled = m(image_tensor)
    # display_image(pooled)

    m = torch.nn.AvgPool2d(kernel_size=3,stride=2)
    r_pool = image_tensor[0]
    r_pool = m(r_pool)
    # g_pool = torch.Tensor(m(image_tensor[1]))
    # b_pool = torch.Tensor(m(image_tensor[2]))
    # seperate_pools = torch.empty((3,r_pool.shape[0],r_pool.shape[1]))
    # seperate_pools[0] = r_pool
    # seperate_pools[1] = g_pool
    # seperate_pools[2] = b_pool
    # display_image(r_pool)
    
    return

    
def display_image(image_tensor) -> None:
    # Tensors are in the form: channel, height, width
    # This will change it to: height, width, channel
    try:
        plottable_image = image_tensor.permute(1,2,0)
    except:
        plottable_image = image_tensor
        pass
    print(plottable_image.shape)
    plt.imshow(plottable_image)
    plt.show()
    return

# def 


image = torchvision.io.read_image("deep-learning-two/CNN/fruits2.jpg", ImageReadMode.RGB)
image = image.float()
# display_image(fruit)
print(f"Image's shape: {image.shape}")

red = image[0]
green = image[1]
blue = image[2]
p = torch.nn.MaxPool2d(10,10)
red = red.unsqueeze(2)
red = red.permute(2,0,1)
block_red = p(red)

q = torch.nn.AvgPool2d(10,10)
green = green.unsqueeze(2)
green = green.permute(2,0,1)
block_green = q(green)
blue = blue.unsqueeze(2)
blue = blue.permute(2,0,1)
block_blue = q(blue)

print("="*50)
x = block_red.shape[1]
y = block_red.shape[2]
print(block_red.shape)

blocky = torch.empty([3,x,y])
blocky[0] = block_red
blocky[1] = block_green
blocky[2] = block_blue
# print(big_red_blocks.size)

# image[0] = block_red
# image[1] = block_green
# image[2] = block_blue
test = block_green
print(test)
try:
    plottable_image = test.permute(1,2,0)
except:
    plottable_image = test
plt.imshow(plottable_image)
plt.show()




# max_pool_image(fruit)
# display_image(fruit)



# # print(fruit.shape)
# print(fruit.shape)
# print(fruit[0].shape) # R
# print(fruit[1].shape) # G
# print(fruit[2].shape) # B



# print("@"*50)
# m = torch.nn.MaxPool1d(3, stride=2)
# input = torch.randn(3, 680, 1024)
# output = m(input)
# print(output)



