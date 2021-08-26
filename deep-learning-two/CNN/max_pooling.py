import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.io.image import ImageReadMode

def max_pool_image(image_tensor, kernel_size=50, stride=50) -> None:
    # Convert the tensor to a float so that we can pool it
    image_tensor = image_tensor.float()
    # We'll use average pooling for this
    m = torch.nn.AvgPool2d(kernel_size=kernel_size,stride=stride)
    pooled_image = m(image_tensor)

    print(pooled_image.shape)

    # Debugging info: I was having trouble getting the entire image to display properly/ at all
    # print(pooled_image[0].mean())
    # print(pooled_image[1].mean())
    # print(pooled_image[2].mean())
    # print(pooled_image[0].shape)

    # pooled_image = pooled_image.permute(1,2,0)
    # print("permuting")
    # print(pooled_image.shape)
    
    # All my values are floats, from the conversion before
    # for x in pooled_image:
    #     print(x)

    # imshow() requires ints, so convert the tensor to int
    pooled_image = pooled_image.int()
    return pooled_image

    
def display_image(image_tensor:torch.Tensor()) -> None:
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

def pixellate(jpg_or_png_path:str, kernel_size:int=25, stride:int=25) -> None:
    image = torchvision.io.read_image(jpg_or_png_path, ImageReadMode.RGB)
    image = image.float()
    display_image(max_pool_image(image, kernel_size=kernel_size, stride=stride))
    return



if __name__ == '__main__':
   
    pixellate("deep-learning-two/CNN/swirls.jpg", 50, 50)


    # Old code, it has now been refactored into pixellate()
    # image = torchvision.io.read_image("deep-learning-two/CNN/fruits.jpg", ImageReadMode.RGB)
    # image = image.float()
    # # display_image(fruit)
    # print(f"Image's shape: {image.shape}")
    # pixellated_image = max_pool_image(image, 50, 50)
    # display_image(pixellated_image)



"""
Below is a lot of ugly debugging code, left in to remind myself to:
1) Use a different function or file to figure out what the problem is rather than create a huge mess where I'm working
2) Take a break and consider different approaches if getting something working is posing an issue
"""



# red = image[0]
# green = image[1]
# blue = image[2]
# p = torch.nn.MaxPool2d(10,10)
# red = red.unsqueeze(2)
# red = red.permute(2,0,1)
# block_red = p(red)

# q = torch.nn.AvgPool2d(10,10)
# green = green.unsqueeze(2)
# green = green.permute(2,0,1)
# block_green = q(green)
# blue = blue.unsqueeze(2)
# blue = blue.permute(2,0,1)
# block_blue = q(blue)

# print("="*50)
# x = block_red.shape[1]
# y = block_red.shape[2]
# print(block_red.shape)

# blocky = torch.empty([3,x,y])
# blocky[0] = block_red
# blocky[1] = block_green
# blocky[2] = block_blue
# # print(big_red_blocks.size)

# # image[0] = block_red
# # image[1] = block_green
# # image[2] = block_blue
# test = block_green
# print(test)
# try:
#     plottable_image = test.permute(1,2,0)
# except:
#     plottable_image = test
# plt.imshow(plottable_image)
# plt.show()




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



