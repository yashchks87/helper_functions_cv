# Author: Yash Choksi
# Date: Mar 06th 2022

# import libraries
import PIL
import numpy as np

def show_dataset(thumb_size, cols, rows, ds):
  # create now mosaic paint of image of images
  # Important link: https://towardsdatascience.com/how-to-create-a-photo-mosaic-in-python-45c94f6e8308
  mosaic = PIL.Image.new(mode='RGB', size=(thumb_size*cols + (cols-1), 
                                         thumb_size*rows + (rows-1)))
   
    # For loop to iterate over images
    for idx, data in enumerate(iter(ds)):
        # Try block which will look at images and de couple if they have image and id both
        try:
          img, target_or_imgid = data
        # If only image is present it will go to except block
        except:
          img = data
        # to get co-ordinate position ix will look at x axis and iy will look at y axis
        ix  = idx % cols
        iy  = idx // cols
        # Get the image and change data type to uint8
        img = np.clip(img.numpy() * 255, 0, 255).astype(np.uint8)
        img = PIL.Image.fromarray(img)
        # image resizing upto the given fixed size
        img = img.resize((thumb_size, thumb_size), resample=PIL.Image.BILINEAR)
        # paste created image to the given x-y place with given size.
        mosaic.paste(img, (ix*thumb_size + ix, 
                           iy*thumb_size + iy))


    display(mosaic)

if '__name__' == '__main__':
    show_dataset()