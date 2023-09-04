import numpy as np
from PIL import Image
import PIL

def print_images(files, thumb_size, rows, cols, save = False, save_path = None):
    mosaic = Image.new(
        mode="RGB",
        size=(thumb_size * cols + (cols - 1), thumb_size * rows + (rows - 1)),
    )

    for idx, path in enumerate(files):
        img = Image.open(path)
        img = np.array(img)
        ix = idx % cols
        iy = idx // cols
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = Image.fromarray(img)
        img = img.resize((thumb_size, thumb_size), resample=PIL.Image.BILINEAR)
        mosaic.paste(img, (ix * thumb_size + ix, iy * thumb_size + iy))
    
    display(mosaic)

    if save:
        if save_path is not None:
            mosaic.save(save_path)

