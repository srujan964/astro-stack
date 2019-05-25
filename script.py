import sys
import numpy as np
from astropy.io import fits
from skimage import color
from skimage.io import imsave
from skimage.transform import resize
from skimage.exposure import rescale_intensity


#Color greyscale image with the corresponding hue value
def colorize(image, hue, s=1,v=1):
    hsv = color.rgb2hsv(image)
    hsv[:, :, 0] = hue
    hsv[:, :, 1] = s
    hsv[:, :, 2] *= v
    return color.hsv2rgb(hsv)


#Normalize and resize to 1750 x 1750 resolution
def normalize(image):
    img = image * (255 / np.max(image))
    img = img.transpose()
    img = resize(img, (1750, 1750), mode='constant')
    return img


def rescale_by_percentile(image,perclow=0.1,perchigh=99.9):
    lo, hi = np.percentile(image, (perclow, perchigh))
    return rescale_intensity(image, in_range=(lo, hi))


def main():
    if len(sys.argv) == 1:
        sys.exit(
            "Usage:python script.py low_wavelength.fits mid_wavelength.fits high_wavelength.fits"
        )
    elif len(sys.argv) == 4:
        file_1 = sys.argv[1]
        file_2 = sys.argv[2]
        file_3 = sys.argv[3]
    else:
        sys.exit(
            "Usage: python script.py low_wavelength.fits mid_wavelength.fits high_wavelength.fits"
        )

    print('Reading datasets...')
    raw_one = fits.getdata(file_1)
    raw_two = fits.getdata(file_2)
    raw_three = fits.getdata(file_3)

    low = normalize(raw_one)
    mid = normalize(raw_two)
    high = normalize(raw_three)

    print('Colouring...')
    low_greyRGB = color.gray2rgb(low)
    mid_greyRGB = color.gray2rgb(mid)
    high_greyRGB = color.gray2rgb(high)

    red_hue = 0
    green_hue = 60
    blue_hue = 120

    low_RGB = colorize(low_greyRGB, blue_hue)
    mid_RGB = colorize(mid_greyRGB, green_hue)
    high_RGB = colorize(high_greyRGB, red_hue)

    final_rgb = np.nanmean([low_RGB, mid_RGB, high_RGB], axis=0)

    for i in [0,1,2]:
        final_rgb[:,:,i] = rescale_by_percentile(final_rgb[:,:,i])
    
    print('Saving image...')
    imsave('images/final.png', final_rgb)


if __name__ == "__main__":
    main()