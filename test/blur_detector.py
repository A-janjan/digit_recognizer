import numpy as np
import imutils
import cv2

def detect_blur_fft(image, size=60, thresh=15):
    (h, w) = image.shape
    (cX, cY) = (int(w/2.0), int(h/2.0))

    # compute the FFT to find the frequency transform, then shift
    # the zero-frequency component (i.e., DC component located at
    # the top-left corner) to the center, where it will be more
    # easy to analyze
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0  #remove low frequencies
    fftShift = np.fft.ifftshift(fftShift)   #inverse shift
    recon = np.fft.ifft2(fftShift)          #inverse FFT

    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    magnitude = 20 * np.log(np.abs(recon))
    mean = abs(np.mean(magnitude))

    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value
    return (mean, mean <= thresh)