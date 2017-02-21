import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd
import gzip


from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    
    for s in slices:
        s.SliceThickness = slice_thickness
    
    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))
    
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing


def zeroPad(image,targetSize=(80,95,95)):
    # zero pad the image to target size
    currentSize= image.shape
    paddingSize= np.array(targetSize) - np.array(currentSize)
    if min(paddingSize) < 0:
        print 'Issue: need to cut down!'
    l= []
    for ps in paddingSize:
        if ps%2==0:
            l.append( (int(ps)/2,int(ps)/2) )
        else:
            l.append( (int(ps)/2+1, int(ps)/2) )
    return np.lib.pad(image,l,'constant')


def main():
    dataPath= '/Volumes/Persephone/Kaggle/stage1'
    outPath= '/Volumes/Persephone/Kaggle/numpy'
#    paths= os.listdir(dataPath)
    labelFile= '/Volumes/Persephone/Kaggle/stage1_labels.csv'
    nFiles= 100

    label_df= pd.read_csv(labelFile)
    paths= label_df.id.values
    y= label_df.cancer.values
    
    maxSize= np.zeros(3)
    ctr= 0
    imageData= np.zeros((nFiles,80,95,95))
    for p in paths[1:nFiles+1]:
        print ctr
        path= dataPath + '/'+p
        slices= load_scan(path)
        pixels_hu = get_pixels_hu(slices)
        pix_resampled, spacing = resample(pixels_hu, slices, [5,5,5])
        for i in range(3):
            if pix_resampled.shape[i] > maxSize[i]:
                maxSize[i] = pix_resampled.shape[i]
        pix_padded= zeroPad(pix_resampled)
        imageData[ctr,:,:,:]= pix_padded
        ctr += 1

    dataDict= { 'imageData': imageData,
                'labels': y[0:nFiles] }
    pkl.dump(dataDict,open(outPath + '/' + 'processed.pkl','wb'))

#    plt.imshow(pix_padded[15], cmap=plt.cm.gray)
#    plt.show()

if __name__ == "__main__":
    main()
