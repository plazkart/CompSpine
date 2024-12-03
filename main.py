import pandas as pd
from PIL import Image
import itk

def cropImageIntoSquares(subj):
    import itk
    # open image
    imgSource = itk.imread(subj.pathSub + '\\nii\\' + subj.dixonImg)

    imgSourceArray = itk.array_from_image(imgSource)

    # get central slice
    imgShape = imgSourceArray.shape
    imgCentral = imgSourceArray[0, imgShape[1]//2, :, :]
    imgCroped = itk.image_from_array(imgCentral)
    itk.imwrite(imgCroped, subj.pathSub + '\\nii\\proc\\cropped.nii')



class subjData():
    name: str
    converted: bool
    dixonImg: str
    t1wImg: str
    pathSub: str


def plotImage(img):
    import matplotlib.pyplot as plt
    import itk

    array = itk.array_from_image(img)

    plt.gray()
    plt.imshow(array)
    plt.show()

def convertToNifti(path):
    #this script is dedicated to convert DICOM or .mha files  to nifti
    from nipype.interfaces.dcm2nii import Dcm2niix
    import SimpleITK as sitk

    if 'mha' in path:
        img = sitk.ReadImage(path)
        sitk.WriteImage(img, path[0:-3] + 'nii')
    else:
        converter = Dcm2niix()
        converter.inputs.source_dir = path
        converter.inputs.compression = 5
        converter.inputs.output_dir = path+'\\nii'
        converter.cmdline
        converter.run()


def getImages(path):
    import os
    subj1 = subjData()
    subj1.name = 'sub_01'
    subj1.pathSub = path
    #parse data
    fils = os.listdir(path+"\\nii")
    for i in fils:
        if 'nii' not in i:
            fils.remove(i)
    for i in fils:
        if 'DIX' in i:
            subj1.dixonImg = i
        if ('T1' in i) or ('t1' in i):
            subj1.t1wImg = i
    return subj1

def phaseSymmetryFilter(input_image_file, output_image_file,
            wavelengths=None,
            sigma=0.55,
            polarity=0,
            noise_threshold=10.0):
    import itk
    import numpy as np

    input_image = itk.imread(input_image_file, itk.ctype('float'))
    boundary_condition = itk.PeriodicBoundaryCondition[type(input_image)]()
    padded = itk.fft_pad_image_filter(input_image, boundary_condition=boundary_condition)

    dimension = input_image.GetImageDimension()

    phase_symmetry_filter = itk.PhaseSymmetryImageFilter.New(padded)

    if wavelengths:
        scales = len(wavelengths) / dimension
        wavelength_matrix = itk.Array2D[itk.D](int(scales), dimension)
        np_array = np.array(wavelengths, dtype=np.float64)
        vnl_vector = itk.GetVnlVectorFromArray(np_array)
        wavelength_matrix.copy_in(vnl_vector.data_block())
        phase_symmetry_filter.SetWavelengths(wavelength_matrix)

    if dimension == 2:
        orientation_matrix = itk.Array2D[itk.D](2, dimension)
        np_array = np.array([1, 0, 0, 1], dtype=np.float64)
        vnl_vector = itk.GetVnlVectorFromArray(np_array)
        orientation_matrix.copy_in(vnl_vector.data_block())
        phase_symmetry_filter.SetOrientations(orientation_matrix)
    else:
        orientation_matrix = itk.Array2D[itk.D](3, dimension)
        np_array = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float64)
        vnl_vector = itk.GetVnlVectorFromArray(np_array)
        orientation_matrix.copy_in(vnl_vector.data_block())
        phase_symmetry_filter.SetOrientations(orientation_matrix)

    phase_symmetry_filter.SetSigma(sigma)
    phase_symmetry_filter.SetPolarity(polarity)
    phase_symmetry_filter.SetNoiseThreshold(noise_threshold)

    phase_symmetry_filter.Initialize()
    phase_symmetry_filter.Update()

    output_image = phase_symmetry_filter.GetOutput()


    itk.imwrite(output_image, output_image_file, True)

def thresholding(output_image, output_image_file, thrs):
    output_image = itk.imread(output_image, itk.ctype('float'))
    thresholdFilter = itk.ThresholdImageFilter[type(output_image)].New()

    thresholdFilter.SetInput(output_image)
    thresholdFilter.ThresholdOutside(thrs, 1)
    thresholdFilter.SetOutsideValue(0)
    output_image = thresholdFilter.GetOutput()
    itk.imwrite(output_image, output_image_file, True)

def smooth_image(input_file, sigma, output_image_file):
    input = itk.imread(input_file, itk.ctype('float'))
    smoothFilter = itk.SmoothingRecursiveGaussianImageFilter[type(input), type(input)].New()
    smoothFilter.SetInput(input)
    smoothFilter.SetSigma(sigma)

    output_image = smoothFilter.GetOutput()
    itk.imwrite(output_image, output_image_file, True)

def ASM_method(input_file):
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.color import rgb2gray
    from skimage import data
    from skimage.filters import gaussian
    from skimage.segmentation import active_contour

    img = itk.imread(input_file, itk.ctype('float'))
    #img = rgb2gray(img)

    s = np.linspace(0, 2 * np.pi, 400)
    r = 100 + 100 * np.sin(s)
    c = 220 + 100 * np.cos(s)
    init = np.array([r, c]).T

    snake = active_contour(
        gaussian(img, sigma=3, preserve_range=False),
        init,
        alpha=0.015,
        beta=10,
        gamma=0.001,
    )

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray)
    ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])

    plt.show()

def exploreImage(image_file):
    import numpy as np
    import nibabel as nb
    from matplotlib import pyplot as plt

    img = nb.load(image_file)
    data = img.get_fdata()
    img2 = nb.load(path + '\\nii\\proc\\cropped.nii')
    data2 = img2.get_fdata()
    plt.hist(data2[data>0.7])

def cornerHarris_method(img_filename, output_image_file):
    import cv2 as cv
    import numpy as np

    img = itk.imread(img_filename, itk.ctype('float'))
    #img = cv.imread(img_filename)
    ret, images = cv.imreadmulti(filename = img_filename)
    #img = itk.array_from_image(img)
    #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    gray = np.float32(img)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv.dilate(dst, None)
    dst = itk.image_from_array(dst)
    itk.imwrite(dst, output_image_file, True)
    # Threshold for an optimal value, it may vary depending on the image.
    #img[dst > 0.01 * dst.max()] = [0, 0, 255]

   # cv.imshow('dst', img)
    #if cv.waitKey(0) & 0xff == 27:
    #    cv.destroyAllWindows()

def simpleExample(input, output):
    image = itk.imread(input)
    median = itk.median_image_filter(image, radius=2)
    itk.imwrite(median, output)


#path = "C:\\Users\\user\\Documents\\DATA_for_WORK\\Datasets\\Spinal\\10159290\\images\\images"
#main(path)

#phaseSymmetryFilter('C:\\Users\\user\\Documents\\DATA_for_WORK\\Datasets\\Spinal\\10159290\\images\\images\\nii\\5_t1.nii', 'C:\\Users\\user\\Documents\\DATA_for_WORK\\Datasets\\Spinal\\10159290\\images\\images\\nii\\5_t1.tif')

#exctractSlice('C:\\Users\\user\\YandexDisk\\Work\\data\\openDataset\\Spinal\\my_compression\\nii\\my_compression_WIP_T2_SAG_COUNT_20240629165849_401.nii.gz',
#'C:\\Users\\user\\YandexDisk\\Work\\data\\openDataset\\Spinal\\my_compression\\nii\\my_compression_WIP_T2_SAG_COUNT_20240629165849_401_2.nii.gz',
 #             5)
#phaseSymmetryFilter('C:\\Users\\user\\YandexDisk\\Work\\data\\openDataset\\Spinal\\my_compression\\nii\\spine.png',
#                    'C:\\Users\\user\\YandexDisk\\Work\\data\\openDataset\\Spinal\\my_compression\\nii\\spine_2.tiff')

path = "C:\\Users\\Admin\\YandexDisk\\Work\\data\\openDataset\\Spinal\\my_compression"

if __name__ == '__main__':
    #asas
    #convertToNifti(path)
    subj1 = getImages(path)
    print(subj1.t1wImg)

    #cropImageIntoSquares(subj1)
    #smooth_image(path + '\\nii\\proc\\cropped.nii', 0.7, path + '\\nii\\proc\\cropped_s.nii')
    #phaseSymmetryFilter(path + '\\nii\\proc\\cropped_s.nii', path + '\\nii\\proc\\cropped_sp.nii')
    #thresholding(path + '\\nii\\proc\\cropped_sp.nii', path + '\\nii\\proc\\cropped_spt_' + str(0.8) + '.nii', 0.8)
    exploreImage(path + '\\nii\\proc\\cropped_spt_' + str(0.8) + '.nii')
    #smooth_image(path + '\\nii\\proc\\cropped_psf.tiff', 1.2, path + '\\nii\\proc\\cropped_psf_sm.tiff')
    #for i in range(10):
    #    i = i/10
    #    thresholding(path + '\\nii\\proc\\cropped_sp.tiff', path + '\\nii\\proc\\cropped_spt_' + str(i) + '.tiff', i)
    #cornerHarris_method(path + '\\nii\\proc\\cropped_psf_trh.tiff', path + '\\nii\\proc\\cropped_psf_trh_ch.tiff' )
    #ASM_method(path + '\\nii\\proc\\cropped_psf_trh_ch.tiff')
    pass