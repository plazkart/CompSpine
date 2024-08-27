def main(path):
    #asas
    #convertToNifti(path)
    subj1 = getImages(path)
    print(subj1.t1wImg)

    return None

def preInitiliasize(): 
    #install itk
    #install itk-phasesymmetry
    return None

class subjData():
    name: str
    converted: bool
    dixonImg: str
    t1wImg: str

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
    #parse data
    fils = os.listdir(path+"\\nii")
    for i in fils:
        if 'nii' not in i:
            fils.remove(i)
    for i in fils:
        if 'DIX' in i:
            subj1.dixonImg = i
        if 'T1' or 't1' in i:
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
    getImageSlice(input_image)
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

    output_image = phase_symmetry_filter.GetOutput()
    itk.imwrite(output_image, output_image_file, True)

def getImageSlice(inputImage):
    import itk
    
    Dimension = 3
    PixelType = itk.ctype("short")
    ImageType = itk.Image[PixelType, Dimension]
    extractFilter = itk.ExtractImageFilter.New(inputImage)
    extractFilter.SetDirectionCollapseToSubmatrix()
    
    
    # set up the extraction region [one slice]
    inputRegion = inputImage.GetBufferedRegion()
    size = inputRegion.GetSize()
    size[2] = 1  # we extract along z direction
    start = inputRegion.GetIndex()
    sliceNumber = 7
    start[2] = sliceNumber
    desiredRegion = inputRegion
    desiredRegion.SetSize(size)
    desiredRegion.SetIndex(start)
    
    outputImage = extractFilter.SetExtractionRegion(desiredRegion)
    return outputImage

    #itk.imwrite(pasteFilter.GetOutput(), args.output_3D_image)


def simpleExample(input, output):
    image = itk.imread(input)
    median = itk.median_image_filter(image, radius=2)
    itk.imwrite(median, output)

path = "C:\\Users\\user\\Documents\\DATA_for_WORK\\Datasets\\Spinal\\10159290\\images\\images"
#main(path)

phaseSymmetryFilter('C:\\Users\\Admin\\YandexDisk\\Work\\data\\openDataset\\Spinal\\my_compression\\nii\\my_compression_WIP_SAG_mDIX_TSE_T2_20240629165849_501.nii.gz', 'C:\\Users\\Admin\\YandexDisk\\Work\\data\\openDataset\\Spinal\\my_compression\\nii\\my_compression_WIP_SAG_mDIX_TSE_T2_20240629165849_501.tif')
