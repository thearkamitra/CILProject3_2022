""" This module processes the dataset Massachusetts obtained from https://www.cs.toronto.edu/~vmnih/data/.
The original dataset contains satellite images of Massachusetts and its surroundings. The images are
of size 1500x1500, 3 channels, TIFF format. The dataset also contains separate labels for roads and
buildings.
Processing: The original satellite images are of different zoom level and size than the dataset provided for the project,
it needs to be rescaled and cropped (both the satellite image and its corresponding mask).
From each original image the non-overlapping patches are taken and only those that contain at least `maskWhitePxRatioTh` * 100
percent of roads are kept. The resulting patches are stored in `outputPath` directory.
"""

from PIL import Image
from matplotlib import pyplot as plt
import os
import numpy as np

########################################################################################################################
#                                               INPUT PARAMETERS
########################################################################################################################

# Input dataset path.
inputPath = "./massachusetts-road-dataset/tiff/"
outputPath = "./massachusetts-road-dataset/processed0.05/"
mapDir = "groundtruth"
satDir = "images"

# Threshold for all-white parts of the satellite images - ratio of white pixels (intensity == 255). If the white/other
# ratio is higher than this threshold, the image is dropped.
whitePxRatioTh = 0.001
# Threshold of roads vs. background within mask patch - if the roads/background ratio is lower then this threshold,
# the patch is dropped.
maskWhitePxRatioTh = 0.05

# Upscale image and mask ratio.
upscale = (2.0, 2.0)
patchSize = (400, 400)

########################################################################################################################
#                                                  MAIN SCRIPT
########################################################################################################################

imagesFiles = [im for im in os.listdir(inputPath + satDir) if im.endswith(".tiff")]

numFiles = len(imagesFiles)

for idx, imgFile in enumerate(imagesFiles):

    print("Processing image {im} / {tot}".format(im=idx + 1, tot=numFiles))

    # Load satellite image.
    img = Image.open(inputPath + satDir + "/" + imgFile)
    assert img.mode == "RGB"

    # Get image size.
    imgSize = img.size

    # Convert image to grayscale.
    gsImg = img.convert(mode="L")
    hist = gsImg.histogram()

    whitePxRatio = float(hist[255]) / (imgSize[0] * imgSize[1])

    # If the image contains no or insignificant white parts, process it further.
    if whitePxRatio < whitePxRatioTh:
        # Load ground truth road binary mask
        try:
            gtMask = Image.open(inputPath + mapDir + "/" + imgFile[:-1])
        except:
            print(
                "Error: cannot open ground truth binary mask file {f}".format(
                    f=inputPath + mapDir + "/" + imgFile
                )
            )
            continue

        # Check that mask's size matches the corresponding image.
        assert gtMask.size == imgSize

        # Upscale the image and the mask. For upsampling, nearest neighbour (NEAREST) is used.
        # Another possible option is BICUBIC (only for satellite img), which, however, blurs the image. We need to experiment
        # to find out which one is better.
        newSize = (int(imgSize[0] * upscale[0]), int(imgSize[1] * upscale[1]))
        imgSize = newSize

        # Check that at least one patch can fit in the original image.
        assert newSize[0] // patchSize[0] > 0
        assert newSize[1] // patchSize[1] > 0

        img = img.resize(newSize, resample=Image.NEAREST)
        gtMask = gtMask.resize(newSize, resample=Image.NEAREST)

        # Generate x,y coordinates of centers of patches.
        left = 0
        right = imgSize[0] - patchSize[0]
        top = 0
        bottom = imgSize[1] - patchSize[1]

        numPatchesInRow = imgSize[0] // patchSize[0]
        numPatchesInCol = imgSize[1] // patchSize[1]

        centersInRow = np.linspace(left, right, numPatchesInRow, dtype=np.int32)
        centersInCol = np.linspace(top, bottom, numPatchesInCol, dtype=np.int32)

        # Coordinates of patches (left, top, right, bottom)
        patchesCoords = [
            (l, t, l + patchSize[0], t + patchSize[1])
            for t in centersInCol
            for l in centersInRow
        ]

        # Process each patch
        for pc in patchesCoords:
            # Get a patch of img and mask.
            patchMask = gtMask.crop(pc)
            patchImg = img.crop(pc)

            # Check correct size of a patch.
            assert patchMask.size == patchSize

            # Find the ratio of white pixels (roads) to black pixels (background).
            patchMaskHist = patchMask.histogram()
            maskWhitePxRatio = float(patchMaskHist[255]) / (patchSize[0] * patchSize[1])

            # Check whether there is sufficient amount of roads in this patch and if so, save the patch (img and mask).
            if maskWhitePxRatio > maskWhitePxRatioTh:
                nameSuffix = (
                    "_("
                    + str(pc[1] + patchSize[1] // 2)
                    + ", "
                    + str(pc[0] + patchSize[0] // 2)
                    + ")"
                )
                name = imgFile[:-5] + nameSuffix + ".tiff"

                patchImg.save(outputPath + satDir + "/" + name)
                patchMask.save(outputPath + mapDir + "/" + name)
