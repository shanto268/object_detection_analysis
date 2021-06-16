"""
===================================================
Program : CrateAnalysis/FocusMetricAnalyzer.py
===================================================
Summary:
__author__ =  "Sadman Ahmed Shanto"
__date__ = "06/06/2021"
__email__ = "sadman-ahmed.shanto@ttu.edu"

Usage: python3 generateZstack.py config csv_file threshold

"""
import cv2
import os
import sys
import numpy
import glob
from imutils import paths
import img2pdf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
"""
 =================================================================
                    HELPER FUNCTIONS
 =================================================================
"""


def getFigureOfMerit(zplanes, focus_measure, label, loc):
    plt.plot(zplanes, focus_measure, '--x', label=label)
    plt.xlabel('Z Plane (cm)')
    plt.text(42, 0.12, 'Plane of Interest', rotation=0)
    plt.legend()
    plt.ylabel('Focus Measure')
    plt.title('Figure of Merit', fontsize=15)
    plt.grid()
    plt.savefig(loc + "_" + label + ".png")
    plt.clf()
    plt.close("all")


def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(mypath)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise


"""
args = (image directory, threshold, config name)
"""


def classifyImage(args):
    index = 0
    zarr = []
    interp = args[0].split("/")[-1]
    config = args[2]
    threshold = int(args[1])

    for imagePath in paths.list_images(args[0]):
        # load the image, convert it to grayscale, and compute the
        # focus measure of the image using the Variance of Laplacian
        # method
        index += 1
        image = cv2.imread(imagePath)
        #print(imagePath)
        # zdist = imagePath.split("/")[1].split(".")[0].split("-")[1]
        zdist = imagePath.split("/")[-1].split(".")[0].split("img")[1]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)
        text = "Not Blurry"
        # if the focus measure is less than the supplied threshold,
        # then the image should be considered "blurry"
        if fm < threshold:
            text = "Blurry"
        # show the image
        cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 2)
        #print(str(zdist) + " blur score: " + str(fm))
        # cv2.imshow("Image", image)
        mkdir_p("variance_of_laplacian_classifer/{}_{}".format(config, interp))
        current_dir = "variance_of_laplacian_classifer/{}_{}".format(
            config, interp)
        save_dir = "{}/image_{}.png".format(current_dir, index)
        cv2.imwrite(save_dir, image)
        zarr.append([int(zdist), fm])
        # key = cv2.waitKey(0)

    np.savetxt(
        "{}/variance_of_laplacian_score_{}_{}.csv".format(
            current_dir, config, interp), zarr)
    createPdf(current_dir, current_dir.split("/")[-1])


def createPdf(images, output_pdf):
    output_pdf += ".pdf"
    output_pdf = images + "/" + output_pdf
    with open(output_pdf, "wb") as f:
        f.write(img2pdf.convert(glob.glob(images + "/*.png")))
    os.system("rm {}/*.png".format(images))


def getScoreValues(data, config):
    interp = data.split("/")[-2].split("_")[-1]
    score = np.loadtxt(data)
    wd = data.split("/")[-3]

    df = pd.DataFrame(score, columns=["z", "score"])
    df = df.sort_values(by=['z'])
    df["z"] = 4 * (df["z"].values) + 2
    df["interp"] = interp
    df.plot(x="z", y="score", kind="line")
    plt.title("Interpolation: {}".format(interp))
    plt.savefig("{}/{}_{}.png".format(wd, config, interp))
    plt.clf()
    plt.close("all")
    return df


"""
 =================================================================
                        MAIN  FUNCTION
 =================================================================
"""

if __name__ == "__main__":
    """
    =================================================================
                        INPUTS
    =================================================================
    """

    config = sys.argv[1]
    folder = config
    csv_d = sys.argv[2]
    img_dir = glob.glob("images/{}/*".format(folder))
    if len(sys.argv) < 3:
        threshold = sys.argv[3]
        print("No threshold given. Using 100 as default.")
    else:
        threshold = sys.argv[3]
    base_d = "/Users/sshanto/hep/hep_daq/CAMAC/CrateAnalysis/variance_of_laplacian_classifer"
    data_dir = glob.glob("{}/{}*/*csv".format(base_d, config))
    print("\n" + "~" * 60 + "\nCONFIG = {}\n".format(config) + "~" * 60 + "\n")
    # print("config : {}".format(config))
    # print("csv_d : {}".format(csv_d))
    # print("data_dir : {}".format(data_dir))
    """
    =================================================================
                GENERATE Z-STACK FIG FOR DIFFERENT INTERPOLATION
    =================================================================
    """

    print("\n" + "~" * 60 +
          "\nGenerating z-stack figures for different interpolation...\n" +
          "~" * 60 + "\n")
    # I_matrices_files = glob.glob("{}*csv".format(csv_d))
    file = csv_d
    I_matrices_reshaped = np.loadtxt(file)

    filters = [
        'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36',
        'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom',
        'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
    ]

    for j in range(len(filters)):
        dir_name = "images/{}/{}".format(folder, filters[j])
        mkdir_p(dir_name)
        print("Using {} filter".format(filters[j]))
        for k, i in enumerate(I_matrices_reshaped):
            i = i.reshape(21, 21)
            k = k + 2
            fig = plt.figure(frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            imshowobj = ax.imshow(np.flip(i),
                                  aspect='auto',
                                  interpolation=filters[j])
            # imshowobj.set_clim(0.8, 1.2)

            fname = "{}/img{}.png".format(dir_name, k)
            fig.savefig(fname)
            plt.clf()
            plt.close("all")
        """
            for i in I_matrices_reshaped:
            i = i.reshape(21,21)
            plt.matshow(np.flip(i), interpolation=filters[j])
            plt.savefig('{}/img{}.png'.format(dir_name,k),bbox_inches='tight',transparent=True, pad_inches=0)
            k +=1
        """
