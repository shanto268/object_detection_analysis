"""
===================================================
Program : CrateAnalysis/FocusMetricAnalyzer.py
===================================================
Summary:
__author__ =  "Sadman Ahmed Shanto"
__date__ = "06/06/2021"
__email__ = "sadman-ahmed.shanto@ttu.edu"

Usage: python3 FocusMetricAnalyzer.py config data_directory threshold

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
                FOCUS METRICS DEFINITION
 =================================================================
"""


def LAPV(img):
    """Implements the Variance of Laplacian (LAP4) focus measure
    operator. Measures the amount of edges present in the image.
    :param img: the image the measure is applied to
    :type img: numpy.ndarray
    :returns: numpy.float32 -- the degree of focus
    """
    return numpy.std(cv2.Laplacian(img, cv2.CV_64F))**2


def LAPM(img):
    """Implements the Modified Laplacian (LAP2) focus measure
    operator. Measures the amount of edges present in the image.
    :param img: the image the measure is applied to
    :type img: numpy.ndarray
    :returns: numpy.float32 -- the degree of focus
    """
    kernel = numpy.array([-1, 2, -1])
    laplacianX = numpy.abs(cv2.filter2D(img, -1, kernel))
    laplacianY = numpy.abs(cv2.filter2D(img, -1, kernel.T))
    return numpy.mean(laplacianX + laplacianY)


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


def TENG(img):
    """Implements the Tenengrad (TENG) focus measure operator.
    Based on the gradient of the image.
    :param img: the image the measure is applied to
    :type img: numpy.ndarray
    :returns: numpy.float32 -- the degree of focus
    """
    gaussianX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gaussianY = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    return numpy.mean(gaussianX * gaussianX + gaussianY * gaussianY)


def MLOG(img):
    """Implements the MLOG focus measure algorithm.
    :param img: the image the measure is applied to
    :type img: numpy.ndarray
    :returns: numpy.float32 -- the degree of focus
    """
    return numpy.max(cv2.convertScaleAbs(cv2.Laplacian(img, 3)))


def brenner(Imatrix):
    #print("len: ", len(Imatrix))
    rows, cols = int(len(Imatrix)**0.5), int(len(Imatrix)**0.5)
    Imatrix = Imatrix.reshape(rows, cols)
    brenner = 0  # out of focus
    for i in range(cols):
        for j in range(rows):
            try:
                p = Imatrix[i][j]
                p2 = Imatrix[i][j + 2]
                # print("{} - {}".format(p2,p))
                brenner += (p2 - p)**2
            except:
                pass
    return brenner


# SMD (Grayscale Variance) function
def SMD(Imatrix):
    rows, cols = int(len(Imatrix)**0.5), int(len(Imatrix)**0.5)
    Imatrix = Imatrix.reshape(rows, cols)
    value = 0  # out of focus
    for i in range(cols):
        for j in range(rows):
            try:
                p = Imatrix[i][j]
                p1 = Imatrix[i - 1][j]
                p2 = Imatrix[i][j + 1]
                # print("{} - {}".format(p2,p))
                value += abs(p - p1) + abs(p - p2)
            except:
                pass
    return value


def std_based_correlation(Imatrix):
    pass


def histogram_range(Imatrix):
    pass


def threshold_content(Imatrix):
    pass


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
    img_dir = glob.glob(
        "/Users/sshanto/hep/hep_daq/CAMAC/focus-stacking/images/{}/*".format(
            folder))
    if len(sys.argv) < 3:
        threshold = sys.argv[3]
        print("No threshold given. Using 100 as default.")
    else:
        threshold = sys.argv[3]
    base_d = "/Users/sshanto/hep/hep_daq/CAMAC/CrateAnalysis/variance_of_laplacian_classifer"
    data_dir = glob.glob("{}/{}*/*csv".format(base_d, config))
    # print("config : {}".format(config))
    # print("csv_d : {}".format(csv_d))
    # print("data_dir : {}".format(data_dir))
    """
    =================================================================
                        EVALUATING EACH METRIC
    =================================================================
    """
    print("\n" + "~" * 60 + "\nStarting Evaluating each focus metric...\n" +
          "~" * 60 + "\n")
    zplanes = np.array([2 + (n - 1) * 4 for n in range(1, 27)])
    I_matrices_files = glob.glob("{}*csv".format(csv_d))
    print("I_matrices_files : {}".format(I_matrices_files))
    I_matrices = [np.loadtxt(i) for i in I_matrices_files]
    I_matrices_reshaped = I_matrices[0]  #csv file in data_dir

    focus_measure_LAPV = []
    focus_measure_LAPM = []
    focus_measure_TENG = []
    focus_measure_MLOG = []
    focus_measure_Brenner = []
    focus_measure_SMD = []

    for i in I_matrices_reshaped:
        # print(i)
        focus_measure_LAPV.append(LAPV(i))
        focus_measure_LAPM.append(LAPM(i))
        focus_measure_TENG.append(TENG(i))
        focus_measure_Brenner.append(brenner(i))
        focus_measure_SMD.append(SMD(i))
    # focus_measure_MLOG.append(MLOG(i))

    getFigureOfMerit(zplanes, focus_measure_LAPV, "LAPV", loc=csv_d + config)
    getFigureOfMerit(zplanes, focus_measure_LAPM, "LAPM", loc=csv_d + config)
    getFigureOfMerit(zplanes, focus_measure_TENG, "TENF", loc=csv_d + config)
    getFigureOfMerit(zplanes, focus_measure_SMD, "SMD", loc=csv_d + config)
    getFigureOfMerit(zplanes,
                     focus_measure_Brenner,
                     "Brenner",
                     loc=csv_d + config)
    """
    =================================================================
                GENERATE Z-STACK FIG FOR DIFFERENT INTERPOLATION
    =================================================================
    """

    print("\n" + "~" * 60 +
          "\nGenerating z-stack figures for different interpolation...\n" +
          "~" * 60 + "\n")
    file = I_matrices_files[0]
    I_matrices_reshaped = np.loadtxt(file)

    filters = [
        'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36',
        'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom',
        'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
    ]

    for j in range(len(filters)):
        dir_name = "/Users/sshanto/hep/hep_daq/CAMAC/focus-stacking/images/{}/{}".format(
            folder, filters[j])
        mkdir_p(dir_name)
        k = 0
        print("Using {} filter".format(filters[j]))
        for i in I_matrices_reshaped:
            i = i.reshape(21, 21)
            fig = plt.figure(frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            imshowobj = ax.imshow(np.flip(i),
                                  aspect='auto',
                                  interpolation=filters[j])
            imshowobj.set_clim(0.9, 1.2)
            fname = "{}/img{}.png".format(dir_name, k)
            fig.savefig(fname)
            plt.clf()
            plt.close("all")
            k += 1
        """
            for i in I_matrices_reshaped:
            i = i.reshape(21,21)
            plt.matshow(np.flip(i), interpolation=filters[j])
            plt.savefig('{}/img{}.png'.format(dir_name,k),bbox_inches='tight',transparent=True, pad_inches=0)
            k +=1
        """
    """
    =================================================================
                CLASSIFYING Z-STACK UNDER DIFFERENT INTERPOLATION INTO FOCUS USING Variance of Laplacian
    =================================================================
    """
    print(
        "\n" + "~" * 60 +
        "\nClassifying z-stack under different interpolation into focus using Variance of Laplacian...\n"
        + "~" * 60 + "\n")

    for images in img_dir:
        classifyImage([images, threshold, config])
        print("{} is done!".format(images.split("/")[-1]))

    interp_score_dfs = []
    """
    =================================================================
            SCORE OF Z-STACK UNDER DIFFERENT INTERPOLATION Variance of Laplacian
    =================================================================
    """

    print(
        "\n" + "~" * 60 +
        "\nCreating Score Plot under various interpolations using Variance of Laplacian...\n"
        + "~" * 60 + "\n")
    for data in data_dir:
        interp = getScoreValues(data, config)["interp"].values[0]
        #if (interp != "nearest") or (interp != "spline36"):
        interp_score_dfs.append(getScoreValues(data, config))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = sns.color_palette("hls", len(interp_score_dfs))
    ax.set_prop_cycle('color', colors)

    for df in interp_score_dfs:
        interp = df["interp"].values[0]
        plt.plot(df["z"].values, df["score"].values, label=interp)
        plt.xlabel("Z plane (cm)")
        plt.ylabel("Classification Score")
        plt.title(
            "Comparison of Classification Score for Different Interpolation Schemes"
        )
        plt.yscale('log')
    fig.set_size_inches(8, 6)
    plt.legend(loc=(1.01, 0))
    plt.tight_layout()
    plt.savefig("{}/{}_all_interp.png".format(base_d, config), dpi=100)
