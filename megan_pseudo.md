
for each image in z-stack:
    focus = getFocusScore(image) #this method developed by clive
    if focus == True:
        ImageSegmentation(image)
    else:
        pass


def ImageSegmentation(image):
    """ 
    image is in a given X,Y,Z plane
    """
    num_objects = getNumberObjects(image)
    for object in num_objects:
        xsize, ysize = getSize(object)
    return num_objects, xsize, ysize



