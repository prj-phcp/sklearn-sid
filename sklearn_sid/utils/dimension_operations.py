def to_1D(y):

    yshape = y.shape
    if len(yshape) == 2 and yshape[1] == 1:
        y = y.ravel()
    return y

def to_2D(y):

    yshape = y.shape
    if len(yshape) == 1:
        y = y.reshape(-1,1)
    return y
    