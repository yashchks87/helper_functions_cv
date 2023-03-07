def get_iou(boxA, boxB):
    # This will calculate boxes given in any order...
    # This will get max value between 2 boxes becuase max mening overlapping
    # area just begins for x
    interxA = max(boxA[0], boxB[0])
    # This will get max value between 2 boxes becuase max mening overlapping
    # area just begins for y
    interyA = max(boxA[1], boxB[1])
    # This will get min value between 2 boxes becuase min mening overlapping
    # area just begins for x because bottom right should be less as it's ending
    interxB = min(boxA[2], boxB[2])
    # This will get min value between 2 boxes becuase min mening overlapping
    # area just begins for y because bottom right should be less as it's ending
    interyB = min(boxA[3], boxB[3])

    # Now basic concept to find area of any rectangle in co-ordinate plane we
    # can think in this way, this is values top-left: (xa, ya),
    # bottom-right: (xb, yb)
    # (xb - xa + 1) * (yb - ya + 1) this formula will yield us the area of that
    # rectangle

    # this will find area of intersection rectangle
    # TODO: understand why 0 is there is used?
    interArea = max(0, interxB - interxA + 1) * max(0, interyB - interyA + 1)

    boxAarea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBarea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    return interArea / (boxAarea + boxBarea - interArea)


if __name__ == "__main__":
    get_iou()
