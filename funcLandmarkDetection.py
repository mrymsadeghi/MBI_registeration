
import cv2 as cv
import numpy as np
import time
import os

def funcLandmarkDetection(imgpath, midheight):

    LandmarksT = []

    tissueimg = cv.imread(imgpath)

    timg_gray = cv.cvtColor(tissueimg, cv.COLOR_BGR2GRAY)
    img_gr_mblur = cv.medianBlur(timg_gray, 9)
    _, thresh = cv.threshold(img_gr_mblur, 20, 255, cv.THRESH_BINARY)
    kernel = np.ones((11,11),np.uint8)
    mask = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    wcoords = np.where(mask == 255)

    # Boundaries
    minr, maxr = min(wcoords[0]), max(wcoords[0])
    minc, maxc = min(wcoords[1]), max(wcoords[1])

    avgr = int((maxr - minr) / 2) + minr
    avgc = int((maxc - minc) / 2) + minc

    wcoordstup= list(zip(wcoords[0], wcoords[1]))
    wcoordstupL = list(filter(lambda x: (x[1] < avgc), wcoordstup))
    wcoordstupR = list(filter(lambda x: (x[1] > avgc), wcoordstup))
    wcoordstup.sort(key=lambda x: x[1])
    wcoordstupL.sort(key=lambda x: x[0])
    wcoordstupR.sort(key=lambda x: x[0])

    # Point1
    ptTR = wcoordstupR[0]
    LandmarksT.append(ptTR)

    midw = 120
    midh = midheight

    # Point2
    ptRc = wcoordstup[-1][1]
    ptRr = [t[0] for t in wcoordstup if t[1] == ptRc][0]
    ptR = (ptRr, ptRc)
    LandmarksT.append(ptR)

    # Point3
    ptRB = wcoordstupR[-1]
    LandmarksT.append(ptRB)

    # Point4
    maskmidbott = mask[tissueimg.shape[0] - midh:, avgc - midw:avgc + midw]
    bcoordsmb = np.where(maskmidbott == 0)
    bcoordsmbtup = list(zip(bcoordsmb[0], bcoordsmb[1]))
    bcoordsmbtup.sort(key=lambda x: x[0])
    pmb = bcoordsmbtup[0]
    ptMB = (pmb[0] + tissueimg.shape[0] - midh, pmb[1] + avgc - midw)
    LandmarksT.append(ptMB)

    # Point5
    ptLB = wcoordstupL[-1]
    LandmarksT.append(ptLB)

    # Point6
    ptLc = wcoordstup[0][1]
    ptLr = [t[0] for t in wcoordstup if t[1] == ptLc][0]
    ptL = (ptLr, ptLc)
    LandmarksT.append(ptL)

    # Point7
    ptTL = wcoordstupL[0]
    LandmarksT.append(ptTL)

    # Point8
    maskmidtop = mask[0:midh, avgc - midw:avgc + midw]
    bcoordsmt = np.where(maskmidtop == 0)
    bcoordsmttup = list(zip(bcoordsmt[0], bcoordsmt[1]))
    bcoordsmttup.sort(key=lambda x: x[0])
    pmt = bcoordsmttup[-1]
    ptMT = (pmt[0], pmt[1] + avgc - midw)
    LandmarksT.append(ptMT)

    # Point 9-12
    rr = int((ptMB[0] - ptMT[0])/5)
    cc = int((ptMB[1] - ptMT[1])/5)
    i=1
    print('Middle Landmarks')
    for idist in range(1,5):
        pr = idist * rr + ptMT[0]
        pc = idist * cc + ptMT[1]
        cv.putText(tissueimg, str(i), (pc, pr), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv.LINE_AA)
        LandmarksT.append((int(pr), int(pc)))
        i += 1
    LandmarksT = [(200, 200), (800,200),(200, 1200), (800, 1200)]
    return LandmarksT
