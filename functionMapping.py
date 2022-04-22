
import os
import numpy as np
import cv2 as cv

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


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
    #LandmarksT = [(100, 100), (800, 100),(100, 1200), (800, 1200)]

    return LandmarksT


def funcMapping(savepath, source_img_path, target_img_path, source_lms, target_lms):
    print("source_lms", source_lms)
    print("target_lms", target_lms)

    print("source_img_path", source_img_path)
    target_img = cv.imread(target_img_path)  # Atlas image   ################
    source_img = cv.imread(source_img_path)  # Section image
    target_gray = cv.cvtColor(target_img, cv.COLOR_BGR2GRAY)
    source_gray = cv.cvtColor(source_img, cv.COLOR_BGR2GRAY)
    mask = np.zeros_like(target_gray)
    height, width, channels = target_img.shape
    target_new_hull = np.zeros((height, width, channels), np.uint8)

    source_additional_lms = funcLandmarkDetection(source_img_path, 200)
    target_additional_lms = funcLandmarkDetection(target_img_path, 200)

    source_lms.extend(source_additional_lms)
    target_lms.extend(target_additional_lms)

    source_landmark_points = []
    target_landmark_points = []

    for point in source_lms:
        source_landmark_points.append((point[1], point[0]))

    for point in target_lms:
        target_landmark_points.append((point[1], point[0]))

    points = np.array(target_landmark_points, np.int32)    
    convexhull = cv.convexHull(points)
    rect = cv.boundingRect(convexhull)
    save_target_lm_points = np.array(target_landmark_points, np.int32)
    print(source_img.shape)
    subdiv = cv.Subdiv2D(rect)
    for p in target_landmark_points:
        subdiv.insert(p)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)
        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)
        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)
        
        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)

    points2 = np.array(source_landmark_points, np.int32)
    convexhull2 = cv.convexHull(points2)
    rect2 = cv.boundingRect(convexhull2)

    save_source_lm_points = np.array(source_landmark_points, np.int32)

    # Triangulation of both faces
    transformation_functions = []
    triangles1 = []
    triangles2 = []
    iii = 1
    for triangle_index in indexes_triangles:
        # Triangulation of the first image: Source
        tr1_pt1 = source_landmark_points[triangle_index[0]]
        tr1_pt2 = source_landmark_points[triangle_index[1]]
        tr1_pt3 = source_landmark_points[triangle_index[2]]
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
        triangles1.append(triangle1)
        rect1 = cv.boundingRect(triangle1)
        (x, y, w, h) = rect1
        cropped_triangle = source_img[y: y + h, x: x + w]
        cropped_tr1_mask = np.zeros((h, w), np.uint8)
        subpoints = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                        [tr1_pt2[0] - x, tr1_pt2[1] - y],
                        [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

        cv.fillConvexPoly(cropped_tr1_mask, subpoints, 255)

        # Lines space
        # Triangulation of second face
        tr2_pt1 = target_landmark_points[triangle_index[0]]
        tr2_pt2 = target_landmark_points[triangle_index[1]]
        tr2_pt3 = target_landmark_points[triangle_index[2]]
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
        triangles2.append(triangle2)
        
        rect2 = cv.boundingRect(triangle2)
        (x, y, w, h) = rect2
        cropped_tr2_mask = np.zeros((h, w), np.uint8)
        subpoints2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                            [tr2_pt2[0] - x, tr2_pt2[1] - y],
                            [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)
        cv.fillConvexPoly(cropped_tr2_mask, subpoints2, 255)
        # Warp triangles
        subpoints = np.float32(subpoints)
        subpoints2 = np.float32(subpoints2)
        M = cv.getAffineTransform(subpoints, subpoints2)
        transformation_functions.append(M)
        warped_triangle = cv.warpAffine(cropped_triangle, M, (w, h))
        warped_triangle = cv.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

        # Reconstructing destination face
        target_new_hull_rect_area = target_new_hull[y: y + h, x: x + w]
        target_new_hull_rect_area_gray = cv.cvtColor(target_new_hull_rect_area, cv.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv.threshold(target_new_hull_rect_area_gray, 1, 255, cv.THRESH_BINARY_INV)
        warped_triangle = cv.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)
        target_new_hull_rect_area = cv.add(target_new_hull_rect_area, warped_triangle)
        target_new_hull[y: y + h, x: x + w] = target_new_hull_rect_area
        iii+=1
    #("number of triangles: ", triangles1, triangles2)
    # Face swapped (putting 1st face into 2nd face)
    target_hull_mask = np.zeros_like(target_gray)
    target_inv_hull_mask = cv.fillConvexPoly(target_hull_mask, convexhull2, 255)
    target_hull_mask = cv.bitwise_not(target_inv_hull_mask)
    target_nohull = cv.bitwise_and(target_img, target_img, mask=target_hull_mask)
    registered_img = cv.add(target_nohull, target_new_hull)
    delauney_reg_img_path = os.path.join(savepath, 'images', 'registered_img.png')
    cv.imwrite(delauney_reg_img_path, registered_img)

    # cv.imwrite(os.path.join(path, 'mappingresult.jpg'), result_img)
    return registered_img
