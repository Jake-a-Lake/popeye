import numpy as np
import cv2


if __name__ == '__main__':

    img = cv2.imread("images/gold214.jpg")

    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    #MAC
    #points = np.array([[[3,524],[1149,333],[1554,376],[1558,873],[3,874],[3,522]]])
    
    #ComEmployee
    # points = np.array([[[284,206],[192,217],[180,260],[2,408],[6,715],[1300,718],[1300,661],[628,251],[495,213],[378,204],[289,202]]])
    
    #Gold14
    # points = np.array([[
    # [782,418],[523,425],[128,518],[11,564],[129,1025],
    # [200,1073],[1915,1067],[1808,551],[1600,517],[798,409]
    # ]])

    #method 1 smooth region
    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)

    #method 2 not so smooth region
    # cv2.fillPoly(mask, points, (255))

    res = cv2.bitwise_and(img,img,mask = mask)
    rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

    ## create the white background of the same size of original image
    wbg = np.ones_like(img, np.uint8)*255
    cv2.bitwise_not(wbg,wbg, mask=mask)
    # overlap the resulted cropped image on the white background
    dst = wbg+res

    # cv2.imshow('Original',img)
    # cv2.imshow("Mask",mask)
    # cv2.imshow("Cropped", cropped )
    # cv2.imshow("Samed Size Black Image", res)
    # cv2.imshow("Samed Size White Image", dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imwrite('Original.jpg',img)
    # cv2.imwrite("OnlyMask.jpg",mask)
    # cv2.imwrite("OriginalCropped.jpg", cropped )
    # cv2.imwrite("SamedSizeBlackImage.jpg", res)
    # cv2.imwrite("SamedSizeWhiteImage.jpg", dst)

    cv2.imwrite('GOLD214_Original.jpg',img)
    cv2.imwrite("GOLD214_OnlyMask.jpg",mask)
    cv2.imwrite("GOLD214_OriginalCropped.jpg", cropped )
    cv2.imwrite("GOLD214_SamedSizeBlackImage.jpg", res)
    cv2.imwrite("GOLD214_SamedSizeWhiteImage.jpg", dst)
