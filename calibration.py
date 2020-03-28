import numpy as np
import cv2
import os


class Calibrator():
    def __init__(self, param, imgPath, patternSize, squareSize, savePath, vis = False):
        self.stereo = False
        if len(imgPath) == 2:
            self.stereo = True
        self.setImgPath(imgPath)
        if os.path.exists(param[0]):
            self.fs1 = cv2.FileStorage(param[0], cv2.FILE_STORAGE_READ)
            self.K1 = self.fs1.getNode("K").mat()
            self.distCoeffs1 = self.fs1.getNode("Distortion").mat()
            
        if self.stereo:
            if os.path.exists(param[1]):
            #already have intrincis for two camera
                self.fs2 = cv2.FileStorage(param[1], cv2.FILE_STORAGE_READ)
                self.K2 = self.fs2.getNode("K").mat()
                self.distCoeffs2 = self.fs2.getNode("Distortion").mat()
        self.param = param
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        self.patternSize = patternSize
        self.squareSize = squareSize
        #visualize the corner result
        self.vis = vis

        self.objp = np.zeros((np.prod(self.patternSize), 3), np.float32)
        self.objp[:, :2] = np.indices(self.patternSize).T.reshape(-1, 2)
        self.objp *= self.squareSize

        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d point in real world space
        self.imgpointsL = [] # 2d points in image plane.
        self.imgpointsR = []

        self.savePath = savePath
    
    def findCorners(self):
        ind = 0
        for f in range(len(self.left_imgs)):
            ind += 1
            imgL = cv2.imread(self.left_imgs[f])
            grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            #find the corner
            retL, cornersL = cv2.findChessboardCorners(grayL, self.patternSize, None, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if self.stereo:
                imgR = cv2.imread(self.right_imgs[f])
                grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
                retR, cornersR = cv2.findChessboardCorners(grayR, self.patternSize, None, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if self.vis:
                if retL:
                    print('show the left corners')
                    visL = imgL
                    cv2.drawChessboardCorners(visL, self.patternSize, cornersL, retL)
                    cv2.imshow('left', visL)
                if self.stereo:
                    if self.retR:
                        print('show the right corners')
                        visR = imgR
                        cv2.drawChessboardCorners(visR, self.patternSize, cornersR, retR)
                        cv2.imshow('right', visR)
                cv2.waitKey(0)
                
            self.objpoints.append(self.objp)
            cv2.cornerSubPix(grayL, cornersL, (5, 5), (-1, -1), self.criteria)
            self.imgpointsL.append(cornersL)
            if self.stereo:
                cv2.cornerSubPix(grayR, cornersR, (5, 5), (-1, -1), self.criteria)
                self.imgpointsR.append(cornersR)
            if self.stereo:
                if retL and retR:
                    print('find stereo corners %d'%ind)
                    self.setPath_beforeCorner = False
            else:
                if retL:
                    print('find corners %d'%ind)
                    self.setPath_beforeCorner = False        

    '''
    calibrate the intrinsic
    '''
    def cameraCalibration(self):
        self.findCorners()
        ret1,self.K1,self.distCoeffs1,rvecs2,tvecs2=cv2.calibrateCamera(self.objpoints,self.imgpointsL,(self.w, self.h),None,None)
        print("k1", self.K1)
        print("d1", self.distCoeffs1)
        cv_file1 = cv2.FileStorage(self.param[0], cv2.FILE_STORAGE_WRITE)
        cv_file1.write("width",self.w)
        cv_file1.write("height",self.h)
        cv_file1.write("K",self.K1)
        cv_file1.write("Distortion", self.distCoeffs1)
        if self.stereo:
            ret2,self.K2,self.distCoeffs2,rvecs2,tvecs2=cv2.calibrateCamera(self.objpoints,self.imgpointsR,(self.w, self.h),None,None)
            print("k2", self.K2)
            print("d2", self.distCoeffs2)
            cv_file2 = cv2.FileStorage(self.param[1], cv2.FILE_STORAGE_WRITE)
            cv_file2.write("width",self.w)
            cv_file2.write("height",self.h)
            cv_file2.write("K",self.K2)
            cv_file2.write("Distortion", self.distCoeffs2)
        print('camera calibration finished!')
    '''
    if use different images for intrinsic and extrinsic, need to set imagepath again
    '''
    def setImgPath(self, imgPath):
        self.left_imgs = os.listdir(imgPath[0])
        self.left_imgs.sort()
        self.left_imgs = [os.path.join(imgPath[0],p) for p in self.left_imgs]
        if self.stereo:
            print('stereo mode')
            self.right_imgs = os.listdir(imgPath[1])
            self.right_imgs.sort()
            self.right_imgs = [os.path.join(imgPath[1],p) for p in self.right_imgs]
            if len(self.left_imgs) != len(self.right_imgs):
                print("unequal number of left and right images")
                exit(0)
        self.h, self.w = cv2.imread(self.left_imgs[0]).shape[: 2]
        self.setPath_beforeCorner = True
    
    def stereoCalibration(self):
        if min(len(self.left_imgs), len(self.right_imgs)) < 3:
            print("imgs not enough")
            exit(0)
        if self.setPath_beforeCorner:
            self.findCorners()
        term_crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 50, 1e-5)
        rms, self.K1, self.distCoeffs1, self.K2, self.distCoeffs2, self.R, self.T, E, F = \
        cv2.stereoCalibrate(self.objpoints, self.imgpointsL, self.imgpointsR, self.K1, self.distCoeffs1, self.K2, self.distCoeffs2, (self.w, self.h), None, None, None, None, flags=(cv2.CALIB_FIX_INTRINSIC + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6), criteria=term_crit)
        print("RMS:", rms)
        print("k1", self.K1)
        print("d1", self.distCoeffs1)
        print("k2", self.K2)
        print("d2", self.distCoeffs2)
        print("Rotation matrix", self.R)
        print("Translation vector", self.T)
        print("Result saved to" + self.savePath)
        cv_file = cv2.FileStorage(self.savePath, cv2.FILE_STORAGE_WRITE)
        cv_file.write("width",self.w)
        cv_file.write("height",self.h)
        cv_file.write("K1",self.K1)
        cv_file.write("Distortion1",self.distCoeffs1)
        cv_file.write("K2",self.K2)
        cv_file.write("Distortion2",self.distCoeffs2)
        cv_file.write("T",self.T)
        cv_file.write("R",self.R)


class Rectify():
    def __init__(self, param):
        fs1 = cv2.FileStorage(param, cv2.FILE_STORAGE_READ)
        self.K1 = fs1.getNode("K1").mat()
        self.distCoeffs1 = fs1.getNode("Distortion1").mat()

        self.K2 = fs1.getNode("K2").mat()
        self.distCoeffs2 = fs1.getNode("Distortion2").mat()

        self.R = fs1.getNode("R").mat()
        self.T = fs1.getNode("T").mat()

        self.w = int(fs1.getNode("width").real())
        self.h = int(fs1.getNode("height").real())

        print("k1", self.K1)
        print("d1", self.distCoeffs1)
        print("k2", self.K2)
        print("d2", self.distCoeffs2)
        print("Rotation matrix", self.R)
        print("Translation vector", self.T)

        

        self.R1,self.R2,self.P1,self.P2,self.Q, self.validPixROI1, self.validPixROI2 = cv2.stereoRectify(
            self.K1,
            self.distCoeffs1,
            self.K2,
            self.distCoeffs2,
            (self.w,self.h),
            self.R,self.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
            newImageSize=(0, 0)
        )

        self.maps1 = cv2.initUndistortRectifyMap(self.K1,self.distCoeffs1,self.R1,self.P1,(self.w,self.h),cv2.CV_16SC2)
        self.maps2 = cv2.initUndistortRectifyMap(self.K2,self.distCoeffs2,self.R2,self.P2,(self.w,self.h),cv2.CV_16SC2)
    
    def __call__(self, imgL, imgR):
        imgL_rectify = cv2.remap(imgL, self.maps1[0], self.maps1[1], cv2.INTER_LINEAR)
        imgR_rectify = cv2.remap(imgR, self.maps2[0], self.maps2[1], cv2.INTER_LINEAR)
        return imgL_rectify, imgR_rectify

    
    
    def vis(self, im1, im2):
        cv2.rectangle(im1, (self.validPixROI1[0], self.validPixROI1[1]),
                    (self.validPixROI1[0] + self.validPixROI1[2], self.validPixROI1[1] + self.validPixROI1[3]),
                    (0,0,255))
        cv2.rectangle(im2, (self.validPixROI2[0], self.validPixROI2[1]),
                    (self.validPixROI2[0] + self.validPixROI2[2], self.validPixROI2[1] + self.validPixROI2[3]),
                    (0,0,255))
        
        full_image = np.concatenate((im1, im2), 1)
        full_image = cv2.resize(full_image, (2048,1152))
        for i in range(0, full_image.shape[1],16):
            cv2.line(full_image,(0,i),(full_image.shape[1],i),(0,255,0),1,8)
        return full_image
    '''
    undistort function is not often used, just for single image undistort
    '''
    @staticmethod
    def undistort(img, K, dist):
        dst = cv2.undistort(img, K, dist, None, K)
        res = np.hstack((img, dst))
        return res 
    