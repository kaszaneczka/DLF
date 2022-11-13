import numpy as np
import cv2
import time
asd = cv2.imread('frame25.jpg')[900:1300,200:1000]
asd = cv2.cvtColor(asd,cv2.COLOR_BGR2GRAY)
cv2.imshow('aaa',asd)
cv2.waitKey(0)
print(asd)


class conv:
    def __init__(self,zdj:np.ndarray):
        self.tablica2d = zdj
        self.tablica2d_help = np.ndarray
        self.tablica2d_help1 = np.ndarray
        self.tablica3d = np.ndarray

    def dot_prod(self,wym_maski,maska):
        for a in range(len(self.tablica2d) - wym_maski + 1):
            for b in range(len(self.tablica2d[0]) - wym_maski + 1):
                self.tablica2d_help = np.append(self.tablica2d_help,np.dot(self.tablica2d[a:a+wym_maski,b:b+wym_maski],maska))
                print(self.tablica2d_help)
            self.tablica2d_help1 = np.append(self.tablica2d_help1,self.tablica2d_help)
        self.tablica3d = np.append(self.tablica3d,self.tablica2d_help1)
        return self.tablica3d

aa = conv(asd)
maska = np.array([[1,2,3],[1,2,3],[1,2,3]])
aasd = aa.dot_prod(3,maska)

print(asd.shape,aasd.shape)










