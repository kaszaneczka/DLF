import numpy as np
import cv2
import time
asd = cv2.imread('frame25.jpg')[900:1000,200:500]
asd = cv2.cvtColor(asd,cv2.COLOR_BGR2GRAY)
# print(asd.shape)
#asd = np.moveaxis(asd, -1, 0)
#print(asd.shape)

# cv2.imshow('aaa',asd)
# cv2.waitKey(0)




class conv:
    def __init__(self,zdj:np.ndarray,kernel_size,depth,dim_in,count_kernels = 3):

        self.dim_in = dim_in
        self.count_kernels = count_kernels
        self.kernel_size = kernel_size

        self.depth = depth
        if len(zdj.shape) == 3:
            aa = np.moveaxis(zdj, -1, 0)
            self.tablica2d = np.array(aa)
        elif len(zdj.shape) == 2 :
            self.tablica2d = np.array([zdj])

        self.all_kernels = []
        self.tablica3d = np.array([])
        self.tablica3d_pomoc = np.array([])
        self.tablica3d_pomoc2 = np.array([])
        self.gen_kernels()


    def gen_kernels(self):
        b = 0


        for a in range(self.depth * self.count_kernels):
            #
            # if b < self.count_kernels :
            if len(self.tablica2d.shape) == 3:

                self.kernels_shape = (self.dim_in,self.kernel_size, self.kernel_size)
                self.kernels = np.random.randn(*self.kernels_shape)
                self.all_kernels.append(self.kernels)
                # b +=1
                # else:
                #     self.kernels_shape = ( self.kernel_size, self.kernel_size,self.dim_in)
                #     self.kernels = np.random.randn(*self.kernels_shape)
                #     self.all_kernels.append(self.kernels)
                #     b += 1
            else :
                self.kernels_shape = (self.count_kernels, self.kernel_size, self.kernel_size)
                self.kernels = np.random.randn(*self.kernels_shape)
                self.all_kernels.append(self.kernels)
        #self.all_kernels = (self.all_kernels)



    def dot_prod(self,maska, tablica2d):
        tablica2d_help = np.array([])

        for a in range(len(tablica2d) - self.kernel_size + 1):
            tablica2d_help = np.append(tablica2d_help,
                                       (np.array([np.dot((tablica2d[a:a + self.kernel_size, b:b + self.kernel_size]).flatten(), maska.flatten()) for b in range(len(tablica2d[0]) - self.kernel_size + 1)])))

        tablica2d_help = tablica2d_help.reshape((tablica2d.shape[0] - self.kernel_size + 1, tablica2d.shape[1] - self.kernel_size + 1))

        return np.array([tablica2d_help])

    def calculate(self):

        for b in range(self.count_kernels):
            print(self.tablica2d.shape)
            for a in range(self.count_kernels):
                for c in range(np.array(self.tablica2d).shape[0]):
                    print(1)
                    self.tablica3d = np.append(self.tablica3d,
                                               [self.dot_prod(self.all_kernels[a + b*3][c],
                                                        self.tablica2d[c])])


                self.tablica3d_pomoc = self.tablica3d.reshape((c + 1)*(a+1),self.tablica2d[0].shape[0] - self.kernel_size + 1,self.tablica2d[0].shape[1] - self.kernel_size + 1)

                self.tablica3d_pomoc2 = np.append(self.tablica3d_pomoc2,[sum(self.tablica3d_pomoc)])

            self.tablica3d_pomoc2 = self.tablica3d_pomoc2.reshape(self.count_kernels,self.tablica2d[0].shape[0] - self.kernel_size + 1,self.tablica2d[0].shape[1] - self.kernel_size + 1)

            # self.tablica3d = self.tablica3d.reshape(self.count_kernels,self.tablica2d[0].shape[0] - self.kernel_size + 1,self.tablica2d[0].shape[1] - self.kernel_size + 1)

            self.tablica2d = np.array(self.tablica3d_pomoc2 )
            self.tablica3d = np.array([])
            self.tablica3d_pomoc2 = np.array([])

            print(self.tablica2d.shape)


        return self.tablica2d.flatten()



    def add_padding(self):

        zeros_col = np.zeros((1, self.tablica2d.shape[1]), dtype=np.uint8)

        for a in range(self.kernel_size - 1):
            self.tablica2d = np.array([np.append(self.tablica2d[0], zeros_col.T, axis=1)])
            self.tablica2d = np.array([np.append(zeros_col.T,self.tablica2d[0], axis=1)])


        zeros_row = np.zeros((1,self.tablica2d.shape[2]),dtype=np.uint8)

        for a in range(self.kernel_size - 1):
            self.tablica2d = np.array([np.append(self.tablica2d[0], zeros_row, axis=0)])
            self.tablica2d = np.array([np.append(zeros_row, self.tablica2d[0], axis=0)])

        print(self.tablica2d )






aa = conv(asd,4,4,3)
# cv2.imshow('aaa',asd)
# cv2.waitKey(0)
aa.add_padding()
print(aa.all_kernels[0].shape)
print(len(aa.all_kernels))
print(aa.calculate())
#aa.add_padding()




#print(aa.calculate())










