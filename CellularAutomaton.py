import numpy as np
from numba import jit, cuda
from PIL import Image
import random as rd
import VideoGenerator as vg


import os

os.environ['CUDA_HOME']      = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7' #install cuda toolkit from nvidia and put the correct path here (should not really change much tho, except v11.7)

#custom vars
perc = 30/100 #for custom starting frame 

#IMG SIZE
_SHOWSIMULATION = True
_SIZE = (120, 120) #size of the output image inverse of the resolution you want i.e: 1920x1080 => (1080, 1920)
_STEPS = 15 #number of steps to simulate
_LOOPPAUSETIME = 1 #pause between each calculations (indirectly fps)
_SKIPONEFRAME = True #renders 2 frames but only show one (epilepsy brrrrr)
_SKIPTWOFRAME = False #renders 3 frames but only show one (works only if _SKIPONEFRAME = True)(epilepsy brrrrr)

#_COLORSHIFT = [110/255, 220/255, 255/255] #not working

#GPU NO TOUCHIES
_THREADSPERBLOCK = (6,6)
_BLOCKSPERGRID = (_SIZE[0] + (36 - 1) , _SIZE[1] + (36 - 1))

#VIDEO GENERATOR just use to generate iamge or 10gb videos go brrr
_GENERATEVIDEO = False
_GENERATEVIDEOFRAMES = True
_TEMPVIDFOLDER = "temp"
_OUTPUT = "test.avi"
_VIDEOLENGTH = 10  #seconds
_VIDEOFPS = 60

if _GENERATEVIDEO:
    _STEPS = _VIDEOLENGTH * _VIDEOFPS
    _LOOPPAUSETIME = 1




#SIM
_CUSTOMFRAME = True #whether or not to use the custom function for starting frame
RandomStartingFrame = True #generates a random image to start
RandomIntStartingFrame = True #generates a random image to start with only 1 and 0
ZerosStartingFrame = False #blank 0 frame
_LOOP = False
@cuda.jit
def ActFunction(x):
    #return x*x
    #return -1./pow(2., (0.6*pow(x, 2.)))+1. #worms
    #return -1./(0.9*pow(x, 2.)+1.)+1. #Mitosis
    #return -1./(0.89*pow(x, 2.)+1.)+1. #slime
    #return abs(1.2*x) #waves?
    #return -1./pow(2., (0.6*pow(x, 2.)))+1. #for the weird thing
    """
    #Conway's game of life
    if x == 3. or x == 11. or x == 12. :
        return 1
    else:
        return 0
    """

    #cave generator
    if x>4:
        return 1
    else:
        return 0
"""
"""
#Chose your filter / make a new one
"""
NCA_Filter = np.array([ [0.1 , -0.1  , 0.3   ],
                          [0   ,     1 ,  -0.3 ],
                          [-0.5, -0.1  , 0.2   ]])




NCA_Filter =  np.array([ [-0.214 , -0.715  , -0.11   ],
                           [0.441  ,     -0.712 ,  -0.78 ],
                           [0.225, 0.757  , -0.331   ]])

NCA_Filter =  np.array([ [-0.05 , 0.05  , -0.05   ],
                           [0.05  ,     -0.05 ,  -0.05 ],
                           [0.05, 0.05  , -0.05   ]])


NCA_Filter =  np.array([ [0.463 , -0.996  , 0.557   ],
                           [0.18  ,     -0.49 ,  -0.845 ],
                           [-0.851, 0.175  , -0.205   ]])

NCA_Filter =  np.array([ [-0,602 , 0,647  , -0,602   ],
                           [0,502  ,     0,133 ,  0,502 ],
                           [-0,999, 0,84  , -0,999   ]])


#WORMS
NCA_Filter = np.array([ [0.68 ,    -0.9  ,   0.68   ],
                        [-0.9   ,  -0.66 ,  -0.9 ],
                        [0.68,     -0.9  ,   0.68   ]])

#Mitosis
NCA_Filter = np.array([ [-0.939 ,    0.88  ,   -0.939   ],
                        [0.88   ,  0.4 ,  0.88 ],
                        [-0.939,     0.88  ,   -0.939   ]])


#slime
NCA_Filter = np.array([ [0.8 ,    -0.85  ,   0.8  ],
                        [-0.85   ,  -0.2 ,  -0.85 ],
                        [0.8,     -0.85  ,   0.8   ]])
#to use with sin(x)
NCA_Filter = np.array([ [0.019 ,    0.389  ,   -0.647  ],
                        [0.987   ,  -0.988 ,  -0.999 ],
                        [-0.786,     -0.048  ,   -0.847   ]]

#waves?
NCA_Filter = np.array([ [0.565 ,    -0.716  ,   0.565 ],
                        [-0.716   ,  0.627 ,  -0.716 ],
                        [0.565,     -0.716  ,   0.565   ]])

#Super cool growing spaceships, use x*x activ function
NCA_Filter = np.array([[-0.8300993,  -0.44785473,  0.979766  ],
                       [-0.47983372,  0.07656833, -0.35514032],
                       [ 0.53823346,  0.43503127,  0.2492277 ]])

#Conway  game of life
NCA_Filter = np.array([ [1 ,    1  ,   1  ],
                        [1   ,  9 ,  1 ],
                        [1,     1  ,  1   ]])

#weird looking thing that grows and creates pipe inside itself. also freaky tentacles idk    // use return -1./pow(2., (0.6*pow(x, 2.)))+1. as ActFunction
NCA_Filter = np.array([[-0.99038735,  0.82367216, -0.99038735],
                       [ 0.82367216,  0.31695098,  0.82367216],
                       [-0.99038735,  0.82367216, -0.99038735]])

"""
#cave generator
NCA_Filter = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])





def RandomFilter():
    """#normalFilter
    return np.array([ [rd.uniform(-1, 1) ,    rd.uniform(-1, 1)  ,   rd.uniform(-1, 1)  ],
                      [rd.uniform(-1, 1)   ,  rd.uniform(-1, 1) ,    rd.uniform(-1, 1) ],
                      [rd.uniform(-1, 1),     rd.uniform(-1, 1)  ,   rd.uniform(-1, 1)   ]])
    """
    #FullSymetry Filter
    a = rd.uniform(-1, 1)
    b = rd.uniform(-1, 1)
    c = rd.uniform(-1, 1)
    return np.array([ [a ,    b  ,   a  ],
                      [b   ,  c ,    b ],
                      [a,     b  ,   a   ]])
"""
#NCA_Filter = RandomFilter()
NCA_Filter = np.array([ [0.2565 ,    -0.2716  ,   0.2565 ],
                        [-0.2716   ,  0.627 ,  -0.2716 ],
                        [0.2565,     -0.2716  ,   0.2565   ]])
"""


































@cuda.jit
def GPU_PIXEL(_input, _filter, _output):
    x,y = cuda.grid(2)
    sum = 0.
    xSize = _input.shape[0]
    ySize = _input.shape[1]


    if x <= xSize and y <= ySize:
        for subx in [-1,0,1]:
            for suby in [-1,0,1]:
                subbx = subx+x #relative to grid
                if subbx != -1: subbx = subbx%xSize

                subby = suby+y #relative to grid
                if subby != -1: subby = subby%ySize




                sum += _input[subbx,subby] * _filter[subx+1,suby+1]
                #sum += _input[(subx+x)%(xSize),(suby+y)%(ySize)] * _filter[subx+1,suby+1]
        sum = ActFunction(sum)
        if sum <0:
            sum = 0
        elif sum > 1:
            sum = 1
        _output[x,y] = sum



"""
@cuda.jit
def MakeImage(input, _img):
    x,y = cuda.grid(2)
    val = input[x,y]
    print(val)
    _img[0,0,0] =  val * 0.5
    _img[0,0,1] =  val * 1
    _img[0,0,2] =  val * 1
"""


def NextStep(input, filter, activFunction):
    """takes np array shape = (x, y) (dim 2) and returns next step of the sim"""
    xSize = input.shape[0]
    ySize = input.shape[1]
    output = np.zeros((xSize, ySize))
    threadsperblock = _THREADSPERBLOCK
    blockspergrid = _BLOCKSPERGRID
    GPUInputArray = cuda.to_device(input)
    GPUOutputArray = cuda.to_device(output)
    GPUFilterArray = cuda.to_device(filter)

    #GPU_PIXEL[blockspergrid, threadsperblock](input,  output)
    GPU_PIXEL[blockspergrid, threadsperblock](input, filter, output)
    #output = DoPixel(input, output, xSize, ySize, activFunction, filter)
    return output



if __name__ == "__main__":
    import cv2
    from main import MakeFrame
    from PIL import Image as im
    size = _SIZE
    frame = np.zeros((size[0], size[1]))
    """
    #frame[int(size[0]/2)][int(size[1]/2)] = 1 #starting grid
    #conway drifter
    frame[int(size[0]/2)][int(size[1]/2)] = 1
    frame[int(size[0]/2)][int(size[1]/2)-1] = 1
    frame[int(size[0]/2)-1][int(size[1]/2)] = 1
    frame[int(size[0]/2)-1][int(size[1]/2)+1] = 1
    frame[int(size[0]/2)+1][int(size[1]/2)+1] = 1

    """


    if RandomStartingFrame: frame = np.random.rand(size[0], size[1])
    if RandomIntStartingFrame: frame = np.random.randint(2, size=(size[0], size[1]))
    if ZerosStartingFrame: frame = np.zeros(shape=(size[0], size[1]))

    if _CUSTOMFRAME:
        """set your custom frame algo here"""
        
        for i in range(int(size[0] * size[1] * perc)):
            frame[np.random.randint(size[0])][np.random.randint(size[1])] = 1
    

    #IMG = np.zeros((frame.shape[0], frame.shape[1], 3))
    #GPUIMG = cuda.to_device(IMG)
    for i in range(_STEPS):
        #print(frame)
        frame = NextStep(frame, NCA_Filter, ActFunction)
        if _SKIPONEFRAME: frame = NextStep(frame, NCA_Filter, ActFunction)
        if _SKIPTWOFRAME: frame = NextStep(frame, NCA_Filter, ActFunction)

        if _GENERATEVIDEOFRAMES or _GENERATEVIDEO:
            vg.SaveImage(frame, "./"+_TEMPVIDFOLDER+"/", i)

        #threadsperblock = _THREADSPERBLOCK
        #blockspergrid = _BLOCKSPERGRID
        #MakeImage[blockspergrid, threadsperblock](frame, IMG)

        if _SHOWSIMULATION: cv2.imshow('image', frame)
        print("STEP: " , i, "/", _STEPS)
        cv2.waitKey(_LOOPPAUSETIME)


    if _GENERATEVIDEO:
        vg.MakeVideo(_TEMPVIDFOLDER, _OUTPUT, _VIDEOFPS)









    #=========================================================infinite loop test
    c = 0
    while _LOOP:
        c+=1
        frame = np.random.rand(size[0], size[1])
        NCA_Filter = RandomFilter()
        print(NCA_Filter)
        for i in range(_STEPS):

            frame = NextStep(frame, NCA_Filter, ActFunction)
            if _SKIPONEFRAME: frame = NextStep(frame, NCA_Filter, ActFunction)

            cv2.imshow('image', frame)
            cv2.waitKey(_LOOPPAUSETIME)

        directory = "./Save/"

        frame2 = cv2.normalize(frame, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        cv2.imwrite(directory+str(c)+".bmp", frame2)
        file = open(directory+str(c)+"-Filter.txt", "w+")
        content = str(NCA_Filter)
        file.write(content)
        file.close()















#
