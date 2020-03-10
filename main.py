#!/usr/bin/env python3

import cv2
import sys
import random
import pickle
import multiprocessing
import pic_analyze
from eye_canny import canny
from eye_circle import circle, circle_vectorized
from csv_analysis import read
from plot_data import plotting
from threshold import threshold
from pre_determination import determine 
from glint_detection import circle_glint

#The video frame is mostly 60 fps
class PupilTracking():

    def __init__(self, video,
                 timing_fname="", num_tests=5,
                 fps=60, show=False, ncpu=2):
        """
        Three phases: 
        1. determination the best way to transfrom an image into computer readable, 
        2. track the pupil and glint movement as precise as possible
        3. Use the data provided and the data read, construct amn algorithm to precisely track the slight movement of vector between pupil and glint
        
        Starting with the first step
        pick five instances of image from the colleciton of images framed from the video
         Opens the Video file
        Video will be read from the command line
        Number of image traisl you want
        """
        super().__init__()
        self.video = video
        self.fps = fps
        self.show = show
        self.num_tests = num_tests
        self.ncpu = ncpu
        if timing_fname == "":
            timing_fname = 'input/testing_set/testing_1/10997_20180818_mri_1_view.csv'
            print("Warning: using default timing %s" % timing_fname)
        self.timing_fname = timing_fname 

        # initilze other values
        self.V, self.L, self.H, self.number_frame = (0, 0, 0, 0)
        self.thres_pic = ""

    def set_threshold(self):
        """
        use random frames to guess at best threshold
        """
        # get random frames to test on
        random_num = self.rand()
        try:
            self.pre_test(random_num)
        except Exception as e:
            print(e)
            print('Error: Resizing factors too big to be useful')
            exit()

    def run_all(self):
        # convert video to series of frames
        self.to_frame()
        print('To frame successful')
        
        # find best L nad H thresholds
        self.set_threshold()
        print(self.V, self.L, self.H, self.thres_pic)

        #list of list that contains the whole set of testing data
        sets = self.file_data()
        print(sets)
        # [[6.0, 8.0, 10.0, 16.0],
        #  [20.0, 22.0, 24.0, 30.0],
        #  [40.0, 42.0, 44.0, 50.0], ....


        #Now do the analysis set by set// Starting to code the main part of the program        
        print('pretesting finished, starting analying the collection pictures using the paramaters')

        output_sets = self.frame_retrieve(sets, self.L, self.H)
        #Now the output_sets is obtained, next setp isd to analyze it.
        print('Data gethering complete')
        print('Starting to plot the data')
        plotting(output_sets)

    def frame_retrieve(self, sets):
        output_sets = []
        #Only need to get the frame around the critical area
        #60 frame/second
        ########################################################################
        #Change it to for loop later
        ########################################################################
        # for i in sets:
        i = sets[0]
        t_cue = i[0]
        t_vgs = i[1]
        t_dly = i[2]
        t_mgs = i[3]

        #Now, after the cue, the pupil should be staring at the center 
        show_center = range(self.fps*int(t_cue), self.fps*int(t_vgs))
        #After vgs, the eye should be staring at the picture
        show_loc = range(self.fps*int(t_vgs), self.fps*int(t_dly))
        #After dly, it should be staring at the center
        hide_center = range(self.fps*int(t_dly), self.fps*int(t_mgs))
        #After t_mgs, it should be staring at wherever it remembered
        #Turns out it always gonna be 2s --> for now
        hide_pic = range(self.fps*int(t_mgs), self.fps*(int(t_mgs) + 2))

        collections = [show_center, show_loc, hide_center, hide_pic]

        #Read the critical frame from the folder
        output_sets.append(self.critical_frame(collections))

        return output_sets

    def eye_at_frame(self, frame_i):
        """
        find eye at every frame in collections for threshold L and H
        @param frame_i video frame index
        @param self.L lower threshold (see pre_test)
        @param self.H upper threshold (see pre_test)
        """
        case_name = 'analysis_set/kang%05d.png'%frame_i
        image = threshold(cv2.imread(case_name), self.L, self.H)
        max_cor, max_collec = circle_vectorized(image, msg=str(count))
        return max_cor

    #Append every frame data to the dictionary and return it back in a big listy
    def critical_frame(self, collections):
        """
        find eye at every frame in collections for threshold L and H
        @param collections frame index for each (4) condition type
        >>> collections = [ [1,2,3], [4,5,6], [7,8,9], [10,11,12]]
        >>> self.critical_frame(collections)
        """
        idx = ['s_center', 's_loc', 'h_center','h_loc']
        dic = { k: []  for k in idx }
        ncol = len(collections)
        for i in range(0, ncol):
            p = multiprocessing.Pool(self.ncpu)
            dic[idx[i]] = p.map(self.eye_at_frame(collections[i]))
            p.close()
            print('{}/{} section done'.format(i+1, ncol))
            print('\n\n')
        return dic


        
        # self.video_analyze(self.L, self.H)
    def file_data(self):
        current = []
        cue, vgs, dly, mgs = read(self.timing_fname)
        for i in range(0, len(cue)):
            current.append([cue[i], vgs[i], dly[i], mgs[i]])
        #Now start to narrow down the analysis rang
        #now the video file is 60 fps, and every video file is named according to the sequence
        return current


    def pre_test(self, random_fnum):
        """
        @param random_fnum list of numbers - becomes frame filename to read from
        """
        grand_test = [determine(cv2.imread('frame_testing/kang%05d.png'%i),
                                ncpu=self.ncpu)
                      for i in random_fnum]
        grand_test.sort(reverse=True)

        #Best of the best should be
        self.V, self.L, self.H, self.thres_pic = grand_test[0]

    def rand(self):
        """
        To get NUM_TESTS random frames to test out thres params
        @param self.number_frame
        @param self.num_tests
        """
        rand = []
        for i in range(self.num_tests):
            r = random.randint(0, self.number_frame)
            if r not in rand:
                rand.append(r)
        return rand

    def to_frame(self):
        """
        Method to convert the whole video into frames
        @param self.video video filename to read in
        sets self.number_frame 
        """
        print('Starting to convert video to frames')
        cap = cv2.VideoCapture(self.video)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            #Write out individual frames just to test
            #Downscale the frame
            height = frame.shape[0]
            width = frame.shape[1]
            #Need to conserve one for the later analysis
            keep = frame
            #The first resize is for the real_analysis
            keep = cv2.resize(keep,(int(height), int(width)))
            #the second resize if for the analysis when determining parameters
            frame = cv2.resize(frame,(int(height/8), int(width/8)))
            cv2.imwrite('analysis_set/kang%05d.png'%i,keep)
            cv2.imwrite('frame_testing/kang%05d.png'%i,frame)
            #Then throw the image to threshold to process
            i+=1

        #Total number of frames
        self.number_frame = i


if __name__ == '__main__':

    # usage if wrong number of input args
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("USAGE: %s video.mov [timing.csv]" % sys.argv[0])
        print("\t %s input/testing_set/testing_1/output.mov" % sys.argv[0])
        exit()

    # pass all cli arguments (video, maybe timing) into class
    ######################################################################
    #TODO: change num_tests to 20 later
    ######################################################################
    pup = PupilTracking(*sys.argv[1:], show=True, num_tests=5)
    pup.run_all()


def quick()
    "play with functions"
    pup = PupilTracking('input/testing_set/testing_1/output.mov')
    #pup.to_frame()
    pup.number_frame = 4805
    #pup.set_threshold()
    pup.L = 83
    pup.H = 255
    pup.V = 65
    pup.thres_pic = 'frame_testing/kang00033.png'
    img = cv2.imread(pup.thres_pic)
    frame = threshold(img, pup.L, pup.H)
    [xyr, v, img ]= circle_vectorized(frame,draw=True, show=True)
