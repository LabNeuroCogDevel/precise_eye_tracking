from threshold import threshold
from eye_circle import circle, circle_vectorized
import multiprocessing
import functools
import cv2

def circles_at_thres(image, uppert, tup):
    (count, low) = tup
    outcome = threshold(image, low, uppert)
    #Run the hough transform on each outcome
    #count is simply the naning convention
    max_cor, max_collec = circle_vectorized(outcome, msg=str(count))
    circle_1, circle_2 = max_collec
    #circle_1+circle_2 is the summation of two highest votes with coordinates
    outname = 'frame_testing/kang%05d.png' % count
    return [(circle_1+circle_2), low, uppert, outname]


def determine(image, ncpu=2):
    """
    Get all the possible canny image throught all the threshold
    The eyeball should always be the darkest pixel in the image
    so the uppert would be set to 255 for now
    
    @param image cv2.imread matrix with eye to find
    @param ncpu number cores to use
    """
    uppert = 255

    # Set lower t between 50 and 120 
    lowert = range(50,120)
    pool= multiprocessing.Pool(ncpu)
    f = functools.partial(circles_at_thres, image, uppert)
    guessing = pool.map(f, enumerate(lowert))
    pool.close()


    #Sort the best overall and get the best result and picture overall
    guessing.sort(reverse=True)
    result = guessing[0]
    #result_image = cv2.imread(result[3])
    # cv2.imwrite('pretesting/%s.png'%i, circled_cases)
    # #Write the image to the destinated folder to better examine
    # cv2.imwrite('testing_result/%s.png'%i, result_image)
    return result


