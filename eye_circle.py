
#########################################
#the image generated with radius set to two will just be messed up, Hence, I let it to find radius itself
########################################

import cv2,sys
import functools
import math
import multiprocessing
import numpy as np
import operator

def circle(name_count, frame):
  original_image = frame # Input file image
  edged_image = frame #Input image for the edged image
  height = edged_image.shape[0]
  width = edged_image.shape[1]


  Rmin = 20
  Rmax = 50

  # Initialise Accumulator as a Dictionary with x0, y0 and r as tuples and votes as values
  accumulator = {}

  # Loop over the image
  for y in range(0,height):
    for x in range(0,width):
      # If an edge pixel is found..
      if edged_image.item(y,x) >= 255:

        for r in range(Rmin,Rmax,2):
          for t in range(0,360,2):

            #Cast it to a new coordinates
            x0 = int(x-(r*math.cos(math.radians(t))))
            y0 = int(y-(r*math.sin(math.radians(t))))

            # Checking if the center is within the range of image
            if x0>0 and x0<width and y0>0 and y0<height:
              if (x0,y0,r) in accumulator:
                accumulator[(x0,y0,r)]=accumulator[(x0,y0,r)]+1
              else:
                accumulator[(x0,y0,r)]=0
  #print(accumulator
  #Iterate through the dictionary to find the max values
  max_cor = [] #Store the coordinates
  max_collec = [] #Store the max number
  max_coordinate = None
  max_value = 0
  count = 2 #First try 2

  #Somehow it is ranked from highest to lowest
  for i in range(count):
      for k, v in accumulator.items():
          if v > max_value:
            #Find the max votes in the accumulator
            max_value = v
            max_coordinate = k

      max_collec.append(max_value)
      #Zero out max
      max_value = 0
      #Append max position
      max_cor.append(max_coordinate)
      #Zero out that position
      accumulator[max_coordinate] = 0

  print('big_circle')
  #Show the x,y,and radius
  #If the x and y diverge too far away from each other, just cateforize it as the eye close
  print(max_cor)
  #Show the vote on each circle
  print(max_collec)

  original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

  for x, y, r in max_cor:
    circled_cases = cv2.circle(original_image, (x,y), r, (0,0,255))

  return max_cor, max_collec, circled_cases

def ccast(yx, r, t, width, height):
    """
    at angle t and radius r
    return 2d matrix: 1 where (x,y) within. 0 otherwise
    yx should be 2 item list: vector of y pos, vector of x pos
    >>> plt.imshow(ccast(yx,22,180,width,height))
    """
    #Cast it to a new coordinates
    y0 = (yx[0] - (r * math.sin(t))).astype(int) 
    x0 = (yx[1] - (r * math.cos(t))).astype(int) 
    # Checking if the center is within the range of image
    i = np.logical_and.reduce((y0 > 0, y0< height, x0>0, x0<width))
    # create matrix with one if in range
    v = np.zeros([height, width])
    if(len(i) > 0):
        v[y0[i],x0[i]] = 1
    return v

def sum_angles(yx, r, w, h, by=2):
    """ sum the circles for over all angles """
    angles = [math.radians(t) for t in range(0, 360, by)]
    s = np.sum([ccast(yx, r, a, w, h) for a in angles], axis=0)
    return(s)

def plot_radlist(rs, radi):
    import matplotlib.pyplot as plt
    ns = math.sqrt(len(rs))
    fg, ax = plt.subplots(math.ceil(ns),math.ceil(ns))
    for i, r in enumerate(rs):
        x=math.floor(i/ns)
        y=i%math.ceil(ns)
        mi=np.unravel_index(np.argmax(r), r.shape)
        mv=r[mi]
        ax[x,y].imshow(r)
        ax[x,y].set_title("r=%d; m=%d @ %s" % (radi[i], mv, mi))
        ax[x,y].axis('off')
        ax[x,y].annotate("%d" % mv, xy=mi, color='white')

def draw_circles(frame, xyrs, show=False):
    """
    draw circles on FRAME wih x,y,r provided by XYRS
    return new frame wih circles drawn
    circles are colored red, green, blue based on input order
    """
    circled_cases = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    for i, xyr in enumerate(xyrs):
        (x,y,r) = xyr
        # change color for each circle
        # 1st=red, 2nd=green, 3rd=blue, 4th=red, ....
        color=[0,0,0]
        color[i%3] = 255

        # cv2.circle side-effect: updates input
        cv2.circle(circled_cases, (x,y), r, color)
        
    if show:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(circled_cases)

    return(circled_cases)

def circle_vectorized(frame, N=2, msg="", show=False, draw=False):
    """
    use vectorized matricies to calc N circle from FRAME
    >>> frame=threshold(cv2.imread('./analysis_set/kang00013.png'),100,8)
    >>> # import matplotlib.pyplot as plt; plt.ion(); plt.imshow(frame)
    >>> [xyr, cnts, circled_cases] = circle_vectorized(frame, N=2, show=True)
    """
    (height, width) = frame.shape
    (Rmin, Rmax) = (20, 50)
  
    # find coords of edges
    yx = np.where(frame >= 255)
    # large list of r,t pairs for every r_min-r_max and degree (by 2)
    # TODO: there is porbably math to show some of this space is redundant?
    #   e.g. exclude x,y Rmin from image sides?
    #   xy = xy[xy[:,0] > Rmin and xy[:,0] < width - Rmax, xy[0,:] > Rmin and xy[0,:] < height - Rmax]
    #rng = [(r,math.radians(t)) for r in range(Rmin, Rmax, 2) for t in range(0, 360, 2)]
    radi = range(Rmin,Rmax,2)
    # for all the xy pairs on an edge,
    #  make a cirlce matrix for each angle
    #  and sum
    rs = [sum_angles(yx, r, width, height) for r in radi]

    # view it maybe
    if show: plot_radlist(rs, radi)
    
    # # -- best only -- 
    # (ri, yi, xi) = np.unravel_index(np.argmax(rs),shape)
    # max_value = rs[ri][yi,xi]
    # max_coordinate = (xi,yi,radi[ri])

    # -- top N --
    rs = np.array(rs)
    i = np.argpartition(rs.flatten(), -N)[-N:]
    max3didx = [np.unravel_index(ii, rs.shape) for ii in i]
    # grab the counts (v) from the 3d matrix rs. get the radius, y, and x pos
    vryx = [[rs[ii], radi[ii[0]], *ii[1:]] for ii in max3didx]
    # argpartition is quick, but does not return in order
    vryx = sorted(vryx, key=lambda a: a[0], reverse=True)
  
    print(msg + ' big_circle vryx')
    print(vryx)
  
    # want in two vars: list of xyr, and list of values
    max_cor = [z[3:0:-1] for z in vryx] 
    max_collec = [z[0] for z in vryx]
    
    # don't draw the cirlces if we dont need them
    if not draw:
      return max_cor, max_collec

    # we want circles. draw them. maybe show them too
    circled_cases = draw_circles(frame, max_cor, show)
    return max_cor, max_collec, circled_cases
