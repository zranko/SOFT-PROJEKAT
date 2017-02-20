import sys
sys.path.append('code/')

import cv2
import numpy as np
from scipy import ndimage
from vector import distance, pnt2line
from matplotlib.pyplot import cm
from keras.models import model_from_json
import itertools
import time
import matplotlib.pyplot as plt
from skimage.io import imread

file= open("out.txt", "w")
file.write("file \t sum\n")



json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")



for vid in range(0,10):

    cap = cv2.VideoCapture("videos/video-{}.avi".format(vid))
    #cap = cv2.VideoCapture("videos/video-6.avi")

    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150,apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10,lines=np.array([]), minLineLength=370, maxLineGap=20)

    print lines
    hough = np.zeros(img.shape, np.uint8)

    for line1 in lines:
        x1, y1, x2, y2 = line1[0]
        cv2.line(hough, (x1, y1), (x2, y2), (255, 255, 255), 2)
        line = [(x1, y1), (x2, y2)]
    print line

    #line = [(48,398), (447, 98)]
    #line = findLine();
    print line
    print("Linija: {}".format(line))

    #line = [(100,450), (500, 100)]


    cc = -1
    def nextId():
        global cc
        cc += 1
        return cc

    def inRange(r, item, items):
        retVal = []
        for obj in items:
            mdist = distance(item['center'], obj['center'])
            if(mdist<r):
                retVal.append(obj)
        return retVal


    # color filter
    kernel = np.ones((2,2),np.uint8)
    lower = np.array([230, 230, 230])
    upper = np.array([255, 255, 255])

    #boundaries = [
    #    ([230, 230, 230], [255, 255, 255])
    #]


    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.cv.CV_FOURCC(*'XVID')
    out = cv2.VideoWriter('images/output-rezB.avi',fourcc, 20.0, (640,480))

    elements = []
    t =0
    counter = 0
    times = []
    suma = 0
    while (1):
        start_time = time.time()
        ret, img = cap.read()
        #print ret, img
        file_name = 'images/frame-' + str(t) + '.png'
        cv2.imwrite(file_name, img)
        if not (ret):
            break
        # (lower, upper) = boundaries[0]
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(img, lower, upper)
        img0 = 1.0 * mask

        img0 = cv2.dilate(img0, kernel)  # cv2.erode(img0,kernel)
        img0 = cv2.dilate(img0, kernel)

        labeled, nr_objects = ndimage.label(img0)
        objects = ndimage.find_objects(labeled)
        for i in range(nr_objects):
            loc = objects[i]
            (xc, yc) = ((loc[1].stop + loc[1].start) / 2,
                        (loc[0].stop + loc[0].start) / 2)
            (dxc, dyc) = ((loc[1].stop - loc[1].start),
                          (loc[0].stop - loc[0].start))

            if (dxc > 11 or dyc > 11):
                cv2.circle(img, (xc, yc), 16, (25, 25, 255), 1)
                elem = {'center': (xc, yc), 'size': (dxc, dyc), 't': t}
                # find in range
                lst = inRange(20, elem, elements)
                nn = len(lst)
                if nn == 0:
                    elem['id'] = nextId()
                    elem['t'] = t
                    elem['pass'] = False
                    elem['history'] = [{'center': (xc, yc), 'size': (dxc, dyc), 't': t}]
                    elem['future'] = []
                    elements.append(elem)

                    img = imread('images/frame-{}.png'.format(elem['t']))
                    blok_size = (28, 28)
                    blok_center = (xc, yc)
                    blok_loc = (blok_center[1] - blok_size[0] / 2, blok_center[0] - blok_size[1] / 2)
                    print blok_center
                    imgB = img[blok_loc[0]:blok_loc[0] + blok_size[0],
                           blok_loc[1]:blok_loc[1] + blok_size[1], 0]
                    cv2.imwrite('imagess2/' + str(elem['id']) + '.jpg', imgB)

                elif nn == 1:
                    lst[0]['center'] = elem['center']
                    lst[0]['t'] = t
                    lst[0]['history'].append({'center': (xc, yc), 'size': (dxc, dyc), 't': t})
                    lst[0]['future'] = []

        for el in elements:
            tt = t - el['t']
            if (tt < 3):
                dist, pnt, r = pnt2line(el['center'], line[0], line[1])
                if r > 0:
                    cv2.line(img, pnt, el['center'], (0, 255, 25), 1)
                    c = (25, 25, 255)
                    if (dist < 9):
                        c = (0, 255, 160)
                        if el['pass'] == False:
                            el['pass'] = True
                            counter += 1

                            img = 'imagess2' + '/' + str(el['id']) + '.jpg'
                            img = imread(img)

                            imgB = img.reshape(1, 1, 28, 28).astype('float32')
                            imgB = imgB / 255

                            imgB_test = imgB.reshape(784)
                            imgB_test = imgB_test / 255.
                            tt = model.predict(imgB, verbose=1)
                            answer = np.argmax(tt)

                            suma += answer

                cv2.circle(img, el['center'], 16, c, 2)

                id = el['id']
                cv2.putText(img, str(el['id']),
                            (el['center'][0] + 10, el['center'][1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
                for hist in el['history']:
                    ttt = t - hist['t']
                    if (ttt < 100):
                        cv2.circle(img, hist['center'], 1, (0, 255, 255), 1)

                for fu in el['future']:
                    ttt = fu[0] - t
                    if (ttt < 100):
                        cv2.circle(img, (fu[1], fu[2]), 1, (255, 255, 0), 1)

        elapsed_time = time.time() - start_time
        times.append(elapsed_time * 1000)
        cv2.putText(img, 'Suma: ' + str(suma), (400, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 90, 255), 2)

        # print nr_objects
        t += 1
        if t % 10 == 0:
            print t
        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        out.write(img)
    out.release()
    cap.release()
    cv2.destroyAllWindows()

    et = np.array(times)
    print 'mean %.2f ms' % (np.mean(et))
    # print np.std(et)
    print suma
    file.write('videos/video-{}.avi\t{}.0\n'.format(vid, suma))