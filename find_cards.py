import numpy as np
import cv2

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1)*np.dot(d2, d2)))

def find_cards(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    cards = []

    for gray in cv2.split(img):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            bin, contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
                rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
                topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
                bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < 0.1 and cv2.contourArea(cnt) < 50000:
                        cards.append(cv2.boundingRect(np.array([leftmost, rightmost, topmost, bottommost])))
    return cards

def find_shapes(image):
    count = 0
    for gray in cv2.split(image):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            bin, contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 800:
                    cv2.drawContours(card_img, [cnt], 0, (255,0,0), 1)
                    count += 1
    cv2.imshow('squares2', card_img)
    return count

if __name__ == '__main__':
    from glob import glob
    for fn in glob('gameboard2.jpg'):
        img = cv2.imread(fn)
        cards = find_cards(img)
        cards = cv2.groupRectangles(cards, 1)[0]
        for card in cards:
            card_img = img[card[1]:card[1]+card[3], card[0]:card[0]+card[2]]
            num_shapes = find_shapes(card_img)
            print num_shapes
            cv2.rectangle(img, (card[0], card[1]), (card[0]+card[2], card[1]+card[3]), (255, 0, 0), thickness=2)
        print "Found {num} cards.".format(num=len(cards))
        cv2.imshow('squares', img)
        ch = 0xFF & cv2.waitKey()
        if ch == 27:
            break
    cv2.imwrite('output.jpg', img)
    cv2.destroyAllWindows()