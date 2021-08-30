import datetime as datetime
import numpy as np
import cv2
import pytesseract
import re
import datetime


def ageVerification():
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
    # reading of image
    image = cv2.imread("./DATASET/DL America/licenses/Hawaii's.jpg")

    if image is None:
        print('Could not open or find the image: ')
        exit(0)

    new_image = np.zeros(image.shape, image.dtype)
    alpha = 1.4  # contrast control
    beta = 0  # brightness control

    print('OUTPUT BEGINS')
    print('-------------------------')

    # see readme file for explanation.
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_image[y, x, c] = np.clip(alpha * image[y, x, c] + beta, 0, 255)

    # displaying contrasted image.
    # cv2.imshow('Contrast', new_image)

    # grayscale contrasted image
    gray_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

    # thresholding image
    ret, thresh1 = cv2.threshold(gray_image, 127, 255, cv2.THRESH_TOZERO)
    # cv2.imshow('Threshold Binary', thresh1)

    # cv2.waitKey(0)

    # saving image
    cv2.imwrite('image1.jpg', thresh1)
    cv2.waitKey(1)

    # tesseract configuration
    config = '-l eng --oem 1 --psm 3'

    # Read image from disk
    im = cv2.imread('image1.jpg', cv2.IMREAD_COLOR)

    # tesseract OCR on image
    text = pytesseract.image_to_string(im, config=config)
    print(text.split('\n'))

    text = re.sub('-', '/', text)
    date_compile = re.compile(r"[\d]{1,2}/[\d]{1,2}/[\d]{4}")
    date_extract = date_compile.findall(text)
    print(date_extract)
    extracted_date = []

    for date in date_extract:
        if int(date[6:]) > 1900:
            date_format = datetime.datetime.strptime(date, "%m/%d/%Y")
            extracted_date.append(date_format)
        else:
            continue

    if len(date_extract) >= 1:
        min_year = min(extracted_date).year
        max_year = max(extracted_date).year
        current_year = datetime.date.today().year
        if max_year > current_year or current_year > min_year:
            if current_year - min_year > 21:
                age = current_year - min_year
                print("Your age : ", age)
            print('You are allowed to purchase the item')
        else:
            print('Sorry, you are not allowed to purchase this item')
    else:
        print('Couldn\'t verify your age, Please Show your id card to the Associate for verification')


    cv2.waitKey(0)
    cv2.waitKey()
    cv2.destroyAllWindows()

ageVerification()

