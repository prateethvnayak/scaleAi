import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
import matplotlib.pyplot as plt
import tensorflow as tf

IMAGE_SIZE = 200
MAX_RAD = 50
N_LVL = 2

# utils for drawing
def draw(orig, img):

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(orig)
    ax2.imshow(img)
    plt.show()
    plt.savefig("./images.png")

    pdb.set_trace()


def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (rr >= 0) & (rr < img.shape[0]) & (cc >= 0) & (cc < img.shape[1])
    img[rr[valid], cc[valid]] = val[valid]


def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)

    # Noise
    orig_img = img.copy()
    img += noise * np.random.rand(*img.shape)
    # draw(orig_img, img)
    return (row, col, rad), img, orig_img


def find_circle(noisy_img, model):
    # Fill in this function
    noisy_img = np.expand_dims(np.expand_dims(noisy_img, axis=0), axis=-1)
    img = model.predict(noisy_img / np.amax(noisy_img)).reshape(IMAGE_SIZE, IMAGE_SIZE)

    edges = canny(img, low_threshold=0.5, high_threshold=1.0)
    try_radii = np.arange(10, MAX_RAD, 1)
    hough_res = hough_circle(edges, try_radii)
    _, cy, cx, radii = hough_circle_peaks(hough_res, try_radii, total_num_peaks=1)

    if cx.size == 0:
        cx = 0
    else:
        cx = cx[-1]
    if cy.size == 0:
        cy = 0
    else:
        cy = cy[-1]

    if radii.size == 0:
        radii = 0
    else:
        radii = radii[-1]

    return (cx, cy, radii)


def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return shape0.intersection(shape1).area / shape0.union(shape1).area


def main():

    model = tf.keras.models.load_model(filepath="./noise_detection_autoenc.h5")
    results = []
    for i in range(1000):
        params, img, _ = noisy_circle(200, 50, 2)
        detected = find_circle(img, model)
        results.append(iou(params, detected))
        print("image {}\tOriginal {}\tDetected {}".format(i, params, detected))
    results = np.array(results)
    print("\nThe IOU Precision AP@0.7 :")
    print((results > 0.7).mean())


if __name__ == "__main__":
    main()
