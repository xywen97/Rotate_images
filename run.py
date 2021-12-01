import cv2
from math import fabs, sin, cos, radians
import numpy as np
from scipy.stats import mode


def get_img_rot_broa(img, degree=45, filled_color=-1):
    """
    Desciption:
            Get img rotated a certain degree,
        and use some color to fill 4 corners of the new img.
    """

    # 获取旋转后4角的填充色
    if filled_color == -1:
        filled_color = mode([img[0, 0], img[0, -1],
                             img[-1, 0], img[-1, -1]]).mode[0]
    if np.array(filled_color).shape[0] == 2:
        if isinstance(filled_color, int):
            filled_color = (filled_color, filled_color, filled_color)
    else:
        filled_color = tuple([int(i) for i in filled_color])

    height, width = img.shape[:2]

    # 旋转后的尺寸
    height_new = int(width * fabs(sin(radians(degree))) +
                     height * fabs(cos(radians(degree))))
    width_new = int(height * fabs(sin(radians(degree))) +
                    width * fabs(cos(radians(degree))))

    mat_rotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    mat_rotation[0, 2] += (width_new - width) / 2
    mat_rotation[1, 2] += (height_new - height) / 2

    # Pay attention to the type of elements of filler_color, which should be
    # the int in pure python, instead of those in numpy.
    img_rotated = cv2.warpAffine(img, mat_rotation, (width_new, height_new),
                                 borderValue=filled_color)
    # 填充四个角
    mask = np.zeros((height_new + 2, width_new + 2), np.uint8)
    mask[:] = 0
    seed_points = [(0, 0), (0, height_new - 1), (width_new - 1, 0),
                   (width_new - 1, height_new - 1)]
    for i in seed_points:
        cv2.floodFill(img_rotated, mask, i, filled_color)

    return img_rotated

if __name__ == "__main__":
    raw_img = cv2.imread("./automobile5.jpg")

    counter = 0
    for dgr in range(-20,20):
        img = get_img_rot_broa(raw_img, degree=dgr)
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        img = cv2.resize(img, (32,32))
        cv2.imwrite("./rotated_images/automobile/automobile"+str(counter)+".png", img)
        counter += 1