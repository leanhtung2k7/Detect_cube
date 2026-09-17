import cv2 as cv

from functions import (
    create_mask,
    find_contours,
    get_largest_contours,
    approximate_contour,
    get_y_range,
    calculate_distances
)
# =========================
# Đọc ảnh
# =========================

image_cube = cv.imread("../dataset/cube2.jpg")

HSV = cv.cvtColor(
    image_cube,
    cv.COLOR_BGR2HSV
)


# =========================
# Tìm CUBE
# =========================

lower_cube_blue = (100, 150, 150)
upper_cube_blue = (140, 255, 255)

mask_cube = create_mask(
    HSV,
    lower_cube_blue,
    upper_cube_blue
)

contours_cube = find_contours(mask_cube)

cube = max(
    contours_cube,
    key=cv.contourArea
)
# =========================
# Tìm tâm cube
# =========================

M = cv.moments(cube)

cx = int(M["m10"] / M["m00"])
cy = int(M["m01"] / M["m00"])

print("Tâm cube:", cx, cy)


# =========================
# Tìm WHITE LINE
# =========================

lower_cube_white = (100, 30, 220)
upper_cube_white = (130, 60, 255)

mask_white = create_mask(
    HSV,
    lower_cube_white,
    upper_cube_white
)

contours_white = find_contours(mask_white)


# =========================
# Lấy 2 contour lớn nhất
# =========================

contours_white = get_largest_contours(
    contours_white,
    n=2,
    min_area=1000
)


# =========================
# Approx
# =========================

for contour in contours_white:

    approx = approximate_contour(contour)

    print("Số điểm:", len(approx))


# =========================
# Tìm y trên / dưới
# =========================

y_min, y_max = get_y_range(
    contours_white
)

print("y_min =", y_min)
print("y_max =", y_max)


# =========================
# Tính khoảng cách
# =========================

d_top, d_bottom = calculate_distances(
    cy,
    y_min,
    y_max
)

print("Distance top =", d_top)
print("Distance bottom =", d_bottom)


# =========================
# Vẽ
# =========================

cv.drawContours(
    image_cube,
    contours_white,
    -1,
    (0, 255, 0),
    3
)

cv.circle(
    image_cube,
    (cx, cy),
    8,
    (0, 0, 255),
    -1
)

cv.putText(
    image_cube,
    f"d_top = {d_top:.1f}",
    (cx + 20, int(y_min)),
    cv.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
)

cv.putText(
    image_cube,
    f"d_bottom = {d_bottom:.1f}",
    (cx + 20, int(y_max)),
    cv.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
)


# =========================
# Hiển thị
# =========================
#resize lại window cho nhỏ
cv.namedWindow("Image", cv.WINDOW_NORMAL)
cv.resizeWindow("Image", 200, 100)
cv.imshow("Image", image_cube)
cv.imshow("Mask White", mask_white)

cv.waitKey(0)
cv.destroyAllWindows()