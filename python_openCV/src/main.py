import cv2 as cv

# imread image
image_cube = cv.imread("../dataset/cube.jpg")
"""
convert color image gray/ hsv 
gray biến pixel từ BGR -> gray value 0 ->255
exp: gray =cv.cvtColor(image_cube,cv.COLOR_BGR2GRAY)
HSV{ 
hue :màu gì
saturation: độ bão hòa (mức độ đậm nhạt hay cường độ màu sắc trong ảnh)
value}
"""
HSV = cv.cvtColor(image_cube,cv.COLOR_BGR2HSV)
# grading : phân ngưỡng theo HSV
# thredhold :ngưỡng
lower_blue = (90, 50, 30) # cần có hàm để nhận 
upper_blue = (140, 255, 255)
# MASK chuyển vùng inrange thành white else thành black
mask= cv.inRange(HSV,lower_blue, upper_blue)
# Tìm contour từ mask
contours, hierarchy= cv.findContours( #hierarchy là để xác định cha con contour
    mask,
    cv.RETR_EXTERNAL, #RETRIEVE → cách lấy contour. #EXTERNAL = chỉ lấy contour ngoài cùng
    cv.CHAIN_APPROX_SIMPLE #rút gọn các điểm không cần thiết.
)

# Vẽ contour lên ảnh gốc
cv.drawContours(image_cube,contours,
    -1,
    (0, 255, 0),
    3
)
#resize lại window cho nhỏ
cv.namedWindow("Image", cv.WINDOW_NORMAL)
cv.resizeWindow("Image", 200, 100)
#imshow
cv.imshow("Image",image_cube)
cv.waitKey(0)
cv.destroyAllWindows()