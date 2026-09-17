import cv2

# Đọc ảnh

img = cv2.imread("../dataset/cube.jpg")

# Chuyển BGR -> HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
def get_hsv(event, x,y,flags, param  ):
    if event == cv2.EVENT_FLAG_LBUTTON:
        # Lấy HSV tại vị trí click
        h, s, v = hsv[y, x]

        print("Vị trí:", x, y)
        print("H =", h)
        print("S =", s)
        print("V =", v)
        print("----------------")
# tạo window
cv2.namedWindow("Image")
# gắn hàm click chuột 
cv2.setMouseCallback("Image", get_hsv)
#resize lại window cho nhỏ
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", 200, 100)
while True:
    cv2.imshow("Image",img)
    # thoát bằng Esc
    if cv2.waitKey(1)== 27:
        break
cv2.destroyAllWindows
