import cv2
image = cv2.imread(r"C:\Users\alami\Desktop\Al-Amin Stuffs\ML\eggs_crate.jpeg")
# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Apply Gaussian blur to the grayscale image
blur = cv2.GaussianBlur(gray, (7, 7), 0)
# Apply Canny edge detection
edges = cv2.Canny(blur, 30, 150)
# Find contours in the edged image
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Filter contours based on area
egg_count = 0
for contour in contours:
    area = cv2.contourArea(contour)
    if 500 < area < 3000:  # Adjust based on your image
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        if 15 < radius < 50:
            cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            egg_count += 1
# Display the result
cv2.putText(image, f'Total Eggs: {egg_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imshow("Egg Counting", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Save the result image
cv2.imwrite('egg_counted.jpg', image)


