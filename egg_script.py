import cv2

# Load image
image_path = r"C:\Users\alami\Desktop\Al-Amin Stuffs\ML\Egg Count\egg_crate.jpeg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError("Image not found at specified path.")

output = image.copy()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# Apply adaptive thresholding
binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

# Apply morphological operation
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
morphed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# Detect circles using blurred (recommended)
circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=40,
    param1=100,
    param2=30,
    minRadius=40,
    maxRadius=70
)

egg_count = 0

if circles is not None:
    circles = circles[0].astype("int")
    egg_count = len(circles)
    for (x, y, r) in circles:
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

# Annotate and display result
cv2.putText(output, f"Total Eggs: {egg_count}", (10, 30),
            cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)
cv2.imshow("Detected Eggs", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save result
cv2.imwrite("egg_counted.jpg", output)
print(f"Total Eggs Detected: {egg_count}")
