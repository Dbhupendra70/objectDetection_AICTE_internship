import cv2
import os

try:
    # Load cascade classifier
    cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Cascade file not found at: {cascade_path}")
    
    plat_detector = cv2.CascadeClassifier(cascade_path)
    if plat_detector.empty():
        raise ValueError("Failed to load Haar cascade classifier.")

    # Load input image
    img_path = 'data/img3.png'
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found or unsupported format: {img_path}")

    # Detect plates
    plates = plat_detector.detectMultiScale(
        img,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(25, 25)
    )

    if len(plates) == 0:
        print("No license plates detected in the image.")

    # Process each detected plate
    for (x, y, w, h) in plates:
        cv2.putText(
            img,
            text='License Plate',
            org=(x - 3, y - 3),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            color=(0, 255, 0),
            thickness=2,
            fontScale=0.6
        )
        # Blur the plate region
        img[y:y+h, x:x+w] = cv2.blur(img[y:y+h, x:x+w], ksize=(50, 50))
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Ensure output directory exists
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # Save result
    output_path = os.path.join(output_dir, 'blurred3.jpg')
    success = cv2.imwrite(output_path, img)
    if not success:
        raise IOError(f"Failed to write output image to: {output_path}")
    
    print(f"Output saved as '{output_path}'")

except FileNotFoundError as e:
    print(f"[ERROR] File not found: {e}")
except ValueError as e:
    print(f"[ERROR] Invalid data: {e}")
except IOError as e:
    print(f"[ERROR] I/O issue: {e}")
except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")