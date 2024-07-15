import cv2
import easyocr

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Start the webcam feed
cap = cv2.VideoCapture(1)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for better OCR results
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform OCR on the frame
    results = reader.readtext(gray)

    # Loop through the results and draw bounding boxes and text
    for (bbox, text, prob) in results:
        # Unpack the bounding box coordinates
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple([int(val) for val in top_left])
        bottom_right = tuple([int(val) for val in bottom_right])

        # Draw the bounding box
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        # Put the text above the bounding box
        cv2.putText(frame, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Webcam OCR', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
