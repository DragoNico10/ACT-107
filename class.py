import cv2

# Create a video capture object
cap = cv2.VideoCapture("footvolleyball.mp4")

# Read the first frame
ret, frame = cap.read()

# Select the object to track
bbox = cv2.selectROI("Tracking", frame, False)

# Initialize the tracker with the KCF algorithm
tracker = cv2.TrackerKCF_create()
tracker.init(frame, bbox)

while True:
    # Read the next frame
    ret, frame = cap.read()

    # Exit if the video is finished
    if not ret:
        break

    # Update the tracker
    success, bbox = tracker.update(frame)

    # Draw the bounding box around the object
    if success:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

    # Display the frame
    cv2.imshow("Tracking", frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy the window
cap.release()
cv2.destroyAllWindows()
