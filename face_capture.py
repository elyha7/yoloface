import cv2
import os

# Function to capture multiple face images and save to database folder
def capture_face_images(database_folder, input_name):
    # Create the input_name folder in the database folder if it doesn't exist
    input_folder = os.path.join(database_folder, input_name)
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)

    # Initialize the webcam
    video_capture = cv2.VideoCapture(0)  # Use 0 for the primary webcam, adjust accordingly if multiple webcams are present

    i = len(os.listdir(input_folder)) + 1 # Count the number of existing face images in the input_name folder
    while True:
        # Capture a frame from the webcam
        ret, frame = video_capture.read()
        if not ret:
            break

        # Display the frame to the user
        cv2.imshow("Capture Face Image", frame)

        # Allow the user to save the captured face image
        key = cv2.waitKey(1)
        if key == ord('s'):  # Press 's' key to save the image
            # Save the captured face image in the input_name folder
            face_image_path = os.path.join(input_folder, f"{input_name}_{i}.jpg")
            cv2.imwrite(face_image_path, frame)
            print(f"Face image {i} saved for {input_name}.")
            i += 1


        elif key == 27 or key == ord('q'):  # Press 'Esc' or 'q' key to quit capturing
            break

    # Release the video capture and close the OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    database_folder = "database"  # Folder to store the captured face images
    input_name = input("Please enter your name: ").capitalize() # Name of the person to capture face images
    print('You input name is: ', input_name)
    print(f"Please look at the camera and press 's' to save the captured face images for {input_name}.")
    capture_face_images(database_folder, input_name)
