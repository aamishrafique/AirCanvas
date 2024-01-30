# Import necessary libraries
import cv2
import numpy as np
from collections import deque
import time
from multiprocessing import Process, Queue
import mediapipe as mp
import pytesseract
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Initialize deques to store different color points
blue_points = [deque(maxlen=1024)]
green_points = [deque(maxlen=1024)]
red_points = [deque(maxlen=1024)]
yellow_points = [deque(maxlen=1024)]

# Initialize indices for each color
blue_index = green_index = red_index = yellow_index = 0

# Initialize a kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)

# Define RGB values for different colors
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
color_index = 0

# Create a blank paint window
paint_window = np.zeros((471, 636, 3)) + 255

# Create a named window for displaying the paint application
cv2.namedWindow("Paint", cv2.WINDOW_AUTOSIZE)

# Create a named window for displaying the OCR results
cv2.namedWindow("OCR Result", cv2.WINDOW_AUTOSIZE)

# Set up MediaPipe hands module for hand tracking
media_pipe_hands = mp.solutions.hands
hands = media_pipe_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
media_pipe_draw = mp.solutions.drawing_utils

# Open the webcam for capturing video
capture = cv2.VideoCapture(0)
return_value = True
start_time = time.time()

# Load the fine-tuned model
model_path = "Model"  # Path to model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-handwritten"
)  # Change to model's path
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")


# Function to perform OCR in a separate process
def perform_ocr(queue):
    while True:
        # Get the paint_window from the queue
        paint_window = queue.get()

        # Convert the paint_window from 64-bit to 8-bit
        paint_window_8bit = cv2.convertScaleAbs(paint_window)

        # Convert the paint_window from BGR to RGB
        paint_window_rgb = cv2.cvtColor(paint_window_8bit, cv2.COLOR_BGR2RGB)

        # Convert the numpy array to an Image object
        paint_window_image = Image.fromarray(paint_window_rgb)

        # Apply OCR on the canvas (paint_window) using the fine-tuned TrOCR model
        pixel_values = processor(paint_window_image, return_tensors="pt").pixel_values

        # Perform OCR using the model
        with torch.no_grad():
            outputs = model.generate(pixel_values.to(device))

        # Decode the predicted text
        predicted_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # Display the OCR result in the new window
        result_image = np.zeros((50, 300, 3), np.uint8) + 255
        cv2.putText(
            result_image,
            f"OCR Result: {predicted_text}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow("OCR Result", result_image)


if __name__ == "__main__":
    # Initialize a multiprocessing queue
    ocr_queue = Queue()

    # Create a Process for OCR
    ocr_process = Process(target=perform_ocr, args=(ocr_queue,))
    ocr_process.start()

    # Main loop for capturing video frames and processing them
    while return_value:
        return_value, frame = capture.read()

        # Get dimensions of the frame
        x, y, c = frame.shape

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB format for hand tracking
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Draw rectangles and labels for different colors on the video frame
        frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
        frame = cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), 2)
        frame = cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), 2)
        frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 255), 2)
        frame = cv2.rectangle(frame, (505, 1), (600, 65), (0, 255, 255), 2)

        # Set up text labels for different options on the video frame
        cv2.putText(
            frame,
            "CLEAR",
            (49, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "BLUE",
            (185, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "GREEN",
            (298, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "RED",
            (420, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "YELLOW",
            (520, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        # Process the frame with MediaPipe hands to detect hand landmarks
        result = hands.process(frame_rgb)

        # Check if hand landmarks are detected
        if result.multi_hand_landmarks:
            landmarks = []
            # Extract landmarks for each detected hand
            for hands_landmarks in result.multi_hand_landmarks:
                for landmark in hands_landmarks.landmark:
                    landmark_x = int(landmark.x * 640)
                    landmark_y = int(landmark.y * 480)
                    landmarks.append([landmark_x, landmark_y])

                # Draw hand landmarks on the frame
                media_pipe_draw.draw_landmarks(
                    frame, hands_landmarks, media_pipe_hands.HAND_CONNECTIONS
                )

            # Extract key points for hand drawing
            fore_finger = (landmarks[8][0], landmarks[8][1])
            center = fore_finger
            thumb = (landmarks[4][0], landmarks[4][1])
            cv2.circle(frame, center, 3, (0, 255, 0), -1)

            # Check hand gestures for color selection and clearing
            if thumb[1] - center[1] < 30:
                # Clear all color points if the thumb is close to the forefinger
                blue_points.append(deque(maxlen=512))
                blue_index += 1
                green_points.append(deque(maxlen=512))
                green_index += 1
                red_points.append(deque(maxlen=512))
                red_index += 1
                yellow_points.append(deque(maxlen=512))
                yellow_index += 1
            elif center[1] <= 65:
                # Select color based on the horizontal position of the hand
                # Clear canvas for the "CLEAR" option
                # Set color_index for other color options
                if 40 <= center[0] <= 140:
                    # Clear all color points
                    blue_points = [deque(maxlen=512)]
                    green_points = [deque(maxlen=512)]
                    red_points = [deque(maxlen=512)]
                    yellow_points = [deque(maxlen=512)]

                    blue_index = green_index = red_index = yellow_index = 0

                    paint_window[67:, :, :] = 255  # Clear the canvas
                elif 160 <= center[0] <= 255:
                    color_index = 0
                elif 275 <= center[0] <= 370:
                    color_index = 1
                elif 390 <= center[0] <= 485:
                    color_index = 2
                elif 505 <= center[0] <= 600:
                    color_index = 3
            else:
                # Draw points based on color selection
                if color_index == 0:
                    blue_points[blue_index].appendleft(center)
                elif color_index == 1:
                    green_points[green_index].appendleft(center)
                elif color_index == 2:
                    red_points[red_index].appendleft(center)
                elif color_index == 3:
                    yellow_points[yellow_index].appendleft(center)

        else:
            # If no hand is detected, add empty points to each color deque
            blue_points.append(deque(maxlen=512))
            blue_index += 1
            green_points.append(deque(maxlen=512))
            green_index += 1
            red_points.append(deque(maxlen=512))
            red_index += 1
            yellow_points.append(deque(maxlen=512))
            yellow_index += 1

        # Combine points for different colors and draw lines on the frame and paint window
        points = [blue_points, green_points, red_points, yellow_points]
        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):
                    if points[i][j][k - 1] is None or points[i][j][k] is None:
                        continue
                    # Draw lines on the frame and paint window
                    cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                    cv2.line(
                        paint_window, points[i][j][k - 1], points[i][j][k], colors[i], 2
                    )

        # Display the output frame and paint window
        cv2.imshow("Output", frame)
        cv2.imshow("Paint", paint_window)

        # Put the paint_window into the queue for OCR processing
        ocr_queue.put(paint_window.copy())

        elapsed_time = time.time() - start_time

        if elapsed_time >= 2:
            # Convert the paint_window from 64-bit to 8-bit
            paint_window_8bit = cv2.convertScaleAbs(paint_window)

            # Convert the paint_window from BGR to RGB
            paint_window_rgb = cv2.cvtColor(paint_window_8bit, cv2.COLOR_BGR2RGB)

            # Convert the numpy array to an Image object
            paint_window_image = Image.fromarray(paint_window_rgb)

            # Apply OCR on the canvas (paint_window)
            result = pytesseract.image_to_string(paint_window_image)

            # Display the OCR result in the new window
            result_image = np.zeros((50, 300, 3), np.uint8) + 255
            cv2.putText(
                result_image,
                f"OCR Result: {result}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow("OCR Result", result_image)

            start_time = time.time()  # Reset the start time

        # Exit the loop if "q" key is pressed
        if cv2.waitKey(1) == ord("q"):
            break

    # Release the webcam and close all windows
    capture.release()
    cv2.destroyAllWindows()

    # Terminate the OCR process
    ocr_process.terminate()
