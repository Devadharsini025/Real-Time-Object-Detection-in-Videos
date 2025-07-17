import tkinter as tk  # Import the Tkinter library for creating GUIs.
from tkinter import ttk  # Import themed Tkinter components.
from tkinter import filedialog  # Import file dialog for file selection.
from imutils.video import VideoStream  # Import video streaming from imutils library.
from imutils.video import FPS  # Import FPS calculation from imutils library.
import numpy as np  # Import NumPy for numerical operations.
import imutils  # Import utility functions for OpenCV.
import cv2  # Import OpenCV for computer vision tasks.
import threading  # Import threading for parallel execution.
class ObjectDetectionGUI:
    def __init__(self, root):
        self.root = root  # Initialize the Tkinter root window.
        self.root.title("Real-Time Object Detection")  # Set the title of the window.
        self.root.geometry("800x600")  # Set the dimensions of the window.

        # Variables to store file paths, confidence level, video stream, and stop event.
        self.prototxt_path = tk.StringVar()
        self.model_path = tk.StringVar()
        self.confidence = tk.DoubleVar(value=0.2)
        self.vs = None
        self.stopEvent = None

        # Create the layout of the GUI.
        self.create_layout()

    def create_layout(self):
        # Entry for Prototxt file path.
        prototxt_label = ttk.Label(self.root, text="Prototxt File:")
        prototxt_label.grid(row=0, column=0, padx=10, pady=10, sticky="e")
        prototxt_entry = ttk.Entry(self.root, textvariable=self.prototxt_path, width=50)
        prototxt_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        prototxt_button = ttk.Button(self.root, text="Browse", command=self.browse_prototxt)
        prototxt_button.grid(row=0, column=2, padx=10, pady=10, sticky="w")

        # Entry for Model file path.
        model_label = ttk.Label(self.root, text="Model File:")
        model_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")
        model_entry = ttk.Entry(self.root, textvariable=self.model_path, width=50)
        model_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        model_button = ttk.Button(self.root, text="Browse", command=self.browse_model)
        model_button.grid(row=1, column=2, padx=10, pady=10, sticky="w")

        # Entry for confidence level.
        confidence_label = ttk.Label(self.root, text="Confidence:")
        confidence_label.grid(row=2, column=0, padx=10, pady=10, sticky="e")
        confidence_entry = ttk.Entry(self.root, textvariable=self.confidence)
        confidence_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")

        # Button to start object detection.
        start_button = ttk.Button(self.root, text="Start Detection", command=self.start_detection)
        start_button.grid(row=3, column=0, columnspan=3, pady=20)
    def browse_prototxt(self):
        # Open a file dialog to select Prototxt file and set the file path.
        file_path = filedialog.askopenfilename(filetypes=[("Prototxt files", "*.prototxt")])
        if file_path:
            self.prototxt_path.set(file_path)

    def browse_model(self):
        # Open a file dialog to select Caffe Model file and set the file path.
        file_path = filedialog.askopenfilename(filetypes=[("Caffe Model files", "*.caffemodel")])
        if file_path:
            self.model_path.set(file_path)
    def start_detection(self):
        # Retrieve values from the GUI.
        prototxt_file = self.prototxt_path.get()
        model_file = self.model_path.get()
        confidence_value = self.confidence.get()

        # Disable the start button to prevent multiple threads.
        self.root.children['!button']['state'] = 'disable'

        # Start detection in a new thread using VideoStream.
        self.stopEvent = threading.Event()
        self.vs = VideoStream(src=0).start()
        threading.Thread(target=self.detect_objects, args=(prototxt_file, model_file, confidence_value)).start()

    def detect_objects(self, prototxt_file, model_file, confidence_value):
        # Load the pre-trained model from Caffe.
        print("[INFO] loading model...")
        net = cv2.dnn.readNetFromCaffe(prototxt_file, model_file)

        # Initialize the FPS counter.
        fps = FPS().start()

        while not self.stopEvent.is_set():
            try:
                # Read a frame from the video stream and resize it.
                frame = self.vs.read()
                frame = imutils.resize(frame, width=400)

                # Preprocess the frame for object detection.
                (h, w) = frame.shape[:2]
                resized_image = cv2.resize(frame, (300, 300))
                blob = cv2.dnn.blobFromImage(resized_image, (1 / 127.5), (300, 300), 127.5, swapRB=True)

                # Set the input to the neural network and forward pass.
                net.setInput(blob)
                predictions = net.forward()

                # Process the predictions and draw bounding boxes.
                for i in np.arange(0, predictions.shape[2]):
                    confidence = predictions[0, 0, i, 2]
                    if confidence > confidence_value:
                        idx = int(predictions[0, 0, i, 1])
                        box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                        print("Object detected: ", label)
                        cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                # Display the frame with annotations.
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                # Update the FPS counter.
                fps.update()
            except Exception as e:
                print(f"An error occurred: {e}")
                break

        # Stop the FPS counter and display the elapsed time and FPS.
        fps.stop()
        print("[INFO] Elapsed Time: {:.2f}".format(fps.elapsed()))
        print("[INFO] Approximate FPS: {:.2f}".format(fps.fps()))

        # Close OpenCV windows and stop the video stream.
        cv2.destroyAllWindows()
        self.vs.stop()
        self.stopEvent.clear()

        # Enable the start button after detection is finished.
        self.root.children['!button']['state'] = 'normal'
if __name__ == "__main__":
    # Initializations of classes and variables.
    CLASSES = ["aeroplane", "background", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # Create Tkinter window and start the GUI.
    root = tk.Tk()
    app = ObjectDetectionGUI(root)
    root.mainloop()
