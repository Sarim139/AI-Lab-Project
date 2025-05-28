import cv2
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
from PIL import Image, ImageTk

class SmokeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Smoke Material Detection Software")
        self.root.configure(bg="#282C34")

        self.model = YOLO("D:/ids project/runs/detect/train/weights/best.pt")

        self.create_welcome_screen()

    def create_welcome_screen(self):
        self.clear_frame()

        tk.Label(self.root, text="AI SMOKE MATERIAL DETECTION SOFTWARE", font=("Calibri", 18, "bold"), pady=20, bg="#282C34", fg="#61AFEF").pack()
        tk.Label(self.root, text="WELCOME!", font=("Arial", 16), bg="#282C34", fg="#98C379").pack(pady=10)
        tk.Label(self.root, text="This software detects cigarettes and e-cigarettes in images and live camera feeds. Press Next to continue.", font=("Arial", 12), pady=10, bg="#282C34", fg="#ABB2BF").pack()

        next_button = tk.Button(self.root, text="NEXT", command=self.create_choice_screen, font=("Arial", 12), width=12, bg="#61AFEF", fg="white", activebackground="#4CAF50")
        next_button.pack(pady=20)

    def create_choice_screen(self):
        self.clear_frame()

        tk.Label(self.root, text="AI SMOKE MATERIAL DETECTION SOFTWARE", font=("Arial", 18, "bold"), pady=20, bg="#282C34", fg="#61AFEF").pack()
        tk.Label(self.root, text="CHOOSE ONE:", font=("Arial", 16), bg="#282C34", fg="#98C379").pack(pady=10)

        upload_button = tk.Button(self.root, text="UPLOAD IMAGE", command=self.upload_image, font=("Arial", 12), width=20, bg="#56B6C2", fg="white", activebackground="#61AFEF")
        upload_button.pack(pady=10)

        camera_button = tk.Button(self.root, text="CAMERA", command=self.start_camera, font=("Arial", 12), width=20, bg="#56B6C2", fg="white", activebackground="#61AFEF")
        camera_button.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", ".jpg;.jpeg;*.png")])
        if file_path:
            self.detect_in_image(file_path)

    def detect_in_image(self, file_path):
        image = cv2.imread(file_path)
        results = self.model(image)

        for result in results:  # Loop through detected results
            for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                x1, y1, x2, y2 = map(int, box)
                label = f"{self.model.names[int(cls)]} ({conf:.2f})"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        self.display_detected_image(image)

    def display_detected_image(self, image):
        self.clear_frame()

        image_width, image_height = image.size
        self.root.geometry(f"{image_width+20}x{image_height+100}")

        tk.Label(self.root, text="DETECTED IMAGE", font=("Arial", 18, "bold"), pady=20, bg="#282C34", fg="#61AFEF").pack()

        image = ImageTk.PhotoImage(image)
        canvas = tk.Label(self.root, image=image, bg="#282C34")
        canvas.image = image
        canvas.pack(pady=10)

        back_button = tk.Button(self.root, text="BACK", command=self.create_choice_screen, font=("Arial", 12), width=10, bg="#61AFEF", fg="white", activebackground="#4CAF50")
        back_button.pack(pady=20)

    def start_camera(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame)

            for result in results:  # Loop through detected results
                for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{self.model.names[int(cls)]} ({conf:.2f})"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def clear_frame(self):
        for widget in self.root.winfo_children():
            widget.destroy()

#if __name__ == "_main_":
root = tk.Tk()
app = SmokeDetectionApp(root)
root.mainloop()
