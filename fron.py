import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import threading
import time
from ultralytics import YOLO
import tkinter.font as tkFont

model = YOLO('D:/ids project/runs/detect/train/weights/best.pt')

class SmokingDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smoking Material Detection System")
        self.root.geometry("1000x600")
        self.current_frame = None
        self.last_detection = None
        self.capture = None
        self.build_welcome_screen()

    def clear_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def build_welcome_screen(self):
        self.clear_screen()
        canvas = tk.Canvas(self.root, width=1000, height=600)
        canvas.pack(fill="both", expand=True)

        # Simulate gradient using solid steps of purple
        gradient_colors = ["#9c88ff", "#a29bfe", "#b388ff", "#c084f5", "#dfe6e9"]
        for i, color in enumerate(gradient_colors):
            canvas.create_rectangle(0, i * 120, 1000, (i + 1) * 120, fill=color, outline="")

        title_font = tkFont.Font(family="Helvetica", size=26, weight="bold")
        button_font = tkFont.Font(family="Helvetica", size=16)

        canvas.create_text(500, 150, text="🚭 Welcome to the Smoking Detection System",
                       font=title_font, fill="white")

        def on_enter(e): start_btn['bg'] = "#6c5ce7"
        def on_leave(e): start_btn['bg'] = "#a29bfe"

        start_btn = tk.Button(self.root, text="Start", font=button_font, bg="#a29bfe", fg="white",
                          activebackground="#6c5ce7", relief="raised", command=self.build_mode_selection_screen)
        start_btn.place(x=440, y=300, width=120, height=50)

        start_btn.bind("<Enter>", on_enter)
        start_btn.bind("<Leave>", on_leave)
    def build_mode_selection_screen(self):
        self.clear_screen()
        canvas = tk.Canvas(self.root, width=1000, height=600)
        canvas.pack(fill="both", expand=True)

    # Simulate gradient using solid blue colors
        gradient_colors = ["#74b9ff", "#81ecec", "#00cec9", "#0984e3", "#dfe6e9"]
        for i, color in enumerate(gradient_colors):
            canvas.create_rectangle(0, i * 120, 1000, (i + 1) * 120, fill=color, outline="")

        label_font = tkFont.Font(family="Helvetica", size=22, weight="bold")
        button_font = tkFont.Font(family="Helvetica", size=14)

        canvas.create_text(500, 100, text="Select Detection Mode", font=label_font, fill="white")

        def create_styled_button(text, y_pos, command):
            btn = tk.Button(self.root, text=text, font=button_font, bg="#0984e3", fg="white",
                        activebackground="#74b9ff", relief="groove", command=command)
            btn.place(x=370, y=y_pos, width=260, height=60)
            btn.bind("<Enter>", lambda e: btn.config(bg="#74b9ff"))
            btn.bind("<Leave>", lambda e: btn.config(bg="#0984e3"))

        create_styled_button("📷 Open Camera", 200, self.start_camera_screen)
        create_styled_button("🖼️ Upload from Gallery", 300, self.upload_from_gallery)

    def upload_from_gallery(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = cv2.imread(file_path)
            self.display_detection(image)

    def display_detection(self, image):
        self.clear_screen()

        # Run YOLO detection
        results = model(image)
        annotated = results[0].plot()

        image_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        img_tk = ImageTk.PhotoImage(img_pil.resize((900, 500)))

        label = tk.Label(self.root, image=img_tk)
        label.image = img_tk
        label.pack(pady=20)

        back_btn = tk.Button(self.root, text="Back", command=self.build_mode_selection_screen)
        back_btn.pack(pady=10)

    def start_camera_screen(self):
        self.clear_screen()

        self.cam_frame = tk.Label(self.root)
        self.cam_frame.pack(side="left", padx=10, pady=10)

        self.snapshot_frame = tk.Label(self.root)
        self.snapshot_frame.pack(side="right", padx=10, pady=10)

        self.capture = cv2.VideoCapture(0)
        self.running = True
        threading.Thread(target=self.update_camera_feed).start()

    def update_camera_feed(self):
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                break

            results = model(frame)
            annotated = results[0].plot()

            # If any detections, store the still frame
            if results[0].boxes:
                self.last_detection = (annotated.copy(), time.strftime("%H:%M:%S"))

            # Update live camera
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(img_pil.resize((600, 400)))
            self.cam_frame.configure(image=img_tk)
            self.cam_frame.image = img_tk

            # Update snapshot
            if self.last_detection:
                snapshot_img, timestamp = self.last_detection
                snapshot_rgb = cv2.cvtColor(snapshot_img, cv2.COLOR_BGR2RGB)
                snap_pil = Image.fromarray(snapshot_rgb)
                snap_tk = ImageTk.PhotoImage(snap_pil.resize((300, 200)))
                self.snapshot_frame.configure(image=snap_tk, text=f"Detected at {timestamp}",
                                              compound="bottom", font=("Arial", 10))
                self.snapshot_frame.image = snap_tk

            time.sleep(0.03)

    def on_close(self):
        self.running = False
        if self.capture:
            self.capture.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SmokingDetectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
