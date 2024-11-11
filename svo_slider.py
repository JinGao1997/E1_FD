# Filename: svo_slider_gui_with_scrollbar_fixed.py
import tkinter as tk
from tkinter import ttk
import math

class SVOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SVO Slider Measure")

        # Create a canvas and scrollbar
        self.canvas = tk.Canvas(root, borderwidth=0)
        self.frame = ttk.Frame(self.canvas)
        self.scrollbar = ttk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.create_window((0, 0), window=self.frame, anchor="nw")

        self.frame.bind("<Configure>", self.onFrameConfigure)

        # Define SVO items with endpoints and tick mark positions
        self.svo_items = [
            # Item 1
            {
                'self_left': 85,
                'other_left': 85,
                'self_right': 85,
                'other_right': 15,
                'marks': [0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100],
                'self_marks': [85]*9,
                'other_marks': [85, 76, 68, 59, 50, 41, 33, 24, 15]
            },
            # Item 2
            {
                'self_left': 85,
                'other_left': 15,
                'self_right': 100,
                'other_right': 50,
                'marks': [0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100],
                'self_marks': [85, 87, 89, 91, 93, 94, 96, 98, 100],
                'other_marks': [15, 19, 24, 28, 33, 37, 41, 46, 50]
            },
            # Item 3
            {
                'self_left': 50,
                'other_left': 100,
                'self_right': 85,
                'other_right': 85,
                'marks': [0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100],
                'self_marks': [50, 54, 59, 63, 68, 72, 76, 81, 85],
                'other_marks': [100, 98, 96, 94, 93, 91, 89, 87, 85]
            },
            # Item 4
            {
                'self_left': 50,
                'other_left': 100,
                'self_right': 85,
                'other_right': 15,
                'marks': [0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100],
                'self_marks': [50, 54, 59, 63, 68, 72, 76, 81, 85],
                'other_marks': [100, 89, 79, 68, 58, 47, 36, 26, 15]
            },
            # Item 5
            {
                'self_left': 100,
                'other_left': 50,
                'self_right': 50,
                'other_right': 100,
                'marks': [0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100],
                'self_marks': [100, 94, 88, 81, 75, 69, 63, 56, 50],
                'other_marks': [50, 56, 63, 69, 75, 81, 88, 94, 100]
            },
            # Item 6
            {
                'self_left': 100,
                'other_left': 50,
                'self_right': 85,
                'other_right': 85,
                'marks': [0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100],
                'self_marks': [100, 98, 96, 94, 93, 91, 89, 87, 85],
                'other_marks': [50, 54, 59, 63, 68, 72, 76, 82, 85]
            }
        ]

        self.num_items = len(self.svo_items)
        self.slider_values = [50] * self.num_items  # Initialize slider values to 50
        self.create_widgets()

    def onFrameConfigure(self, event):
        """Reset the scroll region to encompass the inner frame"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def create_widgets(self):
        self.frames = []
        self.scales = []
        self.self_labels = []
        self.other_labels = []

        # Define canvas width and padding
        canvas_width = 540  # Increased width to accommodate labels
        start_x = 20        # Left padding
        end_x = canvas_width - 20  # Right padding

        for i in range(self.num_items):
            frame = ttk.Frame(self.frame, padding="10 10 10 10")
            frame.grid(row=i, column=0, sticky="W")

            # Item label
            label = ttk.Label(frame, text=f"Item {i+1}", font=('Arial', 12, 'bold'))
            label.grid(row=0, column=0, sticky="W")

            item = self.svo_items[i]
            marks = item['marks']
            self_marks = item['self_marks']
            other_marks = item['other_marks']

            # "You receive" label
            self_label_above = ttk.Label(frame, text="You receive", font=('Arial', 10))
            self_label_above.grid(row=1, column=1, sticky='S', padx=10)

            # Self payoff canvas (above the slider)
            self_canvas = tk.Canvas(frame, width=canvas_width, height=30)
            self_canvas.grid(row=2, column=1, padx=10)
            for idx, mark in enumerate(marks):
                x = start_x + (mark / 100) * (end_x - start_x)
                self_canvas.create_line(x, 20, x, 30)
                self_canvas.create_text(x, 15, text=f"{self_marks[idx]}", anchor='s', font=('Arial', 8))

            # Slider
            scale_var = tk.DoubleVar(value=50.0)
            scale_length = end_x - start_x
            scale = tk.Scale(
                frame, from_=0, to=100, orient='horizontal',
                variable=scale_var, showvalue=False,
                length=scale_length, resolution=0.1,
                command=lambda value, idx=i: self.update_values(idx, float(value))
            )
            scale.grid(row=3, column=1, padx=(start_x + 10, start_x + 10))
            scale_frame = ttk.Frame(frame, width=canvas_width)
            scale_frame.grid(row=3, column=1, padx=10)

            # Other payoff canvas (below the slider)
            other_canvas = tk.Canvas(frame, width=canvas_width, height=30)
            other_canvas.grid(row=4, column=1, padx=10)
            for idx, mark in enumerate(marks):
                x = start_x + (mark / 100) * (end_x - start_x)
                other_canvas.create_line(x, 0, x, 10)
                other_canvas.create_text(x, 15, text=f"{other_marks[idx]}", anchor='n', font=('Arial', 8))

            # "Other receives" label
            other_label_below = ttk.Label(frame, text="Other receives", font=('Arial', 10))
            other_label_below.grid(row=5, column=1, sticky='N', padx=10)

            # Real-time payoff display
            payoff_frame = ttk.Frame(frame)
            payoff_frame.grid(row=3, column=2, padx=20, sticky='N')

            self_label = ttk.Label(payoff_frame, text="Your payoff:", font=('Arial', 10))
            self_label.grid(row=0, column=0, sticky='W')

            self_value_label = ttk.Label(payoff_frame, text="0.00", font=('Arial', 10))
            self_value_label.grid(row=0, column=1, sticky='W')

            other_label = ttk.Label(payoff_frame, text="Other's payoff:", font=('Arial', 10))
            other_label.grid(row=1, column=0, sticky='W')

            other_value_label = ttk.Label(payoff_frame, text="0.00", font=('Arial', 10))
            other_value_label.grid(row=1, column=1, sticky='W')

            # Save references
            self.frames.append(frame)
            self.scales.append(scale)
            self.self_labels.append(self_value_label)
            self.other_labels.append(other_value_label)

            # Initialize payoff display
            self.update_values(i, 50.0)

        # Calculate button
        calculate_button = ttk.Button(self.frame, text="Calculate SVO Angle", command=self.calculate_svo)
        calculate_button.grid(row=self.num_items + 1, column=0, pady=20)

        # Result display
        self.result_label = ttk.Label(self.frame, text="", font=('Arial', 12))
        self.result_label.grid(row=self.num_items + 2, column=0, pady=5)

    def update_values(self, idx, slider_value):
        self.slider_values[idx] = slider_value

        item = self.svo_items[idx]
        p = slider_value / 100.0  # Compute proportion factor

        # Calculate payoffs
        self_alloc = item['self_left'] + p * (item['self_right'] - item['self_left'])
        other_alloc = item['other_left'] + p * (item['other_right'] - item['other_left'])

        # Update payoff display
        self.self_labels[idx].config(text=f"{self_alloc:.2f}")
        self.other_labels[idx].config(text=f"{other_alloc:.2f}")

    def calculate_svo(self):
        allocations = []

        for i in range(self.num_items):
            slider_value = self.slider_values[i]
            p = slider_value / 100.0  # Compute proportion factor

            item = self.svo_items[i]

            # Calculate payoffs
            self_alloc = item['self_left'] + p * (item['self_right'] - item['self_left'])
            other_alloc = item['other_left'] + p * (item['other_right'] - item['other_left'])

            allocations.append({'self': self_alloc, 'other': other_alloc})

        # Compute average payoffs
        A = sum([alloc['self'] for alloc in allocations]) / len(allocations)
        B = sum([alloc['other'] for alloc in allocations]) / len(allocations)

        # Calculate SVO angle
        theta_rad = math.atan2(B - 50, A - 50)
        theta_deg = math.degrees(theta_rad)

        # Ensure angle is between 0 and 360 degrees
        if theta_deg < 0:
            theta_deg += 360

        # Classify SVO type
        if 22.45 <= theta_deg <= 157.5:
            svo_type = "Prosocial"
        elif 157.5 < theta_deg <= 202.5:
            svo_type = "Altruistic"
        elif (0 <= theta_deg < 22.45) or (337.5 < theta_deg <= 360):
            svo_type = "Individualistic"
        elif 202.5 < theta_deg <= 337.5:
            svo_type = "Competitive"
        else:
            svo_type = "Unclassified"

        # Display result
        result_text = f"SVO Angle: {theta_deg:.2f}°\nSocial Value Orientation: {svo_type}"
        self.result_label.config(text=result_text)

def main():
    root = tk.Tk()
    root.geometry("750x600")  # Adjusted window size
    app = SVOApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
