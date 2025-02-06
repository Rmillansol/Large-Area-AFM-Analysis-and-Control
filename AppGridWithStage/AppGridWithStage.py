"""
File: AppGridWithStage.py
Author: Ruben Millan-Solsona
Date: August 2024

Description:
This Python script creates a graphical user interface (GUI) for controlling a grid with a Piezo scanner or motor stage.
It allows the user to set parameters such as grid length, resolution, and scan rate, and provides
controls to connect, send, start, pause, abort, and clear tasks related to the scanner.

The GUI also displays a status label and a helpful information message regarding the running software.
Errors and important events are logged to a file named 'error.log'.

Dependencies:
- logging
- os
- tkinter (for GUI elements)
- PIL (for image handling)
- Help_form (custom module for help functionality)
- actions (custom module for handling tasks)

"""

import logging
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from Help_form import HelpForm  # Import the HelpForm class
from actions import connect_task, send_task, start_task, pause_task, abort_task, clear_task, clearAll_task, Indices

# Configure error logging
logging.basicConfig(
    filename='error.log', 
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_session_start():
    """Logs the start of a session."""
    logging.info("\n" + "="*40 + " SESSION STARTED " + "="*40)

def log_session_end():
    """Logs the end of a session."""
    logging.info("\n" + "="*40 + " SESSION ENDED " + "="*40)

class AppGridPiezo(tk.Tk):
    def __init__(self):
        """Initializes the main application window and its components."""
        super().__init__()
        # Initialize attributes
        self.entry_change_id = None
        self.current_entry = None
        try:
            # Create the main window
            self.title("App Grid with Piezo Scanner")
            self.geometry("650x410")
            self.resizable(False, False)  # Make the window non-resizable

            # Set the window icon
            icon_path = os.path.join('..', 'resources', 'icon_ORNL.ico')
            self.iconbitmap(icon_path)  # Replace with the path to your .ico file

            # Load and display the image
            image_path = os.path.join('..', 'resources', 'grid.png')
            image = Image.open(image_path)  # Relative path to the image file
            image = image.resize((200, 150), Image.LANCZOS)  # Resize the image
            self.photo = ImageTk.PhotoImage(image)

            # Display the image on the form
            self.image_label = tk.Label(self, image=self.photo)
            self.image_label.place(x=320, y=60)  # Adjust the position as needed

            # Labels and input fields
            labels = ["Grid length (um):", "# n:", "# m:", "Overscan (%):", "Resolution (px):", "Scan rate (Hz):", "Center X (um):", "Center Y (um):"]
            txtvalue = ["50", "3", "3", "10", "128", "1", "0", "0"]
            self.entries = []
            y_positions = [20, 50, 80, 110, 140, 170, 200, 230]

            for i, text in enumerate(labels):
                label = tk.Label(self, text=text)
                label.place(x=20, y=y_positions[i])
                entry = tk.Entry(self, width=12)
                entry.place(x=150, y=y_positions[i])
                entry.insert(0, txtvalue[i])
                # Bind key release events only to specific text boxes
                if text in ["Grid length (um):", "# n:", "Overscan (%):"]:
                    entry.bind("<KeyRelease>", self.reset_timer)

                self.entries.append(entry)

            # Status label
            self.status_label = tk.Label(self, text="Status: Ready", anchor='w', justify='left')
            self.status_label.place(x=20, y=290)

            # Info label
            self.info_label = tk.Label(self, text="Studio software must be running before starting", anchor='w', justify='left')
            self.info_label.place(x=20, y=310)
            # Store initial label colors
            self.default_bg = self.info_label.cget("bg")
            self.default_fg = self.info_label.cget("fg")

            # Buttons with associated actions
            self.connect_button = ttk.Button(self, text="Connect", command=lambda: connect_task(self))
            self.connect_button.place(x=550, y=20)

            self.send_button = ttk.Button(self, text="Send", command=lambda: send_task(self))
            self.send_button.config(state=tk.DISABLED)
            self.send_button.place(x=550, y=50)

            self.start_button = ttk.Button(self, text="Start", command=lambda: start_task(self))
            self.start_button.config(state=tk.DISABLED)
            self.start_button.place(x=550, y=80)

            self.pause_button = ttk.Button(self, text="Pause/Resume", command=lambda: pause_task(self))
            self.pause_button.config(state=tk.DISABLED)
            self.pause_button.place(x=550, y=110)

            self.abort_button = ttk.Button(self, text="Abort", command=lambda: abort_task(self))
            self.abort_button.config(state=tk.DISABLED)
            self.abort_button.place(x=550, y=140)

            self.clear_button = ttk.Button(self, text="Clear", command=lambda: clear_task(self))
            self.clear_button.config(state=tk.DISABLED)
            self.clear_button.place(x=550, y=170)

            self.clearAll_button = ttk.Button(self, text="Clear All", command=lambda: clearAll_task(self))
            self.clearAll_button.config(state=tk.DISABLED)
            self.clearAll_button.place(x=550, y=200)

            self.help_button = ttk.Button(self, text="Help", command=self.help_task)
            self.help_button.place(x=20, y=365)

            self.close_button = ttk.Button(self, text="Close", command=self.quit)
            self.close_button.place(x=550, y=300)

            # Create a Canvas to draw a line
            self.canvas = tk.Canvas(self, width=700, height=2)
            self.canvas.place(x=10, y=350)
            self.canvas.create_line(10, 2, 615, 2, fill="black")

            # About us section
            autor = 'Functional Atomic Force Microscopy Group\nRuben Millan-Solsona\t\t\t\t\tsolsonrm@ornl.gov'
            self.label_text = tk.Label(self, text=autor, wraplength=450, justify="left")
            self.label_text.place(x=170, y=360)  # Adjust the position as needed

        except Exception as e:
            logging.error("Error initializing the application", exc_info=True)

    def reset_timer(self, event):
        """Resets the timer for delayed actions when a key is released in specific entries."""
        if self.entry_change_id is not None:
            self.after_cancel(self.entry_change_id)
        # Save the reference of the entry that triggered the event
        self.current_entry = event.widget
        self.entry_change_id = self.after(500, self.on_entry_change)

    def on_entry_change(self):
        """Performs an action after a period of inactivity in specific text entries."""
        if self.current_entry is not None:
            L = float(app.entries[0].get())
            n = int(app.entries[1].get())
            p = float(app.entries[3].get()) / 100
            listafinal, l, c = Indices(n, L, p)
            app.info_label.config(text=f"Image length: {l:.2f} um. It must be less than 90um")
            self.entry_change_id = None
            self.current_entry = None
            # Change the background color of the info label based on the image length
            if l <= 90:
                self.info_label.config(bg=self.default_bg, fg=self.default_fg)  # Normal color
            else:
                self.info_label.config(bg='red', fg='white')  # Warning color

    def help_task(self):
        """Opens the help form."""
        try:
            HelpForm(self)
        except Exception as e:
            logging.error("Error in help task", exc_info=True)

    def quit(self):
        """Closes the application safely."""
        try:
            self.destroy()
        except Exception as e:
            logging.error("Error in quit", exc_info=True)
        finally:
            log_session_end()

if __name__ == "__main__":
    try:
        log_session_start()
        app = AppGridPiezo()
        app.mainloop()
    except Exception as e:
        logging.error("Error in the main block", exc_info=True)
    finally:
        log_session_end()
