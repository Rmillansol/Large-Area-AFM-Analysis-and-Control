"""
File: Help_form.py
Author: Ruben Millan-Solsona
Date: August 2024

Description:
This Python script provides a set of functions to control a nanosurf device via a graphical user interface (GUI).
The functions handle connecting to the device, sending grid parameters, starting, pausing, aborting tasks, 
and clearing the workspace. It also includes utility functions for calculating grid indices based on the 
provided parameters.

Dependencies:
- logging
- tkinter (for GUI elements)
- PIL

"""

import tkinter as tk
from PIL import Image, ImageTk

class HelpForm:
    def __init__(self, master):
        # Create a new window
        self.new_window = tk.Toplevel(master)
        self.new_window.title("Help")
        self.new_window.geometry("600x300")
        self.new_window.resizable(False, False)  # Make the window non-resizable
        
        # Change the window icon
        self.new_window.iconbitmap('resources/icon_ORNL.ico')  # Replace with the path to your .ico file

        # Add explanatory text
        # Read the text file
        with open('resources/help.txt', 'r') as file:
            text = file.read()

        # Display the text in a Label
        self.label_text = tk.Label(self.new_window, text=text, wraplength=460, justify="left")
        self.label_text.place(x=120, y=10)  # Adjust the position as needed

        # About us section
        author_info = 'Functional Atomic Force Microscopy Group\nRuben Millan-Solsona\t\t\t\tsolsonrm@ornl.gov'
        self.label_text = tk.Label(self.new_window, text=author_info, wraplength=400, justify="left")
        self.label_text.place(x=170, y=250)  # Adjust the position as needed

        # Create a Canvas to draw a line
        self.canvas = tk.Canvas(self.new_window, width=600, height=2)
        self.canvas.place(x=4, y=240)
        self.canvas.create_line(10, 2, 580, 2, fill="black")

        # Resize the image to display it in the form
        image = Image.open('resources/CNMS_Logo.png')  # Relative path to the image
        image = image.resize((100, 100), Image.LANCZOS)  # Resize the image
        self.photo = ImageTk.PhotoImage(image)

        # Display the image in the form
        self.image_label = tk.Label(self.new_window, image=self.photo)
        self.image_label.image = self.photo  # Keep a reference to the image
        self.image_label.place(x=10, y=10)  # Adjust the position as needed

        # Add a close button
        self.close_button = tk.Button(self.new_window, text="Close",
                                      width=20, height=2, command=self.new_window.destroy)
        self.close_button.place(x=10, y=250)
