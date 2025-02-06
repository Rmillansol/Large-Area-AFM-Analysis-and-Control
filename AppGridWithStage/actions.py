"""
File: actionstrol.py
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
- nanosurf (for controlling the device)

"""

import logging
import tkinter as tk
from datetime import datetime
import threading
import time
from time import sleep
from os import path
import nanosurf

# Global variables for nanosurf control
global studio
global workspace 
global automation
global state
global id_list

def Indices(n, L, p):
    """
    Calculate grid indices based on the number of points, grid length, and overscan percentage.

    Parameters:
    n (int): Number of grid points.
    L (float): Length of the grid.
    p (float): Overscan percentage.

    Returns:
    list: Final calculated list of grid indices.
    float: Individual length of each grid point.
    float: Corrected length considering overscan.
    """
    l = L / (n - p * n + p)
    c = l - l * p
    
    if n % 2 == 0:  # If the number is even
        nmax = (n / 2) - 1
        lista = [0.5]
        numero = 1
        while numero <= nmax:
            lista.append(numero + 0.5)
            numero += 1.0
        
        listaR = list(reversed(lista))
        listaR = [x * -1 for x in listaR]
        listafinal = listaR + lista

    else:  # If the number is odd
        nmax = (n - 1) / 2
        lista = []
        numero = 1
        while numero <= nmax:
            lista.append(numero)
            numero += 1.0
        
        listaR = list(reversed(lista))
        listaR = [x * -1 for x in listaR]
        listafinal = listaR + [0.0] + lista
    
    listafinal = [x * c for x in listafinal]
    return listafinal, l, c

def Check_running_task(app):
    """
    Monitor the running state of the automation task, updating the status label in the GUI.

    Parameters:
    app (tk.Tk): The main application instance.
    """
    cont = 0
    while automation.is_running():
        cont += 1
        time.sleep(1)  # Simulate a time-consuming task
        app.status_label.config(text=f"Status: Connected\nConnected with session {studio.session_id}\nRunning grid{cont*'.'}")
        if cont > 6:
            cont = 0

    app.status_label.config(text=f"Status: Connected\nConnected with session {studio.session_id}\nDone!")
    app.pause_button.config(state=tk.DISABLED) 
    app.abort_button.config(state=tk.DISABLED)  

def connect_task(app):
    """
    Establish a connection to the nanosurf studio and initialize the workspace and automation objects.

    Parameters:
    app (tk.Tk): The main application instance.
    """
    global studio
    global workspace 
    global automation
    try:
        app.status_label.config(text="Connecting")
        # Studio must be running before executing the following!
        studio = nanosurf.Studio()
        i = 0
        while not studio.connect():
            app.status_label.config(text=f"Connecting.{i*'.'}")
            i += 1
            if i > 10:
                i = 0
            sleep(0.5)

        workspace = studio.spm.workflow.workspace
        automation = studio.spm.workflow.automation
        app.status_label.config(text=f"Status: Connected\nConnected with session {studio.session_id}")
        app.send_button.config(state=tk.NORMAL)  
        app.clearAll_button.config(state=tk.NORMAL)
        
    except Exception as e:
        logging.error("Error in the connect method", exc_info=True)

def send_task(app):
    """
    Send the grid parameters to the nanosurf workspace and queue the automation task.

    Parameters:
    app (tk.Tk): The main application instance.
    """
    global workspace 
    global automation
    global id_list
    try:
        try:
            # Retrieve values from the input fields
            grid_length = float(app.entries[0].get())
            n_grid = int(app.entries[1].get())
            m_grid = int(app.entries[2].get())
            overscan = float(app.entries[3].get()) / 100
            resolution = int(app.entries[4].get())
            scan_rate = float(app.entries[5].get())
            center_X = float(app.entries[6].get())
            center_Y = float(app.entries[7].get())

        except Exception as e:
            app.status_label.config(text="Status: NOT Connected!\nSome of the parameters have an invalid value")
            return
        
        # Observe what's happening in the ViewPort.
        id_list = []
        
        indX, iml, c = Indices(n_grid, grid_length, overscan)
        grid_length_Y = m_grid * (iml - overscan * iml) + overscan * iml
        indY, imly, cy = Indices(m_grid, grid_length_Y, overscan)
        i = 0
        for iY in indY:
            for iX in indX:
                id_list.append(workspace.add_frame("MyFrame" + str(i + 1), 1e-6 * (iX + center_X), 1e-6 * (iY + center_Y)))
                i += 1

        # Adjust item properties and add each item to the queue
        for i in range(n_grid * m_grid):
            workspace.set_item_resolution(id_list[i], resolution, resolution)
            workspace.set_item_size(id_list[i], 1e-6 * iml, 1e-6 * iml)                           
            workspace.set_item_line_rate(id_list[i], scan_rate, True)
            automation.add_to_queue(id_list[i])

        # How many items are in the queue now?
        automation.queue_size()
        app.status_label.config(text=f"Status: Connected\nConnected with session {studio.session_id}\nSent grid!")
        app.info_label.config(text=f"Connected with session {studio.session_id}\nSent grid!")
        app.start_button.config(state=tk.NORMAL)
        app.clear_button.config(state=tk.NORMAL)   

    except Exception as e:
        logging.error("Error in the send_task method", exc_info=True)

def start_task(app):
    """
    Start the automation task in the nanosurf system.

    Parameters:
    app (tk.Tk): The main application instance.
    """
    global state
    try:
        automation.start()
        app.status_label.config(text=f"Status: Connected\nConnected with session {studio.session_id}\nLaunch grid!")
        app.pause_button.config(state=tk.NORMAL) 
        app.abort_button.config(state=tk.NORMAL) 
        state = 'running'

        wire = threading.Thread(target=Check_running_task, args=(app,))
        wire.start()
        
    except Exception as e:
        logging.error("Error in the start_task method", exc_info=True)

def pause_task(app): 
    """
    Pause or resume the automation task depending on the current state.

    Parameters:
    app (tk.Tk): The main application instance.
    """
    global state
    try:
        if state == 'running':
            automation.pause()
            app.status_label.config(text=f"Status: Connected\nConnected with session {studio.session_id}\nPaused grid, must be resumed!")
            state = 'pause'
        else:
            automation.resume()
            app.status_label.config(text=f"Status: Connected\nConnected with session {studio.session_id}\nLaunch grid!")
            state = 'running'
    except Exception as e:
        logging.error("Error in the pause_task method", exc_info=True)
   

def abort_task(app):
    """
    Abort the automation task and reset the GUI buttons.

    Parameters:
    app (tk.Tk): The main application instance.
    """
    try:
        automation.abort()
        app.pause_button.config(text='Pause')
        app.status_label.config(text=f"Status: Connected\nConnected with session {studio.session_id}\nAbort grid!")
        state = 'abort'
        app.pause_button.config(state=tk.DISABLED) 
        app.abort_button.config(state=tk.DISABLED)   
    except Exception as e:
        logging.error("Error in the abort_task method", exc_info=True)

def clear_task(app):
    """
    Clear the current grid items from the workspace.

    Parameters:
    app (tk.Tk): The main application instance.
    """
    try:
        # Clear the workspace
        for i in id_list:
            workspace.delete_item(i)

        app.status_label.config(text=f"Status: Connected\nConnected with session {studio.session_id}\nGrid cleaned, must be sent!")
        app.start_button.config(state=tk.DISABLED)
        app.pause_button.config(state=tk.DISABLED) 
        app.abort_button.config(state=tk.DISABLED)  
        app.clear_button.config(state=tk.DISABLED) 
    except Exception as e:
        logging.error("Error in the clear_task method", exc_info=True)

def clearAll_task(app):
    """
    Clear all items from the workspace.

    Parameters:
    app (tk.Tk): The main application instance.
    """
    try:
        workspace.delete_all_items()
        app.status_label.config(text=f"Status: Connected\nConnected with session {studio.session_id}\nGrid cleaned, must be sent!")
        app.start_button.config(state=tk.DISABLED)
        app.pause_button.config(state=tk.DISABLED) 
        app.abort_button.config(state=tk.DISABLED)
        app.clear_button.config(state=tk.DISABLED)   
    except Exception as e:
        logging.error("Error in the clearAll_task method", exc_info=True)
