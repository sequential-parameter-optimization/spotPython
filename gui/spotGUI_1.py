import tkinter as tk
import json
from spotPython.hyperparameters.values import get_default_values, get_bound_values

def get_default_values(elements):
    pass

def get_bound_values(elements):
    pass

def create_gui(dict_file):
    # Load the dictionary from the file
    with open(dict_file, 'r') as f:
        elements = json.load(f)

    # Apply the functions to the dictionary
    default_values = get_default_values(elements)
    bound_values = get_bound_values(elements)

    # Create a tkinter window
    root = tk.Tk()

    # Loop over the dictionary and create labels and entries for each key-value pair
    for i, (key, value) in enumerate(default_values.items()):
        # Create a label with the key as text
        label = tk.Label(root, text=key)
        label.grid(row=i, column=0, sticky="W")

        # Create an entry with the default value as the default text
        default_entry = tk.Entry(root)
        default_entry.insert(0, value)
        default_entry.grid(row=i, column=1, sticky="W")

        # Create an entry with the bound value as the default text
        bound_entry = tk.Entry(root)
        bound_entry.insert(0, bound_values[key])
        bound_entry.grid(row=i, column=2, sticky="W")

    # Run the tkinter main loop
    root.mainloop()

# Call the function to create the GUI
create_gui("elements.json")

