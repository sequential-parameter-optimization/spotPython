import tkinter as tk
import json
from spotPython.hyperparameters.values import get_default_values, get_bound_values
from spotPython.hyperdict.light_hyper_dict import LightHyperDict


def create_gui(model):
    lhd = LightHyperDict()
    # generate a dictionary fun_control with the key "core_model_hyper_dict" and the value lhd.hyper_dict['NetLightRegression']
    fun_control = {"core_model_hyper_dict": lhd.hyper_dict['NetLightRegression']}

    # Apply the functions to the dictionary
    default_values = get_default_values(fun_control)
    bound_values = get_bound_values(fun_control)

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
create_gui(model = 'NetLightRegression')

