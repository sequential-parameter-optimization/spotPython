import tkinter as tk
from spotPython.hyperparameters.values import get_default_values

def create_gui(elements):
    # Create a tkinter window
    root = tk.Tk()

    # Loop over the dictionary and create labels and entries for each key-value pair
    for i, (key, value) in enumerate(elements.items()):
        # Create a label with the key as text
        label = tk.Label(root, text=key)
        label.grid(row=i, column=0, sticky="W")

        # Create an entry with the value as the default text
        entry = tk.Entry(root)
        entry.insert(0, value)
        entry.grid(row=i, column=1, sticky="W")

    # Run the tkinter main loop
    root.mainloop()

# Create a dictionary with keys and default values
# elements = {"A": "1", "B": "2", "C": "3"}
d = {"core_model_hyper_dict":{
    "leaf_prediction": {
        "levels": ["mean", "model", "adaptive"],
        "type": "factor",
        "default": "mean",
        "core_model_parameter_type": "str"},
    "leaf_model": {
        "levels": ["linear_model.LinearRegression", "linear_model.PARegressor", "linear_model.Perceptron"],
        "type": "factor",
        "default": "LinearRegression",
        "core_model_parameter_type": "instance"},
    "splitter": {
        "levels": ["EBSTSplitter", "TEBSTSplitter", "QOSplitter"],
        "type": "factor",
        "default": "EBSTSplitter",
        "core_model_parameter_type": "instance()"},
    "binary_split": {
        "levels": [0, 1],
        "type": "factor",
        "default": 0,
        "core_model_parameter_type": "bool"},
    "stop_mem_management": {
        "levels": [0, 1],
        "type": "factor",
        "default": 0,
        "core_model_parameter_type": "bool"}}}
elements = get_default_values(d)



# Call the function to create the GUI
create_gui(elements)
