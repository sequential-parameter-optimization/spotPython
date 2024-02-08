import tkinter as tk
from tkinter import ttk

from spotRiver.tuner.run import run_spot_river_experiment, compare_tuned_default, contour_plot, parallel_plot, importance_plot, progress_plot

result = None
fun_control = None


def run_experiment():
    global result, fun_control
    MAX_TIME = float(max_time_entry.get())
    INIT_SIZE = int(init_size_entry.get())
    PREFIX = prefix_entry.get()
    horizon = int(horizon_entry.get())
    n_total = n_total_entry.get()
    if n_total == "None" or n_total == "All":
        n_total = None
    else:
        n_total = int(n_total)
    perc_train = float(perc_train_entry.get())
    oml_grace_period = oml_grace_period_entry.get()
    if oml_grace_period == "None" or oml_grace_period == "n_train":
        oml_grace_period = None
    else:
        oml_grace_period = int(oml_grace_period)
    data_set = data_set_combo.get()
    prep_model = prep_model_combo.get()
    core_model = core_model_combo.get()

    result, fun_control = run_spot_river_experiment(
        MAX_TIME=MAX_TIME,
        INIT_SIZE=INIT_SIZE,
        PREFIX=PREFIX,
        horizon=horizon,
        n_total=n_total,
        perc_train=perc_train,
        oml_grace_period=oml_grace_period,
        data_set=data_set,
        prepmodel=prep_model,
        coremodel=core_model,
        log_level=20,
    )


def call_compare_tuned_default():
    if result is not None and fun_control is not None:
        compare_tuned_default(result, fun_control)


def call_parallel_plot():
    if result is not None:
        parallel_plot(result)


def call_contour_plot():
    if result is not None:
        contour_plot(result)


def call_importance_plot():
    if result is not None:
        importance_plot(result)


def call_progress_plot():
    if result is not None:
        progress_plot(result)


# Create the main application window
app = tk.Tk()
app.title("Spot River Hyperparameter Tuning GUI")

# Create a notebook (tabbed interface)
notebook = ttk.Notebook(app)
# notebook.pack(fill='both', expand=True)

# Create and pack entry fields for the "Run" tab
run_tab = ttk.Frame(notebook)
notebook.add(run_tab, text="Binary classification")

# colummns 0+1: Data

data_label = tk.Label(run_tab, text="Data options:")
data_label.grid(row=0, column=0, sticky="W")

data_set_label = tk.Label(run_tab, text="Select data_set:")
data_set_label.grid(row=1, column=0, sticky="W")
data_set_values = ["Bananas", "CreditCard", "Elec2", "Higgs", "HTTP", "MaliciousURL", "Phishing", "SMSSpam", "SMTP", "TREC07", "USER"]
data_set_combo = ttk.Combobox(run_tab, values=data_set_values)
data_set_combo.set("Phishing")  # Default selection
data_set_combo.grid(row=1, column=1)


n_total_label = tk.Label(run_tab, text="n_total:")
n_total_label.grid(row=2, column=0, sticky="W")
n_total_entry = tk.Entry(run_tab)
n_total_entry.insert(0, "All")
n_total_entry.grid(row=2, column=1, sticky="W")

perc_train_label = tk.Label(run_tab, text="perc_train:")
perc_train_label.grid(row=3, column=0, sticky="W")
perc_train_entry = tk.Entry(run_tab)
perc_train_entry.insert(0, "0.60")
perc_train_entry.grid(row=3, column=1, sticky="W")


# colummns 2+3: Model
model_label = tk.Label(run_tab, text="Model options:")
model_label.grid(row=0, column=2, sticky="W")

prep_model_label = tk.Label(run_tab, text="Select preprocessing model")
prep_model_label.grid(row=1, column=2, sticky="W")
prep_model_values = ["MinMaxScaler", "StandardScaler", "None"]
prep_model_combo = ttk.Combobox(run_tab, values=prep_model_values)
prep_model_combo.set("StandardScaler")  # Default selection
prep_model_combo.grid(row=1, column=3)


core_model_label = tk.Label(run_tab, text="Select core model")
core_model_label.grid(row=2, column=2, sticky="W")
core_model_values = ["AMFClassifier", "HoeffdingAdaptiveTreeClassifier", "LogisticRegression"]
core_model_combo = ttk.Combobox(run_tab, values=core_model_values)
core_model_combo.set("LogisticRegression")  # Default selection
core_model_combo.grid(row=2, column=3)


# columns 4+5: Experiment
experiment_label = tk.Label(run_tab, text="Experiment options:")
experiment_label.grid(row=0, column=4, sticky="W")

max_time_label = tk.Label(run_tab, text="MAX_TIME:")
max_time_label.grid(row=1, column=4, sticky="W")
max_time_entry = tk.Entry(run_tab)
max_time_entry.insert(0, "1")
max_time_entry.grid(row=1, column=5)

init_size_label = tk.Label(run_tab, text="INIT_SIZE:")
init_size_label.grid(row=2, column=4, sticky="W")
init_size_entry = tk.Entry(run_tab)
init_size_entry.insert(0, "3")
init_size_entry.grid(row=2, column=5)

prefix_label = tk.Label(run_tab, text="PREFIX:")
prefix_label.grid(row=3, column=4, sticky="W")
prefix_entry = tk.Entry(run_tab)
prefix_entry.insert(0, "00")
prefix_entry.grid(row=3, column=5)

horizon_label = tk.Label(run_tab, text="horizon:")
horizon_label.grid(row=4, column=4, sticky="W")
horizon_entry = tk.Entry(run_tab)
horizon_entry.insert(0, "1")
horizon_entry.grid(row=4, column=5)

oml_grace_period_label = tk.Label(run_tab, text="oml_grace_period:")
oml_grace_period_label.grid(row=5, column=4, sticky="W")
oml_grace_period_entry = tk.Entry(run_tab)
oml_grace_period_entry.insert(0, "n_train")
oml_grace_period_entry.grid(row=5, column=5)

# column 6: Run button
run_button = ttk.Button(run_tab, text="Run Experiment", command=run_experiment)
run_button.grid(row=7, column=6, columnspan=2, sticky="E")

# Create and pack the "Regression" tab with a button to run the analysis
regression_tab = ttk.Frame(notebook)
notebook.add(regression_tab, text="Regression")

# colummns 0+1: Data

regression_data_label = tk.Label(regression_tab, text="Data options:")
regression_data_label.grid(row=0, column=0, sticky="W")

# colummns 2+3: Model
regression_model_label = tk.Label(regression_tab, text="Model options:")
regression_model_label.grid(row=0, column=2, sticky="W")

# columns 4+5: Experiment
regression_experiment_label = tk.Label(regression_tab, text="Experiment options:")
regression_experiment_label.grid(row=0, column=4, sticky="W")


# Create and pack the "Analysis" tab with a button to run the analysis
analysis_tab = ttk.Frame(notebook)
notebook.add(analysis_tab, text="Analysis")

notebook.pack()


# Add the Logo image in both tabs
logo_image = tk.PhotoImage(file="images/spotlogo.png")
logo_label = tk.Label(run_tab, image=logo_image)
logo_label.grid(row=0, column=6, rowspan=1, columnspan=1)

analysis_label = tk.Label(analysis_tab, text="Analysis options:")
analysis_label.grid(row=0, column=1, sticky="W")

progress_plot_button = ttk.Button(analysis_tab, text="Progress plot", command=call_progress_plot)
progress_plot_button.grid(row=1, column=1, columnspan=2, sticky="W")

compare_tuned_default_button = ttk.Button(analysis_tab, text="Compare tuned vs. default", command=call_compare_tuned_default)
compare_tuned_default_button.grid(row=2, column=1, columnspan=2, sticky="W")

importance_plot_button = ttk.Button(analysis_tab, text="Importance plot", command=call_importance_plot)
importance_plot_button.grid(row=3, column=1, columnspan=2, sticky="W")

contour_plot_button = ttk.Button(analysis_tab, text="Contour plot", command=call_contour_plot)
contour_plot_button.grid(row=4, column=1, columnspan=2, sticky="W")

parallel_plot_button = ttk.Button(analysis_tab, text="Parallel plot (Browser)", command=call_parallel_plot)
parallel_plot_button.grid(row=5, column=1, columnspan=2, sticky="W")


analysis_logo_label = tk.Label(analysis_tab, image=logo_image)
analysis_logo_label.grid(row=0, column=6, rowspan=1, columnspan=1)

regression_logo_label = tk.Label(regression_tab, image=logo_image)
regression_logo_label.grid(row=0, column=6, rowspan=1, columnspan=1)

# Run the mainloop

app.mainloop()
