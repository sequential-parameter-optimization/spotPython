def save_or_show_plot(plt, filename=None) -> None:
    """
    Save or show the plot based on the provided filename.

    Args:
        plt (matplotlib.pyplot):
            The matplotlib pyplot object to save or show.
        filename (str, optional):
            The name of the file to save the plot. If None, the plot will be shown instead.
            Supported formats: 'pdf', 'png'.
            If a filename is provided, it must end with either '.pdf' or '.png'.
            If the filename is invalid, a ValueError will be raised.

    Returns:
        None

    Raises:
        ValueError: If the filename does not end with '.pdf' or '.png'.

    Examples:
        >>> from spotpython.plot.utils import save_or_show_plot
        >>> save_or_show_plot("plot.pdf")
        >>> save_or_show_plot("plot.png")
        >>> save_or_show_plot()  # This will show the plot
    """
    # Save or show the plot
    if filename:
        if filename.endswith(".pdf") or filename.endswith(".png"):
            plt.savefig(filename, format=filename.split(".")[-1], bbox_inches="tight")
        else:
            raise ValueError("Filename must have a valid suffix: '.pdf' or '.png'.")
    else:
        plt.show()
