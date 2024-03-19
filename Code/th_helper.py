"""
This Python script contains some helper functions.

Authors - Belinda Fleischmann, Dirk Ostwald
"""


def humanreadable_time(time_in_seconds: float) -> str:
    """Function to transform time values into a format that is more convenient
    to read

    Args:
        time_in_seconds (float): time value in seconds, e.g. output from time.time()

    Returns:
        str: human readable time
    """
    if time_in_seconds < 0.01:
        hr_time = "< 0.01 sec."
    # elif time_in_seconds < 1:
    #     hr_time = f"{round(time_in_seconds, 2)} sec."
    elif time_in_seconds < 60:
        hr_time = f"{round(time_in_seconds, 2)} sec."
    else:
        hr_time = (f"{int(round(time_in_seconds/60, 0))}:" +
                   f"{int(round(time_in_seconds%60, 0)):02d} min.")
    return hr_time
