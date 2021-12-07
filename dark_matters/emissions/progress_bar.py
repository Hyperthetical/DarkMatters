def progress(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar

    Arguments
    ---------------------------
    iteration : int 
        Current iteration
    total : int
        Total iterations
    prefix : str
        Prefix string
    suffix : str
        Suffix string
    decimals : int 
        Positive number of decimals in percent complete
    length : int, optional 
        Character length of bar
    fill: str, optional
        Bar fill character 

    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()
