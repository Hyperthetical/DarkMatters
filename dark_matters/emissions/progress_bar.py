import sys

def progress(iteration, total,prefix = '', suffix = '', decimals = 1, length = 30, fill = '█'):
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
    #print('\r%s %s%% %s' % (prefix, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

def printProgressBar(i,max,postText):
    n_bar =10 #size of progress bar
    j= i/max
    sys.stdout.write('\r')
    sys.stdout.write(f"{int(100 * j)}%  {postText}")
    #sys.stdout.write(f"[{'=' * int(n_bar * j):{n_bar}s}] {int(100 * j)}%  {postText}")
    sys.stdout.flush()
