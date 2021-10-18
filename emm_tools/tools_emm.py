#cython: language_level=3
import sys
#from numba import jit

def progressOld(j,jmax):
    p = j*100.0/jmax
    div = 3
    prog = int(p/div)
    bar = "  "+"-"*prog
    if prog == 100/div:
        bar += " Done"
    sys.stdout.write(bar+"\r%d%%" % p)
    sys.stdout.flush()
#@jit
def progress(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
        ---------------------------
        Parameters
        ---------------------------
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()
