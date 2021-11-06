from functools import wraps
import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")

from nle.env.base import NLE
import matplotlib.pyplot as plt

def add_method(cls):
    def decorator(func):
        @wraps(func) 
        def wrapper(self, *args, **kwargs): 
            return func(*args, **kwargs)
        setattr(cls, func.__name__, wrapper)
        # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
        return func # returning func means func can still be used normally
    return decorator


im1 = None

def _start_rendering(img):
    fig = plt.figure()

    #create axes
    ax1 = fig.add_subplot(111)
    ax1.has_been_closed = False

    #create image plot
    im1 = ax1.imshow(img)

    plt.ion()
    fig.canvas.mpl_connect('close_event', on_close)

    return im1

@add_method(NLE)
def render(obs, sleep=1e-2):
    global im1

    try:

        img = obs['pixel']
        mean = img.mean(axis=2)==0
        img = img[~np.all(mean, axis=1), :]
        img = img[:, ~np.all(mean, axis=0)]
        if im1 is None:
            im1 = _start_rendering(img)
        else:
            im1.set_data(img)
            plt.pause(sleep)
    except:
        pass

def on_close(event):
    stop_rendering()

def stop_rendering():
    global im1
    plt.ioff() # due to infinite loop, this gets never called.
    im1 = None