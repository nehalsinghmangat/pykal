import numpy as np
from numpy.typing import NDArray

def forward_v(magnitude_v: float = 0.25): # m/s (max for turtlebot4)
    return np.array([[magnitude_v], [0]])


def backward_v(magnitude_v: float = 0.25):
    return np.array([[-magnitude_v], [0]])


def ccw_w(magnitude_w: float = 0.5): # rad/s (max for turtlebot4)
    return np.array([[0], [magnitude_w]])

def cw_w(magnitude_w: float = 0.5):
    return np.array([[0],[-magnitude_w]])

def no_vw():
    return np.array([[0],[0]])

def move_forward(tk: float, t_start: float, t_end: float, magnitude_v: float = 1) -> NDArray:
    if t_start <= tk < t_end:
        return forward_v(magnitude_v)

def circle_cw(tk: float, t_start: float, t_end: float, magnitude_v: float = 1, magnitude_w: float = 1) -> NDArray:
    if t_start <= tk < t_end:
        return forward_v(magnitude_v) + cw_w(magnitude_w)

def circle_ccw(tk: float, t_start: float, t_end: float, magnitude_v: float = 1, magnitude_w: float = 1) -> NDArray:
    if t_start <= tk < t_end:
        return forward_v(magnitude_v) + ccw_w(magnitude_w)

def straight_loop_straight(tk: float) -> NDArray:
    return next((
        movement for movement in [
            move_forward(tk,0, 2,magnitude_v=0.25),
            circle_ccw(tk,2, 10,magnitude_v=0.25,magnitude_w=0.5), 
            move_forward(tk,10, 15,magnitude_v=0.25), 
            circle_cw(tk,15, 25,magnitude_v=0.25,magnitude_w=0.5),
            move_forward(tk,25, 35,magnitude_v=0.25),
            no_vw()
        ] if movement is not None
    ), None)    
