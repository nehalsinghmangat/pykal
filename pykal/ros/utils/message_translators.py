# pykal/ros/utils/message_translators.py

import numpy as np
from numpy.typing import NDArray
from geometry_msgs.msg import Twist


def ndarray_to_twist(u: NDArray) -> Twist:
    if not isinstance(u, np.ndarray) or u.shape != (2, 1):
        raise ValueError(f"Expected input of shape (2, 1), got {u}")
    msg = Twist()
    msg.linear.x = float(u[0, 0])
    msg.angular.z = float(u[1, 0])
    return msg


# Map from message name → translator function
MESSAGE_TRANSLATORS = {
    "Twist": ndarray_to_twist,
}

# Map from message name → message class
MESSAGE_TYPES = {
    "Twist": Twist,
}
