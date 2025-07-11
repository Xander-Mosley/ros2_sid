from pymavlink import mavutil
import time
import math
import string
from typing import List, Dict, Any
import copy


class DroneCommander():
    #Interface with pymavutil 
    """
    Args:
        master_link (str): The connection string for the drone.
        use_serial (bool): Whether to use a serial connection.
        baud_rate (int): The baud rate for the serial connection.
    
    """
    def __init__(
            self,
            master_link: str = 'udpin:127.0.0.1:14551',
            use_serial: bool = False,
            baud_rate: int = 115200
            ) -> None:
        self.master_link: str = master_link
        self.use_serial: bool = use_serial
        self.baud_rate: int = baud_rate
        if self.use_serial:
            self.master = mavutil.mavlink_connection(self.master_link, baud=self.baud_rate)
        else:
            # should be UDP connection
            self.master = mavutil.mavlink_connection(self.master_link)
        self.master: mavutil = mavutil.mavlink_connection(self.master_link)
        self.master.wait_heartbeat()
        print('Connected')


    def check_rc_channel(self, rc_channel: int = 7) -> int:
        # Get the number value of specified channel
        message = self.master.recv_match(type='RC_CHANNELS', blocking=True)
        if message:
            channel_value = getattr(message, f'chan{rc_channel}_raw', None)
            if channel_value is not None:
                return channel_value
        return None
    

    def is_trigger_on(
            self,
            rc_channel: int = 7,
            channel_threshold: float = 1550
            ) -> bool:
        #return the state of the switch
        #TODO: get these values checked to see if they are standard
        #returning int is better or are these strings fine?
        value = self.check_rc_channel(rc_channel=rc_channel)
        if value is None:
            return False
        elif value >= channel_threshold:
            return True
        else:
            return False