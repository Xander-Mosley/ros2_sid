#!/usr/bin/env python3

import array
import math
import os
import sqlite3

import numpy as np
import pandas as pd

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from rotation_utils import euler_from_quaternion


# --- Connection Management ---
def connect(sqlite_file):
    conn = sqlite3.connect(sqlite_file)
    c = conn.cursor()
    return conn, c

def close(conn):
    conn.close()

# --- Generic Table Query ---
def getAllElements(cursor, table_name, print_out=False):
    """ Returns all rows from a table as a list of tuples. """
    cursor.execute('SELECT * from({})'.format(table_name))
    records = cursor.fetchall()
    if print_out:
        print(f"\nAll elements in {table_name}:")
        for row in records:
            print(row)
    return records

def countRows(cursor, table_name, print_out=False):
    """ Returns the total number of rows in the database. """
    cursor.execute('SELECT COUNT(*) FROM {}'.format(table_name))
    count = cursor.fetchall()
    if print_out:
        print('\nTotal rows: {}'.format(count[0][0]))
    return count[0][0]

def getHeaders(cursor, table_name, print_out=False):
    """ Returns a list of tuples with column informations:
    (id, name, type, notnull, default_value, primary_key)
    """
    cursor.execute('PRAGMA TABLE_INFO({})'.format(table_name))
    info = cursor.fetchall()
    if print_out:
        print("\nColumn Info:\nID, Name, Type, NotNull, DefaultVal, PrimaryKey")
        for col in info:
            print(col)
    return info

# --- Topic Metadata Access ---
def getAllTopicsNames(cursor, print_out=False):
    records = getAllElements(cursor, 'topics', print_out=False)
    names = [row[1] for row in records]
    if print_out:
        print("\nTopics names:", names)
    return names

def getAllMsgsTypes(cursor, print_out=False):
    records = getAllElements(cursor, 'topics', print_out=False)
    types = [row[2] for row in records]
    if print_out:
        print("\nMessage types:", types)
    return types

def getMsgType(cursor, topic_name, print_out=False):
    names = getAllTopicsNames(cursor, print_out=False)
    types = getAllMsgsTypes(cursor, print_out=False)
    msg_type = ""
    for i, n in enumerate(names):
        if n == topic_name:
            msg_type = types[i]
            break
    if print_out:
        print(f"\nMessage type for '{topic_name}' is {msg_type}")
    return msg_type

def isTopic(cursor, topic_name, print_out=False):
    """ Returns the topic row (id, name, type, ...) if it exists, else []. """
    records = getAllElements(cursor, 'topics', print_out=False)
    topicFound = []
    for row in records:
        if row[1] == topic_name:
            topicFound = row
    if print_out:
        if topicFound:
            print(f"\nTopic '{topicFound[1]}' exists with id {topicFound[0]}\n")
        else:
            print(f"\nTopic '{topic_name}' could not be found.\n")
    return topicFound

# --- Message Content Access ---
def getAllMessagesInTopic(cursor, topic_name, print_out=False):
    """
    Returns two lists for a given topic:
      - timestamps: list of int nanoseconds
      - messages:   list of raw BLOB data
    """
    topicFound = isTopic(cursor, topic_name, print_out=False)
    if not topicFound:
        print(f"Topic '{topic_name}' not found.")
        return [], []

    records = getAllElements(cursor, 'messages', print_out=False)
    timestamps = []
    messages   = []
    for row in records:
        if row[1] == topicFound[0]:
            timestamps.append(row[2])
            messages.append(row[3])
    if print_out:
        print(f"\nThere are {len(timestamps)} messages on '{topic_name}'")
    return timestamps, messages

# ————————————————————————————————————————————————————————————

def parse_ols(label, msg, relative_time):
    data = msg.data

    if not isinstance(data, (list, tuple, np.ndarray, array.array)):
        raise ValueError(f"{label} message data must be a sequence, got {type(data)}")
    if len(data) < 3 or (len(data) - 1) % 2 != 0:
        raise ValueError(f"{label} message data must follow the format: [output, regressors..., parameters...] with equal number of regressors and parameters.")

    num_regressors = (len(data) - 1) // 2

    result = {
        'timestamp': relative_time,
        f'{label}_measured_output': data[0]
    }
    for i in range(num_regressors):
        result[f'{label}_regressor_{i+1}'] = data[1 + i]
    for i in range(num_regressors):
        result[f'{label}_parameter_{i+1}'] = data[1 + num_regressors + i]

    return result

def parse_telem(msg, relative_time):
    return {
        'timestamp': relative_time,
        'ax': msg.accel_x,
        'ay': msg.accel_y,
        'az': msg.accel_z,
        'gx': msg.gyro_x,
        'gy': msg.gyro_y,
        'gz': msg.gyro_z
    }


def parse_imu(msg, relative_time):
    return {
        'timestamp': relative_time,
        'ax': msg.linear_acceleration.x,
        'ay': msg.linear_acceleration.y,
        'az': msg.linear_acceleration.z,
        'gx': msg.angular_velocity.x,
        'gy': msg.angular_velocity.y,
        'gz': msg.angular_velocity.z
    }

def parse_imu_raw(msg, relative_time):
    return {
        'timestamp': relative_time,
        'ax_raw': msg.linear_acceleration.x,
        'ay_raw': msg.linear_acceleration.y,
        'az_raw': msg.linear_acceleration.z,
        'gx_raw': msg.angular_velocity.x,
        'gy_raw': msg.angular_velocity.y,
        'gz_raw': msg.angular_velocity.z
    }

def parse_filt_duration(msg, relative_time):
    return {
        'timestamp': relative_time,
        'elapsed': msg.data[0],
        'ema_elapsed': msg.data[1],
        'max_elapsed': msg.data[2],
        'min_elapsed': msg.data[3]
    }
    
def parse_imu_diff(msg, relative_time):
    return {
        'timestamp': relative_time,
        # 'gax': msg.data[0],
        # 'gay': msg.data[1],
        # 'gaz': msg.data[2]
        'gax': msg.angular_velocity.x,
        'gay': msg.angular_velocity.y,
        'gaz': msg.angular_velocity.z
    }

def parse_diff_duration(msg, relative_time):
    return {
        'timestamp': relative_time,
        'elapsed': msg.data[0],
        'ema_elapsed': msg.data[1],
        'max_elapsed': msg.data[2],
        'min_elapsed': msg.data[3]
    }
    
def parse_rcout(msg, relative_time):
    return {
        'timestamp': relative_time,
        **{f'rcout_ch{i+1}': ch for i, ch in enumerate(msg.channels)}
    }

def parse_rcin(msg, relative_time):
    return {
        'timestamp': relative_time,
        **{f'rcin_ch{i+1}': ch for i, ch in enumerate(msg.channels)}
    }

def parse_odometry(msg, relative_time):
    pos = msg.pose.pose.position
    ori = msg.pose.pose.orientation
    roll, pitch, yaw = euler_from_quaternion(ori.x, ori.y, ori.z, ori.w)
    vel = msg.twist.twist.linear
    airspeed = np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
    return {
        'timestamp': relative_time,
        'x': pos.x, 'y': pos.y, 'z': pos.z,
        'roll_deg': np.rad2deg(roll),
        'pitch_deg': -np.rad2deg(pitch),
        'yaw_deg': np.rad2deg(yaw),
        'airspeed': airspeed
    }

def parse_gps(msg, relative_time):
    return {
        'timestamp': relative_time,
        'lat': msg.latitude,
        'lon': msg.longitude,
        'alt': msg.altitude
    }

def parse_gps_vel(msg, relative_time):
    return {
        'timestamp': relative_time,
        'lin_x_velo': msg.twist.linear.x,
        'lin_y_velo': msg.twist.linear.y,
        'lin_z_velo': msg.twist.linear.z,
        'ang_x_velo': msg.twist.angular.x,
        'ang_y_velo': msg.twist.angular.y,
        'ang_z_velo': msg.twist.angular.z
    }

def parse_altitude(msg, relative_time):
    return {
        'timestamp': relative_time,
        'altitude': msg.data
    }


def parse_trajectory(msg, relative_time):
    idx = msg.idx
    return {
        'timestamp': relative_time,
        'roll_cmd': np.rad2deg(msg.roll[idx]),
        'pitch_cmd': np.rad2deg(msg.pitch[idx]),
        'yaw_cmd': np.rad2deg(msg.yaw[idx])
    }


def parse_ros_message(label, msg, relative_time):
    if label.startswith('ols'):
        return parse_ols(label, msg, relative_time)
    match label:
        case 'imu':
            return parse_imu(msg, relative_time)
        case 'imu_raw':
            return parse_imu_raw(msg, relative_time)
        case 'filt_duration':
            return parse_filt_duration(msg, relative_time)
        case 'imu_diff':
            return parse_imu_diff(msg, relative_time)
        case 'diff_duration':
            return parse_diff_duration(msg, relative_time)
        case 'telem':
            return parse_telem(msg, relative_time)
        case 'rcout':
            return parse_rcout(msg, relative_time)
        case 'rcin':
            return parse_rcin(msg, relative_time)
        case 'odometry':
            return parse_odometry(msg, relative_time)
        case 'gps':
            return parse_gps(msg, relative_time)
        case 'gps_vel':
            return parse_gps_vel(msg, relative_time)
        case 'altitude':
            return parse_altitude(msg, relative_time)
        case 'trajectory':
            return parse_trajectory(msg, relative_time)
        case _:
            base = {'timestamp': relative_time}
            if hasattr(msg, 'data') and isinstance(msg.data, (list, tuple, np.ndarray, array.array)):
                for i, val in enumerate(msg.data):
                    base[f'{label}_data_{i}'] = val
            elif hasattr(msg, 'data'):
                base[f'{label}_data'] = msg.data
            else:
                base[f'{label}_unknown'] = str(msg)
            return base

# ————————————————————————————————————————————————————————————

def main(bag_file, topics_to_extract, output_directory):
    nanoseconds_per_second = 1e9
    tolerance = 0.02

    os.makedirs(output_directory, exist_ok=True)

    db_connection, db_cursor = connect(bag_file)
    all_topic_names = getAllTopicsNames(db_cursor, print_out=True)
    all_topic_types = getAllMsgsTypes(db_cursor)
    topic_type_map = {all_topic_names[i]: all_topic_types[i] for i in range(len(all_topic_names))}

    topic_dataframes = {}
    global_start_timestamp = None

    for topic_path, label in topics_to_extract.items():
        if topic_path not in topic_type_map:
            print(f"Warning: Topic {topic_path} not found in bag.")
            continue

        msg_timestamps, msg_raw_blobs = getAllMessagesInTopic(db_cursor, topic_path, print_out=True)
        if not msg_timestamps:
            print(f"No messages found for topic {topic_path}")
            continue

        if global_start_timestamp is None:
            global_start_timestamp = msg_timestamps[0]

        relative_timestamps = [(ts - global_start_timestamp) / nanoseconds_per_second for ts in msg_timestamps]
        message_type_class = get_message(topic_type_map[topic_path])
        message_data_rows = []

        for relative_time, serialized_msg in zip(relative_timestamps, msg_raw_blobs):
            deserialized_msg = deserialize_message(serialized_msg, message_type_class)
            parsed_row = parse_ros_message(label, deserialized_msg, relative_time)
            message_data_rows.append(parsed_row)

        topic_dataframe = pd.DataFrame(message_data_rows)
        topic_dataframes[label] = topic_dataframe
        topic_dataframe.to_csv(os.path.join(output_directory, f"{label}_data.csv"), index=False)
        print(f"Saved {label} data to {label}_data.csv")


    reference_label = list(topics_to_extract.values())[0]
    if reference_label in topic_dataframes:
        reference_df = topic_dataframes[reference_label].sort_values('timestamp')
        merged_synced_df = reference_df.copy()

        for label, df in topic_dataframes.items():
            if (label == reference_label):
                continue
            sorted_df = df.sort_values('timestamp')
            merged_synced_df = pd.merge_asof(
                merged_synced_df, sorted_df,
                on='timestamp',
                direction='nearest',
                tolerance=tolerance, # type: ignore
                suffixes=(None, f'_{label}')
            )

        merged_synced_df.to_csv(os.path.join(output_directory, 'synced_all_data.csv'), index=False)
        print("\nSaved fully synced data to synced_all_data.csv\n")

    close(db_connection)

if __name__ == "__main__":
    bag_file = '/develop_ws/bag_files/2026-01-22_ReplayedBin91/rosbag2_2026_01_22-18_38_12_0.db3'
    
    topics_to_extract = {
        # '/mavros/imu/data': 'imu',
        # '/mavros/imu/data_raw': 'imu_raw',
        '/imu_filt': 'imu',
        '/imu_filt_duration': 'filt_duration',
        '/imu_diff': 'imu_diff',
        '/imu_diff_duration': 'diff_duration',

        '/telem': 'telem',

        '/mavros/rc/out': 'rcout',
        '/mavros/rc/in': 'rcin',
        '/mavros/local_position/odom': 'odometry',
        # '/mavros/global_position/global': 'gps',
        # '/mavros/global_position/raw/gps_vel': 'gps_vel',
        '/mavros/global_position/rel_alt': 'altitude',
        '/mavros/imu/diff_pressure': 'diff_pressure',
        # '/mavros/imu/static_pressure': 'static_pressure',
        # '/mavros/imu/temperature_baro': 'temperature_baro',
        '/trajectory': 'trajectory',

        '/ols_rol': 'ols_rol',
        # '/ols_rol_slowed': 'ols_rol_slowed',
        # '/ols_rol_nondim': 'ols_rol_nondim',
        # '/ols_rol_nondim_inertias': 'ols_rol_nondim_inertias',
        # '/ols_rol_ssa': 'ols_rol_ssa',
        # '/ols_rol_ssa_nondim': 'ols_rol_ssa_nondim',
        # '/ols_rol_ssa_nondim_inertias': 'ols_rol_ssa_nondim_inertias',
        
        '/ols_rol_large': 'ols_rol_large',
        # '/ols_rol_large_nondim': 'ols_rol_large_nondim',
        # '/ols_rol_large_nondim_inertias': 'ols_rol_large_nondim_inertias',
        # '/ols_rol_large_ssa': 'ols_rol_large_ssa',
        # '/ols_rol_large_ssa_nondim': 'ols_rol_large_ssa_nondim',
        # '/ols_rol_large_ssa_nondim_inertias': 'ols_rol_large_ssa_nondim_inertias',

        '/ols_pit': 'ols_pit',
        # '/ols_pit_nondim': 'ols_pit_nondim',
        # '/ols_pit_nondim_inertias': 'ols_pit_nondim_inertias',
        # '/ols_pit_aoa': 'ols_pit_aoa',
        # '/ols_pit_aoa_nondim': 'ols_pit_aoa_nondim',
        # '/ols_pit_aoa_nondim_inertias': 'ols_pit_aoa_nondim_inertias',

        '/ols_yaw': 'ols_yaw',
        # '/ols_yaw_nondim': 'ols_yaw_nondim',
        # '/ols_yaw_nondim_inertias': 'ols_yaw_nondim_inertias',
        # '/ols_yaw_ssa': 'ols_yaw_ssa',
        # '/ols_yaw_ssa_nondim': 'ols_yaw_ssa_nondim',
        # '/ols_yaw_ssa_nondim_inertias': 'ols_yaw_ssa_nondim_inertias',
        
        '/ols_yaw_large': 'ols_yaw_large',
        # '/ols_yaw_large_nondim': 'ols_yaw_large_nondim',
        # '/ols_yaw_large_nondim_inertias': 'ols_yaw_large_nondim_inertias',
        # '/ols_yaw_large_ssa': 'ols_yaw_large_ssa',
        # '/ols_yaw_large_ssa_nondim': 'ols_yaw_large_ssa_nondim',
        # '/ols_yaw_large_ssa_nondim_inertias': 'ols_yaw_large_ssa_nondim_inertias'
        }
    
    output_directory = '/develop_ws/src/ros2_sid/ros2_sid/ros2_sid/topic_data_files'

    main(bag_file, topics_to_extract, output_directory)