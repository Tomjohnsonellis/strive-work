import pandas as pd
import numpy as np


# Each sensor has slightly different readings so will need a function for each
# let's start with accelerometer


def extract_accelerometer(file_contents):
    sensor_readings = []
    wanted_reading_info = []
    for line in file_contents:
        if "accelerometer" in line:
            sensor_readings.append(line)

    for reading in sensor_readings:
        info = reading.split(",")
        wanted_reading_info.append([info[0], info[2], info[3], info[4]])

    accelerometer_df = pd.DataFrame(data=wanted_reading_info, columns=["id", "accel_x", "accel_y", "accel_z"])
    return accelerometer_df


def extract_gyroscope(file_contents):
    sensor_readings = []
    wanted_reading_info = []
    for line in file_contents:
        if "gyroscope," in line:
            sensor_readings.append(line)

    for reading in sensor_readings:
        info = reading.split(",")
        wanted_reading_info.append([info[0], info[2], info[3], info[4]])

    gyroscope_df = pd.DataFrame(data=wanted_reading_info, columns=["id", "gyro_x", "gyro_y", "gyro_z"])
    return gyroscope_df


def extract_geomagnetic_rot(file_contents):
    """
    https://developer.android.com/reference/android/hardware/SensorEvent#sensor.type_rotation_vector:
    Readings are: x,y,z,cos(theta),estimated accuracy
    We'll only use x,y,z
    """
    sensor_readings = []
    wanted_reading_info = []
    for line in file_contents:
        if "geomag" in line:
            sensor_readings.append(line)

    for reading in sensor_readings:
        info = reading.split(",")
        wanted_reading_info.append([info[0], info[2], info[3], info[4]])

    geomagnetic_rot_df = pd.DataFrame(data=wanted_reading_info,
                                      columns=["id", "geomagnetic_rot_x", "geomagnetic_rot_y", "geomagnetic_rot_z"])
    return geomagnetic_rot_df


def extract_gravity(file_contents):
    sensor_readings = []
    wanted_reading_info = []
    for line in file_contents:
        if "gravity" in line:
            sensor_readings.append(line)
    for reading in sensor_readings:
        info = reading.split(",")
        wanted_reading_info.append([info[0], info[2], info[3], info[4]])
    gravity_df = pd.DataFrame(data=wanted_reading_info, columns=["id", "gravity", "X_gravity extra", "Y_gravity extra"])
    return gravity_df


def extract_game_rotation_vector(file_contents):
    sensor_readings = []
    wanted_reading_info = []
    for line in file_contents:
        if "game_rotation_vector" in line:
            sensor_readings.append(line)
    for reading in sensor_readings:
        info = reading.split(",")
        wanted_reading_info.append([info[0], info[2], info[3], info[4]])
    game_rotation_df = pd.DataFrame(data=wanted_reading_info,
                                    columns=["id", "gravity", "X_gravity extra", "Y_gravity extra"])
    return game_rotation_df


def extract_rotation_vector(file_contents):
    sensor_readings = []
    wanted_reading_info = []
    for line in file_contents:
        if "rotation_vector" in line:
            sensor_readings.append(line)
    for reading in sensor_readings:
        info = reading.split(",")
        wanted_reading_info.append([info[0], info[2], info[3], info[4]])
    rotation_df = pd.DataFrame(data=wanted_reading_info,
                               columns=["id", "gravity", "X_gravity extra", "Y_gravity extra"])
    return rotation_df


def extract_orientation(file_contents):
    sensor_readings = []
    wanted_reading_info = []
    for line in file_contents:
        if "orientation" in line:
            sensor_readings.append(line)
    for reading in sensor_readings:
        info = reading.split(",")
        wanted_reading_info.append([info[0], info[2], info[3], info[4]])
    orientation_df = pd.DataFrame(data=wanted_reading_info,
                                  columns=["id", "gravity", "X_gravity extra", "Y_gravity extra"])
    return orientation_df


def extract_magnetic_field(file_contents):
    sensor_readings = []
    wanted_reading_info = []
    for line in file_contents:
        if "magnetic_field" in line:
            sensor_readings.append(line)
    for reading in sensor_readings:
        info = reading.split(",")
        wanted_reading_info.append([info[0], info[2], info[3], info[4]])
    magnetic_field_df = pd.DataFrame(data=wanted_reading_info,
                                     columns=["id", "gravity", "X_gravity extra", "Y_gravity extra"])
    return magnetic_field_df



def extract_gyroscope_uncalibrated(file_contents):
    sensor_readings = []
    wanted_reading_info = []
    for line in file_contents:
        if "gyroscope_uncalibrated" in line:
            sensor_readings.append(line)
    for reading in sensor_readings:
        info = reading.split(",")
        wanted_reading_info.append([info[0], info[2], info[3], info[4]])
    gyroscope_uncalibrated_df = pd.DataFrame(data=wanted_reading_info,
                                             columns=["id", "gravity", "X_gravity extra", "Y_gravity extra"])
    return gyroscope_uncalibrated_df


def extract_linear_acceleration(file_contents):
    sensor_readings = []
    wanted_reading_info = []
    for line in file_contents:
        if "linear_acceleration" in line:
            sensor_readings.append(line)
    for reading in sensor_readings:
        info = reading.split(",")
        wanted_reading_info.append([info[0], info[2], info[3], info[4]])
    linear_acceleration_df = pd.DataFrame(data=wanted_reading_info,
                                          columns=["id", "gravity", "X_gravity extra", "Y_gravity extra"])
    return linear_acceleration_df


def extract_orientation(file_contents):
    sensor_readings = []
    wanted_reading_info = []
    for line in file_contents:
        if "orientation" in line:
            sensor_readings.append(line)
    for reading in sensor_readings:
        info = reading.split(",")
        wanted_reading_info.append([info[0], info[2], info[3], info[4]])
    orientation_df = pd.DataFrame(data=wanted_reading_info,
                                  columns=["id", "gravity", "X_gravity extra", "Y_gravity extra"])
    return orientation_df


# for thing in sensor_readings:
#         print(thing)

# fp = "/work/Data/Row Data/U1/sensorfile_U1_Bus_1480485018989.csv"
# f = open(fp, "r")
# content = f.readlines()



def extract_light(file_contents):
    sensor_readings = []
    wanted_reading_info = []
    for line in file_contents:
        if "light" in line:
            sensor_readings.append(line)

    for reading in sensor_readings:
        info = reading.split(",")
        wanted_reading_info.append([info[0], info[2]])
    light_df = pd.DataFrame(data  = wanted_reading_info, columns=["id", "light"])
    return light_df




if __name__ == '__main__':
    fp = "sensorfile_U4_Car_1482579574027.csv"
    f = open(fp, "r")
    content = f.readlines()
    oof = extract_light(content)
    #madina = old_light(content)
    print(oof)
