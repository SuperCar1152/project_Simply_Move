import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

df = pd.read_csv('../dataCSV/leg/OtherMove_20240411_163510.csv')

# Function to convert Euler angles to rotation matrix
def euler_to_rotmat(euler_x, euler_y, euler_z):
    # Convert Euler angles to radians
    euler_x_rad = np.radians(euler_x)
    euler_y_rad = np.radians(euler_y)
    euler_z_rad = np.radians(euler_z)

    # Initialize empty list for rotation matrices
    rot_mat_list = []

    # Iterate over each set of Euler angles
    for i in range(len(euler_x)):
        # Calculate rotation matrices for each set of Euler angles
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(euler_x_rad[i]), -np.sin(euler_x_rad[i])],
                       [0, np.sin(euler_x_rad[i]), np.cos(euler_x_rad[i])]])

        Ry = np.array([[np.cos(euler_y_rad[i]), 0, np.sin(euler_y_rad[i])],
                       [0, 1, 0],
                       [-np.sin(euler_y_rad[i]), 0, np.cos(euler_y_rad[i])]])

        Rz = np.array([[np.cos(euler_z_rad[i]), -np.sin(euler_z_rad[i]), 0],
                       [np.sin(euler_z_rad[i]), np.cos(euler_z_rad[i]), 0],
                       [0, 0, 1]])

        # Combine rotation matrices
        rot_mat = np.dot(Rz, np.dot(Ry, Rx))

        # Append the rotation matrix to the list
        rot_mat_list.append(rot_mat)

    return np.array(rot_mat_list)

# Function to integrate acceleration to estimate position
def integrate_acceleration(acc_data, time_interval):
    # Integrate acceleration twice to get position
    velocity = np.cumsum(acc_data * time_interval, axis=0)
    position = np.cumsum(velocity * time_interval, axis=0)
    return position

# Function to create the animation
def animate_positions(frame):
    ax.clear()

    # Plot the body segments
    ax.plot(left_wrist_abs_position[frame, :, 0], left_wrist_abs_position[frame, :, 1],
            left_wrist_abs_position[frame, :, 2], color='blue')
    ax.plot(right_ankle_abs_position[frame, :, 0], right_ankle_abs_position[frame, :, 1],
            right_ankle_abs_position[frame, :, 2], color='red')
    ax.plot(left_ankle_abs_position[frame, :, 0], left_ankle_abs_position[frame, :, 1],
            left_ankle_abs_position[frame, :, 2], color='green')
    ax.plot(right_wrist_abs_position[frame, :, 0], right_wrist_abs_position[frame, :, 1],
            right_wrist_abs_position[frame, :, 2], color='orange')
    ax.plot(middle_abs_position[frame, :, 0], middle_abs_position[frame, :, 1], middle_abs_position[frame, :, 2],
            color='purple')

    # Set axis limits
    ax.set_xlim([-0.2, 0.2])
    ax.set_ylim([-0.2, 0.2])
    ax.set_zlim([-0.2, 0.2])

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set title
    ax.set_title('Human Body Animation')

# Filter data for left wrist and right ankle
left_wrist_data = df[df['Device'] == 'D4:22:CD:00:05:6E'].reset_index(drop=True)
right_ankle_data = df[df['Device'] == 'D4:22:CD:00:05:5D'].reset_index(drop=True)
left_ankle_data = df[df['Device'] == 'D4:22:CD:00:05:5E'].reset_index(drop=True)
right_wrist_data = df[df['Device'] == 'D4:22:CD:00:05:6B'].reset_index(drop=True)
middle_data = df[df['Device'] == 'D4:22:CD:00:49:AA'].reset_index(drop=True)


## Convert Euler angles to rotation matrices
left_wrist_rotmat = euler_to_rotmat(left_wrist_data['EulerX'], left_wrist_data['EulerY'], left_wrist_data['EulerZ'])
right_ankle_rotmat = euler_to_rotmat(right_ankle_data['EulerX'], right_ankle_data['EulerY'], right_ankle_data['EulerZ'])
left_ankle_rotmat = euler_to_rotmat(left_ankle_data['EulerX'], left_ankle_data['EulerY'], left_ankle_data['EulerZ'])
right_wrist_rotmat = euler_to_rotmat(right_wrist_data['EulerX'], right_wrist_data['EulerY'], right_wrist_data['EulerZ'])
middle_rotmat = euler_to_rotmat(middle_data['EulerX'], middle_data['EulerY'], middle_data['EulerZ'])

# Integrate acceleration data to estimate position
time_interval = df['Timestamp'].diff().mean()  # Assuming constant time interval
left_wrist_acc = left_wrist_data[['FreeAccX', 'FreeAccY', 'FreeAccZ']].values
right_ankle_acc = right_ankle_data[['FreeAccX', 'FreeAccY', 'FreeAccZ']].values
left_ankle_acc = left_ankle_data[['FreeAccX', 'FreeAccY', 'FreeAccZ']].values
right_wrist_acc = right_wrist_data[['FreeAccX', 'FreeAccY', 'FreeAccZ']].values
middle_acc = middle_data[['FreeAccX', 'FreeAccY', 'FreeAccZ']].values


left_wrist_position = integrate_acceleration(left_wrist_acc, time_interval)
right_ankle_position = integrate_acceleration(right_ankle_acc, time_interval)
left_ankle_position = integrate_acceleration(left_ankle_acc, time_interval)
right_wrist_position = integrate_acceleration(right_wrist_acc, time_interval)
middle_position = integrate_acceleration(middle_acc, time_interval)

# Transform positions to absolute positions using rotation matrices
left_wrist_abs_position = np.dot(left_wrist_rotmat, left_wrist_position.T).T
right_ankle_abs_position = np.dot(right_ankle_rotmat, right_ankle_position.T).T
right_wrist_abs_position = np.dot(right_wrist_rotmat, left_wrist_position.T).T
left_ankle_abs_position = np.dot(left_ankle_rotmat, right_ankle_position.T).T
middle_abs_position = np.dot(middle_rotmat, right_ankle_position.T).T

# Initialize the figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the animation
ani = FuncAnimation(fig, animate_positions, frames=len(left_wrist_abs_position), interval=20)

# Show the animation
plt.show()
