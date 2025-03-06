import rclpy
from rclpy.node import Node
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import math

class TestStateFunction(Node):
    def __init__(self):
        super().__init__('test_state_function')

        self.state = np.zeros(8)
        self.previous_state = np.zeros(8)
        self.input = np.zeros(4)
        self.timestamp = 0.0
        self.freq = 25
        self.time_step = 1 / self.freq
        self.simulation_duration = 10  # seconds

        self.A = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.04, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.04],
            [0.0, 0.0, 0.0, 0.0, 0.99377999, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.99596168, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.87923344, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.90079292]
        ])

        self.B = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0013678, 0.0, 0.0, 0.0],
            [0.0, 0.00137339, 0.0, 0.0],
            [0.0, 0.0, 0.0011884, 0.0],
            [0.0, 0.0, 0.0, 0.00126393]
        ])

        self.simulated_states = []
        self.real_states = {
            'x': [],
            'y': [],
            'z': [],
            'yaw': []
        }
        self.time_axis = []
        
        file_names = [
            ('drone_state_data_x.csv', 'x'),
            ('drone_state_data_y.csv', 'y'),
            ('drone_state_data_z.csv', 'z'),
            ('drone_state_data_yaw.csv', 'yaw'),
        ]

        for file_name, state_key in file_names:
            self.read_real_data(file_name, state_key)


        self.run_simulation()
        self.plot_states()

    def read_real_data(self,file_name, state_key):
        data_dir = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data')
        csv_file = os.path.join(data_dir, file_name)

        with open(csv_file, mode='r') as file:
            reader = csv.DictReader(file)
            initial_position = None
            for row in reader:
                timestamp = float(row['elapsed_time'])
                if timestamp <= self.simulation_duration:
                    yaw = float(row['yaw'])
                    if yaw < 0 and initial_position is not None:
                       yaw += 2 * math.pi
                    
                    current_position = [
                        float(row['x']), float(row['y']), float(row['z']), yaw,
                        float(row['x_speed']), float(row['y_speed']), float(row['z_speed']), float(row['yaw_speed'])
                    ]
                    if initial_position is None:
                        initial_position = current_position[:4]  # Save initial position (x, y, z, yaw)
                    normalized_position = [
                        current_position[0] - initial_position[0],
                        current_position[1] - initial_position[1],
                        current_position[2] - initial_position[2],
                        current_position[3] - initial_position[3],
                        current_position[4],  # Speeds don't need normalization
                        current_position[5],
                        current_position[6],
                        current_position[7]
                    ]
                    self.real_states[state_key].append(normalized_position)
                    self.time_axis.append(timestamp)


    def run_simulation(self):
        input_sequences = {
            'x': [20, 0, 0, 0] * (self.freq * 5) + [-20, 0, 0, 0] * (self.freq * 5),
            'y': [0, 20, 0, 0] * (self.freq * 5) + [0, -20, 0, 0] * (self.freq * 5),
            'z': [0, 0, 50, 0] * (self.freq * 5) + [0, 0, -50, 0] * (self.freq * 5),
            'yaw': [0, 0, 0, 100] * (self.freq * 5) + [0, 0, 0, -100] * (self.freq * 5),
        }

        for key, input_sequence in input_sequences.items():
            simulated_states = self.simulate(input_sequence)
            self.simulated_states.append(simulated_states)

    def simulate(self, input_sequence):
        states = []
        self.previous_state = np.zeros(8)
        for i in range(len(input_sequence) // 4):
            self.input = input_sequence[i * 4:(i + 1) * 4]
            self.state = self.A.dot(self.previous_state) + self.B.dot(self.input)
            states.append(self.state.copy())
            self.previous_state = self.state
        return np.array(states)

    def plot_states(self):
        fig, axs = plt.subplots(4, 2, figsize=(15, 20))
        fig.suptitle('Comparison of Simulated and Real States')

        labels = ['x', 'y', 'z', 'yaw', 'x_speed', 'y_speed', 'z_speed', 'yaw_speed']

        for i, key in enumerate(self.real_states.keys()):
            real_states = np.array(self.real_states[key])
            simulated_states = self.simulated_states[i]
            time_axis = self.time_axis[:len(real_states)]
            min_length = min(len(simulated_states), len(real_states), len(time_axis))

            # Position plot
            ax_pos = axs[i, 0]
            ax_pos.plot(time_axis[:min_length], simulated_states[:min_length, i], label='Simulated')
            if min_length > 0:
                ax_pos.plot(time_axis[:min_length], real_states[:min_length, i], label='Real', linestyle='dashed')
            ax_pos.set_title(labels[i])
            ax_pos.legend()
            ax_pos.grid()

            # Speed plot
            ax_speed = axs[i, 1]
            ax_speed.plot(time_axis[:min_length], simulated_states[:min_length, i + 4], label='Simulated')
            if min_length > 0:
                ax_speed.plot(time_axis[:min_length], real_states[:min_length, i + 4], label='Real', linestyle='dashed')
            ax_speed.set_title(labels[i + 4])
            ax_speed.legend()
            ax_speed.grid()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = TestStateFunction()
    rclpy.shutdown()

if __name__ == '__main__':
    main()