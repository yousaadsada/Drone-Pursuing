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

        data_dir = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data','real_drone_state_function','smoothed_data02','state_function')

        self.A = np.loadtxt(os.path.join(data_dir, 'A_matrix.csv'), delimiter=',')
        self.B = np.loadtxt(os.path.join(data_dir, 'B_matrix.csv'), delimiter=',')

        self.simulated_states = []
        self.real_states = {
            'x': [],
            'y': [],
            'z': [],
            'yaw': []
        }


        self.time_axis = []
        
        file_names = [
            ('state_data_1.csv', 'x'),
            ('state_data_2.csv', 'y'),
            ('state_data_3.csv', 'z'),
            ('state_data_4.csv', 'yaw'),
        ]

        for file_name, state_key in file_names:
            self.read_real_data(file_name, state_key)


        self.run_simulation()
        self.plot_states()

    def read_real_data(self,file_name, state_key):
        data_dir = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data','real_drone_state_function','raw_data02')
        csv_file = os.path.join(data_dir, file_name)

        with open(csv_file, mode='r') as file:
            reader = csv.DictReader(file)
            initial_position = None
            for row in reader:
                timestamp = float(row['elapsed_time'])
                if timestamp <= self.simulation_duration:
                    yaw = float(row['pitch'])
                    if yaw < 0 and initial_position is not None:
                       yaw += 2 * math.pi
                    
                    current_position = [
                        float(row['x']), float(row['y']), float(row['z']), yaw,
                        float(row['x_speed']), float(row['y_speed']), float(row['z_speed']), float(row['pitch_speed'])
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
            'x': [20, 0, 0, 0] * (self.freq ) + [0, 0, 0, 0] * (self.freq *10),
            'y': [0, 20, 0, 0] * (int(0.5 * self.freq) + 1) + [0, 0, 0, 0] * (self.freq * 10),
            'z': [0, 0, 20, 0] * (self.freq ) + [0, 0, 0, 0] * (self.freq ) + [0, 0, -20, 0] * (self.freq ) + [0, 0, 0, 0] * (self.freq ),
            'yaw': [0, 0, 0, 100] * (self.freq ) + [0, 0, 0, 0] * (self.freq ) + [0, 0, 0, -100] * (self.freq ) + [0, 0, 0, 0] * (self.freq )
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
        savedir = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data','drone_state_comparison')
        os.makedirs(savedir, exist_ok=True)
        labels = ['x', 'y', 'z', 'yaw', 'x_speed', 'y_speed', 'z_speed', 'yaw_speed']

        for i, key in enumerate(self.real_states.keys()):
            fig, axs = plt.subplots(2, 1, figsize=(10, 8))
            fig.suptitle(f'Comparison in {labels[i]} direction', fontsize=18, fontweight='bold')

            real_states = np.array(self.real_states[key])
            if i == 3:  # Adjust yaw values to be within [-pi, pi]
                real_states[real_states > np.pi] -= 2 * np.pi

            simulated_states = self.simulated_states[i]
            time_axis = self.time_axis[:len(real_states)]
            min_length = min(len(simulated_states), len(real_states), len(time_axis))

            # Position plot
            ax_pos = axs[0]
            ax_pos.plot(time_axis[:min_length], simulated_states[:min_length, i], label='Simulated')
            if min_length > 0:
                ax_pos.plot(time_axis[:min_length], real_states[:min_length, i], label='Real', linestyle='dashed')
            ax_pos.set_title('position', fontsize=16)
            ax_pos.set_xlabel("Time (s)", fontsize=12)
            ax_pos.set_ylabel("Distance (m)", fontsize=12)
            if i == 3:
                ax_pos.set_ylabel("Angle (rad)", fontsize=12)
            ax_pos.legend(loc='upper right', fontsize=12)
            ax_pos.grid()

            # Speed plot
            ax_speed = axs[1]
            ax_speed.plot(time_axis[:min_length], simulated_states[:min_length, i + 4], label='Simulated')
            if min_length > 0:
                ax_speed.plot(time_axis[:min_length], real_states[:min_length, i + 4], label='Real', linestyle='dashed')
            ax_speed.set_title('speed', fontsize=16)
            ax_speed.set_xlabel("Time (s)", fontsize=12)
            ax_speed.set_ylabel("Speed (m/s)", fontsize=12)
            if i == 3:
                ax_speed.set_ylabel("Angular Speed (rad/s)", fontsize=12)
            ax_speed.legend(loc='upper right', fontsize=12)
            ax_speed.grid()

            plt.subplots_adjust(hspace=0.4)

            save_path = os.path.join(savedir, f"comparison_{labels[i]}.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

            plt.tight_layout(rect=[0, 0, 1, 1.0])
            plt.show()


def main(args=None):
    rclpy.init(args=args)
    node = TestStateFunction()
    rclpy.shutdown()

if __name__ == '__main__':
    main()