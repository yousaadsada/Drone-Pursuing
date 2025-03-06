import numpy as np
import csv
import os
from sklearn.linear_model import LinearRegression
import rclpy
from rclpy.node import Node
from sklearn.linear_model import Ridge
from scipy.optimize import minimize

class CalculateStateFunction(Node):
    def __init__(self):
        super().__init__('calculate_state_function')

        
        self.A = np.eye(8)
        self.A[0,4] = 0.04
        self.A[1,5] = 0.04
        self.A[2,6] = 0.04
        self.A[3,7] = 0.04
        self.B = np.zeros(( 8, 4))
        
        data_dir = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data','real_drone_state_function','smoothed_data02')
        input_csv_files = [
            os.path.join(data_dir, 'smoothed_x_state.csv'),
            os.path.join(data_dir, 'smoothed_y_state.csv'),
            os.path.join(data_dir, 'smoothed_z_state.csv'),
            os.path.join(data_dir, 'smoothed_yaw_state.csv'),
        ]

        self.pth_dir = os.path.join(data_dir, 'pth')
        self.npy_dir = os.path.join(data_dir, 'npy')
        self.csv_dir = os.path.join(data_dir, 'state_function')

        os.makedirs(self.pth_dir, exist_ok=True)
        os.makedirs(self.npy_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)

        self.positions_x, self.speeds_x, self.controls_x = self.load_and_extract_data(input_csv_files[0])
        self.positions_y, self.speeds_y, self.controls_y = self.load_and_extract_data(input_csv_files[1])
        self.positions_z, self.speeds_z, self.controls_z = self.load_and_extract_data(input_csv_files[2])
        self.positions_yaw, self.speeds_yaw, self.controls_yaw = self.load_and_extract_data(input_csv_files[3])
        
        self.train_models()

    def load_and_extract_data(self, filename):
        with open(filename, 'r') as file:
            positions = []
            speeds = []
            controls = []
            reader = csv.reader(file)
            headers = next(reader)
            for row in reader:
                positions.append(float(row[1]))
                speeds.append(float(row[2]))
                controls.append(float(row[3]))
            print(speeds)

        return np.array(positions).reshape(-1,1), np.array(speeds).reshape(-1, 1), np.array(controls).reshape(-1, 1)
    


    def train_models(self):

        def error_function(params, x_real, v_k, u_real, delta_t):
            A, B = params
            error = 0.0
            x_estimated = x_real[0]  # Start with the initial x
            v_k_current = np.copy(v_k)  # Create a copy to avoid modifying the original array

            for k in range(len(v_k_current) - 1):
                # Estimate the next position
                v_k_plus_1 = A * v_k_current[k] + B * u_real[k]
                x_estimated = x_estimated + delta_t * v_k_current[k]
                
                # Compute the error
                error += (x_real[k + 1] - x_estimated) ** 2
                
                # Update velocity for the next iteration
                v_k_current[k + 1] = v_k_plus_1

            return error

        delta_t = 0.04

        print("Training model for x...")
        initial_guess = np.array([1.0, 0.0])
        self.result_x = minimize(error_function, initial_guess, args=(self.positions_x, self.speeds_x, self.controls_x, delta_t))
        print("Training model for y...")
        self.result_y = minimize(error_function, initial_guess, args=(self.positions_y, self.speeds_y, self.controls_y, delta_t))
        print("Training model for z...")
        self.result_z = minimize(error_function, initial_guess, args=(self.positions_z, self.speeds_z, self.controls_z, delta_t))
        print("Training model for yaw...")
        self.result_yaw = minimize(error_function, initial_guess, args=(self.positions_yaw, self.speeds_yaw, self.controls_yaw, delta_t))


        # Extract and save matrices
        self.save_matrices()

    def save_matrices(self):
        def save_matrix(result, prefix, idx):
            A_opt, B_opt = result.x
            
            # Fill the calculated A values
            self.A[4 + idx, 4 + idx] = A_opt
            self.B[4 + idx, idx] = B_opt


            np.save(os.path.join(self.npy_dir, f'A_{prefix}.npy'), self.A)
            np.save(os.path.join(self.npy_dir, f'B_{prefix}.npy'), self.B)
            
            # Save A and B to separate CSV files
            if idx == 3:
                np.savetxt(os.path.join(self.csv_dir, 'A_matrix.csv'), self.A, delimiter=',', fmt='%.5f')
                # with open(os.path.join(self.csv_dir, f'A_matrix.csv'), 'w', newline='') as file:
                #     writer = csv.writer(file)
                #     writer.writerows(self.A)

                np.savetxt(os.path.join(self.csv_dir, f'B_matrix.csv'), self.B, delimiter=',', fmt='%.5f')    
                # with open(os.path.join(self.csv_dir, f'B_matrix.csv'), 'w', newline='') as file:
                #     writer = csv.writer(file)
                #     writer.writerows(self.B)

        save_matrix(self.result_x, 'x', 0)
        save_matrix(self.result_y, 'y', 1)
        save_matrix(self.result_z, 'z', 2)
        save_matrix(self.result_yaw, 'yaw', 3)

def main(args=None):
    rclpy.init(args=args)
    node = CalculateStateFunction()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
