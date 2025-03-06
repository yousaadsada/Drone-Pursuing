import numpy as np
import csv
import os
from sklearn.linear_model import LinearRegression
import rclpy
from rclpy.node import Node

class CalculateStateFunction(Node):
    def __init__(self):
        super().__init__('calculate_state_function')

        
        self.A = np.eye(8)
        self.A[0,4] = 0.04
        self.A[1,5] = 0.04
        self.A[2,6] = 0.04
        self.A[3,7] = 0.04
        self.B = np.zeros(( 8, 4))
        
        data_dir = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data','real_drone_state_function','dataset')
        input_csv_files = [
            os.path.join(data_dir, 'state_data_1.csv'),
            os.path.join(data_dir, 'state_data_2.csv'),
            os.path.join(data_dir, 'state_data_3.csv'),
            os.path.join(data_dir, 'state_data_4.csv'),
        ]

        self.pth_dir = os.path.join(data_dir, 'pth')
        self.npy_dir = os.path.join(data_dir, 'npy')
        self.csv_dir = os.path.join(data_dir, 'state_function')

        os.makedirs(self.pth_dir, exist_ok=True)
        os.makedirs(self.npy_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)

        self.speeds_x, self.controls_x = self.load_and_extract_data(input_csv_files[0], index=1)
        self.speeds_y, self.controls_y = self.load_and_extract_data(input_csv_files[1], index=2)
        self.speeds_z, self.controls_z = self.load_and_extract_data(input_csv_files[2], index=3)
        self.speeds_yaw, self.controls_yaw = self.load_and_extract_data(input_csv_files[3], index=4)
        
        self.train_models()

    def load_and_extract_data(self, filename, index):
        with open(filename, 'r') as file:
            speeds = []
            controls = []
            reader = csv.reader(file)
            headers = next(reader)
            for row in reader:
                speeds.append(float(row[6 + index]))
                controls.append(float(row[12 + index]))

        return np.array(speeds).reshape(-1, 1), np.array(controls).reshape(-1, 1)
    
    def create_time_series(self, speeds, controls):
        X, y = [], []
        for i in range(len(speeds) - 1):
            X.append(np.hstack((speeds[i - 1], controls[i - 1])))
            y.append(speeds[i])
        return np.array(X), np.array(y)

    def train_model(self, X, y):
        model = LinearRegression(positive=True)  # Ensure non-negative coefficients
        model.fit(X, y)
        return model

    def train_models(self):
        
        X_x, y_x = self.create_time_series(self.speeds_x, self.controls_x)
        X_y, y_y = self.create_time_series(self.speeds_y, self.controls_y)
        X_z, y_z = self.create_time_series(self.speeds_z, self.controls_z)
        X_yaw, y_yaw = self.create_time_series(self.speeds_yaw, self.controls_yaw)

        print("Training model for x...")
        self.model_x = self.train_model(X_x, y_x)
        print("Training model for y...")
        self.model_y = self.train_model(X_y, y_y)
        print("Training model for z...")
        self.model_z = self.train_model(X_z, y_z)
        print("Training model for yaw...")
        self.model_yaw = self.train_model(X_yaw, y_yaw)

        # Save the trained models
        np.save(os.path.join(self.pth_dir, 'model_x.npy'), self.model_x.coef_)
        np.save(os.path.join(self.pth_dir, 'model_y.npy'), self.model_y.coef_)
        np.save(os.path.join(self.pth_dir, 'model_z.npy'), self.model_z.coef_)
        np.save(os.path.join(self.pth_dir, 'model_yaw.npy'), self.model_yaw.coef_)

        # Extract and save matrices
        self.save_matrices()

    def save_matrices(self):
        def save_matrix(model, prefix, idx):
            coef = model.coef_
            
            # Fill the calculated A values
            self.A[4 + idx, 4 + idx] = coef[0, 0]
            self.B[4 + idx, idx] = coef[0, 1]
            print(coef[0, 1])


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

        save_matrix(self.model_x, 'x', 0)
        save_matrix(self.model_y, 'y', 1)
        save_matrix(self.model_z, 'z', 2)
        save_matrix(self.model_yaw, 'yaw', 3)

def main(args=None):
    rclpy.init(args=args)
    node = CalculateStateFunction()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
