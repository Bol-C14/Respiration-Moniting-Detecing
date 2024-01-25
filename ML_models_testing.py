import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report


class Data_Loader:
    def __init__(self, test_data_path):
        self.columns_of_interest = ['accel_x', 'accel_y', 'accel_z',
                                     'gyro_x', 'gyro_y', 'gyro_z',
                                     'file_id', 'class', 'activity_type', 'activity_subtype']
        self.columns_of_interest_training = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y',
                                             'gyro_z']
        self.columns_of_accel_only = ['accel_x', 'accel_y', 'accel_z']

        self.window_size = 50
        self.step_size = 25

        self.test_data_df = pd.read_csv(test_data_path)
        self.transform_df()

    def transform_df(self):
        self.test_data_df = self.test_data_df.dropna(subset=self.columns_of_interest_training).reset_index(drop=True)
        self.test_data_df[['activity_type', 'activity_subtype']] = self.test_data_df['class'].str.split('_',
                                                                                                        expand=True)

        scaler = RobustScaler()
        scaler = scaler.fit(self.test_data_df[self.columns_of_accel_only].values)
        self.test_data_df.loc[:, self.columns_of_accel_only] = scaler.transform(
            self.test_data_df[self.columns_of_accel_only].to_numpy())

    def Task1_test_df(self):
        Task1_test_df = self.test_data_df[self.test_data_df["activity_subtype"] == 'breathingNormal']
        Task1_acts = sorted(Task1_test_df.activity_type.unique())
        Task1_class_label = {act: idx for idx, act in enumerate(Task1_acts)}

        Task1_test_df['activity_label'] = Task1_test_df['activity_type'].map(Task1_class_label)
        Task1_X_test_df = Task1_test_df.reset_index(drop=True)

        X_test_sliding_windows = self.group_into_sliding_windows(Task1_X_test_df, self.window_size, self.step_size)
        X_test_generated, y_test_generated = self.generate_task1_data_from_sliding_windows(X_test_sliding_windows,
                                                                                           Task1_class_label)
        X_test, y_test = self.convert_values_to_numpy_array(X_test_generated, y_test_generated)

        return X_test, y_test

    def Task2_test_df(self):
        stationary_acts_resp_list = [
            'lyingBack_hyperventilating',
            'sitStand_hyperventilating',
            'lyingStomach_hyperventilating',
            'lyingLeft_breathingNormal',
            'sitStand_breathingNormal',
            'lyingLeft_coughing',
            'sitStand_coughing',
            'lyingStomach_coughing',
            'lyingRight_breathingNormal',
            'lyingLeft_hyperventilating',
            'lyingRight_coughing',
            'lyingRight_hyperventilating',
            'lyingStomach_breathingNormal',
            'lyingBack_coughing',
            'lyingBack_breathingNormal']

        Task2_test_df = self.test_data_df[self.test_data_df["class"].isin(stationary_acts_resp_list)]
        Task2_acts = sorted(stationary_acts_resp_list)
        Task2_class_labels = {act: idx for idx, act in enumerate(Task2_acts)}
        Task2_test_df['activity_label'] = Task2_test_df['class'].map(Task2_class_labels)

        columns_of_interest_task2 = ['accel_x', 'accel_y', 'accel_z',
                                     'gyro_x', 'gyro_y', 'gyro_z',
                                     'file_id', 'class', 'activity_type', 'activity_subtype']

        Task2_X_test_df = Task2_test_df[columns_of_interest_task2].reset_index(drop=True)

        X_test_sliding_windows = self.group_into_sliding_windows(Task2_X_test_df, self.window_size, self.step_size)
        X_test_generated, y_test_generated = self.generate_task23_data_from_sliding_windows(X_test_sliding_windows,
                                                                                           Task2_class_labels)
        X_test, y_test = self.convert_values_to_numpy_array(X_test_generated, y_test_generated)

        return X_test, y_test

    def Task3_test_df(self):
        task3_acts_list = ['lyingBack_other', 'lyingBack_hyperventilating', 'lyingLeft_other',
                           'lyingRight_other', 'sitStand_hyperventilating',
                           'sitStand_other',
                           'lyingStomach_hyperventilating', 'lyingLeft_breathingNormal',
                           'sitStand_breathingNormal', 'lyingStomach_other',
                           'lyingLeft_coughing', 'sitStand_coughing',
                           'lyingStomach_coughing',
                           'lyingRight_breathingNormal',
                           'lyingLeft_hyperventilating',
                           'lyingRight_coughing',
                           'lyingRight_hyperventilating',
                           'lyingStomach_breathingNormal', 'lyingBack_coughing',
                           'lyingBack_breathingNormal']

        Task3_test_df = self.test_data_df[self.test_data_df["class"].isin(task3_acts_list)]
        Task3_class_labels = {act: idx for idx, act in enumerate(sorted(task3_acts_list))}
        Task3_test_df['activity_label'] = Task3_test_df['class'].map(Task3_class_labels)
        Task3_X_test_df = Task3_test_df[self.columns_of_interest].reset_index(drop=True)

        X_test_sliding_windows = self.group_into_sliding_windows(Task3_X_test_df, self.window_size, self.step_size)
        X_test_generated, y_test_generated = self.generate_task23_data_from_sliding_windows(X_test_sliding_windows,
                                                                                           Task3_class_labels)
        X_test, y_test = self.convert_values_to_numpy_array(X_test_generated, y_test_generated)

        return X_test, y_test

    # Splitting data into sliding windows
    def group_into_sliding_windows(self, df, window_size, step_size):

        window_number = 0  # start a counter at 0 to keep track of the window number

        all_overlapping_windows = []

        for rid, group in df.groupby("file_id"):
            large_enough_windows = [window for window in group.rolling(window=window_size, min_periods=window_size) if
                                    len(window) == window_size]

            overlapping_windows = large_enough_windows[::step_size]
            if overlapping_windows:
                for window in overlapping_windows:
                    window.loc[:, 'window_id'] = window_number
                    window_number += 1

                all_overlapping_windows.append(pd.concat(overlapping_windows).reset_index(drop=True))

        final_sliding_windows = pd.concat(all_overlapping_windows).reset_index(drop=True)

        return final_sliding_windows

    def generate_task1_data_from_sliding_windows(self, final_sliding_windows, class_labels):
        X = []
        y = []
        for window_id, group in final_sliding_windows.groupby('window_id'):
            shape = group[self.columns_of_accel_only].values.shape

            X.append(group[self.columns_of_accel_only].values)
            y.append(class_labels[group["activity_type"].values[0]])

        return X, y

    def generate_task23_data_from_sliding_windows(self, final_sliding_windows, class_labels):
        X = []
        y = []
        for window_id, group in final_sliding_windows.groupby('window_id'):
            shape = group[self.columns_of_accel_only].values.shape

            X.append(group[self.columns_of_accel_only].values)
            y.append(class_labels[group["class"].values[0]])

        return X, y

    def convert_values_to_numpy_array(self, X_test_regenerated, y_test_regenerated):
        X_test = np.asarray(X_test_regenerated).astype('float32')
        y_test = np.asarray(pd.get_dummies(y_test_regenerated), dtype=np.float32)

        return X_test, y_test


def evaluate_model(model_path, test_data_path):
    # Load the model from the specified path
    model = load_model(model_path)
    # Load the test data from the CSV file
    data_loader = Data_Loader(test_data_path)

    if 'Task1' in model_path:
        X_test, y_test = data_loader.Task1_test_df()
    elif 'Task2' in model_path:
        X_test, y_test = data_loader.Task2_test_df()
    elif 'Task3' in model_path:
        X_test, y_test = data_loader.Task3_test_df()
    else:
        raise ValueError('No model found nor named properly')

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)
    print(classification_report(y_true_labels, y_pred_labels))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")



def main():
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description="Evaluate a TensorFlow model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the .h5 model file')
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to the test data CSV file')

    args = parser.parse_args()

    # Evaluate the model with the provided arguments
    evaluate_model(args.model_path, args.test_data_path)


if __name__ == "__main__":
    main()

