import numpy as np
from sklearn.model_selection import train_test_split


class DataProcessing:

    def DataProcessing(self):
        # File path containing the binary representations and labels
        file_path = "../file.txt"

        # Read data from the file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Process the data
        data = []
        current_sequence = ""
        current_label = None

        for line in lines:
            line = line.strip()
            if line:
                if len(line) == 1:
                    # If the line starts with a digit, treat it as a label
                    if current_label is not None:
                        data.append((current_sequence, current_label))
                    current_label = int(line)
                    current_sequence = ""
                else:
                    # If the line is not a label, append it to the current sequence
                    current_sequence += line

        # Add the last data point
        if current_label is not None and current_sequence:
            data.append((current_sequence, current_label))

        # Convert binary sequences to NumPy arrays
        X = []
        y = []

        for binary_sequence, label in data:
            # Assuming each line of the binary sequence represents a row in the matrix
            matrix = np.array([list(map(int, line)) for line in binary_sequence.split('\n') if line])
            flattened_image = matrix.flatten()
            X.append(flattened_image)
            y.append(label)

        # Convert lists to NumPy arrays
        X = np.array(X)
        y = np.array(y)

        # Check if there are enough samples for splitting
        if len(X) > 0:
            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        else:
            print("Not enough samples for splitting.")
