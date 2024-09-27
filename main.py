from helpers import load_csv_data, create_csv_submission
import numpy as np
import datetime

from plots import gradient_descent_visualization
from implementations import mean_squared_error_gd, mean_squared_error_sgd
from implementations import standardize


def main():
    """
    Main function to execute the workflow.
    """
    # Step 1: Load the dataset
    dir_path = './dataset_to_release/'
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(dir_path)

    print("x_train shape: ", x_train.shape)
    print("x_test shape: ", x_test.shape)
    print("y_train shape: ", y_train.shape)
    print("train_ids shape: ", train_ids.shape)
    print("test_ids shape: ", test_ids.shape)

    # Step 2: Standardize the data

    x_train = standardize(x_train)
    x_test = standardize(x_test)

    # Define the parameters of the algorithm.
    max_iters = 50
    gamma = 0.1

    # Initialization
    w_initial = np.ones(x_train.shape[1])

    # Start gradient descent.
    start_time = datetime.datetime.now()
    w, loss = mean_squared_error_gd(y_train, x_train, w_initial, max_iters, gamma)
    end_time = datetime.datetime.now()

    # Print result
    print("Gradient Descent: loss={l}, w={w}".format(l=loss, w=w))
    exection_time = (end_time - start_time).total_seconds()
    print("GD: execution time={t:.3f} seconds".format(t=exection_time))

    start_time = datetime.datetime.now()
    w_stoch, loss_stoch = mean_squared_error_sgd(y_train, x_train, w_initial, max_iters, gamma)
    end_time = datetime.datetime.now()

    # Print result
    print("Stochastic Gradient Descent: loss={l}, w={w}".format(l=loss_stoch, w=w_stoch))
    exection_time = (end_time - start_time).total_seconds()
    print("GD: execution time={t:.3f} seconds".format(t=exection_time))



if __name__ == "__main__":
    main()