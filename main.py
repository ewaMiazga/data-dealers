from helpers import load_csv_data, create_csv_submission


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



if __name__ == "__main__":
    main()