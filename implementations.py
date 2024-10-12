import numpy as np 
from collections import Counter

################### DROP FEATURES FUNCTION ###################
def drop_features(headers, data, constant_threshold=0.9, missing_threshold=0.9, correlation_threshold=0.95): 
    # Step 1: Identify columns to keep based on missing values and low variability
    columns_to_keep = []

    for i in range(data.shape[1]):
        # Calculate the percentage of missing values
        nan_ratio = np.isnan(data[:, i]).sum() / data.shape[0]

        # If more than `missing_threshold` of the column is NaN, consider it mostly missing and drop it
        if nan_ratio > missing_threshold:
            continue

        # Remove NaN values
        non_nan_values = data[:, i][~np.isnan(data[:, i])]
        unique_values = np.unique(non_nan_values)

        # Check if column has more than one unique value
        if len(unique_values) > 1:
            # Calculate the frequency of the most common value
            value_counts = Counter(non_nan_values)
            most_common_value, most_common_count = value_counts.most_common(1)[0]
            ratio = most_common_count / len(non_nan_values)

            # Keep column if it is not 90% constant
            if ratio < constant_threshold:
                columns_to_keep.append(i)

    # Step 2: Filter data and headers based on the identified columns to keep
    filtered_data = data[:, columns_to_keep]
    filtered_headers = [headers[i] for i in columns_to_keep]

    # Step 3: Drop highly correlated features (more than 95% correlation)
    corr_matrix = np.corrcoef(filtered_data, rowvar=False)
    corr_matrix = np.abs(corr_matrix)  # Take the absolute value of correlations

    # Create a boolean mask to identify highly correlated columns in the upper triangle
    num_features = corr_matrix.shape[0]
    correlated_features = set()

    for i in range(num_features):
        for j in range(i + 1, num_features):
            if corr_matrix[i, j] > correlation_threshold:
                correlated_features.add(j)

    # Get the indices of the columns to keep
    final_columns_to_keep = [i for i in range(num_features) if i not in correlated_features]

    # Filter data and headers based on the correlation threshold
    final_data = filtered_data[:, final_columns_to_keep]
    final_headers = [filtered_headers[i] for i in final_columns_to_keep]

    # Print number of dropped columns
    print(f"Dropped {data.shape[1] - len(final_headers)} columns that were either mostly missing, had low variability, or were highly correlated (>95%).")
    
    return final_headers, final_data, final_columns_to_keep

################### STANDARDIZE FUNCTION ###################
def standardize(x):
    """Stadartize the input data x

    Args:
        x: numpy array of shape=(num_samples - N, num_features - D)

    Returns:
        standartized data, shape=(num_samples - N, num_features - D)
    """
    # ***************************************************
    std_data = (x - x.mean(axis=0)) / x.std(axis=0)
    return std_data


##################### FIRST TWO FUNCTIONS #####################

def MSE(y, tx, w):
    """Compute the mean square error."""
    e = y - tx @ w
    return e.T @ e / (2 * len(y))


def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    return MSE(y, tx, w)
    
def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D, ). The vector of model parameters.

    Returns:
        An numpy array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = y - tx @ w
    return -tx.T @ e / len(y)


# FOR GRADING 
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm.
    
    Args: 
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess for the model parameters.
        
    Returns:
        An numpy array of shape (2, ) containing, w and loss for the last iteration.
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        w = w - gamma * gradient
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}".format(bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0]))
    return ws[-1], losses[-1]


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from a data sample batch of size B, where B < N, and their corresponding labels.

    Args:
        y: numpy array of shape=(B, )
        tx: numpy array of shape=(B,D)
        w: numpy array of shape=(D, ). The vector of model parameters.

    Returns:
        A numpy array of shape (D, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    return compute_gradient(y, tx, w)

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        num_batches: a scalar denoting the number of batches to generate
        shuffle: a boolean indicating whether to shuffle the data before creating batches

    Yields:
        A tuple of two numpy arrays of shape (batch_size, ) and (batch_size, D), respectively containing the labels and the input data of the batch.

    """
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]



# FOR GRADING 
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        An numpy array of shape (2, ) containing, w and loss for the last iteration.
    """

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        
        # this for loop actually generates only one batch consisting of one element inside it
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            grad = compute_stoch_gradient(y_batch, tx_batch, w)
            loss = compute_loss(y_batch, tx_batch, w)
            w = w - gamma * grad
            ws.append(w)
            losses.append(loss)
        #raise NotImplementedError

        print(
            "SGD iter. {bi}/{ti}: loss={l}, w0={w0}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0]
            )
        )
    return w, loss

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    """

    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    mse = compute_loss(y, tx, w)

    return w, mse 


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    """
    
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))
    return loss

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)/len(y)
    return grad

def learning_by_gradient_descent(y, tx, w, gamma):
    """Do one step of gradient descent using logistic regression. Return the loss and the updated w."""
    # Compute the gradient
    gradient = calculate_gradient(y, tx, w)
    
    # Update the weights
    w = w - gamma * gradient
    
    # Compute the loss with the updated weights
    loss = calculate_loss(y, tx, w)
    
    return loss, w


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Perform logistic regression using gradient descent."""
    w = initial_w
    for iter in range(max_iters):
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        if iter % 100 == 0: 
            print(f"Iteration {iter}/{max_iters}, loss={loss}")
    
    return w, loss
    
def penalized_logistic_regression(y, tx, w, lambda_):
    """
    Compute the loss and gradient for penalized logistic regression.
    Includes an L2 regularization term.
    """
    # Compute the loss with the regularization term
    loss = calculate_loss(y, tx, w) + (lambda_ / 2) * np.sum(w**2)
    
    # Compute the gradient with the regularization term
    gradient = calculate_gradient(y, tx, w) + lambda_ * w
    
    return loss, gradient

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Perform one step of gradient descent using penalized logistic regression.
    Return the loss and updated w.
    """
    # Compute the loss and gradient
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    
    # Update the weights
    w -= gamma * gradient
    
    return loss, w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Perform regularized logistic regression using gradient descent.
    Args:
        y: shape=(N, 1)
        tx: shape=(N, D)
        lambda_: scalar (regularization parameter)
        initial_w: shape=(D, 1) (initial weight vector)
        max_iters: int (number of iterations)
        gamma: scalar (learning rate)

    Returns:
        w: Final weights after training
        loss: Final loss value
    """
    w = initial_w
    for iter in range(max_iters):
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        if iter % 100 == 0:
            print(f"Iteration {iter}/{max_iters}, loss={loss}")

    return w, loss