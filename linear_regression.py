""" Stochastic Gradient Descent """
def SGD(f, theta0, alpha, num_iters):
    """
       Arguments:
       f -- the function to optimize, it takes a single argument
            and yield two outputs, a cost and the gradient
            with respect to the arguments
       theta0 -- the initial point to start SGD from
       num_iters -- total iterations to run SGD for
       Return:
       theta -- the parameter value after SGD finishes
    """
    start_iter = 0
    theta= theta0
    for iter in xrange(start_iter + 1, num_iters + 1):
        _, grad = f(theta)
        theta = theta - (alpha * grad) # there is NO dot product!
    return theta

""" gradientDescent """
def gradientDescent(X, y, theta, alpha, num_iters):
    """
       Performs gradient descent to learn theta
    """
    m = y.size  # number of training examples
    for i in range(num_iters):
        y_hat = np.dot(X, theta)
        theta = theta - alpha * (1.0/m) * np.dot(X.T, y_hat-y)
    return theta

"""
Mini-Batch Gradient Descent

It is similar like SGD, it uses n samples instead of 1 at each iteration.
Summary:

In a summary, explained about the following topics in detail.

Linear regression’s independent and dependent variables
Ordinary Least Squares (OLS) method and Sum of Squared Errors (SSE) details
Gradient descent for linear regression model and types gradient descent algorithms.

https://towardsdatascience.com/linear-regression-simplified-ordinary-least-square-vs-gradient-descent-48145de2cf76
"""
