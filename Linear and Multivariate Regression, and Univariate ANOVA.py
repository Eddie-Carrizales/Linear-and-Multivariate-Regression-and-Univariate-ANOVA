import numpy as np
import scipy.stats as stats
import pandas as pd
from tabulate import tabulate
from itertools import product
import math

# Name: Edilberto Carrizales
# Language: Python
# Language version: 3.9
def problem1():
    print("\n--------------- Problem 1. Linear Regression and Univariate ANOVA ---------------")
    print("Given:")
    predictors = [21, 26, 32, 37, 42, 47, 52, 57, 63, 69, 75]
    observations = [112, 86, 71, 195, 175, 102, 270, 292, 197, 146, 309]
    print("Predictors: " + str(predictors))
    print("Observations: " + str(observations))
    print("Total Predictors: " + str(len(predictors)))
    print("Total Observations: " + str(len(observations)))

    mean_of_predictors = np.mean(predictors)
    mean_of_observations = np.mean(observations)

    # The sample standard deviation of the Predictors Sx
    sum = 0
    for i in range(len(predictors)):
        sum += (predictors[i] - mean_of_predictors) ** 2
    std_of_predictors_sx = math.sqrt(sum / len(predictors) - 1)

    # The sample standard deviation of the Observations Sy
    # sum = 0
    # for i in range(len(observations)):
    #     sum += (observations[i] - mean_of_observations) ** 2
    # std_of_observations_sy = math.sqrt(sum / len(observations) - 1)
    std_of_observations_sy = np.std(observations, ddof= 1)

    print("\nAbout the input: ")
    print("The mean of the Predictors Xbar: " + str(mean_of_predictors))
    print("The mean of the Observations Ybar: " + str(mean_of_observations))
    print("The sample standard deviation of the Predictors Sx: " + str(std_of_predictors_sx))
    print("The sample standard deviation of the Observations Sy: " + str(std_of_observations_sy))

    # The sum of the squares of the Predictors Sxx:
    sum = 0
    for i in range(len(predictors)):
        sum += (predictors[i] - mean_of_predictors) ** 2
    sum_of_squares_sxx = sum

    # The sum of the products of the predictors and Observations Sxy:
    sum = 0
    for i in range(len(predictors)):
        sum += (predictors[i] - mean_of_predictors) * (observations[i] - mean_of_observations)
    sum_of_products_sxy = sum

    # The estimation of the slope b1:
    estimation_of_slope_b1 = sum_of_products_sxy / sum_of_squares_sxx

    # The estimation of the intercept b0
    estimation_of_intercept_b0 = mean_of_observations - (estimation_of_slope_b1 * mean_of_predictors)

    print("\nSolving for the Coefficients: ")
    print("The sum of the squares of the Predictors Sxx: " + str(sum_of_squares_sxx))
    print("The sum of the products of the Predictors and Observations Sxy:" + str(sum_of_products_sxy))
    print("The estimation of the slope b1: " + str(estimation_of_slope_b1))
    print("The estimation of the intercept b0: " + str(estimation_of_intercept_b0))

    # Find the estimators Y_hat
    estimators_y_hat = []
    for i in range(len(predictors)):
        y_hat = (estimation_of_slope_b1 * predictors[i]) + estimation_of_intercept_b0
        estimators_y_hat.append(y_hat)

    # The sum of squares of the Model SS_REG:
    sum = 0
    for i in range(len(observations)):
        sum += (estimators_y_hat[i] - mean_of_observations) ** 2
    sum_of_squares_reg = sum

    # The sum of squares of the Error SS_ERR:
    sum = 0
    for i in range(len(observations)):
        e_i = estimators_y_hat[i] - observations[i]
        sum += e_i ** 2
    sum_of_squares_err = sum

    # The sum of squares of the Total SS_TOT:
    sum_of_squares_total = sum_of_squares_reg + sum_of_squares_err

    # Check that the total is equal to the model plus the error:
    total_sum_of_squares_using_formula = (len(observations) - 1) * std_of_observations_sy * std_of_observations_sy

    print("\nSolving for the Squares: ")
    print("The sum of squares of the Model SS_REG: " + str(sum_of_squares_reg))
    print("The sum of squares of the Error SS_ERR: " + str(sum_of_squares_err))
    print("The sum of squares of the Total SS_TOT: " + str(sum_of_squares_total))
    print("\nCheck that the total is equal to the model plus the error: ")
    print("SS_TOT calculation using formula: ")
    print("total_sum_of_squares_using_formula = (len(observations) - 1) * std_of_observations_sy * std_of_observations_sy")
    print("total_sum_of_squares_using_formula: " + str(total_sum_of_squares_using_formula))
    print("SS_TOT calculation by adding model plus error (SS_TOT = SS_REG + SS_ERR): " + str(sum_of_squares_total))

    # F Test value:
    ms_reg = sum_of_squares_reg
    ms_err = sum_of_squares_err / (len(predictors) - 2)
    f_value = ms_reg / ms_err
    # f_value2 = ss_reg / ss_err

    # The R^2 as a percentage:
    r_squared = sum_of_squares_reg / sum_of_squares_total
    r_squared = round(r_squared, 4)

    print("\nShowing the goodness to fit values: ")
    print("The F value: " + str(f_value))
    print("The R^2 as a percentage: " + str(r_squared * 100) + "%")

def problem2():
    print("\n----------------------- Problem 2. Multivariate Regression -----------------------")
    print("Given:")
    predictor1_days = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    predictor2_temps = [40, 39, 30, 29, 28, 35, 35, 30, 25, 20, 20, 20, 21, 30, 31]
    observations_icicle_lengths = [0, 0, 1, 2, 3, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    print("Predictor 1 Days of the Month: " + str(predictor1_days))
    print("Predictor 2 Air Temperature: " + str(predictor2_temps))
    print("Observations Length of Icicle: " + str(observations_icicle_lengths))
    print("Total Predictors 1 Days: " + str(len(predictor1_days)))
    print("Total Predictors 2 Temps: " + str(len(predictor2_temps)))
    print("Total Observations Icicle Lengths: " + str(len(observations_icicle_lengths)))

    mean_of_predictor1 = np.mean(predictor1_days)
    mean_of_predictor2 = np.mean(predictor2_temps)
    mean_of_observations = np.mean(observations_icicle_lengths)

    # The sample standard deviation of the Predictor1 Sx:
    sum = 0
    for i in range(len(predictor1_days)):
        sum += (predictor1_days[i] - mean_of_predictor1) ** 2
    std_of_predictor1_sx = math.sqrt(sum / len(predictor1_days) - 1)

    # The sample standard deviation of the Predictor2 Sx:
    sum = 0
    for i in range(len(predictor2_temps)):
        sum += (predictor2_temps[i] - mean_of_predictor2) ** 2
    std_of_predictor2_sx = math.sqrt(sum / len(predictor2_temps) - 1)

    # The sample standard deviation of the Observations Sy
    # sum = 0
    # for i in range(len(observations_icicle_lengths)):
    #     sum += (observations_icicle_lengths[i] - mean_of_observations) ** 2
    # std_of_observations_sy = math.sqrt(sum / len(observations_icicle_lengths) - 1)
    std_of_observations_sy = np.std(observations_icicle_lengths, ddof=1)

    print("\nAbout the input: ")
    print("The mean of the Predictors 1 Xbar1: " + str(mean_of_predictor1))
    print("The mean of the Predictors 2 Xbar2: " + str(mean_of_predictor2))
    print("The mean of the Observations Ybar: " + str(mean_of_observations))
    print("The sample standard deviation of the Predictors 1 Sx: " + str(std_of_predictor1_sx))
    print("The sample standard deviation of the Predictors 2 Sx: " + str(std_of_predictor2_sx))
    print("The sample standard deviation of the Observations Sy: " + str(std_of_observations_sy))

    print("\nFind the formula:")
    # Create Matrices
    predictors_matrix = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], predictor1_days, predictor2_temps])
    observations_matrix = np.matrix(observations_icicle_lengths)
    #predictors_matrix = np.array([[1, 1, 1, 1, 1, 1, 1], [6, 7, 7, 8, 10, 10, 15], [4, 20, 20, 10, 10, 2, 1]])
    # observations_matrix = np.matrix([40, 55, 50, 41, 17, 26, 16])

    predictors_transpose = np.transpose(predictors_matrix)
    print("\nX:")
    print(str(predictors_transpose))

    predictors_transpose_predictors = np.dot(predictors_matrix, predictors_transpose)
    print("\nXT_X:")
    print(str(predictors_transpose_predictors))

    predictors_transpose_observations = np.dot(observations_matrix, predictors_transpose)
    print("\nXT_Y:")
    print(str(np.transpose(predictors_transpose_observations)))

    b_values = predictors_transpose_observations.dot(np.linalg.inv(predictors_transpose_predictors))
    print("\nb Values:")
    print(str(b_values))

    print("\nFormula:")
    print("𝑖𝑐𝑖𝑐𝑙𝑒 𝑙𝑒𝑛𝑔𝑡ℎ = (𝑏2)(𝑡𝑒𝑚𝑝𝑒𝑟𝑎𝑡𝑢𝑟𝑒) + (𝑏1)(𝑑𝑎𝑦) + (𝑏0)(1)")
    print("𝑖𝑐𝑖𝑐𝑙𝑒 𝑙𝑒𝑛𝑔𝑡ℎ = " + str(b_values[0, 2]) + "(𝑡𝑒𝑚𝑝𝑒𝑟𝑎𝑡𝑢𝑟𝑒) + " + str(b_values[0, 1]) + "(𝑑𝑎𝑦) + " + str(b_values[0, 0]))

    # Find the estimators Y_hat
    estimators_y_hat = predictors_transpose.dot(np.transpose(b_values))
    print("\nY Hat Estimators:")
    print(estimators_y_hat)

    # The sum of squares of the Model SS_REG:
    sum = 0
    for i in range(len(estimators_y_hat)):
        sum += (estimators_y_hat[i, 0] - mean_of_observations) ** 2
    sum_of_squares_reg = sum

    # The sum of squares of the Error SS_ERR:
    sum = 0
    for i in range(len(estimators_y_hat)):
        e_i = estimators_y_hat[i, 0] - observations_icicle_lengths[i]
        sum += e_i ** 2
    sum_of_squares_err = sum

    # The sum of squares of the Total SS_TOT:
    sum_of_squares_total = sum_of_squares_reg + sum_of_squares_err

    # Check that the total is equal to the model plus the error:
    total_sum_of_squares_using_formula = (len(observations_icicle_lengths) - 1) * (std_of_observations_sy**2)

    print("\nThe values of the Sum of Squares: ")
    print("The sum of squares of the Model SS_REG: " + str(sum_of_squares_reg))
    print("The sum of squares of the Error SS_ERR: " + str(sum_of_squares_err))
    print("The sum of squares of the Total SS_TOT: " + str(sum_of_squares_total))
    print("\nCheck that the total is equal to the model plus the error: ")
    print("SS_TOT calculation using formula: " + str(total_sum_of_squares_using_formula))
    print("SS_TOT calculation by adding model plus error (SS_TOT = SS_REG + SS_ERR): " + str(sum_of_squares_total))

    # F Test value:
    ms_reg = sum_of_squares_reg / 2
    ms_err = sum_of_squares_err / (len(predictor1_days) - 2 - 1)
    f_value = ms_reg / ms_err

    # The R^2 as a percentage:
    r_squared = sum_of_squares_reg / sum_of_squares_total
    r_squared = round(r_squared, 4)

    print("\nThe values of the Mean Squares: ")
    print("The mean squares of the model MS_REG: " + str(ms_reg))
    print("The mean squares of the model MS_ERR: " + str(ms_err))

    print("\nThe values for goodness of fit: ")
    print("The F value: " + str(f_value))
    print("The R^2 as a percentage: " + str(r_squared * 100) + "%")

def main():
    problem1()
    problem2()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

