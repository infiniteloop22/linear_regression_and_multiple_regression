import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import statsmodels.api as sm

def load_data(file_path):
    data = []
    total_sales = []
    advert_budget = []
    airplay_times = []
    attractiveness_score = []
    with open(file_path, "r") as file:
        for line in file:
            data.append(line.strip().split(','))
    
    for row in data[1:]:
        total_sales.append(int(row[0]))
        advert_budget.append(int(row[1]))
        airplay_times.append(int(row[2]))
        attractiveness_score.append(int(row[3]))
    
    total_sales = np.array(total_sales)
    advert_budget = np.array(advert_budget)
    airplay_times = np.array(airplay_times)
    attractiveness_score = np.array(attractiveness_score)
    
    return total_sales, advert_budget, airplay_times, attractiveness_score

def visualize(sales, advertising, airplay, attractiveness):
    variables = {"Sales" : sales,
                 "Advertising" : advertising,
                 "Airplay" : airplay,
                 "Attractiveness" : attractiveness}
    
    pairs = [("Advertising", "Sales"), 
             ("Airplay", "Sales"), 
             ("Attractiveness", "Sales")]
    
    for var1, var2 in pairs:
        plt.scatter(variables[var1], variables[var2])
        plt.title(f"{var2} and {var1}")
        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.show()
    
def linear_regression(x, y):
    m = len(x)
    numerator = 0
    denominator = 0

    for i in range(m):
        numerator += (x[i] - np.mean(x)) * (y[i] - np.mean(y))
    for i in range(m):
        denominator += (x[i] - np.mean(x)) ** 2
        
    slope = numerator / denominator
    intercept = np.mean(y) - slope * np.mean(x)

    return slope, intercept
    
def predict(x, slope, intercept):
    return slope * x + intercept # prediction

def total_sum_squares(y):
    m = len(y)
    sst = 0

    for i in range(m):
        sst += (y[i] - np.mean(y)) ** 2
    
    return sst

def sum_squared_errors(y, prediction):
    m = len(y)
    sse = 0
    for i in range(m):
        sse += (y[i] - prediction[i]) ** 2

    return sse

def sum_squares_regression(y, prediction):
    m = len(y)
    ssr = 0

    for i in range(m):
        ssr += (prediction[i] - np.mean(y)) ** 2

    return ssr

def f_statistic(y, prediction, r):
    m = len(y)

    mean_squared_error = sum_squared_errors(y, prediction) / (m - r - 1)
    mean_squared_regression = sum_squares_regression(y, prediction) / r

    return mean_squared_regression / mean_squared_error # f-statistic

def r_squared(y, prediction):
    return sum_squares_regression(y, prediction) / total_sum_squares(y)

def main():
    sales, advertising, airplay, attractiveness = load_data("Lab Album Sales.csv")

    print("Visualization of variable relationships:\n")
    visualize(sales, advertising, airplay, attractiveness)

    print("Linear regression, f-statistic and p-value for sales and adverts:")
    slope, intercept = linear_regression(advertising, sales)
    print(f"Linear model: y = {slope} * advertising + {intercept}")

    p = stats.pearsonr(advertising, sales)[1]
    prediction = predict(advertising, slope, intercept)
    f_stat = f_statistic(sales, prediction, 1)
    print(f"F_Statistic for Sales and Advertising: {f_stat:.2f}")
    print(f"P-value for Sales and Advertising: {p}\n")

    print("Model coefficients:")
    print(f"Intercept value: {intercept:.2f}")
    print(f"Coefficient value: {slope:.4f}\n")

    print("Manual entry:")
    print(f"For $135,000 spent on advertising: {predict(135000, slope, intercept):.0f} records sold\n")

    print("Multiple regression:")
    X = np.column_stack((advertising, airplay, attractiveness))
    X = sm.add_constant(X)
    model = sm.OLS(sales, X).fit()

    print(f"F-Statistic: {model.fvalue:.2f}")
    print(f"P-value: {model.f_pvalue}\n")

    print("Comparison of models:")
    r_sq = model.rsquared
    print(f"r^2 value for model 1: {r_squared(sales, prediction):.3f}")
    print(f"r^2 value for model 2: {r_sq:.3f}") 

if __name__ == "__main__":
    main()