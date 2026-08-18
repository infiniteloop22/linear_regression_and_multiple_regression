# Linear and Multiple Regression Analysis

A Python implementation of single-variable and multiple linear regression models to analyze product performance.

## Features

- **Model:** Single-variable linear regression ($y = mx + c$) and Multiple Linear Regression from statsmodels library.
- **Error Measure:** Total Sum of Squares ($SST$), Sum of Squared Errors ($SSE$), and Sum of Squares Regression ($SSR$).

## Technologies Used

- **Language:** Python 3.8+
- **Array Modeling:** `numpy` (matrix operations and array stacking)
- **Statistical Modeling:** `scipy` (Pearson correlation and $p$-value computation) and `statsmodels` (multiple regression fitting)
- **Data Visualization:** `matplotlib` (scatter plot generation)

## Dataset

This exercise uses the **Lab Album Sales Dataset** from the `Lab Album Sales.csv` file. 

## Technical Notes

*   **Slope ($m$):** Measures how much sales are expected to change for every single dollar spent on advertising. It is calculated by looking at how advertising and sales move together compared to how much the advertising budget varies on its own.
*   **Intercept ($c$):** Predicts the baseline album sales if the advertising budget were set to zero. It is found by taking the average sales amount and subtracting the pattern created by the slope and average advertising spend.
*   **R-Squared ($R^2$):** Measures the percentage of the total variation in album sales that can be explained by the marketing variables. It calculates how much better our regression line predicts sales compared to just guessing the overall average sales price for every album.
*   **F-Statistic:** Measures if our predictive model is actually useful or just finding random patterns. It divides the amount of variation the model successfully explains by the amount of variation or error the model misses, while adjusting for the number of data points and predictors used.