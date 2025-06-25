"""
Car Price Prediction Analysis
----------------------------
This script performs regression analysis on automobile data to predict car prices using:
1. Simple Linear Regression
2. Multiple Linear Regression
3. Polynomial Regression
4. Pipeline with Feature Scaling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def download (url , filename) :
    """Download file from URL and save locally."""
    response = requests.get(url)
    if response.status_code == 200 :
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"File '{filename}' downloaded successfully.")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

Data_URL= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv"
DOWNLOADED_FILE_NAME = "usedcars.csv"
DATA_FILE_TO_READ = DOWNLOADED_FILE_NAME 

try:
    download (Data_URL, DOWNLOADED_FILE_NAME)
    df = pd.read_csv(DATA_FILE_TO_READ)
    print ("\nData loaded successfully.")
    print("Data Overview:")
    print(df.head())
    print("\nData Types:")
    print(df.dtypes)
except FileNotFoundError:
    print(f"Error: The file '{DATA_FILE_TO_READ}' was not found. Please check the download step or file path.")
    exit()
except Exception as e :
    print(f"Error loading data: {e}")
    exit()


# --- Simple Linear Regression ---
# Highway MPG vs Price
print("\n--- Simple Linear Regression: Highway MPG vs Price ---")
lm_highway = LinearRegression()
X_highway = df[['highway-mpg']]
Y_price = df['price']

# Drop rows with NaN in 'price' or 'highway-mpg' for this specific model
# This is important if these columns have missing values that were not handled before
temp_df_highway = df[['highway-mpg', 'price']].dropna()
X_highway_clean = temp_df_highway[['highway-mpg']]
Y_price_clean_highway = temp_df_highway['price']

if not X_highway_clean.empty:
    lm_highway.fit(X_highway_clean, Y_price_clean_highway)
    Yhat_highway = lm_highway.predict(X_highway_clean)
    print("First 5 predictions (Highway MPG):")
    print(Yhat_highway[0:5])
    print("Intercept (Highway MPG):", lm_highway.intercept_)
    print("Coefficient (Highway MPG):", lm_highway.coef_)
else:
    print("Not enough data to fit Highway MPG model after dropping NaNs.")


# Engine Size vs Price
print("\n--- Simple Linear Regression: Engine Size vs Price ---")
lm_engine = LinearRegression()
X_engine = df[["engine-size"]]
# Y_price is already defined

temp_df_engine = df[['engine-size', 'price']].dropna()
X_engine_clean = temp_df_engine[['engine-size']]
Y_price_clean_engine = temp_df_engine['price']

if not X_engine_clean.empty:
    lm_engine.fit(X_engine_clean, Y_price_clean_engine)
    Yhat_engine = lm_engine.predict(X_engine_clean)
    print("First 5 predictions (Engine Size):")
    print(Yhat_engine[0:5])
    print("Intercept (Engine Size):", lm_engine.intercept_)
    print("Coefficient (Engine Size):", lm_engine.coef_)

    # Predicted prices using equation
    Price_engine_eq = lm_engine.intercept_ + lm_engine.coef_[0] * X_engine_clean['engine-size']
    print("\nPredicted prices using equation (Engine Size - first 5):")
    print(Price_engine_eq.head())
else:
    print("Not enough data to fit Engine Size model after dropping NaNs.")


# --- Multiple Linear Regression ---
print("\n--- Multiple Linear Regression ---")
# Features for Model 1
features_model1 = ['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']
df_multi1 = df[features_model1 + ['price']].dropna() # Drop NaNs for this specific model

if not df_multi1.empty and len(df_multi1) > len(features_model1): # Check if enough data
    multi_var_model1 = LinearRegression()
    Z1 = df_multi1[features_model1]
    Y_multi1 = df_multi1['price']
    multi_var_model1.fit(Z1, Y_multi1)
    # Yhat_multi1 = multi_var_model1.predict(Z1) # Prediction on training data
    print("\nMulti-variable Model 1 (horsepower, curb-weight, engine-size, highway-mpg):")
    print("Intercept:", multi_var_model1.intercept_)
    print("Coefficients:", multi_var_model1.coef_)

    # Calculate predicted prices using equation for multiple regression
    Price_multi1_eq = (multi_var_model1.intercept_ +
                       multi_var_model1.coef_[0] * Z1['horsepower'] +
                       multi_var_model1.coef_[1] * Z1['curb-weight'] +
                       multi_var_model1.coef_[2] * Z1['engine-size'] +
                       multi_var_model1.coef_[3] * Z1['highway-mpg'])
    print("\nPredicted prices using equation (Multi-variable Model 1 - first 5):")
    print(Price_multi1_eq.head())
else:
    print("\nNot enough data for Multi-variable Model 1 after dropping NaNs, or too few samples.")


# Features for Model 2
features_model2 = ["normalized-losses", "highway-mpg"]
df_multi2 = df[features_model2 + ['price']].dropna()

if not df_multi2.empty and len(df_multi2) > len(features_model2):
    multi_var_model2 = LinearRegression()
    A2 = df_multi2[features_model2]
    Y_multi2 = df_multi2['price']
    multi_var_model2.fit(A2, Y_multi2)
    # Yhat_multi2 = multi_var_model2.predict(A2)
    print("\nMulti-variable Model 2 (normalized-losses, highway-mpg):")
    print("Intercept:", multi_var_model2.intercept_)
    print("Coefficients:", multi_var_model2.coef_)

    # Calculate predicted prices using equation
    Price_multi2_eq = (multi_var_model2.intercept_ +
                       multi_var_model2.coef_[0] * A2['normalized-losses'] +
                       multi_var_model2.coef_[1] * A2['highway-mpg'])
    print("\nPredicted prices using equation (Multi-variable Model 2 - first 5):")
    print(Price_multi2_eq.head())
else:
    print("\nNot enough data for Multi-variable Model 2 after dropping NaNs, or too few samples.")


# --- Model Evaluation Using Visualization ---
print("\n--- Model Evaluation Using Visualization ---")
width = 9
height = 7

# Regression Plot: Highway MPG vs Price
if not X_highway_clean.empty:
    plt.figure(figsize=(width,height))
    sns.regplot(x="highway-mpg", y="price", data=temp_df_highway) 
    plt.title('Regression Plot: Highway MPG vs Price')
    plt.ylim(0,)
    plt.show()

# Regression Plot: Peak RPM vs Price
df_peak_rpm = df[['peak-rpm', 'price']].dropna()
if not df_peak_rpm.empty:
    plt.figure(figsize=(width, height))
    sns.regplot(x="peak-rpm", y="price", data=df_peak_rpm)
    plt.title('Regression Plot: Peak RPM vs Price')
    plt.ylim(0,)
    plt.show()

# Correlation matrix for these variables
correlation_features = ["peak-rpm", "highway-mpg", "price"]
# Ensure no NaNs in columns used for correlation, or corr() will produce NaNs
df_corr = df[correlation_features].dropna()
if not df_corr.empty:
    correlation_matrix = df_corr.corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)
else:
    print("\nNot enough data for correlation matrix after dropping NaNs.")


# Residual Plot: Highway MPG vs Price
if not X_highway_clean.empty: 
    plt.figure(figsize=(width,height))
    sns.residplot(x=X_highway_clean["highway-mpg"], y=Y_price_clean_highway)
    plt.title('Residual Plot: Highway MPG vs Price')
    plt.show()

# Distribution plot for Multiple Linear Regression (Model 1)
if not df_multi1.empty and len(df_multi1) > len(features_model1):
    Y_hat_multi1_dist = multi_var_model1.predict(Z1) # Predictions from the fitted model
    plt.figure(figsize=(width,height))
    sns.kdeplot(Y_multi1, color='r', label="Actual Value", fill=True, alpha=0.5)
    sns.kdeplot(Y_hat_multi1_dist, color='b', label="Fitted Values", fill=True, alpha=0.5)
    plt.title('Actual vs Fitted Values for Price (Multi-variable Model 1)')
    plt.xlabel('Price (in dollars $)')
    plt.ylabel('Density') 
    plt.legend()
    plt.show()
   


# --- Polynomial Regression and Pipelines ---
print("\n--- Polynomial Regression and Pipelines ---")
def PlotPolly (model_func, independent_variable, dependent_variable_actual, Name) :
    x_new = np.linspace(independent_variable.min(), independent_variable.max(), 100) 
    y_new = model_func(x_new) 

    plt.figure(figsize=(width, height)) # Create a new figure for each plot
    plt.plot(independent_variable, dependent_variable_actual, '.', label='Actual Data')
    plt.plot(x_new, y_new, '-', label='Polynomial Fit')
    plt.title(f'Polynomial Fit ({Name}) for Price')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')
    plt.legend()
    plt.show()

# Use cleaned highway-mpg and price data
x_poly = X_highway_clean['highway-mpg']
y_poly = Y_price_clean_highway

if not x_poly.empty:
    # Fit the polynomial using the function polyfit, then use poly1d
    # Degree 3
    f_poly3 = np.polyfit(x_poly, y_poly, 3)
    p_poly3 = np.poly1d(f_poly3)
    print("\nPolynomial function (degree 3) for highway-mpg vs price:")
    print(p_poly3)
    PlotPolly(p_poly3, x_poly, y_poly, 'Highway MPG (3rd Deg Poly)')

    # Degree 11 (Caution: High-degree polynomials can overfit)
    f_poly11 = np.polyfit(x_poly, y_poly, 11)
    p_poly11 = np.poly1d(f_poly11)
    print("\nPolynomial function (degree 11) for highway-mpg vs price:")
    print(p_poly11)
    PlotPolly(p_poly11, x_poly, y_poly, 'Highway MPG (11th Deg Poly)')


# Polynomial Features for Multiple Regression (using features from Model 1)
if not df_multi1.empty and len(df_multi1) > len(features_model1):
    pr = PolynomialFeatures(degree=2, include_bias=False)
    Z1_pr = pr.fit_transform(Z1) # Z1 is from the cleaned df_multi1

    print(f"\nOriginal shape (Multi-variable Model 1 features): {Z1.shape}")
    print(f"Polynomial shape (degree 2): {Z1_pr.shape}")
    print(f"Number of new features created: {Z1_pr.shape[1] - Z1.shape[1]}")

    # Pipeline
    print("\n--- Pipeline with StandardScaler, PolynomialFeatures, and LinearRegression ---")
    pipeline = Pipeline( [
        ('scale', StandardScaler()),
        ('polynomial', PolynomialFeatures(degree=2, include_bias=False)), # Using degree 2
        ('model', LinearRegression())
    ] )

    # Prepare data (ensure numeric type - Z1 and Y_multi1 are already from cleaned df)
    # Z1 = Z1.astype(float) # Already handled by dropna and selection if original cols were numeric
    # Y_multi1 = Y_multi1.astype(float)

    pipeline.fit(Z1, Y_multi1)
    y_pipe_pred = pipeline.predict(Z1)
    print("First 4 predictions from pipeline:")
    print(y_pipe_pred[:4])
else:
    print("\nSkipping PolynomialFeatures and Pipeline for Z due to insufficient data for Multi-variable Model 1.")


# --- Model Performance Metrics (R-squared & MSE) ---
print("\n--- Model Performance Metrics ---")

# Model 1: Simple Linear Regression (Highway MPG)
if not X_highway_clean.empty:
    print("\nSLR - Highway MPG Model:")
    # lm_highway is already fitted
    r_sq_highway = lm_highway.score(X_highway_clean, Y_price_clean_highway)
    print(f'R-square (Highway MPG): {r_sq_highway:.4f}')
    # Yhat_highway is already predicted
    print('First four predicted values (Highway MPG): ', Yhat_highway[0:4])
    mse_highway = mean_squared_error(Y_price_clean_highway, Yhat_highway)
    print(f'Mean Squared Error (Highway MPG): {mse_highway:.2f}')
else:
    print("\nSkipping metrics for SLR Highway MPG model due to insufficient data.")

# Model 2: Multiple Linear Regression (using features_model1)
if not df_multi1.empty and len(df_multi1) > len(features_model1):
    print("\nMLR - Model 1 (horsepower, curb-weight, etc.):")
    # multi_var_model1 is already fitted
    r_sq_multi1 = multi_var_model1.score(Z1, Y_multi1)
    print(f'R-square (Multi-variable Model 1): {r_sq_multi1:.4f}')
    Y_predict_multi1 = multi_var_model1.predict(Z1)
    mse_multi1 = mean_squared_error(Y_multi1, Y_predict_multi1)
    print(f'Mean Squared Error (Multi-variable Model 1): {mse_multi1:.2f}')
else:
    print("\nSkipping metrics for MLR Model 1 due to insufficient data.")

# Model 3: Polynomial Fit (3rd degree on highway-mpg)
if not x_poly.empty:
    print("\nPolynomial Fit (3rd Degree - Highway MPG):")
    r_sq_poly3 = r2_score(y_poly, p_poly3(x_poly)) # Use p_poly3 which is the poly1d object
    print(f'R-square (Polynomial 3rd Deg): {r_sq_poly3:.4f}')
    mse_poly3 = mean_squared_error(y_poly, p_poly3(x_poly))
    print(f'Mean Squared Error (Polynomial 3rd Deg): {mse_poly3:.2f}')
else:
    print("\nSkipping metrics for Polynomial Fit (3rd Degree) due to insufficient data.")


# --- Prediction and Decision Making (Example with SLR Highway MPG model) ---
print("\n--- Prediction Example (using SLR Highway MPG model) ---")
if not X_highway_clean.empty:
    new_input_mpg = np.arange(15, 56, 1).reshape(-1, 1) # Range similar to PlotPolly
    # lm_highway is already fitted with X_highway_clean, Y_price_clean_highway
    yhat_new_mpg = lm_highway.predict(new_input_mpg)
    print("\nPredicted prices for new Highway MPG inputs (15 to 55):")
    for mpg, pred_price in zip(new_input_mpg.flatten()[:5], yhat_new_mpg[:5]):
        print(f"MPG: {mpg}, Predicted Price: ${pred_price:.2f}")

    plt.figure(figsize=(width, height))
    plt.plot(new_input_mpg, yhat_new_mpg, label="Predicted Price Trend")
    plt.scatter(X_highway_clean['highway-mpg'], Y_price_clean_highway, color='red', alpha=0.5, label='Actual Data')
    plt.xlabel("Highway MPG")
    plt.ylabel("Predicted Price")
    plt.title("Price Prediction Trend based on Highway MPG")
    plt.legend()
    plt.show()
else:
    print("\nSkipping prediction example plot due to insufficient data for Highway MPG model.")

print("\nAnalysis complete.")