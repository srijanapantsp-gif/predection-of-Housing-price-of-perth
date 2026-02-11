"""
Perth Housing Price Prediction Code
Updated and corrected version with all fixes applied
"""

# Google Colab specific code - commented out for local execution
# from google.colab import drive
# drive.mount('/content/drive')

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import sys

print("Starting Perth Housing Price Prediction Analysis...")
print("="*60)

# Load your Excel file
df = pd.read_excel("Perth_Housing_Data.xlsx")

# Check the first rows
print(df.head())
print(df.info())
print(df.columns.tolist())

# Data cleaning
years = list(range(2015, 2026))

for year in years:
    df[year] = (
        df[year]
        .astype(str)
        .str.replace(r'[\$, \t,]', '', regex=True)
    )
    df[year] = pd.to_numeric(df[year], errors='coerce')

# Fill missing prices using previous year values
df[years] = df[years].ffill(axis=1)

# Fill remaining numeric missing values
df.fillna(df.median(numeric_only=True), inplace=True)

df = df.copy()

# EDA
# Boxplot for land area
sns.boxplot(x=df['Land_Area'])
plt.title("Distribution of Land Area")
plt.xlabel("Land Area (sqm)")
plt.savefig('plot_land_area_distribution.png')
plt.close()
print("Saved: plot_land_area_distribution.png")

# Boxplot for Build_Year
sns.boxplot(x=df['Build_Year'])
plt.title("Distribution of Build Year")
plt.xlabel("Build Year")
plt.savefig('plot_build_year_distribution.png')
plt.close()
print("Saved: plot_build_year_distribution.png")

# Boxplot for price in a sample year, e.g., 2025
sns.boxplot(x=df[2025])
plt.title("Distribution of Property Prices in 2025")
plt.xlabel("Price")
plt.savefig('plot_price_distribution_2025.png')
plt.close()
print("Saved: plot_price_distribution_2025.png")

#removing outliners in the section of Land_Area
Q1 = df['Land_Area'].quantile(0.25)
Q3 = df['Land_Area'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[
    (df['Land_Area'] >= lower_bound) &
    (df['Land_Area'] <= upper_bound)
]

plt.figure()
sns.boxplot(x=df['Land_Area'])
plt.title("Land Area Distribution After Outlier Removal")
plt.xlabel("Land Area (sqm)")
plt.savefig('plot_land_area_after_outlier_removal.png')
plt.close()
print("Saved: plot_land_area_after_outlier_removal.png")

# Count unique houses per suburb
houses_per_suburb = df.drop_duplicates('Property_Address')['Suburb'].value_counts()

print(houses_per_suburb)

#Average price over time
years = list(range(2015, 2026))

# Now compute mean and plot
df[years].mean().plot(title="Average Price Over Time", figsize=(8,5))
plt.ylabel("Price")
plt.savefig('plot_avg_price_over_time.png')
plt.close()
print("Saved: plot_avg_price_over_time.png")

#price distribution by suburb in 2025
sns.boxplot(data=df, x='Suburb', y=2025)  # Choose a year to compare
plt.title("Price Distribution by Suburb in 2025")
plt.savefig('plot_price_by_suburb_2025.png')
plt.close()
print("Saved: plot_price_by_suburb_2025.png")

#price distribution of houses by features
fig, axes = plt.subplots(2, 2, figsize=(14,10))

# Bedrooms (DISCRETE)
sns.countplot(x='Bedrooms', data=df, ax=axes[0,0], color='skyblue')
axes[0,0].set_title("Distribution of Bedrooms")

# Bathrooms (DISCRETE)
sns.countplot(x='Bathrooms', data=df, ax=axes[0,1], color='lightgreen')
axes[0,1].set_title("Distribution of Bathrooms")

# Garage (DISCRETE)
sns.countplot(x='Garage', data=df, ax=axes[1,0], color='salmon')
axes[1,0].set_title("Distribution of Garage Spaces")

# Land Area (CONTINUOUS)
sns.histplot(df['Land_Area'], bins=10, ax=axes[1,1], color='orchid')
axes[1,1].set_title("Distribution of Land Area (sqm)")

plt.tight_layout()
plt.savefig('plot_feature_distributions.png')
plt.close()
print("Saved: plot_feature_distributions.png")

#house price distribution per year 
years = list(range(2015, 2026))

plt.figure(figsize=(12,6))
df[years].boxplot()
plt.title("House Price Distribution per Year")
plt.ylabel("Price ($)")
plt.xticks(rotation=45)
plt.savefig('plot_price_distribution_per_year.png')
plt.close()
print("Saved: plot_price_distribution_per_year.png")

# Price as average over years
year_cols = list(range(2015, 2026))
df['Avg_Price'] = df[year_cols].mean(axis=1)

# Land bins (reasonable ranges)
bins = [0, 300, 450, 600, 800, 1000, 1500, 2500]
labels = ['<300', '300–450', '450–600', '600–800', '800–1000', '1000–1500', '1500+']
df['Land_Bin'] = pd.cut(df['Land_Area'], bins=bins, labels=labels, include_lowest=True)

# Average price per bin
avg_price = df.groupby('Land_Bin', observed=True)['Avg_Price'].mean()

# Plot
plt.figure(figsize=(8,5))
avg_price.plot(kind='bar', color='orchid')
plt.title("Average Price vs Land Area")
plt.xlabel("Land Area (sqm)")
plt.ylabel("Average Price ($)")
plt.xticks(rotation=45)
plt.savefig('plot_avg_price_vs_land_area.png')
plt.close()
print("Saved: plot_avg_price_vs_land_area.png")

#average house price of suburb 2015-2025
avg_price_suburb = df.groupby('Suburb')[years].mean().T

plt.figure(figsize=(12,6))
for suburb in avg_price_suburb.columns:
    plt.plot(avg_price_suburb.index, avg_price_suburb[suburb], marker='o', label=suburb)

plt.title("Average House Prices by Suburb (2015–2025)")
plt.ylabel("Average Price ($)")
plt.xlabel("Year")
plt.legend()
plt.savefig('plot_avg_prices_by_suburb.png')
plt.close()
print("Saved: plot_avg_prices_by_suburb.png")

#historic distribution of house price on 2025
sns.histplot(df[2025], bins=15, kde=True)
plt.title("Distribution of House Prices (2025)")
plt.xlabel("Price ($)")
plt.savefig('plot_historic_price_distribution_2025.png')
plt.close()
print("Saved: plot_historic_price_distribution_2025.png")

#boxplot of land area catogery vs price in 2025

bins = [0, 400, 600, 800, 1000, 2000]
labels = ['Very Small', 'Small', 'Medium', 'Large', 'Very Large']
df['Land_Category'] = pd.cut(df['Land_Area'], bins=bins, labels=labels)

sns.boxplot(x='Land_Category', y=2025, data=df)
plt.title("Land Area Category vs Price (2025)")
plt.ylabel("Price ($)")
plt.savefig('plot_land_category_vs_price.png')
plt.close()
print("Saved: plot_land_category_vs_price.png")

bedroom_price = df.groupby('Bedrooms')[2025].mean()

plt.figure(figsize=(8,5))
bedroom_price.plot(kind='bar', color='skyblue')
plt.title("Average Price by Number of Bedrooms (2025)")
plt.ylabel("Average Price ($)")
plt.xlabel("Bedrooms")
plt.savefig('plot_avg_price_by_bedrooms.png')
plt.close()
print("Saved: plot_avg_price_by_bedrooms.png")

bathroom_price = df.groupby('Bathrooms')[2025].mean()

plt.figure(figsize=(8,5))
bathroom_price.plot(kind='bar', color='lightgreen')
plt.title("Average Price by Number of Bathrooms (2025)")
plt.ylabel("Average Price ($)")
plt.xlabel("Bathrooms")
plt.savefig('plot_avg_price_by_bathrooms.png')
plt.close()
print("Saved: plot_avg_price_by_bathrooms.png")

# Suppose df is already loaded
# Melt using integer column names
df_long = df.melt(
    id_vars=['Property_Address', 'Suburb', 'Bedrooms', 'Bathrooms', 'Garage', 'Land_Area', 'Build_Year'],
    value_vars=list(range(2015, 2026)),  # integers instead of strings
    var_name='Year',
    value_name='Price'
)

# Optional: convert Year to int (already int, but safe)
df_long['Year'] = df_long['Year'].astype(int)

# Add House_Age
df_long['House_Age'] = df_long['Year'] - df_long['Build_Year']

# Drop rows with missing Price (if any)
df_long = df_long.dropna(subset=['Price'])

# Preview
print(df_long.head(10))

years = list(range(2015, 2026))

df_suburb_long = df.melt(
    id_vars=['Suburb'],
    value_vars=years,
    var_name='Year',
    value_name='Price'
)
df_suburb_long['Year'] = df_suburb_long['Year'].astype(int)
df_suburb_long.dropna(subset=['Price'], inplace=True)

# average price over time of suburb
suburb_trend = (
    df_suburb_long
    .groupby(['Year', 'Suburb'])['Price']
    .mean()
    .reset_index()
)

plt.figure(figsize=(12,6))
sns.lineplot(
    data=suburb_trend,
    x='Year',
    y='Price',
    hue='Suburb'
)
plt.title("Historic Sold Price Trend by Suburb (2015–2025)")
plt.ylabel("Average Price")
plt.savefig('plot_suburb_trend.png')
plt.close()
print("Saved: plot_suburb_trend.png")

# year to year growth rate analysis
suburb_growth = suburb_trend.copy()
suburb_growth['Growth_Rate'] = (
    suburb_growth
    .groupby('Suburb')['Price']
    .pct_change() * 100
)

plt.figure(figsize=(12,6))
sns.lineplot(
    data=suburb_growth,
    x='Year',
    y='Growth_Rate',
    hue='Suburb'
)
plt.title("Year-on-Year Price Growth by Suburb")
plt.ylabel("Growth Rate (%)")
plt.savefig('plot_growth_rate.png')
plt.close()
print("Saved: plot_growth_rate.png")

suburb_volatility = (
    df_suburb_long
    .groupby('Suburb')['Price']
    .std()
    .reset_index()
)

plt.figure(figsize=(10,5))
sns.barplot(
    data=suburb_volatility,
    x='Suburb',
    y='Price'
)
plt.title("Price Volatility by Suburb (2015–2025)")
plt.xticks(rotation=45)
plt.savefig('plot_volatility.png')
plt.close()
print("Saved: plot_volatility.png")

cagr_list = []

for suburb, group in df_suburb_long.groupby('Suburb'):
    start_price = group[group['Year'] == 2015]['Price'].mean()
    end_price = group[group['Year'] == 2025]['Price'].mean()
    if start_price > 0:
        cagr = ((end_price / start_price) ** (1/10) - 1) * 100
        cagr_list.append({'Suburb': suburb, 'CAGR (%)': cagr})

cagr_df = pd.DataFrame(cagr_list).sort_values('CAGR (%)', ascending=False)
print("CAGR by Suburb:")
print(cagr_df.head())

# modelling
from sklearn.model_selection import train_test_split

# Features
X = df_long[['Suburb', 'Bedrooms', 'Bathrooms', 'Garage', 'Land_Area', 'House_Age']]

# One-hot encode Suburb
X = pd.get_dummies(X, columns=['Suburb'], drop_first=True)

# Target
y = df_long['Price']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_columns = X_train.columns
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Initialize model
rf_model = RandomForestRegressor(n_estimators=300, random_state=42)

# Train
rf_model.fit(X_train, y_train)

# Predict on test set
y_pred = rf_model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: ${mae:,.0f}")
print(f"R^2 Score: {r2:.2f}")

def predict_price(model, beds, baths, garage, land_area, build_year, year, suburb, X_train_columns):
    house_age = year - build_year
    
    # Create a dictionary for numeric features
    input_dict = {
        'Bedrooms': beds,
        'Bathrooms': baths,
        'Garage': garage,
        'Land_Area': land_area,
        'House_Age': house_age
    }
    
    # Add one-hot encoded suburb columns
    for col in X_train_columns:
        if col.startswith('Suburb_'):
            input_dict[col] = 1 if col == f'Suburb_{suburb}' else 0
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # Make sure the column order matches X_train
    input_df = input_df[X_train_columns]
    
    # Predict
    return model.predict(input_df)[0]

price = predict_price(
    model=rf_model,
    beds=4,
    baths=2,
    garage=2,
    land_area=500,
    build_year=2005,
    year=2025,
    suburb='Nedlands',
    X_train_columns=X_train_columns
)

print(f"Predicted Price: ${price:,.0f}")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Initialize the model
linear_model = LinearRegression()

# Train
linear_model.fit(X_train, y_train)

# Predict
y_pred = linear_model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Linear Regression -> MAE: ${mae:,.0f}, R²: {r2:.2f}")

from sklearn.tree import DecisionTreeRegressor

# Initialize the model
tree_model = DecisionTreeRegressor(random_state=42)

# Train
tree_model.fit(X_train, y_train)

# Predict
y_pred_tree = tree_model.predict(X_test)

# Evaluate
mae_tree = mean_absolute_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)
print(f"Decision Tree -> MAE: ${mae_tree:,.0f}, R²: {r2_tree:.2f}")

# gradient boosting
from xgboost import XGBRegressor

# Initialize model
xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42, objective='reg:squarederror')

# Train
xgb_model.fit(X_train, y_train)

# Predict
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
print(f"XGBoost -> MAE: ${mae_xgb:,.0f}, R²: {r2_xgb:.2f}")

import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score

# Initialize model
lgb_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1, random_state=42)

# Train
lgb_model.fit(X_train, y_train)

# Predict
y_pred_lgb = lgb_model.predict(X_test)

# Evaluate
mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
r2_lgb = r2_score(y_test, y_pred_lgb)
print(f"LightGBM -> MAE: ${mae_lgb:,.0f}, R²: {r2_lgb:.2f}")

# feature Selection
feat_importances = pd.Series(
    rf_model.feature_importances_,
    index=X_train.columns
)
feat_importances.sort_values().plot(kind='barh', figsize=(8,6))
plt.title("Random Forest Feature Importance")
plt.savefig('plot_feature_importance.png')
plt.close()
print("Saved: plot_feature_importance.png")

# future predection for next 10 years
years = list(range(2015, 2026))

suburb_cagr = {}

for suburb in df['Suburb'].unique():
    temp = df[df['Suburb'] == suburb]

    start_price = temp[2015].mean()
    end_price = temp[2025].mean()

    n_years = 2025 - 2015
    cagr = (end_price / start_price) ** (1 / n_years) - 1

    suburb_cagr[suburb] = cagr

print("Suburb CAGR:", suburb_cagr)

future_years = range(2026, 2036)
predictions = []

for _, row in df.iterrows():

    # ---- RF BASE PRICE (2025) ----
    input_dict = {
        'Bedrooms': row['Bedrooms'],
        'Bathrooms': row['Bathrooms'],
        'Garage': row['Garage'],
        'Land_Area': row['Land_Area'],
        'House_Age': 2025 - row['Build_Year']
    }

    for col in X_train_columns:
        if col.startswith('Suburb_'):
            input_dict[col] = 1 if col == f"Suburb_{row['Suburb']}" else 0

    base_df = pd.DataFrame([input_dict])[X_train_columns]
    base_price = rf_model.predict(base_df)[0]

    # ---- APPLY CAGR ----
    for year in future_years:
        growth_factor = (1 + suburb_cagr[row['Suburb']]) ** (year - 2025)

        predictions.append({
            'Property_Address': row['Property_Address'],
            'Suburb': row['Suburb'],
            'Year': year,
            'Predicted_Price': base_price * growth_factor
        })

future_df = pd.DataFrame(predictions)
print("Future predictions preview:")
print(future_df.head())

# Ensure numeric
future_df['Year'] = future_df['Year'].astype(int)
future_df['Predicted_Price'] = future_df['Predicted_Price'].astype(float)

# Unique suburbs
suburbs = future_df['Suburb'].unique()

# Prepare subplots
n = len(suburbs)
cols = 2
rows = (n + 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(14, 5*rows), sharey=True)
axes = axes.flatten()

for i, suburb in enumerate(suburbs):
    ax = axes[i]
    subset = future_df[future_df['Suburb'] == suburb]

    # Plot mean trend only
    mean_data = subset.groupby('Year')['Predicted_Price'].mean().reset_index()
    ax.plot(mean_data['Year'], mean_data['Predicted_Price'],
            marker='o', color='orange', label='Mean Price')

    ax.set_title(suburb)
    ax.set_xlabel("Year")
    ax.set_ylabel("Price ($)")
    ax.grid(True)
    ax.legend()

# Remove empty subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('plot_future_predictions_by_suburb.png')
plt.close()
print("Saved: plot_future_predictions_by_suburb.png")

plt.figure(figsize=(14,7))

# Mean trend per suburb with standard deviation band
sns.lineplot(
    data=future_df,
    x='Year',
    y='Predicted_Price',
    hue='Suburb',
    errorbar='sd',          # Changed from ci='sd' to errorbar='sd'
    estimator='mean', # Plot mean per year
    marker='o'
)

plt.title("Forecasted House Prices by Suburb (All Houses Aggregated)")
plt.ylabel("Price ($)")
plt.xlabel("Year")
plt.grid(True)
plt.legend(title='Suburb')
plt.savefig('plot_forecasted_prices_by_suburb.png')
plt.close()
print("Saved: plot_forecasted_prices_by_suburb.png")

top_houses = future_df.groupby('Property_Address')['Predicted_Price'].mean().nlargest(5).index
subset = future_df[future_df['Property_Address'].isin(top_houses)]

plt.figure(figsize=(14,7))
sns.lineplot(data=subset, x='Year', y='Predicted_Price', hue='Property_Address', marker='o')
plt.title("Top 5 Most Expensive Houses Forecast")
plt.ylabel("Price ($)")
plt.savefig('plot_top5_expensive_houses_forecast.png')
plt.close()
print("Saved: plot_top5_expensive_houses_forecast.png")

print("\n" + "="*60)
print("Analysis completed successfully!")
print("="*60)





