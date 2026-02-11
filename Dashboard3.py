"""
Perth Housing Price Prediction Dashboard
Streamlit dashboard for predicting house prices in Perth, Western Australia
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Perth Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the housing data"""
    try:
        df = pd.read_excel("Perth_Housing_Data.xlsx")
        
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
        
        # Outlier removal for Land_Area
        Q1 = df['Land_Area'].quantile(0.25)
        Q3 = df['Land_Area'].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        df = df[
            (df['Land_Area'] >= lower) &
            (df['Land_Area'] <= upper)
        ]
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def prepare_modeling_data(df):
    """Prepare data for modeling"""
    years = list(range(2015, 2026))
    
    # Melt data to long format
    df_long = df.melt(
        id_vars=['Property_Address', 'Suburb', 'Bedrooms', 'Bathrooms', 'Garage', 'Land_Area', 'Build_Year'],
        value_vars=years,
        var_name='Year',
        value_name='Price'
    )
    
    df_long['Year'] = df_long['Year'].astype(int)
    df_long['House_Age'] = df_long['Year'] - df_long['Build_Year']
    df_long = df_long.dropna(subset=['Price'])
    
    return df_long

@st.cache_data
def calculate_suburb_cagr(df):
    """Calculate CAGR for each suburb"""
    years = list(range(2015, 2026))
    suburb_cagr = {}
    
    for suburb in df['Suburb'].unique():
        temp = df[df['Suburb'] == suburb]
        start_price = temp[2015].mean()
        end_price = temp[2025].mean()
        n_years = 2025 - 2015
        if start_price > 0:
            cagr = (end_price / start_price) ** (1 / n_years) - 1
            suburb_cagr[suburb] = cagr
        else:
            suburb_cagr[suburb] = 0
    
    return suburb_cagr

def predict_price_cagr(df, df_long, beds, baths, garage, land_area, build_year, year, suburb):
    """
    Predict house price using CAGR method
    Based on suburb average price and CAGR growth rate
    """
    years = list(range(2015, 2026))
    
    # Get suburb average price for 2025
    suburb_data = df[df['Suburb'] == suburb]
    if len(suburb_data) == 0:
        return None
    
    # Calculate base price (2025 average for this suburb)
    base_price_2025 = suburb_data[2025].mean()
    
    # Calculate CAGR for this suburb
    start_price = suburb_data[2015].mean()
    end_price = suburb_data[2025].mean()
    n_years = 2025 - 2015
    
    if start_price > 0:
        cagr = (end_price / start_price) ** (1 / n_years) - 1
    else:
        cagr = 0
    
    # Adjust base price based on property features relative to suburb average
    # Get average features for the suburb
    avg_bedrooms = suburb_data['Bedrooms'].mean()
    avg_bathrooms = suburb_data['Bathrooms'].mean()
    avg_garage = suburb_data['Garage'].mean()
    avg_land_area = suburb_data['Land_Area'].mean()
    avg_build_year = suburb_data['Build_Year'].mean()
    
    # Calculate adjustments (simple multipliers based on differences)
    bedroom_factor = 1 + (beds - avg_bedrooms) * 0.05  # 5% per bedroom difference
    bathroom_factor = 1 + (baths - avg_bathrooms) * 0.03  # 3% per bathroom difference
    garage_factor = 1 + (garage - avg_garage) * 0.02  # 2% per garage difference
    land_factor = 1 + ((land_area - avg_land_area) / avg_land_area) * 0.5  # Proportional to land size
    
    # Age factor (newer houses worth more)
    house_age_2025 = 2025 - build_year
    avg_age_2025 = 2025 - avg_build_year
    age_factor = 1 - (house_age_2025 - avg_age_2025) * 0.001  # 0.1% per year difference
    
    # Apply adjustments
    adjusted_price_2025 = base_price_2025 * bedroom_factor * bathroom_factor * garage_factor * land_factor * age_factor
    
    # Apply CAGR to get price for the requested year
    if year <= 2025:
        # Historical: work backwards from 2025
        years_from_2025 = year - 2025
        predicted_price = adjusted_price_2025 * ((1 + cagr) ** years_from_2025)
    else:
        # Future: work forwards from 2025
        years_from_2025 = year - 2025
        predicted_price = adjusted_price_2025 * ((1 + cagr) ** years_from_2025)
    
    return max(predicted_price, 0)  # Ensure non-negative

# Main App
def main():
    # Header with welcome message
    st.markdown('<h1 class="main-header">🏠 Perth Housing Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; padding: 10px 0; color: #666; font-size: 18px;'>
    <p><strong>Your trusted tool for estimating property values in Perth, Western Australia</strong></p>
    <p style='font-size: 14px;'>Powered by CAGR (Compound Annual Growth Rate) analysis and 10+ years of real estate data</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading and preprocessing data..."):
        df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check if 'Perth_Housing_Data.xlsx' exists in the current directory.")
        return
    
    # Sidebar with user-friendly navigation
    st.sidebar.title("📑 Navigation Menu")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "**What would you like to do?**",
        [
            "🏠 Get Property Value Estimate",
            "📊 View Market Statistics",
            "📈 Explore Price Trends",
            "🤖 How Accurate Are Our Predictions?",
            "🔮 Future Market Forecast",
            "📊 Batch Predictions"
        ]
    )
    
    # Map user-friendly names to function names
    page_mapping = {
        "🏠 Get Property Value Estimate": "🏠 Price Prediction",
        "📊 View Market Statistics": "📊 Data Overview",
        "📈 Explore Price Trends": "📈 Trends & Analysis",
        "🤖 How Accurate Are Our Predictions?": "🤖 CAGR Explanation",
        "🔮 Future Market Forecast": "🔮 Future Predictions",
        "📊 Batch Predictions": "📊 Batch Predictions"
    }
    
    # Add helpful descriptions
    st.sidebar.markdown("---")
    descriptions = {
        "🏠 Get Property Value Estimate": "Enter your property details to get an instant value estimate",
        "📊 View Market Statistics": "See overall market data and property distributions",
        "📈 Explore Price Trends": "Analyze how prices have changed over time by suburb",
        "🤖 How Accurate Are Our Predictions?": "Learn how CAGR works and why it's reliable",
        "🔮 Future Market Forecast": "See predicted property values for the next 10 years",
        "📊 Batch Predictions": "Upload CSV/Excel file and generate predictions for all properties"
    }
    
    if page in descriptions:
        st.sidebar.info(f"💡 **{descriptions[page]}**")
    
    # Convert to internal page name
    internal_page = page_mapping.get(page, page)
    
    # Prepare data
    with st.spinner("Preparing data..."):
        df_long = prepare_modeling_data(df)
    
    # Calculate CAGR for all suburbs (cached)
    with st.spinner("Calculating growth rates..."):
        suburb_cagr = calculate_suburb_cagr(df)
    
    # Page routing
    if internal_page == "🏠 Price Prediction":
        show_prediction_page(df, df_long, suburb_cagr)
    elif internal_page == "📊 Data Overview":
        show_data_overview(df, df_long)
    elif internal_page == "📈 Trends & Analysis":
        show_trends_analysis(df, df_long)
    elif internal_page == "🤖 CAGR Explanation":
        show_cagr_explanation(df, suburb_cagr)
    elif internal_page == "🔮 Future Predictions":
        show_future_predictions_cagr(df, suburb_cagr)
    elif internal_page == "📊 Batch Predictions":
        show_batch_predictions(df, df_long, suburb_cagr)

def show_prediction_page(df, df_long, suburb_cagr):
    """Price prediction page - User-friendly version"""
    
    # Welcome section with explanation
    st.header("🏠 Estimate Your Property Value")
    st.markdown("""
    <div style='background-color: #e8f4f8; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h3 style='color: #1f77b4; margin-top: 0;'>How It Works</h3>
    <p style='font-size: 16px; line-height: 1.6;'>
    Simply fill in the details about your property below, and our prediction system will estimate 
    its market value using <strong>CAGR (Compound Annual Growth Rate)</strong> analysis. This method uses 
    real historical data from Perth, Western Australia, to provide accurate price estimates based on:
    </p>
    <ul style='font-size: 16px; line-height: 1.8;'>
        <li><strong>📍 Location:</strong> Which suburb the property is in (most important factor)</li>
        <li><strong>🏠 Property Features:</strong> Number of bedrooms, bathrooms, and garage spaces</li>
        <li><strong>📐 Land Size:</strong> The size of your land in square meters</li>
        <li><strong>📅 Property Age:</strong> How old the house is</li>
        <li><strong>📈 Growth Rate:</strong> Historical price growth patterns in your suburb (CAGR)</li>
    </ul>
    <p style='font-size: 14px; color: #666; margin-top: 10px;'>
    <strong>How CAGR works:</strong> We calculate the average annual growth rate for each suburb based on 
    actual property prices from 2015 to 2025, then use this growth rate to estimate prices for any year.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Make the results panel (right side) wider than the input panel (left side)
    col1, col2 = st.columns([1, 1.6])
    
    with col1:
        st.subheader("📝 Tell Us About Your Property")
        st.markdown("**Fill in the details below to get your property value estimate:**")
        
        suburb = st.selectbox(
            "📍 **Which suburb is your property located in?**",
            sorted(df['Suburb'].unique()),
            help="Select the suburb where your property is located. This is one of the most important factors in determining property value."
        )
        
        bedrooms = st.number_input(
            "🛏️ **How many bedrooms?**",
            min_value=1,
            max_value=10,
            value=3,
            help="Enter the total number of bedrooms in the property"
        )
        
        bathrooms = st.number_input(
            "🚿 **How many bathrooms?**",
            min_value=1,
            max_value=10,
            value=2,
            help="Enter the total number of bathrooms (including powder rooms)"
        )
        
        garage = st.number_input(
            "🚗 **How many car parking spaces?**",
            min_value=0,
            max_value=10,
            value=2,
            help="Enter the number of covered parking spaces or garage spaces"
        )
        
        land_area = st.number_input(
            "📐 **What is the land size? (in square meters)**",
            min_value=100,
            max_value=5000,
            value=500,
            step=50,
            help="Enter the total land area in square meters. Tip: 1 square meter = 10.76 square feet"
        )
        
        build_year = st.number_input(
            "📅 **What year was the house built?**",
            min_value=1900,
            max_value=2025,
            value=2000,
            help="Enter the year the property was originally constructed"
        )
        
        prediction_year = st.number_input(
            "📆 **What year do you want the price estimate for?**",
            min_value=2015,
            max_value=2035,
            value=2025,
            help="Select the year for which you want the property value estimate (can be past, present, or future)"
        )
    
    with col2:
        st.subheader("💰 Your Property Value Estimate")
        st.markdown("**Click the button below to get your estimate:**")
        
        if st.button("🔍 Get My Property Value", type="primary", use_container_width=True):
            try:
                predicted_price = predict_price_cagr(
                    df, df_long, bedrooms, bathrooms, garage, land_area,
                    build_year, prediction_year, suburb
                )
                
                if predicted_price is None:
                    st.error(f"❌ **Unable to calculate price for {suburb}.** Please try a different suburb.")
                    return
                
                # Main result display
                st.markdown("---")
                st.markdown(f"""
                <div style='background-color: #d4edda; padding: 30px; border-radius: 10px; text-align: center; border: 3px solid #28a745;'>
                <h2 style='color: #155724; margin: 0; font-size: 28px;'>Estimated Property Value</h2>
                <h1 style='color: #155724; margin: 10px 0; font-size: 48px; font-weight: bold;'>${predicted_price:,.0f}</h1>
                <p style='color: #155724; font-size: 14px; margin: 0;'>Based on similar properties in {suburb}</p>
                </div>
                """, unsafe_allow_html=True)

                # Price over time chart (easy-to-understand)
                st.markdown("---")
                st.markdown("### 📈 Estimated Price Over Time")
                st.markdown("""
                <div style='background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 15px;'>
                <p style='margin: 0; font-size: 14px;'>
                <strong>📊 What this chart shows:</strong> This graph displays how your property's estimated value changes over time. 
                The <strong style='color: #1f77b4;'>blue line (2015-2025)</strong> shows historical estimates based on actual market data. 
                The <strong style='color: #ff7f0e;'>orange line (2026-2035)</strong> shows future forecasts using the suburb's average growth rate.
                </p>
                </div>
                """, unsafe_allow_html=True)

                # Get CAGR for this suburb
                cagr_value = suburb_cagr.get(suburb, 0)
                
                # Build curve for all years
                curve_years_hist = list(range(2015, 2026))
                curve_years_future = list(range(2026, 2036))
                all_years = curve_years_hist + curve_years_future

                # Calculate prices for all years using CAGR method
                hist_prices = []
                for yr in curve_years_hist:
                    price = predict_price_cagr(df, df_long, bedrooms, bathrooms, garage, land_area, build_year, yr, suburb)
                    hist_prices.append(price if price else 0)

                # For future years, use CAGR growth from 2025
                base_price_2025 = hist_prices[-1] if hist_prices else predicted_price
                future_prices = []
                for yr in curve_years_future:
                    if cagr_value != 0:
                        years_from_2025 = yr - 2025
                        future_price = base_price_2025 * ((1 + cagr_value) ** years_from_2025)
                    else:
                        future_price = base_price_2025
                    future_prices.append(future_price)

                curve_df = pd.DataFrame({
                    "Year": all_years,
                    "Estimated Price": hist_prices + future_prices,
                    "Period": (["Historical Estimates (2015-2025)"] * len(curve_years_hist)) + 
                             (["Future Forecast (2026-2035)"] * len(curve_years_future))
                })

                fig_curve = px.line(
                    curve_df,
                    x="Year",
                    y="Estimated Price",
                    color="Period",
                    markers=True,
                    title="📈 Your Property Value Over Time",
                    labels={"Estimated Price": "Estimated Value ($)", "Year": "Year"}
                )
                fig_curve.update_yaxes(tickprefix="$", separatethousands=True)
                fig_curve.update_layout(
                    legend_title_text="",
                    height=450,
                    margin=dict(l=10, r=10, t=60, b=10),
                    hovermode='x unified'
                )
                
                # Update colors for better visibility
                fig_curve.update_traces(
                    line=dict(width=3),
                    marker=dict(size=8)
                )

                # Highlight the selected prediction year
                fig_curve.add_vline(
                    x=prediction_year,
                    line_width=3,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Your selected year: {prediction_year}",
                    annotation_position="top",
                    annotation_font_size=12,
                    annotation_font_color="red"
                )

                st.plotly_chart(fig_curve, use_container_width=True)

                if cagr_value != 0:
                    st.info(f"💡 **Growth Rate:** {suburb} has an average annual growth rate of **{cagr_value * 100:.2f}%** per year (calculated from 2015-2025 data). This growth rate is used for future predictions.")
                else:
                    st.warning(f"⚠️ Growth rate for {suburb} could not be calculated. Future predictions will remain constant.")
                
                # Property summary
                house_age = prediction_year - build_year
                st.markdown("---")
                st.markdown("### 📋 Property Details Summary")
                st.markdown(f"""
                <div style='background-color: #d1ecf1; padding: 20px; border-radius: 10px; border-left: 5px solid #0c5460;'>
                <table style='width: 100%; font-size: 16px;'>
                <tr><td style='padding: 8px;'><strong>📍 Location:</strong></td><td style='padding: 8px;'>{suburb}</td></tr>
                <tr><td style='padding: 8px;'><strong>🏠 Property Age:</strong></td><td style='padding: 8px;'>{house_age} years old</td></tr>
                <tr><td style='padding: 8px;'><strong>🛏️ Bedrooms:</strong></td><td style='padding: 8px;'>{bedrooms}</td></tr>
                <tr><td style='padding: 8px;'><strong>🚿 Bathrooms:</strong></td><td style='padding: 8px;'>{bathrooms}</td></tr>
                <tr><td style='padding: 8px;'><strong>🚗 Parking:</strong></td><td style='padding: 8px;'>{garage} spaces</td></tr>
                <tr><td style='padding: 8px;'><strong>📐 Land Size:</strong></td><td style='padding: 8px;'>{land_area} square meters</td></tr>
                <tr><td style='padding: 8px;'><strong>📅 Estimate Year:</strong></td><td style='padding: 8px;'>{prediction_year}</td></tr>
                </table>
                </div>
                """, unsafe_allow_html=True)
                
                # Price comparison with explanation
                if suburb in df['Suburb'].values:
                    suburb_avg = df_long[df_long['Suburb'] == suburb]['Price'].mean()
                    diff = predicted_price - suburb_avg
                    diff_pct = (diff / suburb_avg) * 100
                    
                    st.markdown("---")
                    st.markdown("### 📊 How Does This Compare?")
                    
                    if diff > 0:
                        comparison_color = "#d4edda"
                        border_color = "#28a745"
                        arrow = "📈"
                        explanation = "Your property is estimated to be **above average** for this suburb. This could be due to larger land size, more bedrooms/bathrooms, newer construction, or better location within the suburb."
                    else:
                        comparison_color = "#fff3cd"
                        border_color = "#ffc107"
                        arrow = "📉"
                        explanation = "Your property is estimated to be **below average** for this suburb. This could be due to smaller land size, fewer bedrooms/bathrooms, older construction, or location factors."
                    
                    st.markdown(f"""
                    <div style='background-color: {comparison_color}; padding: 20px; border-radius: 10px; border-left: 5px solid {border_color};'>
                    <h4 style='margin-top: 0;'>Compared to Average Property in {suburb}</h4>
                    <p style='font-size: 18px; margin: 10px 0;'><strong>{arrow} Difference:</strong> ${abs(diff):,.0f} ({abs(diff_pct):.1f}%)</p>
                    <p style='font-size: 14px; color: #666; margin-bottom: 0;'>{explanation}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show average for reference
                    st.info(f"💡 **Average property value in {suburb}:** ${suburb_avg:,.0f}")
                
                # Disclaimer
                st.markdown("---")
                st.markdown("""
                <div style='background-color: #f8d7da; padding: 15px; border-radius: 5px; border-left: 4px solid #dc3545;'>
                <p style='margin: 0; font-size: 12px; color: #721c24;'>
                <strong>⚠️ Important Note:</strong> This is an estimated value based on historical data and property features. 
                Actual market value may vary based on property condition, renovations, market conditions, and other factors. 
                For an official valuation, please consult with a licensed property valuer or real estate agent.
                </p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ **Oops! Something went wrong:** {str(e)}")
                st.info("Please try again or contact support if the problem persists.")
        
        else:
            # Show placeholder when button hasn't been clicked
            st.markdown("---")
            st.markdown("""
            <div style='background-color: #f0f0f0; padding: 40px; border-radius: 10px; text-align: center; border: 2px dashed #999;'>
            <p style='font-size: 18px; color: #666; margin: 0;'>
            👆 Fill in your property details on the left, then click <strong>"Get My Property Value"</strong> to see your estimate here!
            </p>
            </div>
            """, unsafe_allow_html=True)

def show_data_overview(df, df_long):
    """Data overview page"""
    st.header("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Properties", len(df))
    with col2:
        st.metric("Unique Suburbs", df['Suburb'].nunique())
    with col3:
        st.metric("Total Records", len(df_long))
    with col4:
        st.metric("Avg Price (2025)", f"${df[2025].mean():,.0f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Properties by Suburb")
        suburb_counts = df.drop_duplicates('Property_Address')['Suburb'].value_counts()
        fig = px.bar(
            x=suburb_counts.index,
            y=suburb_counts.values,
            labels={'x': 'Suburb', 'y': 'Number of Properties'},
            title="Number of Properties per Suburb"
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Price Distribution (2025)")
        fig = px.box(
            df,
            x='Suburb',
            y=2025,
            labels={'x': 'Suburb', 'y': 'Price ($)'},
            title="Price Distribution by Suburb in 2025"
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Property Features Distribution")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = px.histogram(df, x='Bedrooms', nbins=10, title="Bedrooms Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(df, x='Bathrooms', nbins=10, title="Bathrooms Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = px.histogram(df, x='Land_Area', nbins=30, title="Land Area Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Sample Data")
    st.dataframe(df.head(20), use_container_width=True)

def show_trends_analysis(df, df_long):
    """Trends and analysis page"""
    st.header("📈 Trends & Analysis")
    
    years = list(range(2015, 2026))
    
    # Average price trend
    st.subheader("Average Property Price Trend (2015-2025)")
    avg_prices = df[years].mean()
    fig = px.line(
        x=years,
        y=avg_prices.values,
        labels={'x': 'Year', 'y': 'Average Price ($)'},
        title="Average Property Price Over Time",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Suburb trends
    st.subheader("Price Trends by Suburb")
    df_suburb_long = df.melt(
        id_vars=['Suburb'],
        value_vars=years,
        var_name='Year',
        value_name='Price'
    )
    df_suburb_long['Year'] = df_suburb_long['Year'].astype(int)
    df_suburb_long = df_suburb_long.dropna(subset=['Price'])
    
    suburb_trend = (
        df_suburb_long
        .groupby(['Year', 'Suburb'])['Price']
        .mean()
        .reset_index()
    )
    
    fig = px.line(
        suburb_trend,
        x='Year',
        y='Price',
        color='Suburb',
        labels={'Price': 'Average Price ($)'},
        title="Historic Price Trend by Suburb",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Growth rate
    st.subheader("Year-on-Year Growth Rate by Suburb")
    suburb_growth = suburb_trend.copy()
    suburb_growth['Growth_Rate'] = (
        suburb_growth
        .groupby('Suburb')['Price']
        .pct_change() * 100
    )
    
    fig = px.line(
        suburb_growth,
        x='Year',
        y='Growth_Rate',
        color='Suburb',
        labels={'Growth_Rate': 'Growth Rate (%)'},
        title="Year-on-Year Price Growth by Suburb",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # CAGR
    st.subheader("Compound Annual Growth Rate (CAGR) by Suburb")
    cagr_list = []
    for suburb, group in df_suburb_long.groupby('Suburb'):
        start_price = group[group['Year'] == 2015]['Price'].mean()
        end_price = group[group['Year'] == 2025]['Price'].mean()
        if start_price > 0:
            cagr = ((end_price / start_price) ** (1/10) - 1) * 100
            cagr_list.append({'Suburb': suburb, 'CAGR (%)': cagr})
    
    cagr_df = pd.DataFrame(cagr_list).sort_values('CAGR (%)', ascending=False)
    fig = px.bar(
        cagr_df,
        x='Suburb',
        y='CAGR (%)',
        title="CAGR by Suburb (2015-2025)",
        color='CAGR (%)',
        color_continuous_scale='Viridis'
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature relationships
    st.subheader("Feature Relationships")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            df,
            x='Bedrooms',
            y=2025,
            labels={'y': 'Price ($)'},
            title="Bedrooms vs Price (2025)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            df,
            x='Land_Area',
            y=2025,
            labels={'y': 'Price ($)'},
            title="Land Area vs Price (2025)"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_cagr_explanation(df, suburb_cagr):
    """CAGR explanation page - User-friendly"""
    st.header("📊 How Our Predictions Work")
    
    st.markdown("""
    <div style='background-color: #e8f4f8; padding: 25px; border-radius: 10px; margin-bottom: 20px;'>
    <h3 style='color: #1f77b4; margin-top: 0;'>Understanding CAGR (Compound Annual Growth Rate)</h3>
    <p style='font-size: 16px; line-height: 1.8;'>
    Our prediction system uses <strong>CAGR</strong> - a simple and reliable method used by financial experts 
    to calculate average growth rates over time. Here's how it works:
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 What is CAGR?")
        st.markdown("""
        **CAGR (Compound Annual Growth Rate)** tells us the average yearly growth rate 
        of property prices in each suburb.
        
        **Example:** If a suburb's average property price was $1,000,000 in 2015 and 
        $1,500,000 in 2025, the CAGR would be approximately **4.14% per year**.
        
        This means prices in that suburb grew by about 4.14% each year on average.
        """)
    
    with col2:
        st.markdown("### 🏠 How We Use It")
        st.markdown("""
        1. **Calculate Base Price:** We find the average property price in your suburb for 2025
        2. **Adjust for Your Property:** We adjust this average based on your property's features 
           (bedrooms, bathrooms, land size, age)
        3. **Apply Growth Rate:** We use the suburb's CAGR to estimate prices for any year
        4. **Show You the Result:** You get an accurate estimate with a visual graph
        """)
    
    st.markdown("---")
    
    # Show CAGR for all suburbs
    st.subheader("📊 Growth Rates by Suburb (2015-2025)")
    st.markdown("**Here are the average annual growth rates we calculated for each suburb:**")
    
    cagr_display = pd.DataFrame(list(suburb_cagr.items()), columns=['Suburb', 'CAGR'])
    cagr_display['CAGR (%)'] = (cagr_display['CAGR'] * 100).round(2)
    cagr_display = cagr_display.sort_values('CAGR (%)', ascending=False)
    
    fig = px.bar(
        cagr_display,
        x='Suburb',
        y='CAGR (%)',
        title="Average Annual Growth Rate by Suburb",
        labels={'CAGR (%)': 'Average Annual Growth Rate (%)'},
        color='CAGR (%)',
        color_continuous_scale='Viridis'
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(
        cagr_display[['Suburb', 'CAGR (%)']].style.format({'CAGR (%)': '{:.2f}%'}),
        use_container_width=True
    )
    
    st.markdown("---")
    
    st.markdown("### ✅ Why CAGR is Reliable")
    st.markdown("""
    <div style='background-color: #d4edda; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745;'>
    <ul style='font-size: 15px; line-height: 2;'>
        <li><strong>Based on Real Data:</strong> Uses actual property sales from 2015-2025</li>
        <li><strong>Suburb-Specific:</strong> Each suburb has its own growth rate</li>
        <li><strong>Simple to Understand:</strong> Easy to explain and verify</li>
        <li><strong>Widely Used:</strong> Standard method used by real estate professionals</li>
        <li><strong>Transparent:</strong> You can see exactly how we calculate it</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 💡 Example Calculation")
    st.markdown("""
    **Let's say you want to know the price of a property in Nedlands for the year 2030:**
    
    1. **Base Price (2025):** Average property in Nedlands = $2,000,000
    2. **CAGR for Nedlands:** 5% per year (from 2015-2025 data)
    3. **Years from 2025:** 2030 - 2025 = 5 years
    4. **Calculation:** $2,000,000 × (1.05)⁵ = $2,552,563
    
    **Result:** Estimated price in 2030 = **$2,552,563**
    
    *Note: We also adjust this base price based on your specific property features (bedrooms, bathrooms, land size, etc.)*
    """)

def show_future_predictions_cagr(df, suburb_cagr):
    """Future predictions page - Using only CAGR method"""
    st.header("🔮 Future Market Forecast (2026-2035)")
    st.markdown("""
    <div style='background-color: #fff3cd; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #ffc107;'>
    <p style='margin: 0; font-size: 15px;'>
    <strong>📊 What you'll see:</strong> This page shows predicted property values for the next 10 years (2026-2035) 
    for all properties in our database. Predictions are based on each suburb's historical growth rate (CAGR) 
    calculated from 2015-2025 data.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display CAGR
    st.subheader("📈 Growth Rates by Suburb")
    st.markdown("**These are the average annual growth rates we use for predictions:**")
    
    cagr_display = pd.DataFrame(list(suburb_cagr.items()), columns=['Suburb', 'CAGR'])
    cagr_display['CAGR (%)'] = (cagr_display['CAGR'] * 100).round(2)
    cagr_display = cagr_display.sort_values('CAGR (%)', ascending=False)
    
    fig = px.bar(
        cagr_display,
        x='Suburb',
        y='CAGR (%)',
        title="Average Annual Growth Rate by Suburb (2015-2025)",
        labels={'CAGR (%)': 'Growth Rate (%)'},
        color='CAGR (%)',
        color_continuous_scale='Viridis'
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(
        cagr_display[['Suburb', 'CAGR (%)']].style.format({'CAGR (%)': '{:.2f}%'}),
        use_container_width=True
    )
    
    future_years = list(range(2026, 2036))
    
    # Get unique properties from original df
    unique_properties = df.drop_duplicates('Property_Address')
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    predictions = []
    total = len(unique_properties)
    
    with st.spinner("Generating future predictions for all properties..."):
        for idx, (_, row) in enumerate(unique_properties.iterrows()):
            # Get base price for 2025 using CAGR method
            base_price_2025 = predict_price_cagr(
                df, None, row['Bedrooms'], row['Bathrooms'], row['Garage'],
                row['Land_Area'], row['Build_Year'], 2025, row['Suburb']
            )
            
            if base_price_2025 is None:
                base_price_2025 = df[df['Suburb'] == row['Suburb']][2025].mean()
            
            # Apply CAGR for future years
            for year in future_years:
                if row['Suburb'] in suburb_cagr and suburb_cagr[row['Suburb']] != 0:
                    growth_factor = (1 + suburb_cagr[row['Suburb']]) ** (year - 2025)
                    predicted_price = base_price_2025 * growth_factor
                else:
                    predicted_price = base_price_2025
                
                predictions.append({
                    'Property_Address': row['Property_Address'],
                    'Suburb': row['Suburb'],
                    'Year': year,
                    'Predicted_Price': predicted_price
                })
            
            progress_bar.progress((idx + 1) / total)
            status_text.text(f"Processing property {idx + 1}/{total}...")
    
    progress_bar.empty()
    status_text.empty()
    
    future_df = pd.DataFrame(predictions)
    
    st.success(f"Generated {len(future_df)} predictions for {len(unique_properties)} properties!")
    
    # Visualization
    st.subheader("Predicted Price Trends by Suburb")
    suburb_avg = future_df.groupby(['Year', 'Suburb'])['Predicted_Price'].mean().reset_index()
    
    fig = px.line(
        suburb_avg,
        x='Year',
        y='Predicted_Price',
        color='Suburb',
        labels={'Predicted_Price': 'Predicted Price ($)'},
        title="Forecasted House Prices by Suburb (2026–2035)",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Top 5 most expensive houses forecast
    st.subheader("Top 5 Most Expensive Houses Forecast")
    top_houses = future_df.groupby('Property_Address')['Predicted_Price'].mean().nlargest(5).index
    subset = future_df[future_df['Property_Address'].isin(top_houses)]
    
    fig = px.line(
        subset,
        x='Year',
        y='Predicted_Price',
        color='Property_Address',
        labels={'Predicted_Price': 'Predicted Price ($)'},
        title="Top 5 Most Expensive Houses Forecast",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Average trend
    st.subheader("Average Predicted Price Trend")
    avg_trend = future_df.groupby('Year')['Predicted_Price'].mean().reset_index()
    avg_trend['Growth_Rate'] = avg_trend['Predicted_Price'].pct_change() * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(
            avg_trend,
            x='Year',
            y='Predicted_Price',
            labels={'Predicted_Price': 'Average Predicted Price ($)'},
            title="Average Predicted Price Over Time",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            avg_trend[1:],
            x='Year',
            y='Growth_Rate',
            labels={'Growth_Rate': 'Growth Rate (%)'},
            title="Year-over-Year Growth Rate",
            color='Growth_Rate',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Summary by suburb
    st.subheader("Summary Statistics by Suburb (2026-2035)")
    summary_stats = future_df.groupby('Suburb')['Predicted_Price'].agg([
        'mean', 'min', 'max', 'std'
    ]).round(0)
    summary_stats.columns = ['Mean Price', 'Min Price', 'Max Price', 'Std Dev']
    st.dataframe(summary_stats.style.format('${:,.0f}'), use_container_width=True)
    
    # Download predictions
    st.markdown("---")
    csv = future_df.to_csv(index=False)
    st.download_button(
        label="Download Predictions CSV",
        data=csv,
        file_name="future_predictions_2026_2035.csv",
        mime="text/csv"
    )

def show_batch_predictions(df, df_long, suburb_cagr):
    """Batch prediction page - Simple and easy to understand"""
    st.header("📊 Batch Predictions")
    
    st.markdown("""
    <div style='background-color: #e8f4f8; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <p style='font-size: 16px; line-height: 1.6;'>
    Upload a CSV or Excel file with property data to generate predictions for all properties at once. 
    Your file should have columns: <strong>Suburb, Bedrooms, Bathrooms, Garage, Land_Area, Build_Year</strong>
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Your CSV/Excel File",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a file with property data to generate batch predictions"
    )
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            if uploaded_file.name.endswith('.csv'):
                upload_df = pd.read_csv(uploaded_file)
            else:
                upload_df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File loaded successfully! Found {len(upload_df)} properties.")
            
            # Check required columns
            required_cols = ['Suburb', 'Bedrooms', 'Bathrooms', 'Garage', 'Land_Area']
            missing_cols = [col for col in required_cols if col not in upload_df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.info("**Required columns:** Suburb, Bedrooms, Bathrooms, Garage, Land_Area")
                return
            
            # Prediction year selector
            prediction_year = st.selectbox(
                "Select Prediction Year",
                options=list(range(2015, 2036)),
                index=10,  # Default to 2025
                help="Select the year for which you want predictions"
            )
            
            if st.button("🔮 Generate Predictions", type="primary", use_container_width=True):
                with st.spinner("Generating predictions..."):
                    predictions = []
                    progress_bar = st.progress(0)
                    
                    total = len(upload_df)
                    
                    for idx, row in upload_df.iterrows():
                        beds = row.get('Bedrooms', 3) if pd.notna(row.get('Bedrooms')) else 3
                        baths = row.get('Bathrooms', 2) if pd.notna(row.get('Bathrooms')) else 2
                        garage = row.get('Garage', 2) if pd.notna(row.get('Garage')) else 2
                        land_area = row.get('Land_Area', 500) if pd.notna(row.get('Land_Area')) else 500
                        build_year = row.get('Build_Year', 2000) if pd.notna(row.get('Build_Year')) else 2000
                        suburb = row.get('Suburb', 'Unknown')
                        
                        if suburb == 'Unknown' or pd.isna(suburb):
                            continue
                        
                        predicted_price = predict_price_cagr(
                            df, df_long, beds, baths, garage, land_area,
                            build_year, prediction_year, suburb
                        )
                        
                        if predicted_price is not None:
                            predictions.append({
                                'Suburb': suburb,
                                'Bedrooms': beds,
                                'Bathrooms': baths,
                                'Garage': garage,
                                'Land_Area': land_area,
                                'Build_Year': build_year,
                                'Predicted_Price': predicted_price
                            })
                        
                        progress_bar.progress((idx + 1) / total)
                    
                    progress_bar.progress(1.0)
                
                predictions_df = pd.DataFrame(predictions)
                
                if len(predictions_df) > 0:
                    st.success(f"✅ Generated {len(predictions_df)} predictions!")
                    
                    # Display results table
                    st.subheader("📋 Prediction Results")
                    st.dataframe(
                        predictions_df.style.format({
                            'Predicted_Price': '${:,.0f}',
                            'Land_Area': '{:.0f}'
                        }),
                        use_container_width=True,
                        height=400
                    )
                    
                    # Simple Visualizations
                    st.markdown("---")
                    st.subheader("📊 Simple Visualizations")
                    
                    # 1. Percentage Distribution by Suburb (Bar Graph)
                    st.markdown("### Property Distribution by Suburb (Percentage)")
                    suburb_counts = predictions_df['Suburb'].value_counts()
                    total_properties = len(predictions_df)
                    suburb_percentages = (suburb_counts / total_properties * 100).round(2)
                    
                    suburb_pct_df = pd.DataFrame({
                        'Suburb': suburb_percentages.index,
                        'Percentage (%)': suburb_percentages.values,
                        'Count': suburb_counts.values
                    }).sort_values('Percentage (%)', ascending=False)
                    
                    # Create bar chart with percentages
                    fig = px.bar(
                        suburb_pct_df,
                        x='Suburb',
                        y='Percentage (%)',
                        labels={'Percentage (%)': 'Percentage (%)', 'Suburb': 'Suburb'},
                        title=f"Property Distribution by Suburb - {prediction_year}",
                        color='Percentage (%)',
                        color_continuous_scale='Blues',
                        text='Percentage (%)',
                        hover_data=['Count']
                    )
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig.update_xaxes(tickangle=45)
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show table with percentages
                    display_df = suburb_pct_df.copy()
                    display_df['Percentage (%)'] = display_df['Percentage (%)'].apply(lambda x: f"{x:.2f}%")
                    st.dataframe(
                        display_df[['Suburb', 'Percentage (%)', 'Count']],
                        use_container_width=True,
                        height=300
                    )
                    
                    # 2. Average Price by Suburb
                    st.markdown("### Average Price by Suburb")
                    suburb_avg = predictions_df.groupby('Suburb')['Predicted_Price'].mean().reset_index()
                    suburb_avg = suburb_avg.sort_values('Predicted_Price', ascending=False)
                    
                    fig = px.bar(
                        suburb_avg,
                        x='Suburb',
                        y='Predicted_Price',
                        labels={'Predicted_Price': 'Average Price ($)', 'Suburb': 'Suburb'},
                        title=f"Average Predicted Price by Suburb - {prediction_year}",
                        color='Predicted_Price',
                        color_continuous_scale='Greens',
                        text='Predicted_Price'
                    )
                    fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
                    fig.update_xaxes(tickangle=45)
                    fig.update_yaxes(tickprefix="$", separatethousands=True)
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 3. Price Distribution
                    st.markdown("### Price Distribution")
                    fig = px.histogram(
                        predictions_df,
                        x='Predicted_Price',
                        nbins=20,
                        labels={'Predicted_Price': 'Predicted Price ($)', 'count': 'Number of Properties'},
                        title=f"Price Distribution - {prediction_year}",
                        color_discrete_sequence=['#1f77b4']
                    )
                    fig.update_xaxes(tickprefix="$", separatethousands=True)
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download predictions
                    st.markdown("---")
                    csv = predictions_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv,
                        file_name=f"batch_predictions_{prediction_year}.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("No predictions could be generated. Please check your data.")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.info("👆 Please upload a CSV or Excel file to get started with batch predictions.")

if __name__ == "__main__":
    main()



