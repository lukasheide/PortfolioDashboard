import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set up the sidebar for user inputs
st.sidebar.header('User Input Parameters')

def user_input_features():
    expected_return = st.sidebar.number_input('Expected Return', min_value=-1.0, max_value=1.0, value=0.07)
    volatility = st.sidebar.number_input('Volatility', min_value=0.0, max_value=1.0, value=0.1229)
    years = st.sidebar.number_input('Number of Years to Simulate', min_value=1, max_value=100, value=10)
    experiments = st.sidebar.number_input('Number of Experiments to Simulate', min_value=1, max_value=10000, value=5000)
    start_capital = st.sidebar.number_input('Start Capital', min_value=0.0, max_value=1000000.0, value=20000.0)
    savings_rate = st.sidebar.number_input('Savings Rate', min_value=0.0, max_value=1000000.0, value=10000.0)
    return expected_return, volatility, years, experiments, start_capital, savings_rate

expected_return, volatility, years, experiments, start_capital, savings_rate = user_input_features()

# Simulate all returns at once
np.random.seed(0)
all_yearly_returns = np.random.normal(expected_return, volatility, (experiments, years))

# Create a new page
page = st.sidebar.selectbox("Choose a page", ["Relative Returns", "Portfolio Value", "Portfolio Returns Over Time"])

if page == "Relative Returns":
    final_returns = np.prod(1 + all_yearly_returns, axis=1) - 1

    # Display the mean return
    mean_return = np.mean(final_returns)
    median_return = np.median(final_returns)
    negative_returns_share = (final_returns < 0).mean()

    # Create three columns for the metrics
    col1, col2, col3 = st.columns(3)

    # Display the metrics in each column
    with col1:
        st.metric(label="Mean Return", value=f'{mean_return:.2%}')
    with col2:
        st.metric(label="Median Return", value=f'{median_return:.2%}')
    with col3:
        st.metric(label="Share of Negative Returns", value=f'{negative_returns_share:.2%}')

    # Create a histogram of the returns
    n, bins, patches = plt.hist(final_returns, bins=50,
                                weights=np.ones(len(final_returns)) / len(final_returns),
                                edgecolor='black', alpha=0.7)
    # Change the color of the bars based on the return
    for patch in patches:
        if patch.get_x() < 0:
            patch.set_facecolor('red')
        else:
            patch.set_facecolor('green')

    plt.title(f'Total Return After {years} Years')
    plt.xlabel('')
    plt.ylabel('Proportion')
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
    st.pyplot(plt)

elif page == "Portfolio Value":
    final_returns = np.prod(1 + all_yearly_returns, axis=1) - 1

    # Calculate portfolio value
    portfolio_values = np.full((experiments, years+1), start_capital)
    for year in range(years):
        portfolio_values[:, year+1] = portfolio_values[:, year] * (1 + all_yearly_returns[:, year]) + savings_rate
    final_portfolio_values = portfolio_values[:, -1]

    # Display the mean and median portfolio value
    mean_portfolio_value = np.mean(final_portfolio_values)
    median_portfolio_value = np.median(final_portfolio_values)
    negative_returns_share = (final_returns < 0).mean()
    zero_return_capital = start_capital + savings_rate * years

    # Create three columns for the metrics
    col1, col2, col3, col4 = st.columns(4)

    # Display the metrics in each column
    with col1:
        st.metric(label="Mean Portfolio Value", value=f'{mean_portfolio_value:,.0f} €')
    with col2:
        st.metric(label="Median Portfolio Value", value=f'{median_portfolio_value:,.0f} €')
    with col3:
        st.metric(label="Zero Return Portfolio Value", value=f'{zero_return_capital:,.0f} €')
    with col4:
        st.metric(label="Share of Negative Returns", value=f'{negative_returns_share:.2%}')

    # Create a histogram of the portfolio values
    n, bins, patches = plt.hist(final_portfolio_values, bins=50,
                                weights=np.ones(len(final_portfolio_values)) / len(final_portfolio_values),
                                edgecolor='black', alpha=0.7)
    # Change the color of the bars based on the return
    for patch in patches:
        if patch.get_x() < zero_return_capital:
            patch.set_facecolor('red')
        else:
            patch.set_facecolor('green')

    plt.title(f'Total Portfolio Value After {years} Years')
    plt.xlabel('Value (€)')
    plt.ylabel('Proportion')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{0:.2%}'.format))  # Format y-axis as percentages
    st.pyplot(plt)

elif page == "Portfolio Returns Over Time":
    # Calculate relative returns for each year
    relative_returns = 1 + all_yearly_returns

    # Calculate overall portfolio return after each year
    overall_returns = np.cumprod(relative_returns, axis=1)

    # Calculate mean and quantiles for each year
    mean_overall_returns = np.mean(overall_returns, axis=0) - 1
    q90_overall_returns = np.percentile(overall_returns, 90, axis=0) - 1
    q80_overall_returns = np.percentile(overall_returns, 80, axis=0) - 1
    q20_overall_returns = np.percentile(overall_returns, 20, axis=0) - 1
    q10_overall_returns = np.percentile(overall_returns, 10, axis=0) - 1

    # Create line plots
    plt.plot(range(1, years + 1), mean_overall_returns, label='Mean', color='black')
    plt.plot(range(1, years + 1), q90_overall_returns, label='90% Quantile', color='grey', linestyle=':')
    plt.plot(range(1, years + 1), q80_overall_returns, label='80% Quantile', color='grey', linestyle=':')
    plt.plot(range(1, years + 1), q20_overall_returns, label='20% Quantile', color='grey', linestyle=':')
    plt.plot(range(1, years + 1), q10_overall_returns, label='10% Quantile', color='grey', linestyle=':')

    # Fill the area between the 0.8 and 0.2 quantiles
    plt.fill_between(range(1, years + 1), np.maximum(q80_overall_returns, 0), np.maximum(q20_overall_returns, 0),
                     color='green', alpha=0.2)
    plt.fill_between(range(1, years + 1), np.minimum(q80_overall_returns, 0), np.minimum(q20_overall_returns, 0),
                     color='red', alpha=0.2)
    plt.fill_between(range(1, years + 1), np.maximum(q90_overall_returns, 0), np.maximum(q10_overall_returns, 0),
                     color='green', alpha=0.1)
    plt.fill_between(range(1, years + 1), np.minimum(q90_overall_returns, 0), np.minimum(q10_overall_returns, 0),
                     color='red', alpha=0.1)

    plt.title('Relative Returns Over Time')
    plt.xlabel('Years')
    plt.ylabel('Relative Return')
    plt.xticks(range(1, years + 1))  # Display each year on the x-axis
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))  # Display y-axis in percentages
    plt.legend()
    st.pyplot(plt)