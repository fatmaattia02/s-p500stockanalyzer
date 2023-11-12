# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 22:24:28 2023

@author: Source
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st
import base64
from plotly.subplots import make_subplots
from stocknews import StockNews

# ==============================================================================
# HOT FIX FOR YFINANCE .INFO METHOD
# Ref: https://github.com/ranaroussi/yfinance/issues/1729
# ==============================================================================

import requests
import urllib


class YFinance:
    user_agent_key = "User-Agent"
    user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/58.0.3029.110 Safari/537.36")

    def __init__(self, ticker):
        self.yahoo_ticker = ticker

    def __str__(self):
        return self.yahoo_ticker

    def _get_yahoo_cookie(self):
        cookie = None

        headers = {self.user_agent_key: self.user_agent_value}
        response = requests.get("https://fc.yahoo.com",
                                headers=headers,
                                allow_redirects=True)

        if not response.cookies:
            raise Exception("Failed to obtain Yahoo auth cookie.")

        cookie = list(response.cookies)[0]

        return cookie

    def _get_yahoo_crumb(self, cookie):
        crumb = None

        headers = {self.user_agent_key: self.user_agent_value}

        crumb_response = requests.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers=headers,
            cookies={cookie.name: cookie.value},
            allow_redirects=True,
        )
        crumb = crumb_response.text

        if crumb is None:
            raise Exception("Failed to retrieve Yahoo crumb.")

        return crumb

    @property
    def info(self):
        # Yahoo modules doc informations :
        # https://cryptocointracker.com/yahoo-finance/yahoo-finance-api
        cookie = self._get_yahoo_cookie()
        crumb = self._get_yahoo_crumb(cookie)
        info = {}
        ret = {}

        headers = {self.user_agent_key: self.user_agent_value}

        yahoo_modules = ("assetProfile,"  # longBusinessSummary
                         "summaryDetail,"
                         "financialData,"
                         "indexTrend,"
                         "defaultKeyStatistics")

        url = ("https://query1.finance.yahoo.com/v10/finance/"
               f"quoteSummary/{self.yahoo_ticker}"
               f"?modules={urllib.parse.quote_plus(yahoo_modules)}"
               f"&ssl=true&crumb={urllib.parse.quote_plus(crumb)}")

        info_response = requests.get(url,
                                     headers=headers,
                                     cookies={cookie.name: cookie.value},
                                     allow_redirects=True)

        info = info_response.json()
        info = info['quoteSummary']['result'][0]

        for mainKeys in info.keys():
            for key in info[mainKeys].keys():
                if isinstance(info[mainKeys][key], dict):
                    try:
                        ret[key] = info[mainKeys][key]['raw']
                    except (KeyError, TypeError):
                        pass
                else:
                    ret[key] = info[mainKeys][key]

        return ret

# ==============================================================================
# Header
# ==============================================================================


def download_csv(dataframe, filename='download.csv'):
    csv = dataframe.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href


def render_sidebar():
    # Create two columns in the sidebar
    data_source_col1, data_source_col2 = st.sidebar.columns([1, 1])

    # Display the data source information
    data_source_col1.write("Data source:")
    data_source_col2.image('./img/yahoo_finance.png', width=100)
    global ticker_list
    # Get the list of stock tickers from S&P500
    ticker_list = pd.read_html(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
    global ticker

    # Add the ticker selection box
    ticker = st.sidebar.selectbox("Select Ticker", ticker_list)
    global start_date, end_date

    # Add the date input boxes for start and end dates
    start_date = st.sidebar.date_input(
        "Select Start Date", datetime.today().date() - timedelta(days=30))
    end_date = st.sidebar.date_input(
        "Select End Date", datetime.today().date())

    st.sidebar.subheader("")

    # Add an "Update" button and call the update_stock_data function when clicked
    if st.sidebar.button("Update"):
        if ticker:
            st.write(f"Updating stock data for {ticker}...")
            stock_data = yf.download(ticker)

    stock_data = yf.download(ticker)
    st.sidebar.markdown(download_csv(stock_data, ticker), unsafe_allow_html=True)


    return ticker, start_date, end_date


# ==============================================================================
# Tab 1
# ==============================================================================

@st.cache_data
def GetCompanyInfo(ticker):
    return YFinance(ticker).info


@st.cache_data
def GetStockData(ticker, start_date, end_date):
    stock_df = yf.Ticker(ticker).history(start=start_date, end=end_date)
    stock_df.reset_index(inplace=True)
    stock_df['Date'] = stock_df['Date'].dt.date
    return stock_df


@st.cache_data
def GetStockData2(ticker, start_date, end_date, select_period):
    if select_period == 'Selected Range':
        stock_price = yf.Ticker(ticker).history(
            start=start_date, end=end_date, interval='1d').reset_index()
    else:
        stock_price = yf.Ticker(ticker).history(
            period=select_period, interval='1d').reset_index()

    return stock_price


def render_tab1():

    col4, col5, col6, col7 = st.columns((1, 1, 1, 1))
    company_data = GetCompanyInfo(ticker)

    # Fetch historical stock prices using yfinance
    stock_data = yf.Ticker(ticker).history(period='1y')

    # Extract relevant columns (Date and Close)
    stock_price = stock_data[['Close']].reset_index()
    # Inside the col4 block where you display the price metric
    yesterday_close_price = stock_price.iloc[-2]['Close'] if len(
        stock_price) >= 2 else None
    with col4:
        current_price = company_data.get("currentPrice", 'N/A')

        if yesterday_close_price is not None:
            delta_price = (
                (current_price - yesterday_close_price) / yesterday_close_price) * 100
            delta_price_formatted = f"({delta_price:.2f}%)"
        else:
            delta_price_formatted = 'N/A'

        st.metric(label="Price", value=current_price,
                  delta=delta_price_formatted)

    with col5:
        market_cap_value = company_data.get('marketCap', 'N/A')
        formatted_market_cap = "{:,.3f}B".format(
            market_cap_value / 1e9) if market_cap_value != 'N/A' else 'N/A'
        st.metric(label="Market Cap", value=formatted_market_cap)

    with col6:
        st.metric(label="Currency", value=company_data.get('currency'))

    with col7:
        period = st.selectbox(
            "Select Period", ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "MAX"], index=3)

    st.write("")
    st.write("")

    col1, col2, col3 = st.columns([1, 1, 2])

    if ticker != '':
        info = GetCompanyInfo(ticker)

        first_half_keys = {'previousClose': 'Previous Close',
                           'open': 'Open',
                           'bid': 'Bid',
                           'ask': 'Ask',
                           'fiftyTwoWeekLow': '52 Week Range',
                           'volume': 'Volume',
                           'averageVolume': 'Average Volume'}

        second_half_keys = {'beta': 'Beta',
                            'trailingPE': 'PE Ratio (TTM)',
                            'trailingEps': 'EPS (TTM)',
                            'earningsDate': 'Earnings Date',
                            'dividendRate': 'Forward Dividend',
                            'exDividendDate': 'Ex-Dividend Date',
                            'oneYearTargetPrice': '1y Target Est'}
    with col1:
        st.markdown("<br>" * 1, unsafe_allow_html=True)  # Create 2 empty lines
        for key in first_half_keys:
            if key == 'fiftyTwoWeekLow':
                st.write(
                    f"**{first_half_keys[key]}:** {info.get(key, 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}")
            elif key in {'volume', 'averageVolume'}:
                st.write(
                    f"**{first_half_keys[key]}:** {info.get(key, 'N/A'):,.0f}")
            else:
                st.write(f"**{first_half_keys[key]}:** {info.get(key, 'N/A')}")

    with col2:
        st.markdown("<br>" * 1, unsafe_allow_html=True)  # Create 2 empty lines
        for key in second_half_keys:
            if key == 'exDividendDate':
                ex_dividend_date_timestamp = info.get(key)
                if ex_dividend_date_timestamp is not None:
                    ex_dividend_date = datetime.utcfromtimestamp(
                        ex_dividend_date_timestamp).strftime('%b %d, %Y')
                    st.write(
                        f"**{second_half_keys[key]}:** {ex_dividend_date}")
                else:
                    st.write(f"**{second_half_keys[key]}:** N/A")
            else:
                st.write(
                    f"**{second_half_keys[key]}:** {info.get(key, 'N/A')}")

    with col3:

        @st.cache_data
        def GetStockData(ticker, duration):
            end_date = datetime.now()
            if period == "1M":
                start_date = end_date - timedelta(days=30)
            elif period == "3M":
                start_date = end_date - timedelta(days=90)
            elif period == "6M":
                start_date = end_date - timedelta(days=180)
            elif period == "YTD":
                start_date = datetime(end_date.year, 1, 1)
            elif period == "1Y":
                start_date = end_date - timedelta(days=365)
            elif period == "3Y":
                start_date = end_date - timedelta(days=3*365)
            elif period == "5Y":
                start_date = end_date - timedelta(days=5*365)
            stock_df = yf.Ticker(ticker).history(
                start=start_date, end=end_date, interval="1d")
            stock_df.reset_index(inplace=True)
            stock_df['Date'] = stock_df['Date'].dt.date
            return stock_df

        if ticker != '':
            stock_price = GetStockData(ticker, period)
            st.area_chart(data=stock_price, x='Date', y='Close',
                          color="#ADD8E6", use_container_width=True)


# ==============================================================================
# Tab 2
# ==============================================================================
@st.cache_data
def get_stock_data(ticker, start_date, end_date, selected_duration, selected_interval):
    if selected_duration == 'MAX':
        return yf.Ticker(ticker).history(period='max', interval=selected_interval)
    elif selected_duration == 'Selected Range':
        return yf.Ticker(ticker).history(start=start_date, end=end_date, interval='1d').reset_index()
    else:
        return yf.Ticker(ticker).history(period=selected_duration, interval=selected_interval)


def render_tab2():
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        selected_duration = st.selectbox("Select Period", [
                                         'Selected Range', '1M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y', 'MAX'], index=0)

    with col2:
        interval_options = ['1d', '1mo', '1y']
        selected_interval = st.selectbox("Select Interval", interval_options)

    with col3:
        chart_type_options = ['Line', 'Candle']
        selected_chart_type = st.selectbox(
            "Select Chart Type", chart_type_options)

    show_sma = st.checkbox("Show Moving Average", value=True)
    show_volume = st.checkbox("Show Volume", value=True)

    stock_price = get_stock_data(
        ticker, start_date, end_date, selected_duration, selected_interval)

    M_Average = pd.DataFrame(yf.Ticker(ticker).history(
        period='max', interval=selected_interval)).reset_index()
    M_Average['M50'] = M_Average['Close'].rolling(50).mean()
    M_Average = M_Average[M_Average['Date'].isin(stock_price.index)]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if show_sma:
        fig.add_trace(go.Scatter(name='Moving Average/ 50 days',
                      x=M_Average['Date'], y=M_Average['M50'], marker_color='turquoise'), secondary_y=True)

    if show_volume and not stock_price['Volume'].empty and not stock_price['Volume'].isnull().all():
        max_volume = max(stock_price['Volume']) * 1.3
        fig.add_trace(go.Bar(name='Volume of shares',
                      x=stock_price.index, y=stock_price['Volume']))
        fig.update_layout(yaxis1=dict(range=[0, max_volume]))

    if selected_chart_type == 'Line':
        fig.add_trace(go.Scatter(name='Close Value', x=stock_price.index,
                      y=stock_price['Close']), secondary_y=True)

    elif selected_chart_type == 'Candle':
        fig.add_trace(go.Candlestick(name='Stock value (Open, High, Low, Close)', x=stock_price.index,
                                     open=stock_price['Open'],
                                     high=stock_price['High'],
                                     low=stock_price['Low'],
                                     close=stock_price['Close']), secondary_y=True)

    fig.update_layout(xaxis_rangeslider_visible=True, autosize=True)
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# Tab 3
# ==============================================================================


def render_tab3():
    # Create a Streamlit form for financial statement selection
    financial_form = st.form(key='financial')

    # Dropdown for selecting the type of financial info
    selected_statement_type = financial_form.selectbox("Select financial statement type", [
                                                       'Income Statement', 'Balance Sheet', 'Cash Flow'])

    # Radio buttons for selecting the time frame (Annual or Quarterly)
    selected_time_frame = financial_form.radio(
        '', ['Annual', 'Quarterly'], horizontal=True)

    # Form submission button
    financial_form.form_submit_button('Show statements')

    # Function to retrieve company financials based on user selections
    def get_company_financials(ticker, statement_type, time_frame):
        financial_func_map = {
            'Balance Sheet': yf.Ticker(ticker).balance_sheet if time_frame == 'Annual' else yf.Ticker(ticker).quarterly_balance_sheet,
            'Income Statement': yf.Ticker(ticker).financials if time_frame == 'Annual' else yf.Ticker(ticker).quarterly_financials,
            'Cash Flow': yf.Ticker(ticker).cashflow if time_frame == 'Annual' else yf.Ticker(ticker).quarterly_cashflow
        }
        return pd.DataFrame(financial_func_map.get(statement_type, {}))

    # Get financial data for the selected options, fill NaN values with 0, and format the date columns
    financial_data = get_company_financials(
        ticker, selected_statement_type, selected_time_frame).fillna(0)
    financial_data.columns = pd.to_datetime(financial_data.columns).strftime(
        '%m/%d/%Y')  # Convert to datetime and format

    # Calculate the Trailing Twelve Months (TTM) column for Annual reports
    if selected_time_frame == 'Annual' and 'TTM' not in financial_data.columns:
        financial_data['TTM'] = financial_data.iloc[:, -4:].sum(axis=1)

    financial_data = financial_data.style.format("${:0,.2f}", na_rep="N/A")

    # Apply custom CSS class for highlighting
    financial_data = financial_data.applymap(
        lambda x: 'background-color: #ffcccc' if pd.isnull(x) else '')

    # Display the header and financial data in a table
    st.header(selected_time_frame + ' ' + selected_statement_type)
    st.table(financial_data)


# ==============================================================================
# Tab 4
# ==============================================================================

def render_simulation_form():
    # Use st.form() to wrap the form elements
    with st.form(key='simulation_form'):
        # Use st.columns() to display elements in a horizontal layout
        col1, col2 = st.columns(2)

        # Dropdown for selecting the number of simulations
        simulations = col1.selectbox("Select Number of Simulations", [
                                     200, 500, 1000], key='simulations')

        # Dropdown for selecting the time horizon
        time_horizon = col2.selectbox("Select The Time Horizon (in days)", [
                                      30, 60, 90], key='time_horizon')

        # Button to run the simulation
        run_simulation = st.form_submit_button('Run Simulation')

    return simulations, time_horizon, run_simulation


def run_monte_carlo_simulation(ticker, simulations, time_horizon):
    # Create an empty DataFrame to store simulation results
    simulation_df = pd.DataFrame()

    # Get the Apple stock price history from Yahoo finance
    stock_price = yf.Ticker(ticker).history(
        start='2018-01-01', end='2018-11-16')
    close_price = stock_price['Close']

    # Calculate daily returns and volatility
    daily_return = close_price.pct_change()
    daily_volatility = np.std(daily_return)

    # Run Monte Carlo simulation
    for i in range(simulations):
        next_price = []
        last_price = close_price[-1]

        # Generate future stock prices based on normal distribution
        for j in range(time_horizon):
            future_return = np.random.normal(0, daily_volatility)
            future_price = last_price * (1 + future_return)
            next_price.append(future_price)
            last_price = future_price

        # Create a DataFrame for each simulation and concatenate them
        next_price_df = pd.Series(next_price).rename('sim' + str(i))
        simulation_df = pd.concat([simulation_df, next_price_df], axis=1)

    return close_price, simulation_df


def plot_simulation_results(close_price, simulation_df, time_horizon, ticker):
    # Plot the Monte Carlo simulation results
    fig = plt.figure(figsize=(15, 10))
    plt.plot(simulation_df)
    # Add a horizontal line for the initial stock price
    plt.axhline(y=close_price[-1], color='red')
    plt.title(
        f"Monte Carlo simulation for {ticker} stock price in the next {time_horizon} days")
    plt.xlabel('Day')
    plt.ylabel('Price')
    legend_text = ['Current stock price is: ' +
                   str(np.round(close_price[-1], 2))]
    plt.legend(legend_text)
    plt.gca().get_legend().get_lines()[0].set_color(
        'red')  # Set legend color to red
    st.pyplot(fig)


def render_tab4():
    simulations, time_horizon, run_simulation = render_simulation_form()

    if run_simulation:
        close_price, simulation_df = run_monte_carlo_simulation(
            ticker, simulations, time_horizon)
        plot_simulation_results(
            close_price, simulation_df, time_horizon, ticker)


# ==============================================================================
# Tab 5
# ==============================================================================

def render_tab5():
    st.header(f'News for {ticker}')

    stock = [ticker]
    # Create a StockNews object for the selected stock
    sn = StockNews(stock, save_news=False)

    # Read the RSS feed for the selected stock and get the news data as a DataFrame
    df_news = sn.read_rss()
    # Check if there are articles available
    if df_news.empty:
        st.write("No news articles available.")
        return

    # Initialize variables to keep track of sentiments
    total_sentiment = 0

    # Determine the number of articles to iterate over
    num_articles = min(10, len(df_news))

    # Accumulate sentiments for each article
    for i in range(num_articles):
        title_sentiment = df_news['sentiment_title'][i]
        news_sentiment = df_news['sentiment_summary'][i]
        total_sentiment += (title_sentiment + news_sentiment)

    # Calculate the average sentiment only if there are articles
    if num_articles > 0:
        # Divide by 2 for title and news
        avg_sentiment = total_sentiment / (num_articles * 2)
        # Display the overall sentiment with a symbol (green if positive, red if negative)
        sentiment_display = '😌' if avg_sentiment >= 0.5 else '😥'

        # Add a border to the overall sentiment display
        st.markdown(
            f'<div style="border: 2px solid #e4e4e4; padding: 10px; border-radius: 5px; margin-bottom: 20px;">'
            f'<h3 style="margin: 0; padding: 0;">Overall Sentiment: {avg_sentiment:.2%} {sentiment_display}</h3>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Iterate over the articles and display certain attributes
    for i in range(num_articles):
        st.subheader(f"**{df_news['title'][i]}**")
        published_date = df_news['published'][i]
        parsed_date = datetime.strptime(
            published_date, "%a, %d %b %Y %H:%M:%S %z")
        formatted_date = parsed_date.strftime("%m-%d-%Y")
        st.caption(f"Date Published: {formatted_date}")
        st.write("    ", df_news['summary'][i])

    else:
        st.write("No more news articles available.")


# ==============================================================================
# Main body
# ==============================================================================
st.set_page_config(layout="wide", page_title="My Streamlit App",
                   page_icon=":chart_with_upwards_trend:")
st.title("S&P 500 Stocks Analyzer")
# Render the header
render_sidebar()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Company profile", "Chart", "Financials", "Monte Carlo", "News"])
with tab1:
    render_tab1()
with tab2:
    render_tab2()
with tab3:
    render_tab3()
with tab4:
    render_tab4()
with tab5:
    render_tab5()


###############################################################################
# END
###############################################################################
