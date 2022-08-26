import lifetimes
import streamlit as st
import pandas as pd


header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()

with header:
    st.title('Customer Lifetime Value Prediction App!')
    st.image("https://thumbs.dreamstime.com/z/web-202480731.jpg", use_column_width = True)

with dataset:
    st.header('Retail Dataset')

    df = pd.read_csv('data/Retail_cleaned_dataset.csv')
    st.write(df.head())

    #country_dist = pd.DataFrame(df['Country'].value_counts())
    #st.bar_chart(country_dist)

import altair as alt
import time

import warnings
warnings.filterwarnings("ignore")
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from lifetimes.plotting import plot_frequency_recency_matrix
from lifetimes.plotting import plot_probability_alive_matrix
from lifetimes.plotting import plot_period_transactions
from lifetimes.utils import calibration_and_holdout_data
from lifetimes import ParetoNBDFitter
from lifetimes.plotting import plot_history_alive
from sklearn.metrics import mean_squared_error, r2_score

#st.image("https://thumbs.dreamstime.com/z/web-202480731.jpg", use_column_width = True)
data = st.file_uploader("File Uploader")
st.sidebar.title("Prediction of Customer Lifetime Value")
st.sidebar.markdown("This application is a Streamlit dashboard used "
            "to predict Customer Lifetime Value  🐦")

st.sidebar.title("Input Features :pencil:")
st.sidebar.markdown("""
[Example CSV Input File](https://raw.githubusercontent.com/mukulsinghal001/customer-lifetime-prediction-using-python/main/model_deployment/sample_file.csv)
    """)
days = st.sidebar.slider("Select The No. Of Days", min_value = 1, max_value = 365, step = 1, value = 30)

profit = st.sidebar.slider("Select the Profit Margin", min_value = 0.01, max_value = 0.09, step = 0.01, value = 0.05)
t_days = days

profit_m = profit

slider_data = {
    "Days": t_days,
    "Profit": profit_m
}

st.sidebar.markdown("""
### Selected Input Features :page_with_curl:
    """)

features = pd.DataFrame(slider_data, index = [0])

st.sidebar.write(features)

st.sidebar.markdown("""
Before uploading the file, please select the input features first.
Also, please make sure the columns are in proper format. For reference you can download the [dummy data](https://raw.githubusercontent.com/mukulsinghal001/customer-lifetime-prediction-using-python/main/model_deployment/sample_file.csv).
**Note:** Only Use "CSV" File.
    """)
if data is not None:

    def load_data(data, day=t_days, profit=profit_m):

        input_data = pd.read_csv(data)

        input_data = pd.DataFrame(input_data.iloc[:, 1:])

        # Pareto Model

        pareto_model = lifetimes.ParetoNBDFitter(penalizer_coef=0.1)
        pareto_model.fit(input_data["frequency"], input_data["recency"], input_data["T"])
        input_data["p_not_alive"] = 1 - pareto_model.conditional_probability_alive(input_data["frequency"],
                                                                                   input_data["recency"],
                                                                                   input_data["T"])
        input_data["p_alive"] = pareto_model.conditional_probability_alive(input_data["frequency"],
                                                                           input_data["recency"], input_data["T"])
        t = days
        input_data["predicted_purchases"] = pareto_model.conditional_expected_number_of_purchases_up_to_time(t,
                                                                                                             input_data[
                                                                                                                 "frequency"],
                                                                                                             input_data[
                                                                                                                 "recency"],
                                                                                                             input_data[
                                                                                                                 "T"])

        # Gamma Gamma Model

        idx = input_data[(input_data["frequency"] <= 0.0)]
        idx = idx.index
        input_data = input_data.drop(idx, axis=0)
        m_idx = input_data[(input_data["monetary_value"] <= 0.0)].index
        input_data = input_data.drop(m_idx, axis=0)

        input_data.reset_index().drop("index", axis=1, inplace=True)

        ggf_model = lifetimes.GammaGammaFitter(penalizer_coef=0.1)

        ggf_model.fit(input_data["frequency"], input_data["monetary_value"])

        input_data["expected_avg_sales_"] = ggf_model.conditional_expected_average_profit(input_data["frequency"],
                                                                                          input_data["monetary_value"])

        input_data["predicted_clv_1month"] = ggf_model.customer_lifetime_value(pareto_model, input_data["frequency"],
                                                                        input_data["recency"], input_data["T"],
                                                                        input_data["monetary_value"], time=30, freq='D',
                                                                        discount_rate=0.01)

        input_data["profit_margin"] = input_data["predicted_clv_1month"] * profit

        input_data = input_data.reset_index().drop("index", axis=1)

        # K-Means Model

        col = ["predicted_purchases", "expected_avg_sales_", "predicted_clv_1month", "profit_margin"]

        new_df = input_data[col]
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaled = scaler.fit_transform(new_df)
        kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42).fit(scaled)
        df_segm_kmeans = new_df.copy()
        df_segm_kmeans['Segment K-means'] = kmeans.labels_
        input_data = pd.concat([input_data, df_segm_kmeans['Segment K-means']], axis=1)
        df_segm_analysis = df_segm_kmeans.groupby(['Segment K-means']).mean()
        df_segm_analysis.rename({0: 'Low',
                                 1: 'High',
                                 2: 'Medium'
                                 })
        df_segm_kmeans['Labels'] = df_segm_kmeans['Segment K-means'].map({0: 'Low',
                                                                          1: 'High',
                                                                          2: 'Medium',
                                                                          })
        input_data = pd.concat([input_data, df_segm_kmeans['Labels']], axis=1)
        #input_data["Label"] = input_data["Label"].map(df_segm_kmeans['Labels'])

#saving the input data in the separate variable

        download = input_data

        st.write(input_data)

        #adding a count bar chart

        fig = alt.Chart(input_data).mark_bar().encode(

            y = "Labels:N",
            x = "count(Labels):Q"

            )

        #adding an annotation to the chart

        text = fig.mark_text(

            align = "left",
            baseline = "middle",
            dx = 3

            ).encode(

            text = "count(Labels):Q"

            )


        chart = (fig+text)

        #showing the chart

        st.altair_chart(chart, use_container_width = True)


    #calling the function

    st.markdown("""
        ## Customer Lifetime Prediction Result :bar_chart:
        """)

    load_data(data)

else:
    st.text("Please Upload the CSV File")


