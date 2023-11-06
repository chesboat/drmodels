# app.py, run with 'streamlit run app.py'
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


st.set_page_config(page_title="DR Models",
                page_icon=":bar_chart:",
                layout="wide"
)

df = pd.read_csv("data.csv")  # read a CSV file inside the 'data" folder next to 'app.py'
# df = pd.read_excel(...)  # will work for Excel files


##st.dataframe(df)

df_original = df.copy()

# -----sidebar -------
# Add 'All' option to the selectboxes
st.sidebar.subheader("ADR")

# Get the selected values
adr_model_selected = st.sidebar.selectbox(
    "Select ADR Model",
    options=['All'] + list(df["ADR Model"].unique()),
    index=0,
    key="adr_model"
)
if adr_model_selected != 'All':
    df = df[df['ADR Model'] == adr_model_selected]

adr_confirmation = st.sidebar.selectbox(
    "Select Confirmation",
    options=['All'] + list(df["ADR Confirmation"].unique()),
    index=0,
    key="adr_confirmation"
)
if adr_confirmation != 'All':
    df = df[df['ADR Confirmation'] == adr_confirmation]

adr_box_color = st.sidebar.selectbox(
    "Select Box Color",
    options=['All'] + list(df["ADR Box Color"].unique()),
    index=0,
    key="adr_box_color"
)
if adr_box_color != 'All':
    df = df[df['ADR Box Color'] == adr_box_color]

adr_true = st.sidebar.selectbox(
    "Select Whether the Session is T/F",
    options=['All'] + list(df["ADR True"].unique()),
    index=0,
    key="adr_true"
)
if adr_true != 'All':
    df = df[df['ADR True'] == adr_true]

st.sidebar.markdown("---")

st.sidebar.subheader("ODR")

odr_model = st.sidebar.selectbox(
    "Select ODR Model",
    options=['All'] + list(df["ODR Model"].unique()),
    index=0,
    key="odr_model"
)
if odr_model != 'All':
    df = df[df['ODR Model'] == odr_model]

odr_confirmation = st.sidebar.selectbox(
    "Select Confirmation",
    options=['All'] + list(df["ODR Confirmation"].unique()),
    index=0,
    key="odr_confirmation"
)
if odr_confirmation != 'All':
    df = df[df['ODR Confirmation'] == odr_confirmation]

odr_box_color = st.sidebar.selectbox(
    "Select Box Color",
    options=['All'] + list(df["ODR Box Color"].unique()),
    index=0,
    key="odr_box_color"
)
if odr_box_color != 'All':
    df = df[df['ODR Box Color'] == odr_box_color]

odr_true = st.sidebar.selectbox(
    "Select Whether the ODR Session is T/F",
    options=['All'] + list(df["ODR True"].unique()),
    index=0,
    key="odr_true"
)
if odr_true != 'All':
    df = df[df['ODR True'] == odr_true]

st.sidebar.markdown("---")

st.sidebar.subheader("RDR")


rdr_model = st.sidebar.selectbox(
    "Select RDR Model",
    options=['All'] + list(df["RDR Model"].unique()),
    index=0,
    key="rdr_model"
)
if rdr_model != 'All':
    df = df[df['RDR Model'] == rdr_model]

rdr_confirmation = st.sidebar.selectbox(
    "Select Confirmation",
    options=['All'] + list(df["RDR Confirmation"].unique()),
    index=0,
    key="rdr_confirmation"
)
if rdr_confirmation != 'All':
    df = df[df['RDR Confirmation'] == rdr_confirmation]

rdr_box_color = st.sidebar.selectbox(
    "Select Box Color",
    options=['All'] + list(df["RDR Box Color"].unique()),
    index=0,
    key="rdr_box_color"
)
if rdr_box_color != 'All':
    df = df[df['RDR Box Color'] == rdr_box_color]

rdr_true = st.sidebar.selectbox(
    "Select Whether the RDR Session is T/F",
    options=['All'] + list(df["RDR True"].unique()),
    index=0,
    key="rdr_true"
)
if rdr_true != 'All':
    df = df[df['RDR True'] == rdr_true]

# Check if 'All' is selected for each filter
query = " & ".join([f"`{col}` == '{var}'" for col, var in zip(["ADR Model", "ADR Confirmation", "ADR Box Color", "ADR True"], [adr_model_selected, adr_confirmation, adr_box_color, adr_true]) if var != 'All'])

df_selection = df.query(query) if query else df

## st.dataframe(df_selection)

#----- MAIN PAGE -----

## ---------RDR Tab --------------##
rdr_tab = st.expander("RDR Tab", expanded=True)

with rdr_tab:
    col1, col2 = st.columns(2)

    with col1:
        # RDR Model chartz
        # Add a header
        # Add a header with smaller font size
        st.markdown("#### Remove RDR Models")
        # Get unique RDR models
        rdr_models = df['RDR Model'].unique()

        # Create a dictionary to store checkbox status for each RDR model
        checkbox_status = {}

        # Determine the number of columns you want
        num_columns = 3

        # Create columns for checkboxes
        cols = st.columns(num_columns)

        for i, model in enumerate(rdr_models):
            checkbox_status[model] = cols[i % num_columns].checkbox(f'{model}', key=model)

        # Filter the dataframe based on the checkbox status
        df_filtered = df[~df['RDR Model'].isin([model for model, status in checkbox_status.items() if status])]

        # Now use df_filtered to create your chart
        rdr_model_counts = df_filtered['RDR Model'].value_counts()
        rdr_model_counts_unfiltered = df_original['RDR Model'].value_counts()

                # Get the count of the datasets that match the currently selected filters
        count = len(df_filtered)

        # Display the count
        st.markdown(f"**Count of datasets that match the currently selected filters: {count}**")

        total_filtered = rdr_model_counts.sum()
        total_unfiltered = rdr_model_counts_unfiltered.sum()
        percentages_unfiltered = rdr_model_counts_unfiltered / total_unfiltered * 100

        # Create a series with all categories from unfiltered data, fill missing categories in filtered data with 0
        percentages = rdr_model_counts / total_filtered * 100

        fig = plt.figure(figsize=(5.5, 3.5))
        fig.patch.set_facecolor('#0d1117')
        ax = plt.gca()
        ax.set_facecolor('#0d1117')

        def to_percent(y, position):
            return f'{y:.0f}%'

        formatter = FuncFormatter(to_percent)
        ax.yaxis.set_major_formatter(formatter)

        for spine in ax.spines.values():
            spine.set_visible(False)

        # Plot unfiltered data first
        bars_unfiltered = plt.bar(percentages_unfiltered.index, percentages_unfiltered, color='#83c9ff', alpha=0.3, width=0.7, zorder=2)

        # Plot filtered data on top
        bars = plt.bar(percentages.index, percentages, color='#83c9ff', width=0.3, zorder=3)

        for bar, percentage in zip(bars, percentages):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{round(percentage, 2)}%", ha='center', va='bottom', color='white', fontsize=6, weight='bold')

        plt.title('RDR Model', color='white', fontsize=16)

        for label in ax.get_xticklabels():
            label.set_weight('bold')

        for label in ax.get_yticklabels():
            label.set_weight('bold')

        plt.tick_params(colors='white', labelsize=6)
        plt.grid(axis='y', linestyle=':', color='white', zorder=1, alpha=0.5)

        st.pyplot(fig)



    with col2:
        # For RDR Max Extension
        st.subheader("RDR Max Extension")
        st.bar_chart(df['RDR Max Extension'].value_counts())

    col3, col4, col5 = st.columns(3)

with col3:
    # RDR True chart
    rdr_true_counts = df_filtered['RDR True'].value_counts()
    rdr_true_counts_unfiltered = df_original['RDR True'].value_counts()

    total_filtered = rdr_true_counts.sum()
    total_unfiltered = rdr_true_counts_unfiltered.sum()
    percentages_unfiltered = rdr_true_counts_unfiltered / total_unfiltered * 100

    # Create a series with all categories from unfiltered data, fill missing categories in filtered data with 0
    percentages = rdr_true_counts / total_filtered * 100

    fig = plt.figure(figsize=(5.5, 3.5))
    fig.patch.set_facecolor('#0d1117')
    ax = plt.gca()
    ax.set_facecolor('#0d1117')

    def to_percent(y, position):
        return f'{y:.0f}%'

    formatter = FuncFormatter(to_percent)
    ax.yaxis.set_major_formatter(formatter)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Plot unfiltered data first
    bars_unfiltered = plt.bar(percentages_unfiltered.index, percentages_unfiltered, color='#83c9ff', alpha=0.3, width=0.5, zorder=2)

    # Plot filtered data on top
    bars = plt.bar(percentages.index, percentages, color='#83c9ff', width=0.2, zorder=3)

    for bar, percentage in zip(bars, percentages):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{round(percentage, 2)}%", ha='center', va='bottom', color='white', fontsize=8, weight='bold')

    plt.title('RDR True', color='white', fontsize=16)

    for label in ax.get_xticklabels():
        label.set_weight('bold')

    for label in ax.get_yticklabels():
        label.set_weight('bold')

    plt.tick_params(colors='white', labelsize=8)
    plt.grid(axis='y', linestyle=':', color='white', zorder=1, alpha=0.5)

    st.pyplot(fig)



with col4:
    # RDR Box Color chart
    rdr_box_color_counts = df_filtered['RDR Box Color'].value_counts()
    rdr_box_color_counts_unfiltered = df_original['RDR Box Color'].value_counts()

    total_filtered = rdr_box_color_counts.sum()
    total_unfiltered = rdr_box_color_counts_unfiltered.sum()
    percentages_unfiltered = rdr_box_color_counts_unfiltered / total_unfiltered * 100

    # Create a series with all categories from unfiltered data, fill missing categories in filtered data with 0
    percentages = rdr_box_color_counts / total_filtered * 100

    fig = plt.figure(figsize=(5.5, 3.5))
    fig.patch.set_facecolor('#0d1117')
    ax = plt.gca()
    ax.set_facecolor('#0d1117')

    def to_percent(y, position):
        return f'{y:.0f}%'

    formatter = FuncFormatter(to_percent)
    ax.yaxis.set_major_formatter(formatter)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Plot unfiltered data first
    bars_unfiltered = plt.bar(percentages_unfiltered.index, percentages_unfiltered, color='#83c9ff', alpha=0.3, width=0.5, zorder=2)

    # Plot filtered data on top
    bars = plt.bar(percentages.index, percentages, color='#83c9ff', width=0.2, zorder=3)

    for bar, percentage in zip(bars, percentages):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{round(percentage, 2)}%", ha='center', va='bottom', color='white', fontsize=8, weight='bold')

    plt.title('RDR Box Color', color='white', fontsize=16)

    for label in ax.get_xticklabels():
        label.set_weight('bold')

    for label in ax.get_yticklabels():
        label.set_weight('bold')

    plt.tick_params(colors='white', labelsize=8)
    plt.grid(axis='y', linestyle=':', color='white', zorder=1, alpha=0.5)

    st.pyplot(fig)

with col5:
    # RDR Confirmation chart
    rdr_confirmation_counts = df_filtered['RDR Confirmation'].value_counts()
    rdr_confirmation_counts_unfiltered = df_original['RDR Confirmation'].value_counts()

    total_filtered = rdr_confirmation_counts.sum()
    total_unfiltered = rdr_confirmation_counts_unfiltered.sum()
    percentages_unfiltered = rdr_confirmation_counts_unfiltered / total_unfiltered * 100

    # Create a series with all categories from unfiltered data, fill missing categories in filtered data with 0
    percentages = rdr_confirmation_counts / total_filtered * 100

    fig = plt.figure(figsize=(5.5, 3.5))
    fig.patch.set_facecolor('#0d1117')
    ax = plt.gca()
    ax.set_facecolor('#0d1117')

    def to_percent(y, position):
        return f'{y:.0f}%'

    formatter = FuncFormatter(to_percent)
    ax.yaxis.set_major_formatter(formatter)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Plot unfiltered data first
    bars_unfiltered = plt.bar(percentages_unfiltered.index, percentages_unfiltered, color='#83c9ff', alpha=0.3, width=0.5, zorder=2)

    # Plot filtered data on top
    bars = plt.bar(percentages.index, percentages, color='#83c9ff', width=0.2, zorder=3)

    for bar, percentage in zip(bars, percentages):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{round(percentage, 2)}%", ha='center', va='bottom', color='white', fontsize=8, weight='bold')

    plt.title('RDR Confirmation', color='white', fontsize=16)

    for label in ax.get_xticklabels():
        label.set_weight('bold')

    for label in ax.get_yticklabels():
        label.set_weight('bold')

    plt.tick_params(colors='white', labelsize=8)
    plt.grid(axis='y', linestyle=':', color='white', zorder=1, alpha=0.5)

    st.pyplot(fig)

##-------ODR--------------



