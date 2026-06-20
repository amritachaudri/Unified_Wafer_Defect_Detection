import streamlit as st
import warnings
import pandas as pd
import numpy as np
import holoviews as hv
import hvplot.pandas
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Wafer Map Defect Detection",
    page_icon="",
    layout="wide"
)

def main():
    st.header("Wafer Map Defect Detection")
    col1,col2,col3,col4 = st.columns(4)
    with col4:
        with open(r"C:\Users\Amrita Chaudri\Downloads\Sem4-Final_Term\2023aa05132.pdf", "rb") as pdf_file:
            PDFbytes = pdf_file.read()
        st.download_button(
        label="Download PDF",
        data=PDFbytes,
        file_name="Paper.pdf",
        mime='application/octet-stream'
    )
    # st.sidebar.header("Filters")
    tab1, tab2, tab3 = st.tabs(["Information", "Create Image", "Defect Prediction"])
    with tab1:
        info()
    with tab2:
        genrate_image()
    with tab3:
        predict_defect()

def info():
    st.subheader("Data Information")
    st.subheader("Overview")
    st.write("""
    - Total Samples: 38,015 wafer map images.
    - Format: NumPy.npz file (e.g., Wafer_Map_Dataset.npz).
    - Image Size: 52 x 52 pixels.
    - Classes: 38 total.
        - 1 Normal
        - 8 single-type defects
        - 29 multi-type combinations (up to 4 defect types per image).
    """)

    st.subheader("Data Structure")
    st.write("""
    - arr_0: Wafer map array
        - 0: Blank spot;
        - 1: Normal die (passed test);
        - 2: Broken die (failed test)).
    - arr_1: One-hot encoded labels (8 dimensions for basic defect types).
    """)

def genrate_image():
    st.subheader("Create Image")
    df = genrate_points()
    Pre_processing(df)

def genrate_points():
    # Defining the range forX and Y
    X = np.arange(-11, 14)
    Y = np.arange(-11, 14)
    # Creating a meshgrid(used to create coordinate arrays) of X and Y values
    # Output [[1,2],[2,1]]
    X_grid, Y_grid = np.meshgrid(X, Y)
    X_flat = X_grid.ravel()
    Y_flat = Y_grid.ravel()
    size = X_flat.size

    # Create dataframe, ravel(contiguous flatten the grids) is used for arrays
    # Output [1,2,2,1]
    df = pd.DataFrame({
        'X': X_flat,
        'Y': Y_flat,
        'Defect': np.random.randint(0, 2, size=size),
        'Probe_X': X_flat,
        'Probe_Y': Y_flat,
        'Reticle_X': np.random.randint(0, 2, size=size),
        'Reticle_Y': np.random.randint(0, 2, size=size)
    }).sort_values(by=['X', 'Y'])

    return df

def Pre_processing(df):
    df = df.dropna()
    # Finding min and max for Intra reticle X and Y
    reticle_x_min = df['Reticle_X'].min()
    reticle_x_max = df['Reticle_X'].max()
    reticle_y_min = df['Reticle_Y'].min()
    reticle_y_max = df['Reticle_Y'].max()
    # Pivotting based on reticle and die location
    reticle_x_min_mapping = pd.pivot_table(df, index=['Reticle_X'], values='Probe_X', aggfunc='min').reset_index()
    reticle_x_min_mapping.rename(columns={'Probe_X': 'base_probex'}, inplace=True)
    reticle_y_min_mapping = pd.pivot_table(df, index=['Reticle_Y'], values='Probe_Y', aggfunc='min').reset_index()
    reticle_y_min_mapping.rename(columns={'Probe_Y': 'base_probey'}, inplace=True)
    # merging mapping to df
    df = pd.merge(df, reticle_x_min_mapping, how='left')
    df = pd.merge(df, reticle_y_min_mapping, how='left')
    # assigning reticle corner to zero
    df['if_reticle_corner'] = 0
    # based on intra reticle X and Y assigning reticle corner
    df.loc[
        ((df['Reticle_X'] == reticle_x_min) & (df['Reticle_Y'] == reticle_y_min)) |
        ((df['Reticle_X'] == reticle_x_max) & (df['Reticle_Y'] == reticle_y_min)) |
        ((df['Reticle_X'] == reticle_x_min) & (df['Reticle_Y'] == reticle_y_max)) |
        ((df['Reticle_X'] == reticle_x_max) & (df['Reticle_Y'] == reticle_y_max)),
        'if_reticle_corner'] = 1
    # getting reticle list
    reticle_x_list = df[df['if_reticle_corner'] == 1].dropna()['X'].unique()
    reticle_y_list = df[df['if_reticle_corner'] == 1].dropna()['Y'].unique()

    # Step estimation using pivot logic
    if len(pd.pivot_table(df, index=['Reticle_X'], values=['X'], aggfunc='min').reset_index()['X'].unique()) == 1:
        x_stepwise = pd.pivot_table(df, index=['Reticle_Y'], values=['X'], aggfunc='min').reset_index().shape[0]
    else:
        x_stepwise = pd.pivot_table(df, index=['Reticle_X'], values=['X'], aggfunc='min').reset_index().shape[0]

    if len(pd.pivot_table(df, index=['Reticle_X'], values=['Y'], aggfunc='min').reset_index()['Y'].unique()) == 1:
        y_stepwise = pd.pivot_table(df, index=['Reticle_Y'], values=['Y'], aggfunc='min').reset_index().shape[0]
    else:
        y_stepwise = pd.pivot_table(df, index=['Reticle_X'], values=['Y'], aggfunc='min').reset_index().shape[0]

    # Call reticle_trans logic
    reticle_x_list_adj, reticle_y_list_adj = reticle_trans(df, reticle_x_list, reticle_y_list, 'X', 'Y', x_stepwise, y_stepwise)
    # Plot the wafer map
    plot_wafer_map(df, reticle_x_list_adj, reticle_y_list_adj)

def reticle_trans(df, reticle_x_list, reticle_y_list, colx, coly, x_stepwise, y_stepwise):
    df_reticle_x = pd.DataFrame(reticle_x_list).reset_index(drop=True).rename(columns={0: 'list_x'}).sort_values(by='list_x')
    df_reticle_x['Change'] = df_reticle_x['list_x'] - df_reticle_x['list_x'].shift(1, fill_value=df_reticle_x['list_x'][0])
    df_reticle_x.loc[df_reticle_x['Change'] == 1, 'list_x_pre'] = df_reticle_x['list_x'].shift(1)
    df_reticle_x = df_reticle_x[~df_reticle_x['list_x'].isin(df_reticle_x['list_x_pre'].unique())]
    df_reticle_x['final_value'] = (df_reticle_x['list_x'] + df_reticle_x['list_x_pre']) / 2

    df_reticle_y = pd.DataFrame(reticle_y_list).reset_index(drop=True).rename(columns={0: 'list_x'}).sort_values(by='list_x')
    df_reticle_y['Change'] = df_reticle_y['list_x'] - df_reticle_y['list_x'].shift(1, fill_value=df_reticle_y['list_x'][0])
    df_reticle_y.loc[df_reticle_y['Change'] == 1, 'list_x_pre'] = df_reticle_y['list_x'].shift(1)
    df_reticle_y = df_reticle_y[~df_reticle_y['list_x'].isin(df_reticle_y['list_x_pre'].unique())]
    df_reticle_y['final_value'] = (df_reticle_y['list_x'] + df_reticle_y['list_x_pre']) / 2

    # Define extended axes
    add1_x = np.ceil((df[colx].max() - df_reticle_x['final_value'].max()) / x_stepwise) + 2
    add2_x = np.ceil(np.abs(df[colx].min() - df_reticle_x['final_value'].min()) / x_stepwise)
    add1_y = np.ceil((df[coly].max() - df_reticle_y['final_value'].max()) / y_stepwise) + 2
    add2_y = np.ceil(np.abs(df[coly].min() - df_reticle_y['final_value'].min()) / y_stepwise)

    reticle_x_list_adj = np.arange(
        df_reticle_x['final_value'].min() - add2_x * x_stepwise,
        df_reticle_x['final_value'].max() + add1_x * x_stepwise,
        x_stepwise).tolist()

    reticle_y_list_adj = np.arange(
        df_reticle_y['final_value'].min() - add2_y * y_stepwise,
        df_reticle_y['final_value'].max() + add1_y * y_stepwise,
        y_stepwise).tolist()

    return reticle_x_list_adj, reticle_y_list_adj

def plot_wafer_map(df, reticle_x_list_adj, reticle_y_list_adj):
    value_dimension = hv.Dimension('Defect')
    # Define color range and levels
    min_value = 0
    max_value = 1
    colors = ['#FF0000', '#D1FFBD']
    levels = [min_value, 0.5, max_value]

    # Filter points within a circular boundary
    center_x = df['X'].mean()
    center_y = df['Y'].mean()
    radius = min(df['X'].max() - df['X'].min(), df['Y'].max() - df['Y'].min()) / 2
    df['distance'] = np.sqrt((df['X'] - center_x) ** 2 + (df['Y'] - center_y) ** 2)
    df_circular = df[df['distance'] <= radius]

    # Create heatmap
    vplot = df_circular.hvplot.heatmap(
        x="X", y="Y", C='Defect', color=colors
    ).opts(width=800, height=600, cmap=colors, color_levels=levels)

    # Add labels
    labels = hv.Labels(df_circular, kdims=['Probe_X', 'Probe_Y'], vdims=['Defect']).opts(text_font_size="7pt")
    vplot = vplot * labels.opts(title="Wafer Map")

    # Add grid lines for reticle boundaries
    for i in reticle_x_list_adj:
        vplot = vplot * hv.VLine(i).opts(color='black', line_dash='solid', line_width=0.5)
    for j in reticle_y_list_adj:
        vplot = vplot * hv.HLine(j).opts(color="black", line_dash='solid', line_width=0.5)

    # Add circular outline
    circle = hv.Path([hv.Ellipse(center_x, center_y, radius * 2)]).opts(color="black", line_width=1.5)
    vplot = vplot * circle

    # Render plot
    st.bokeh_chart(hv.render(vplot, backend='bokeh'))
    st.subheader("Raw Data For Wafer Map")
    st.dataframe(df)

# Function to translate the label into defect types 
def read_label(label, defect_types =''):
    label_keys = ["Center, ", "Donut, ", "Edge_Loc, ", "Edge_Ring, ", "Loc,", "Near_Full, ", "Scratch, ", "Random, "]
    if np.sum(label) == 0:
        defect_types = 'Normal wafer'
    else:
        for digit in range(np.shape(label)[0]):
            # Digit will also range from 0 to 8
            if label[digit] == 1:
                defect_types = defect_types + label_keys [digit]

    return defect_types

def predict_defect():
    st.subheader("Prediction Input")

    with st.form(key='prediction_form'):
        input_number = st.number_input(
            "Enter a Number", 
            value=None, 
            placeholder="Number between 1 to 5700"
            )
        input_number = int(input_number) if input_number is not None else None
        submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        if input_number is None:
            st.warning("Please enter a number")
            
        elif input_number is not None:
            st.info(f"Your input value is {input_number}")
            model_file = r"C:\Users\Amrita Chaudri\Downloads\Sem 4\hybrid_model.sav"
            try:
                with open(model_file, 'rb') as f:
                    loaded_model = pickle.load(f)
            except FileNotFoundError:
                st.error(f"Error: Model file '{model_file}' not found. Please ensure it's in the same directory as your script.")
                loaded_model = None
            except Exception as e:
                st.error(f"An error occurred while loading the model: {e}")
                loaded_model = None

            # Getting Data
            # Loading publicly available data from a Numpy .npz file
            data = np.load(r"C:\Users\Amrita Chaudri\Downloads\Sem 4\dataset\Wafer_Map_Datasets.npz")
            train = data["arr_0"]
            label = data["arr_1"]
            # First split into train + val and test
            X_temp, X_test, y_temp, y_test = train_test_split(train, label, test_size = 0.15, random_state = 42)
            # Then split train + val into train and val
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size = 0.15, random_state = 42)
            X_test = np.expand_dims(X_test, -1)
            prediction = loaded_model.predict(X_test, verbose=1)
            image = np.squeeze(X_test[input_number,:,:,0])

            st.subheader("Wafer Image")
            fig, ax = plt.subplots(figsize=(3, 2)) 
            ax.imshow(image)
            ax.set_title("Wafer Image") 
            st.pyplot(fig)

            st.subheader("Prediction Output")
            st.warning("Predicting the Defect")
            st.info(f"Model output: {prediction[input_number, :]}")
            st.info(f"Ground-truth: {np.array(y_test[input_number, :])}")
            st.success(f"Predictions string: [{' '.join([f'{val:.0f}' for val in prediction[input_number]])}]")
            label = np.array([f"{val:.0f}" for val in prediction[input_number]]).astype(float)
            st.success(f"Predictions: {read_label(label)}")

if __name__ =="__main__":
    main()