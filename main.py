from scipy.optimize import root_scalar
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
        page_title="Transformations",
        layout='wide',
        )

st.markdown('<h1 style="color:darkgrey; text-align:center">'
            'Transformations</h1>',
            unsafe_allow_html=True)
st.markdown("***")


def negative_exponential(x, S, beta):
    y = beta*(1-np.exp(-x/S))

    return y


def derivative_negative_exponential(x, S, beta):
    y = beta/S * np.exp(-x/S)

    return y


def saturation_negative_exponential(S, beta):
    y = -S * np.log(S/beta)

    return y


def hill(x, S, K, beta):
    y = beta - beta*K**S / (x**S + K**S)

    return y


def hill_derivative(x, S, K, beta):
    y = beta * S * (K**S) * x**(S-1) / ((x**S + K**S)**2)

    return y


def find_initialization(S, K, beta):

    def objective_function(x, S, K, beta):
        y = hill_derivative(x, S, K, beta) - 1

        return y

    solution = root_scalar(objective_function, args=(S, K, beta), bracket=(0, K), x0=K/2)

    return solution.root


def find_saturation(S, K, beta):

    def objective_function(x, S, K, beta):
        y = hill_derivative(x, S, K, beta) - 1

        return y

    solution = root_scalar(objective_function, args=(S, K, beta), bracket=(K, 4*K), x0=3*K/2)

    return solution.root


def initialization_callback(S, K, beta):
    try:
        solution = find_initialization(S, K, beta)
        st.session_state['initialization'] = solution
    except:
        if 'initialization' in st.session_state:
            del st.session_state['initialization']


def saturation_callback(S, K, beta):
    try:
        solution = find_saturation(S, K, beta)
        st.session_state['saturation'] = solution
    except:
        if 'saturation' in st.session_state:
            del st.session_state['saturation']


def slider_callback(S, K, beta):
    initialization_callback(S, K, beta)
    saturation_callback(S, K, beta)


curve = st.radio(label='Choose Transformation',
         options=['Hill Curve', 'Negative Exponential'],
                 horizontal=True)

S = st.sidebar.slider(label='S',
                      min_value=0.1,
                      max_value=10.,
                      value=5.,)
if curve == 'Hill Curve':
    K = st.sidebar.slider(label='K',
                          min_value=0.1,
                          max_value=100.,
                          value=50.,)

beta = st.sidebar.slider(label='$\\beta$',
                         min_value=0.1,
                         max_value=100.,
                         value=1.,)

X_range = np.linspace(start=0, stop=100, num=1000)
derivative_check = st.sidebar.checkbox(label='Derivative')
fig = make_subplots(specs=[[{'secondary_y': True}]])
Y_range = hill(X_range, S, K, beta)\
        if curve == 'Hill Curve'\
        else negative_exponential(X_range, S, beta)
fig.add_trace(
        go.Scatter(x=X_range,
                   y=Y_range,
                   name=curve,
                   mode='lines'),
        secondary_y=False)
fig.update_layout(
        showlegend=False,
        title=curve
        )
if derivative_check:
    derivative_Y_range = hill_derivative(X_range, S, K, beta)\
            if curve == "Hill Curve"\
            else derivative_negative_exponential(X_range, S,  beta)
    fig.add_trace(
            go.Scatter(x=X_range,
                       y=derivative_Y_range,
                       name="Derivative",
                       mode='lines'),
            secondary_y=True
            )
    fig.update_layout(
            showlegend=True
            )
    fig.update_yaxes(title_text=curve, secondary_y=False)
    fig.update_yaxes(title_text='Derivative', secondary_y=True)
my_plot = st.plotly_chart(fig, use_container_width=True)

if curve == 'Hill Curve':
    slider_callback(S, K, beta)
    if 'initialization' in st.session_state:
        fig.add_trace(
                go.Scatter(x=[st.session_state.initialization],
                           y=[hill(st.session_state.initialization, S, K, beta)],
                           name="Initialization",
                           marker_size=7)
            )
        my_plot.plotly_chart(fig, use_container_width=True)

    if 'initialization' in st.session_state:
        fig.add_trace(
                go.Scatter(x=[st.session_state.saturation],
                           y=[hill(st.session_state.saturation, S, K, beta)],
                           name="Saturation",
                           marker_size=7)
            )
        my_plot.plotly_chart(fig, use_container_width=True)

    if 'initialization' not in st.session_state or 'saturation' not in st.session_state:
        st.warning("Couldn't find initialization and/or saturation, maybe they don't exist...")
else:
    saturation_point = saturation_negative_exponential(S, beta)
    saturation_y = negative_exponential(saturation_point, S, beta)
    fig.add_trace(
            go.Scatter(x=[saturation_point],
                       y=[saturation_y],
                       name='Saturation',
                       marker_size=7)
            )
    my_plot.plotly_chart(fig, use_container_width=True)
