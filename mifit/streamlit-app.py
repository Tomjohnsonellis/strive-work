import streamlit as st
import mifitdataprep as prep

st.write("Mi Fit - Activity Analysis")
user_data = st.file_uploader("Upload your stats for the day...")

st.write("_A design improvement for an actual app would be to either"
            " continuously upload the data or have a button that uploads the most recent batch of info_")
st.write("_E.g._:")
st.button("How did I do today?")


# This is how to make use of an uploaded file
if user_data is not None:
    # Read in the data as text (txt)
    file_data = str(user_data.read())
    # Split it into lines
    file_contents = file_data.split("\\n")
    # Try out our prepro functions
    st.write(prep.extract_accelerometer(file_contents))
    st.write(prep.extract_gyroscope(file_contents))
    st.write(prep.extract_gravity(file_contents))


st.write("Todo: Combine these different dataframes into a big one")
st.write("Add the target to those big dataframes")
st.write("Train models with those")
