from modi_assignment import recommend

import streamlit as st


st.title("CineSuggest")
mname = st.text_input("enter movie name (type -1 to do not give any input)")
mnum = int(st.number_input("enter number of different movies you want to see related to this movie"))


if(st.button("recommend")):
    
    st.success(recommend(mname,mnum))

