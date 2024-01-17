import streamlit as st
import urllib
import requests
from streamlit_image_select import image_select

def predict(img_url):
    # encode url    
    params = {"img": img_url}
    encoded_params = urllib.parse.urlencode(params)
    #st.write(encoded_params)
    # request prediction
    predict_url = "https://mlzoomcampfunctionapprhj.azurewebsites.net/api/predict"  # cloud
    response = requests.post(predict_url, params = encoded_params).json()
    return response

st.title("Animal prediction")
st.markdown("**[MLZoomcamp 2023](https://github.com/DataTalksClub/machine-learning-zoomcamp/?tab=readme-ov-file#readme)** (c) robertohj@gmail.com")
st.write("Select an option to expand")
with st.expander("**Use custom Img URL (may fail with some urls due to formatting)**"):
    with st.form("custom_url_form", clear_on_submit=True):
        default_img_url = "https://thumbnails.cbc.ca/maven_legacy/thumbnails/161/803/ST_PANETTA_PANDAS_clean.jpg"
        img_url = st.text_input('Enter img URL:', default_img_url)
        submitted_custom = st.form_submit_button("Run prediction")
        if submitted_custom and img_url is not None:
            if(img_url.endswith(".jpg") or img_url.endswith(".png")):
                #st.write(img_url)
                with st.spinner('Predicting on ' + img_url.split("/")[-1]):
                    #st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
                    prediction_response = predict(img_url)
                    predicted_class = prediction_response["class_label"]
                    predicted_proba = prediction_response["class_probability"]
                    prediction_string = f"({predicted_class}, {predicted_proba})"
                    st.image(img_url, width=360)
                    st.text("Prediction on " + img_url.split("/")[-1] + ":" +\
                            "\n\t" + prediction_string)
            else:
                st.write("URL must end with .jpg or .png")

with st.expander("**Select from sample images (all of them retrieved from the web)**"):
    with st.form("sample_img_form", clear_on_submit=True):
        img_url = image_select(
            label="Select an animal to predict its class:",
            images=[
                "https://fotos.perfil.com/2020/02/26/trim/1140/641/quokka-02262020-919630.jpg",
                "https://www.trvst.world/wp-content/uploads/2023/06/zebra-on-grass-field.jpg",
                "https://site-547756.mozfiles.com/files/547756/medium/Bat.jpg",
                "https://media.wired.com/photos/593261cab8eb31692072f129/master/w_1600,c_limit/85120553.jpg",
                "https://www.fauna-flora.org/wp-content/uploads/2023/05/Thomas-Retterath-iStock-1323457138-scaled-1.jpg",
                "https://www.mypetsname.com/wp-content/uploads/2019/04/Lizard-Names-Feature.jpg"
            ],
            captions=["Quokka", "Zebra", "Bat",  "Fennec Fox","Wild Dog","Lizzard"],
        )
        submitted_sample = st.form_submit_button("Run prediction")
        if submitted_sample and img_url is not None:
            with st.spinner('Predicting on ' + img_url.split("/")[-1]):
                          
                #st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
                
                prediction_response = predict(img_url)
                predicted_class = prediction_response["class_label"]
                predicted_proba = prediction_response["class_probability"]
                prediction_string = f"({predicted_class}, {predicted_proba})"
                st.image(img_url, width=360)
                st.text("Prediction on " + img_url.split("/")[-1] + ":" +\
                            "\n\t" + prediction_string) 