import streamlit as st
import numpy as np
import xgboost as xgb
import pickle

# About page
def about_page():
    st.title('House Price Prediction: A Machine Learning Approach')
    st.write("This is a machine learning model for predicting House Price Prediction. The model uses historical Housing data to determine the Housing Price Prediction.")
    # st.title("Dataset Source")
    # st.write("In our dataset, we have 25 columns with 30,000 rows, reflecting various customer attributes. The target column is default.payment.next.month, which reflects whether the customer defaulted or not. The aim is to predict the probability of default.")
    # st.write(":link: Kaggle link :- https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset")
    # st.write(":link: UCI Repository :- https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients")

# Prediction page
def prediction_page():
    # Define locations before using it in the selectbox
    locations = ['1st Block Jayanagar', '1st Phase JP Nagar', '2nd Phase Judicial Layout', 
             '2nd Stage Nagarbhavi', '5th Block Hbr Layout', '5th Phase JP Nagar', 
             '6th Phase JP Nagar', '7th Phase JP Nagar', '8th Phase JP Nagar', 
             '9th Phase JP Nagar', 'AECS Layout', 'Abbigere', 'Akshaya Nagar', 
             'Ambalipura', 'Ambedkar Nagar', 'Amruthahalli', 'Anandapura', 'Ananth Nagar', 
             'Anekal', 'Anjanapura', 'Ardendale', 'Arekere', 'Attibele', 'BEML Layout', 
             'BTM 2nd Stage', 'BTM Layout', 'Babusapalaya', 'Badavala Nagar', 'Balagere', 
             'Banashankari', 'Banashankari Stage II', 'Banashankari Stage III', 
             'Banashankari Stage V', 'Banashankari Stage VI', 'Banaswadi', 'Banjara Layout', 
             'Bannerghatta', 'Bannerghatta Road', 'Basavangudi', 'Basaveshwara Nagar', 
             'Battarahalli', 'Begur', 'Begur Road', 'Bellandur', 'Benson Town', 'Bharathi Nagar', 
             'Bhoganhalli', 'Billekahalli', 'Binny Pete', 'Bisuvanahalli', 'Bommanahalli', 
             'Bommasandra', 'Bommasandra Industrial Area', 'Bommenahalli', 'Brookefield', 
             'Budigere', 'CV Raman Nagar', 'Chamrajpet', 'Chandapura', 'Channasandra', 
             'Chikka Tirupathi', 'Chikkabanavar', 'Chikkalasandra', 'Choodasandra', 'Cooke Town', 
             'Cox Town', 'Cunningham Road', 'Dasanapura', 'Dasarahalli', 'Devanahalli', 
             'Devarachikkanahalli', 'Dodda Nekkundi', 'Doddaballapur', 'Doddakallasandra', 
             'Doddathoguru', 'Domlur', 'Dommasandra', 'EPIP Zone', 'Electronic City', 
             'Electronic City Phase II', 'Electronics City Phase 1', 'Frazer Town', 'GM Palaya', 
             'Garudachar Palya', 'Giri Nagar', 'Gollarapalya Hosahalli', 'Gottigere', 
             'Green Glen Layout', 'Gubbalala', 'Gunjur', 'HAL 2nd Stage', 'HBR Layout', 'HRBR Layout', 
             'HSR Layout', 'Haralur Road', 'Harlur', 'Hebbal', 'Hebbal Kempapura', 'Hegde Nagar', 
             'Hennur', 'Hennur Road', 'Hoodi', 'Horamavu Agara', 'Horamavu Banaswadi', 'Hormavu', 
             'Hosa Road', 'Hosakerehalli', 'Hoskote', 'Hosur Road', 'Hulimavu', 'ISRO Layout', 
             'ITPL', 'Iblur Village', 'Indira Nagar', 'JP Nagar', 'Jakkur', 'Jalahalli', 'Jalahalli East', 
             'Jigani', 'Judicial Layout', 'KR Puram', 'Kadubeesanahalli', 'Kadugodi', 'Kaggadasapura', 
             'Kaggalipura', 'Kaikondrahalli', 'Kalena Agrahara', 'Kalyan nagar', 'Kambipura', 
             'Kammanahalli', 'Kammasandra', 'Kanakapura', 'Kanakpura Road', 'Kannamangala', 'Karuna Nagar', 
             'Kasavanhalli', 'Kasturi Nagar', 'Kathriguppe', 'Kaval Byrasandra', 'Kenchenahalli', 
             'Kengeri', 'Kengeri Satellite Town', 'Kereguddadahalli', 'Kodichikkanahalli', 'Kodigehaali', 
             'Kodigehalli', 'Kodihalli', 'Kogilu', 'Konanakunte', 'Koramangala', 'Kothannur', 'Kothanur', 
             'Kudlu', 'Kudlu Gate', 'Kumaraswami Layout', 'Kundalahalli', 'LB Shastri Nagar', 'Laggere', 
             'Lakshminarayana Pura', 'Lingadheeranahalli', 'Magadi Road', 'Mahadevpura', 'Mahalakshmi Layout', 
             'Mallasandra', 'Malleshpalya', 'Malleshwaram', 'Marathahalli', 'Margondanahalli', 'Marsur', 
             'Mico Layout', 'Munnekollal', 'Murugeshpalya', 'Mysore Road', 'NGR Layout', 'NRI Layout', 
             'Nagarbhavi', 'Nagasandra', 'Nagavara', 'Nagavarapalya', 'Narayanapura', 'Neeladri Nagar', 
             'Nehru Nagar', 'OMBR Layout', 'Old Airport Road', 'Old Madras Road', 'Padmanabhanagar', 
             'Pai Layout', 'Panathur', 'Parappana Agrahara', 'Pattandur Agrahara', 'Poorna Pragna Layout', 
             'Prithvi Layout', 'R.T. Nagar', 'Rachenahalli', 'Raja Rajeshwari Nagar', 'Rajaji Nagar', 
             'Rajiv Nagar', 'Ramagondanahalli', 'Ramamurthy Nagar', 'Rayasandra', 'Sahakara Nagar', 
             'Sanjay nagar', 'Sarakki Nagar', 'Sarjapur', 'Sarjapur  Road', 'Sarjapura - Attibele Road', 
             'Sector 2 HSR Layout', 'Sector 7 HSR Layout', 'Seegehalli', 'Shampura', 'Shivaji Nagar', 
             'Singasandra', 'Somasundara Palya', 'Sompura', 'Sonnenahalli', 'Subramanyapura', 
             'Sultan Palaya', 'TC Palaya', 'Talaghattapura', 'Thanisandra', 'Thigalarapalya', 'Thubarahalli', 
             'Thyagaraja Nagar', 'Tindlu', 'Tumkur Road', 'Ulsoor', 'Uttarahalli', 'Varthur', 'Varthur Road', 
             'Vasanthapura', 'Vidyaranyapura', 'Vijayanagar', 'Vishveshwarya Layout', 'Vishwapriya Layout', 
             'Vittasandra', 'Whitefield', 'Yelachenahalli', 'Yelahanka', 'Yelahanka New Town', 
             'Yelenahalli', 'Yeshwanthpur']

    # Load the trained model
    model_file_path = 'C:/Project/Housing Data/archive/model.bst'
    try:
        xgb_reg = xgb.XGBRegressor()  # Initialize a new XGBoost regressor
        xgb_reg.load_model(model_file_path)  # Load the model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    # Load columns for one-hot encoding
    columns_file_path = 'C:/Project/Housing Data/archive/columns.pkl'
    try:
        with open(columns_file_path, 'rb') as f:
            data_columns = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading columns: {e}")
        st.stop()

    # Dropdown for location
    location = st.selectbox('Select Location:', locations)

    # User inputs for other features
    total_sqft = st.number_input('Enter Total Square Feet:', min_value=100, max_value=10000, value=1000)
    bath = st.number_input('Enter Number of Bathrooms:', min_value=1, max_value=10, value=2)
    bhk = st.number_input('Enter Number of Bedrooms (BHK):', min_value=1, max_value=10, value=2)

    # Create an array with the input data for prediction
    input_data = np.zeros(len(data_columns))  # Initialize an array with zeros, matching the number of columns

    # Set the values for the input fields
    input_data[0] = total_sqft  # total_sqft
    input_data[1] = bath  # bath
    input_data[2] = bhk  # bhk

    # Set the location feature (one-hot encoding)
    location_index = data_columns.index(location)
    input_data[location_index] = 1  # Set the corresponding location index to 1

    # Prediction button
    if st.button("Predict Price"):
        try:
            # Predict the price using the model
            prediction = xgb_reg.predict([input_data])
            st.success(f"Predicted Price: ₹{prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"Error predicting price: {e}")

# Main page for navigation
def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a Page", ["About", "Prediction"])

    # Display corresponding page based on selection
    if page == "About":
        about_page()
    elif page == "Prediction":
        prediction_page()

if __name__ == "__main__":
    main()
