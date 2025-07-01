import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model dan PCA
model = joblib.load("model/rdf_model.joblib")
pca = joblib.load("model/pca_1.joblib")

# Daftar fitur kategorikal dan numerik
categorical_features = [
    'Application_mode', 'Course', 'Daytime_evening_attendance',
    'Mothers_qualification', 'Fathers_qualification', 'Mothers_occupation',
    'Fathers_occupation', 'Displaced', 'Educational_special_needs',
    'Debtor', 'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder',
    'International'
]

numerical_features = [
    'Application_order', 'Previous_qualification_grade', 'Admission_grade',
    'Age_at_enrollment', 'Curricular_units_1st_sem_credited',
    'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_evaluations',
    'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
    'Curricular_units_1st_sem_without_evaluations', 'Curricular_units_2nd_sem_credited',
    'Curricular_units_2nd_sem_enrolled', 'Curricular_units_2nd_sem_evaluations',
    'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade',
    'Curricular_units_2nd_sem_without_evaluations'
]

pca_numerical = [
    'Curricular_units_1st_sem_enrolled','Curricular_units_1st_sem_evaluations',
    'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
    'Curricular_units_2nd_sem_credited', 'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_2nd_sem_evaluations', 'Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade',
]

# Mapping label deskriptif ke nilai numerik
categorical_mappings = {
    "Application_mode": {
        1: "1 - 1st phase - general contingent", 2: "2 - Ordinance No. 612/93",
        5: "5 - 1st phase - special contingent (Azores Island)",
        7: "7 - Holders of other higher courses", 10: "10 - Ordinance No. 854-B/99",
        15: "15 - International student (bachelor)", 16: "16 - 1st phase - special contingent (Madeira Island)",
        17: "17 - 2nd phase - general contingent", 18: "18 - 3rd phase - general contingent",
        26: "26 - Ordinance No. 533-A/99, item b2) (Different Plan)", 27: "27 - Ordinance No. 533-A/99, item b3 (Other Institution)",
        39: "39 - Over 23 years old", 42: "42 - Transfer", 43: "43 - Change of course",
        44: "44 - Technological specialization diploma holders", 51: "51 - Change of institution/course",
        53: "53 - Short cycle diploma holders", 57: "57 - Change of institution/course (International)"
    },
    "Course": {
        33: "33 - Biofuel Production Technologies", 171: "171 - Animation and Multimedia Design",
        8014: "8014 - Social Service (evening attendance)", 9003: "9003 - Agronomy",
        9070: "9070 - Communication Design", 9085: "9085 - Veterinary Nursing",
        9119: "9119 - Informatics Engineering", 9130: "9130 - Equinculture",
        9147: "9147 - Management", 9238: "9238 - Social Service", 9254: "9254 - Tourism",
        9500: "9500 - Nursing", 9556: "9556 - Oral Hygiene", 9670: "9670 - Advertising and Marketing Management",
        9773: "9773 - Journalism and Communication", 9853: "9853 - Basic Education",
        9991: "9991 - Management (evening attendance)"
    },
    "Daytime_evening_attendance": {1: "1 - daytime", 0: "0 - evening"},
    "Mothers_qualification": {
        1: "1 - Secondary Education - 12th Year of Schooling or Eq.", 2: "2 - Higher Education - Bachelor's Degree",
        3: "3 - Higher Education - Degree", 4: "4 - Higher Education - Master's",
        5: "5 - Higher Education - Doctorate", 6: "6 - Frequency of Higher Education",
        9: "9 - 12th Year of Schooling - Not Completed", 10: "10 - 11th Year of Schooling - Not Completed",
        11: "11 - 7th Year (Old)", 12: "12 - Other - 11th Year of Schooling",
        14: "14 - 10th Year of Schooling", 18: "18 - General commerce course",
        19: "19 - Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.",
        22: "22 - Technical-professional course", 26: "26 - 7th year of schooling",
        27: "27 - 2nd cycle of the general high school course", 29: "29 - 9th Year of Schooling - Not Completed",
        30: "30 - 8th year of schooling", 34: "34 - Unknown", 35: "35 - Can't read or write",
        36: "36 - Can read without having a 4th year of schooling", 37: "37 - Basic education 1st cycle",
        38: "38 - Basic Education 2nd Cycle", 39: "39 - Technological specialization course",
        40: "40 - Higher education - degree (1st cycle)", 41: "41 - Specialized higher studies course",
        42: "42 - Professional higher technical course", 43: "43 - Higher Education - Master (2nd cycle)",
        44: "44 - Higher Education - Doctorate (3rd cycle)"
    },
    "Fathers_qualification": {
        1: "1 - Secondary Education - 12th Year of Schooling or Eq.", 2: "2 - Higher Education - Bachelor's Degree",
        3: "3 - Higher Education - Degree", 4: "4 - Higher Education - Master's",
        5: "5 - Higher Education - Doctorate", 6: "6 - Frequency of Higher Education",
        9: "9 - 12th Year of Schooling - Not Completed", 10: "10 - 11th Year of Schooling - Not Completed",
        11: "11 - 7th Year (Old)", 12: "12 - Other - 11th Year of Schooling",
        13: "13 - 2nd year complementary high school course", 14: "14 - 10th Year of Schooling",
        18: "18 - General commerce course", 19: "19 - Basic Education 3rd Cycle",
        20: "20 - Complementary High School Course", 22: "22 - Technical-professional course",
        25: "25 - Complementary High School Course - not concluded", 26: "26 - 7th year of schooling",
        27: "27 - 2nd cycle of the general high school course", 29: "29 - 9th Year of Schooling - Not Completed",
        30: "30 - 8th year of schooling", 31: "31 - General Course of Administration and Commerce",
        33: "33 - Supplementary Accounting and Administration", 34: "34 - Unknown", 35: "35 - Can't read or write",
        36: "36 - Can read without having a 4th year of schooling", 37: "37 - Basic education 1st cycle",
        38: "38 - Basic Education 2nd Cycle", 39: "39 - Technological specialization course",
        40: "40 - Higher education - degree (1st cycle)", 41: "41 - Specialized higher studies course",
        42: "42 - Professional higher technical course", 43: "43 - Higher Education - Master (2nd cycle)",
        44: "44 - Higher Education - Doctorate (3rd cycle)"
    },
    "Mothers_occupation": {
        0: "0 - Student", 1: "1 - Legislative/Executive/Directors", 2: "2 - Intellectual and Scientific Activities",
        3: "3 - Intermediate Technicians", 4: "4 - Administrative staff", 5: "5 - Personal services/sellers",
        6: "6 - Farmers/Forestry", 7: "7 - Skilled industrial workers", 8: "8 - Machine operators",
        9: "9 - Unskilled workers", 10: "10 - Armed Forces", 90: "90 - Other", 99: "99 - Blank"
    },
    "Fathers_occupation": {
        0: "0 - Student", 1: "1 - Legislative/Executive/Directors", 2: "2 - Intellectual and Scientific Activities",
        3: "3 - Intermediate Technicians", 4: "4 - Administrative staff", 5: "5 - Personal services/sellers",
        6: "6 - Farmers/Forestry", 7: "7 - Skilled industrial workers", 8: "8 - Machine operators",
        9: "9 - Unskilled workers", 10: "10 - Armed Forces", 90: "90 - Other", 99: "99 - Blank"
    },
    "Displaced": {0: "0 - No", 1: "1 - Yes"},
    "Educational_special_needs": {0: "0 - No", 1: "1 - Yes"},
    "Debtor": {0: "0 - No", 1: "1 - Yes"},
    "Tuition_fees_up_to_date": {0: "0 - No", 1: "1 - Yes"},
    "Gender": {0: "0 - Female", 1: "1 - Male"},
    "Scholarship_holder": {0: "0 - No", 1: "1 - Yes"},
    "International": {0: "0 - No", 1: "1 - Yes"}
}

# Form input
st.title("Prediksi Kecenderungan Dropout Mahasiswa")
user_input = {}

with st.form("user_form"):
    for col in categorical_features:
        options = categorical_mappings[col]
        selected_label = st.selectbox(col, list(options.values()))
        user_input[col] = [k for k, v in options.items() if v == selected_label][0]

    for col in numerical_features:
        user_input[col] = st.number_input(col, step=1.0)

    submitted = st.form_submit_button("Prediksi")

if submitted:
    df = pd.DataFrame([user_input])

    # Scaling
    for col in numerical_features:
        scaler = joblib.load(f"model/scaler_{col}.joblib")
        df[col] = scaler.transform(df[[col]])

    # One-hot encoding
    for col in categorical_features:
        encoder = joblib.load(f"model/onehot_{col}.joblib")
        ohe = encoder.transform(df[[col]])
        ohe_df = pd.DataFrame(ohe, columns=[f"{col}_{cat}" for cat in encoder.categories_[0]])
        df = df.drop(columns=[col])
        df = pd.concat([df.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)

    # PCA
    pca_data = pca.transform(df[pca_numerical])
    df[["pc1_1", "pc1_2"]] = pd.DataFrame(pca_data, columns=["pc1_1", "pc1_2"])
    df = df.drop(columns=pca_numerical)

    # Prediksi
    pred = model.predict(df)[0]
    st.success("Hasil Prediksi: Kemungkinan akan DROPOUT" if pred == 1 else "Hasil Prediksi: TIDAK DROP OUT")
