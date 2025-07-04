# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

Pendidikan adalah fondasi dari suatu bangsa. Kualitas Sumber Daya Manusia (SDM) berbanding lurus dengan kualitas sistem pendidikan. Buruknya sistem pendidikan akan berimbas pada banyak aspek di kemudian hari. Oleh karena itu perlu dilakukan analisa dan evaluasi terhadap hasil pendidikan masa kini agar perbaikan berkelanjutan dapat dilakukan.

Salah satu permasalahan dalam pendidikan adalah putus sekolah atau lebih dikenal dengan 'Dropout'. Ada banyak hal yang dapat menyebabkan seoarng siswa dropout, hal ini perlu dianalisa secara mendalam dan dengan pendekatan data science dan machine learning untuk dicarikan jalan keluarnya.

Dropout bisa dipicu oleh latar belakang keluarga, performa siswa, atau sistem pendidikan yang buruk. Selain itu ada beberapa siswa yang berada pada kategori rentan untuk dropout. Pada akhirnya kuantitas dropout siswa juga akan berpenagruh terhadap jalannya sistem pendidikan.
### Permasalahan Bisnis

#### Berikut adalah Permasalahan bisnis yang akan diselesaikan :
- Apakah kategori marital status berpengaruh terhadap tingkat dropout?
- Bagaimana tingkat pendidikan orang tua siswa yang dropout
- Apa pekerjaan orang tua siswa paling banyak dropout?
- Apa applikation mode paling banyak siswa dropout
- Apa cours paling banyak siswa dropout
- Berapa rata-rata nilai semester pertama siswa dropout
- Berapa rata-rata nilai semester kedua siswa dropout
- Bagaimana cara memperkirakan siswa akan dropout atau tidak

### Cakupan Proyek
Proyek ini mencakup pembuatan dashboard interaktif yang dapat digunakan user dalam memahami faktor-faktor dropout siswa. Selain itu menggunakan pendekatan machine learning dibuat sebuah model yang dapat menghitung posibilitas dropout siswa. Selanjutnya dibuat prototype sistem machine learning yang dapat memprediksi kemungkinan dropout siswa.

### Persiapan
Project ini menggunakan data siswa dari sebuah sekolah. Data dapat di download pada link berikut: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

| Column name | Description |
| --- | --- |
|Marital status | The marital status of the student. (Categorical) 1 – single 2 – married 3 – widower 4 – divorced 5 – facto union 6 – legally separated |
| Application mode | The method of application used by the student. (Categorical) 1 - 1st phase - general contingent 2 - Ordinance No. 612/93 5 - 1st phase - special contingent (Azores Island) 7 - Holders of other higher courses 10 - Ordinance No. 854-B/99 15 - International student (bachelor) 16 - 1st phase - special contingent (Madeira Island) 17 - 2nd phase - general contingent 18 - 3rd phase - general contingent 26 - Ordinance No. 533-A/99, item b2) (Different Plan) 27 - Ordinance No. 533-A/99, item b3 (Other Institution) 39 - Over 23 years old 42 - Transfer 43 - Change of course 44 - Technological specialization diploma holders 51 - Change of institution/course 53 - Short cycle diploma holders 57 - Change of institution/course (International)|
|Application order | The order in which the student applied. (Numerical) Application order (between 0 - first choice; and 9 last choice) |
|Course | The course taken by the student. (Categorical) 33 - Biofuel Production Technologies 171 - Animation and Multimedia Design 8014 - Social Service (evening attendance) 9003 - Agronomy 9070 - Communication Design 9085 - Veterinary Nursing 9119 - Informatics Engineering 9130 - Equinculture 9147 - Management 9238 - Social Service 9254 - Tourism 9500 - Nursing 9556 - Oral Hygiene 9670 - Advertising and Marketing Management 9773 - Journalism and Communication 9853 - Basic Education 9991 - Management (evening attendance)|
|Daytime/evening attendance | Whether the student attends classes during the day or in the evening. (Categorical) 1 – daytime 0 - evening |
|Previous qualification| The qualification obtained by the student before enrolling in higher education. (Categorical) 1 - Secondary education 2 - Higher education - bachelor's degree 3 - Higher education - degree 4 - Higher education - master's 5 - Higher education - doctorate 6 - Frequency of higher education 9 - 12th year of schooling - not completed 10 - 11th year of schooling - not completed 12 - Other - 11th year of schooling 14 - 10th year of schooling 15 - 10th year of schooling - not completed 19 - Basic education 3rd cycle (9th/10th/11th year) or equiv. 38 - Basic education 2nd cycle (6th/7th/8th year) or equiv. 39 - Technological specialization course 40 - Higher education - degree (1st cycle) 42 - Professional higher technical course 43 - Higher education - master (2nd cycle) |
|Previous qualification (grade) | Grade of previous qualification (between 0 and 200) |
| Nacionality | The nationality of the student. (Categorical) 1 - Portuguese; 2 - German; 6 - Spanish; 11 - Italian; 13 - Dutch; 14 - English; 17 - Lithuanian; 21 - Angolan; 22 - Cape Verdean; 24 - Guinean; 25 - Mozambican; 26 - Santomean; 32 - Turkish; 41 - Brazilian; 62 - Romanian; 100 - Moldova (Republic of); 101 - Mexican; 103 - Ukrainian; 105 - Russian; 108 - Cuban; 109 - Colombian|
|Mother's qualification | The qualification of the student's mother. (Categorical) 1 - Secondary Education - 12th Year of Schooling or Eq. 2 - Higher Education - Bachelor's Degree 3 - Higher Education - Degree 4 - Higher Education - Master's 5 - Higher Education - Doctorate 6 - Frequency of Higher Education 9 - 12th Year of Schooling - Not Completed 10 - 11th Year of Schooling - Not Completed 11 - 7th Year (Old) 12 - Other - 11th Year of Schooling 14 - 10th Year of Schooling 18 - General commerce course 19 - Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv. 22 - Technical-professional course 26 - 7th year of schooling 27 - 2nd cycle of the general high school course 29 - 9th Year of Schooling - Not Completed 30 - 8th year of schooling 34 - Unknown 35 - Can't read or write 36 - Can read without having a 4th year of schooling 37 - Basic education 1st cycle (4th/5th year) or equiv. 38 - Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv. 39 - Technological specialization course 40 - Higher education - degree (1st cycle) 41 - Specialized higher studies course 42 - Professional higher technical course 43 - Higher Education - Master (2nd cycle) 44 - Higher Education - Doctorate (3rd cycle)|
|Father's qualification | The qualification of the student's father. (Categorical) 1 - Secondary Education - 12th Year of Schooling or Eq. 2 - Higher Education - Bachelor's Degree 3 - Higher Education - Degree 4 - Higher Education - Master's 5 - Higher Education - Doctorate 6 - Frequency of Higher Education 9 - 12th Year of Schooling - Not Completed 10 - 11th Year of Schooling - Not Completed 11 - 7th Year (Old) 12 - Other - 11th Year of Schooling 13 - 2nd year complementary high school course 14 - 10th Year of Schooling 18 - General commerce course 19 - Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv. 20 - Complementary High School Course 22 - Technical-professional course 25 - Complementary High School Course - not concluded 26 - 7th year of schooling 27 - 2nd cycle of the general high school course 29 - 9th Year of Schooling - Not Completed 30 - 8th year of schooling 31 - General Course of Administration and Commerce 33 - Supplementary Accounting and Administration 34 - Unknown 35 - Can't read or write 36 - Can read without having a 4th year of schooling 37 - Basic education 1st cycle (4th/5th year) or equiv. 38 - Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv. 39 - Technological specialization course 40 - Higher education - degree (1st cycle) 41 - Specialized higher studies course 42 - Professional higher technical course 43 - Higher Education - Master (2nd cycle) 44 - Higher Education - Doctorate (3rd cycle) |
| Mother's occupation | The occupation of the student's mother. (Categorical) 0 - Student 1 - Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers 2 - Specialists in Intellectual and Scientific Activities 3 - Intermediate Level Technicians and Professions 4 - Administrative staff 5 - Personal Services, Security and Safety Workers and Sellers 6 - Farmers and Skilled Workers in Agriculture, Fisheries and Forestry 7 - Skilled Workers in Industry, Construction and Craftsmen 8 - Installation and Machine Operators and Assembly Workers 9 - Unskilled Workers 10 - Armed Forces Professions 90 - Other Situation 99 - (blank) 122 - Health professionals 123 - teachers 125 - Specialists in information and communication technologies (ICT) 131 - Intermediate level science and engineering technicians and professions 132 - Technicians and professionals, of intermediate level of health 134 - Intermediate level technicians from legal, social, sports, cultural and similar services 141 - Office workers, secretaries in general and data processing operators 143 - Data, accounting, statistical, financial services and registry-related operators 144 - Other administrative support staff 151 - personal service workers 152 - sellers 153 - Personal care workers and the like 171 - Skilled construction workers and the like, except electricians 173 - Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like 175 - Workers in food processing, woodworking, clothing and other industries and crafts 191 - cleaning workers 192 - Unskilled workers in agriculture, animal production, fisheries and forestry 193 - Unskilled workers in extractive industry, construction, manufacturing and transport 194 - Meal preparation assistants |
| Father's occupation | The occupation of the student's father. (Categorical) 0 - Student 1 - Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers 2 - Specialists in Intellectual and Scientific Activities 3 - Intermediate Level Technicians and Professions 4 - Administrative staff 5 - Personal Services, Security and Safety Workers and Sellers 6 - Farmers and Skilled Workers in Agriculture, Fisheries and Forestry 7 - Skilled Workers in Industry, Construction and Craftsmen 8 - Installation and Machine Operators and Assembly Workers 9 - Unskilled Workers 10 - Armed Forces Professions 90 - Other Situation 99 - (blank) 101 - Armed Forces Officers 102 - Armed Forces Sergeants 103 - Other Armed Forces personnel 112 - Directors of administrative and commercial services 114 - Hotel, catering, trade and other services directors 121 - Specialists in the physical sciences, mathematics, engineering and related techniques 122 - Health professionals 123 - teachers 124 - Specialists in finance, accounting, administrative organization, public and commercial relations 131 - Intermediate level science and engineering technicians and professions 132 - Technicians and professionals, of intermediate level of health 134 - Intermediate level technicians from legal, social, sports, cultural and similar services 135 - Information and communication technology technicians 141 - Office workers, secretaries in general and data processing operators 143 - Data, accounting, statistical, financial services and registry-related operators 144 - Other administrative support staff 151 - personal service workers 152 - sellers 153 - Personal care workers and the like 154 - Protection and security services personnel 161 - Market-oriented farmers and skilled agricultural and animal production workers 163 - Farmers, livestock keepers, fishermen, hunters and gatherers, subsistence 171 - Skilled construction workers and the like, except electricians 172 - Skilled workers in metallurgy, metalworking and similar 174 - Skilled workers in electricity and electronics 175 - Workers in food processing, woodworking, clothing and other industries and crafts 181 - Fixed plant and machine operators 182 - assembly workers 183 - Vehicle drivers and mobile equipment operators 192 - Unskilled workers in agriculture, animal production, fisheries and forestry 193 - Unskilled workers in extractive industry, construction, manufacturing and transport 194 - Meal preparation assistants 195 - Street vendors (except food) and street service providers |
| Admission grade | Admission grade (between 0 and 200) |
| Displaced | Whether the student is a displaced person. (Categorical) 	1 – yes 0 – no |
| Educational special needs | Whether the student has any special educational needs. (Categorical) 1 – yes 0 – no |
|Debtor | Whether the student is a debtor. (Categorical) 1 – yes 0 – no|
|Tuition fees up to date | Whether the student's tuition fees are up to date. (Categorical) 1 – yes 0 – no|
|Gender | The gender of the student. (Categorical) 1 – male 0 – female |
|Scholarship holder | Whether the student is a scholarship holder. (Categorical) 1 – yes 0 – no |
|Age at enrollment | The age of the student at the time of enrollment. (Numerical)|
|International | Whether the student is an international student. (Categorical) 1 – yes 0 – no|
|Curricular units 1st sem (credited) | The number of curricular units credited by the student in the first semester. (Numerical) |
| Curricular units 1st sem (enrolled) | The number of curricular units enrolled by the student in the first semester. (Numerical) |
| Curricular units 1st sem (evaluations) | The number of curricular units evaluated by the student in the first semester. (Numerical) |
| Curricular units 1st sem (approved) | The number of curricular units approved by the student in the first semester. (Numerical) |

V. Realinho, M. Vieira Martins, J. Machado, and L. Baptista. "Predict Students' Dropout and Academic Success," UCI Machine Learning Repository, 2021. [Online]. Available: https://doi.org/10.24432/C5MC89.

Setup environment:

- Proses machine learning dilakukan mengunakan jupyter notebook. Setup environment dilakukan dengan menjalankan kode berikut:
```sh
conda create --name p-school-ds python=3.9
```
```sh
conda activate school-ds
```
```sh
pip install -r requirements.txt

```

- Menggunakan Streamlit
```sh
streamlit run app.py
```

- Business Dashboard dibuat menggunakan Tableau 

## Business Dashboard
Dashboard dibagi menjadi dua halaman yaitu family background dan the student. Setiap halaman berisi grafik interaktif yang dapat digunakan untuk menganalisa pengaruh dropout. Dashbard dapat di akses memnggunakan link berikut: https://public.tableau.com/shared/FBDC7CPDF?:display_count=n&:origin=viz_share_link&:device=desktop

## Menjalankan Sistem Machine Learning
Prototype sistem machine learning dapat diakses menggunakan link berikut: 
https://school-project-2queaqxz9xictfkbpm7vjy.streamlit.app/
## Conclusion
- Marital status tidak terlalu berpengaruh terhadap dropout
- Pendidikan ibu dan ayah paling banyak siswa dropout adalah Basic education 1st cycle (4th/5th year) or equiv. sebanyak 383 dan 432 siswa secara berurutan.
- Pekerjaan paling banyak ibu dan ayah siswa dropout adalah unskilled workers sebanyak 490 dan 323 siswa
- Paling banyak siswa dropout adalah yang memiliki application mode over 23 years old sebanyak 435 siswa
- Cours paling banyak siswa dropout adalah Management(evening attendance) sebanyak 136 siswa
- Rata-rata nilai semester pertama siswa dropout adalah 7.257
- Rata-rata nilai semester kedua siswa dropout adalah 5.899
- Sistem machine learning dapat digunakan untuk memprediksi kemungkinan dropout siswa yang dapat diakses di link berikut : https://school-project-2queaqxz9xictfkbpm7vjy.streamlit.app/

### Rekomendasi Action Items
- Mengumpulkan siswa dengan kategori rentan dropout {pendidikan kedua orang tua -Basic education 1st cycle (4th/5th year) or equiv, pekerjaan kedua orag tua - unskilled workers, application mode - over 23 years old, cours - mnageent(evening attendance), nilai semester pertama mendekati 7.257, nilai semester kedua mendekati 5.899} selanjutnya memberikan pembekalan dan menjelaskan posisi mereka sebagai kategori rentan dropout. Harapannya ini dapat memicu siswa rentan tersebut untuk mengevaluasi diri agar tidak sampai dropout.
- Background merupakan hal yang tidak bisa diubah, maka siswa dengan background yang berada pada kategori rentan dropout dapat meningkatkan nilai semester 1 dan 2 nya agar terhindar dari dropout.
- Menggunakan prototype machine learning terhadap seluruh siswa secara berkala sehingga pihak sekolah dapat mencegah dropout siswa yang memiliki kemungkinan untuk dropout.
- Pada waktu penerimaan siswa baru sekolah dapat memfokuskan penerimaan pada siswa yang tidak berada pada kategori rentan dropout.
