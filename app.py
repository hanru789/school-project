import streamlit as st

# Judul aplikasi
st.title("Formulir Kontak")

# Membuat formulir menggunakan with st.form
with st.form("form_kontak"):
    nama = st.text_input("Nama Lengkap")
    email = st.text_input("Email")
    pesan = st.text_area("Pesan")

    # Tombol submit
    submitted = st.form_submit_button("Kirim")

    # Tampilkan hasil jika tombol ditekan
    if submitted:
        if nama and email and pesan:
            st.success(f"Terima kasih, {nama}! Pesan Anda telah dikirim.")
            st.write("📧 Email:", email)
            st.write("📝 Pesan:", pesan)
        else:
            st.warning("Harap lengkapi semua kolom sebelum mengirim.")
