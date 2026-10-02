import streamlit as st

st.set_page_config(
    page_title="WFS Auth Test",
    page_icon="🔐",
    layout="centered",
)

st.title("🔐 Wynners Fantasy Spot")
st.subheader("Google Authentication Test")

if not st.user.is_logged_in:
    st.info("You are not signed in.")
    if st.button("Continue with Google", use_container_width=True):
        st.login()
    st.stop()

st.success("Google authentication is working.")

st.write("### Signed-in identity")

st.write(
    {
        "name": st.user.get("name"),
        "email": st.user.get("email"),
        "sub_present": bool(st.user.get("sub")),
    }
)

if st.button("Log out", use_container_width=True):
    st.logout()
