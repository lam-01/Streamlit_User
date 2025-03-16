import importlib
import streamlit as st

# Tạo selectbox để chọn dự án
option = st.sidebar.selectbox(
    "📌 Chọn một dự án để thực hiện:",
    ["Phân tích Titanic", "MNIST","Clustering Algorithms","PCA & t-SNE","Neural Network","Pseudo Labelling"]
)

# Hiển thị nội dung tương ứng với lựa chọn
if option == "Phân tích Titanic":
    titanic_app = importlib.import_module("titanic_app")
    titanic_app.create_streamlit_app()

elif option == "MNIST":
    mnist_app = importlib.import_module("mnist")
    if hasattr(mnist_app, "create_streamlit_app"):
        mnist_app.create_streamlit_app()
    else:
        st.error("❌ Module MNIST không có hàm `create_streamlit_app()`")
elif option=="Clustering Algorithms":
    with open("dbscan.py", "r", encoding="utf-8") as file:
        code = file.read()
        exec(code)
elif option=="PCA & t-SNE":
    with open("pca.py", "r", encoding="utf-8") as file:
        code = file.read()
        exec(code)
elif option=="Neural Network":
    with open("NeuralNetwork.py", "r", encoding="utf-8") as file:
        code = file.read()
        exec(code)
elif option=="Pseudo Labelling":
    with open("Pseudo.py", "r", encoding="utf-8") as file:
        code = file.read()
        exec(code)

