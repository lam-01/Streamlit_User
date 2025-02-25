import streamlit as st
import mlflow
import mlflow.sklearn
import numpy as np
import cv2
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_tags import st_tags
import io
import os
import tempfile
import runpy

# 📌 Tải và xử lý dữ liệu MNIST từ OpenML
@st.cache_data
def load_data():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)  # Chuyển nhãn về kiểu số nguyên
    X = X / 255.0  # Chuẩn hóa về [0,1]
    return X, y

# 📌 Chia dữ liệu thành train, validation, và test
def split_data(X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    # Chia tập train và tập test trước
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Chia tiếp tập train thành train và validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size / (train_size + val_size), random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# 📌 Huấn luyện mô hình
def train_model(model_name,params, X_train, X_val, X_test, y_train, y_val, y_test):
    if model_name == "Decision Tree":
        model = DecisionTreeClassifier(
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            criterion=params["criterion"],
            random_state=42
        )
    elif model_name == "SVM":
        model = SVC(
            kernel=params["kernel"],
            C=params["C"],
            probability=True
        )
    else:
        raise ValueError("Invalid model selected!")

    model.fit(X_train, y_train)

    y_train_pred =model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_val_pred = model.predict(X_val)
    # Tính độ chính xác
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Lưu mô hình vào MLFlow
    with mlflow.start_run(run_name="MNIST_Classification"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.sklearn.log_model(model, model_name)
    
    return model, train_accuracy, val_accuracy, test_accuracy

# 📌 Xử lý ảnh tải lên
def preprocess_uploaded_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    image = image / 255.0
    return image.reshape(1, -1)

# 📌 Xử lý ảnh từ vẽ tay trên canvas
def preprocess_canvas_image(canvas):
    image = np.array(canvas)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    image = cv2.bitwise_not(image)
    image = cv2.resize(image, (28, 28))
    image = image / 255.0
    return image.reshape(1, -1)

def show_sample_images(X, y):
    st.write("**🖼️ Một vài mẫu dữ liệu từ MNIST**")
    fig, axes = plt.subplots(1, 10, figsize=(15, 3))
    for digit in range(10):
        idx = np.where(y == digit)[0][0]
        ax = axes[digit]
        ax.imshow(X[idx].reshape(28, 28), cmap='gray')
        ax.set_title(f"{digit}")
        ax.axis('off')
    st.pyplot(fig)


def display_mlflow_experiments():
    try:
        # Lấy danh sách các thí nghiệm từ MLflow
        experiments = mlflow.list_experiments()
        
        if experiments:
            st.write("#### Danh sách thí nghiệm")
            experiment_data = []
            for exp in experiments:
                experiment_data.append({
                    "Experiment ID": exp.experiment_id,
                    "Experiment Name": exp.name,
                    "Artifact Location": exp.artifact_location
                })
            st.dataframe(pd.DataFrame(experiment_data))
            
            # Chọn thí nghiệm để xem chi tiết
            selected_exp_id = st.selectbox(
                "🔍 Chọn thí nghiệm để xem chi tiết",
                options=[exp.experiment_id for exp in experiments]
            )
            
            # Lấy danh sách runs trong thí nghiệm đã chọn
            runs = mlflow.search_runs(selected_exp_id)
            if not runs.empty:
                st.write("#### Danh sách runs")
                st.dataframe(runs)
                
                # Chọn run để xem chi tiết
                selected_run_id = st.selectbox(
                    "🔍 Chọn run để xem chi tiết",
                    options=runs["run_id"]
                )
                
                # Hiển thị chi tiết run
                run = mlflow.get_run(selected_run_id)
                st.write("##### Thông tin run")
                st.write(f"**Run ID:** {run.info.run_id}")
                st.write(f"**Experiment ID:** {run.info.experiment_id}")
                st.write(f"**Start Time:** {run.info.start_time}")
                
                # Hiển thị metrics
                st.write("##### Metrics")
                st.json(run.data.metrics)
                
                # Hiển thị params
                st.write("##### Params")
                st.json(run.data.params)
                
                # Hiển thị artifacts
                artifacts = mlflow.list_artifacts(selected_run_id)
                if artifacts:
                    st.write("##### Artifacts")
                    for artifact in artifacts:
                        st.write(f"- {artifact.path}")
                else:
                    st.write("Không có artifacts nào.")
            else:
                st.warning("Không có runs nào trong thí nghiệm này.")
        else:
            st.warning("Không có thí nghiệm nào được tìm thấy.")
    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi lấy danh sách thí nghiệm: {e}")

# 📌 Giao diện Streamlit
def create_streamlit_app():
    st.title("🔢 Phân loại chữ số viết tay")
    
    # # Load dữ liệu
    X, y = load_data()
    # Tạo các tab
    tab1, tab2 ,tab3= st.tabs(["📋 Huấn luyện", "🔮 Dự đoán","⚡ Mlflow"])
    with tab1:
        st.write(f"**Số lượng mẫu của bộ dữ liệu MNIST : {X.shape[0]}**")
        # Hiển thị mẫu dữ liệu và phân phối dữ liệu
        show_sample_images(X, y)
        
        st.write("**📊 Tỷ lệ dữ liệu**")
        # Chọn tỷ lệ dữ liệu Test và Validation
        test_size = st.slider("Tỷ lệ Test (%)", min_value=5, max_value=30, value=15, step=5)
        val_size = st.slider("Tỷ lệ Validation (%)", min_value=5, max_value=30, value=15, step=5)

        # Tính toán tỷ lệ Train
        train_size = 100 - test_size  # Tỷ lệ Train là phần còn lại sau khi trừ Test
        val_ratio = val_size / train_size  # Tỷ lệ Validation trên tập Train

        # Kiểm tra tính hợp lệ
        if val_ratio >= 1.0:
            st.error("Tỷ lệ Validation quá lớn so với Train! Vui lòng điều chỉnh lại.")
        else:
            # Chia dữ liệu
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, random_state=42)

            # Hiển thị bảng tỷ lệ
            data_ratios = pd.DataFrame({
                "Tập dữ liệu": ["Train", "Validation", "Test"],
                "Tỷ lệ (%)": [train_size - val_size, val_size, test_size]
            })
            st.table(data_ratios)

            # Hiển thị số lượng mẫu
            st.write(f"🧮 Số lượng mẫu Train: {len(X_train)}")
            st.write(f"🧮 Số lượng mẫu Validation: {len(X_val)}")
            st.write(f"🧮 Số lượng mẫu Test: {len(X_test)}")


        st.write("**🚀 Huấn luyện mô hình**")
        model_name = st.selectbox("🔍 Chọn mô hình", ["Decision Tree", "SVM"])
        params = {}

        if model_name == "Decision Tree":
            params["criterion"] = st.selectbox("📏 Tiêu chí đánh giá", ["gini", "entropy", "log_loss"])
            params["max_depth"] = st.slider("🌳 Độ sâu tối đa (max_depth)", 1, 30, 15)
            params["min_samples_split"] = st.slider("🔄 Số mẫu tối thiểu để chia nhánh (min_samples_split)", 2, 10, 5)
            params["min_samples_leaf"] = st.slider("🍃 Số mẫu tối thiểu ở lá (min_samples_leaf)", 1, 10, 2)

        elif model_name == "SVM":
            params["kernel"] = st.selectbox("⚙️ Kernel", ["linear", "rbf", "poly", "sigmoid"])
            params["C"] = st.slider("🔧 Tham số C ", 0.1, 10.0, 1.0)
        # Huấn luyện mô hình
        if st.button("🚀 Huấn luyện mô hình"):
            with st.spinner("🔄 Đang huấn luyện..."):
                model, train_accuracy, val_accuracy, test_accuracy = train_model(
                model_name,params, X_train, X_val, X_test, y_train, y_val, y_test
            )
            st.success(f"✅ Huấn luyện xong!")
            
            # Hiển thị độ chính xác trên cả 3 tập dữ liệu
            st.write(f"🎯 **Độ chính xác trên tập train: {train_accuracy:.4f}**")
            st.write(f"🎯 **Độ chính xác trên tập validation: {val_accuracy:.4f}**")
            st.write(f"🎯 **Độ chính xác trên tập test: {test_accuracy:.4f}**")

    with tab2:
        # Chọn phương thức nhập ảnh
        option = st.radio("🖼️ Chọn phương thức nhập:", ["📂 Tải ảnh lên", "✏️ Vẽ số"])

        # 📂 Xử lý ảnh tải lên
        if option == "📂 Tải ảnh lên":
            uploaded_file = st.file_uploader("📤 Tải ảnh số viết tay (PNG, JPG)", type=["png", "jpg", "jpeg"])

            if uploaded_file is not None:
                image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
                processed_image = preprocess_uploaded_image(image)

                # Hiển thị ảnh
                st.image(image, caption="📷 Ảnh tải lên", use_column_width=True)

                # Dự đoán số
                if st.button("🔮 Dự đoán"):
                    model, train_accuracy, val_accuracy, test_accuracy= train_model(model_name, X_train, X_val, X_test, y_train, y_val, y_test)
                    prediction = model.predict(processed_image)[0]
                    probabilities = model.predict_proba(processed_image)[0]

                    st.write(f"🎯 **Dự đoán: {prediction}**")
                    st.write(f"🔢 **Độ tin cậy: {probabilities[prediction] * 100:.2f}%**")

        # ✏️ Vẽ số trên canvas
        elif option == "✏️ Vẽ số":
            canvas_result = st_canvas(
                fill_color="white",
                stroke_width=15,
                stroke_color="black",
                background_color="white",
                width=280,
                height=280,
                drawing_mode="freedraw",
                key="canvas"
            )

            if st.button("🔮 Dự đoán"):
                if canvas_result.image_data is not None:
                    processed_canvas = preprocess_canvas_image(canvas_result.image_data)

                    model, train_accuracy, val_accuracy, test_accuracy= train_model(model_name, X_train, X_val, X_test, y_train, y_val, y_test)
                    prediction = model.predict(processed_canvas)[0]
                    probabilities = model.predict_proba(processed_canvas)[0]

                    st.write(f"🎯 **Dự đoán: {prediction}**")
                    st.write(f"🔢 **Độ tin cậy: {probabilities[prediction] * 100:.2f}%**")

    with tab3:
        st.write("### 📊 Tracking MLflow")
        
        try:
            # Lấy danh sách thí nghiệm từ MLflow
            experiments = mlflow.search_experiments()
            
            if experiments:
                st.write("#### Danh sách thí nghiệm")
                experiment_data = []
                for exp in experiments:
                    experiment_data.append({
                        "Experiment ID": exp.experiment_id,
                        "Experiment Name": exp.name,
                        "Artifact Location": exp.artifact_location
                    })
                st.dataframe(pd.DataFrame(experiment_data))
                
                # Chọn thí nghiệm để xem chi tiết
                selected_exp_id = st.selectbox(
                    "🔍 Chọn thí nghiệm để xem chi tiết",
                    options=[exp.experiment_id for exp in experiments]
                )
                
                # Lấy danh sách runs trong thí nghiệm đã chọn
                runs = mlflow.search_runs(selected_exp_id)
                if not runs.empty:
                    st.write("#### Danh sách runs")
                    st.dataframe(runs)
                    
                    # Chọn run để xem chi tiết
                    selected_run_id = st.selectbox(
                        "🔍 Chọn run để xem chi tiết",
                        options=runs["run_id"]
                    )
                    
                    # Hiển thị chi tiết run
                    run = mlflow.get_run(selected_run_id)
                    st.write("##### Thông tin run")
                    st.write(f"**Run ID:** {run.info.run_id}")
                    st.write(f"**Experiment ID:** {run.info.experiment_id}")
                    st.write(f"**Start Time:** {run.info.start_time}")
                    
                    # Hiển thị metrics
                    st.write("##### Metrics")
                    st.json(run.data.metrics)
                    
                    # Hiển thị params
                    st.write("##### Params")
                    st.json(run.data.params)
                    
                    # Hiển thị artifacts
                    artifacts = mlflow.list_artifacts(selected_run_id)
                    if artifacts:
                        st.write("##### Artifacts")
                        for artifact in artifacts:
                            st.write(f"- {artifact.path}")
                else:
                    st.warning("Không có runs nào trong thí nghiệm này.")
            else:
                st.warning("Không có thí nghiệm nào được tìm thấy.")
        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi lấy danh sách thí nghiệm: {e}")

if __name__ == "__main__":
    create_streamlit_app()
