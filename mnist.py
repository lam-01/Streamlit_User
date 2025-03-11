import streamlit as st
import mlflow
import mlflow.sklearn
import numpy as np
import cv2
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import matplotlib.pyplot as plt
import time  # Thêm để giả lập tiến trình

# 📌 Tải và xử lý dữ liệu MNIST từ OpenML
@st.cache_data
def load_data():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)
    X = X / 255.0
    return X, y

# 📌 Chia dữ liệu thành train, validation, và test
@st.cache_data
def split_data(X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size / (train_size + val_size), random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

# 📌 Huấn luyện mô hình với thanh tiến trình
def train_model(custom_model_name, model_name, params, X_train, X_val, X_test, y_train, y_val, y_test, progress_bar):
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

    # Giả lập tiến trình huấn luyện (vì sklearn không có callback trực tiếp)
    progress_bar.progress(0)  # Khởi tạo thanh tiến trình
    time.sleep(0.1)  # Đợi một chút để giao diện cập nhật
    
    # Bắt đầu huấn luyện
    model.fit(X_train, y_train)
    progress_bar.progress(50)  # 50% sau khi huấn luyện xong
    
    # Dự đoán trên các tập
    y_train_pred = model.predict(X_train)
    progress_bar.progress(60)
    y_val_pred = model.predict(X_val)
    progress_bar.progress(70)
    y_test_pred = model.predict(X_test)
    progress_bar.progress(80)
    
    # Tính độ chính xác
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    progress_bar.progress(90)
    
    # Lưu vào MLflow
    with mlflow.start_run(run_name=custom_model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.sklearn.log_model(model, model_name)
    
    progress_bar.progress(100)  # Hoàn tất
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

# 📌 Giao diện Streamlit
def create_streamlit_app():
    st.title("🔢 Phân loại chữ số viết tay")
    
    # Lưu trữ mô hình trong session_state để tránh huấn luyện lại
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    X, y = load_data()
    tab1, tab2, tab3 = st.tabs(["📋 Huấn luyện", "🔮 Dự đoán", "⚡ MLflow"])
    
    with tab1:
        st.write(f"**Số lượng mẫu của bộ dữ liệu MNIST: {X.shape[0]}**")
        show_sample_images(X, y)
        
        st.write("**📊 Tỷ lệ dữ liệu**")
        test_size = st.slider("Tỷ lệ Test (%)", min_value=5, max_value=30, value=15, step=5)
        val_size = st.slider("Tỷ lệ Validation (%)", min_value=5, max_value=30, value=15, step=5)
        
        train_size = 100 - test_size - val_size
        if train_size <= 0:
            st.error("Tổng tỷ lệ vượt quá 100%! Vui lòng điều chỉnh lại.")
        else:
            X_train, X_val, X_test, y_train, y_val, y_test = split_data(
                X, y, train_size=train_size/100, val_size=val_size/100, test_size=test_size/100
            )
            data_ratios = pd.DataFrame({
                "Tập dữ liệu": ["Train", "Validation", "Test"],
                "Tỷ lệ (%)": [train_size, val_size, test_size],
                "Số lượng mẫu": [len(X_train), len(X_val), len(X_test)]
            })
            st.table(data_ratios)

        st.write("**🚀 Huấn luyện mô hình**")
        custom_model_name = st.text_input("Nhập tên mô hình để lưu vào MLflow:", "My_Model")
        model_name = st.selectbox("🔍 Chọn mô hình", ["Decision Tree", "SVM"])
        params = {}

        if model_name == "Decision Tree":
            params["criterion"] = st.selectbox("📏 Tiêu chí đánh giá", ["gini", "entropy", "log_loss"], help="...")
            params["max_depth"] = st.slider("🌳 Độ sâu tối đa (max_depth)", 1, 30, 15, help="...")
            params["min_samples_split"] = st.slider("🔄 Số mẫu tối thiểu để chia nhánh (min_samples_split)", 2, 10, 5, help="...")
            params["min_samples_leaf"] = st.slider("🍃 Số mẫu tối thiểu ở lá (min_samples_leaf)", 1, 10, 2, help="...")
        elif model_name == "SVM":
            params["kernel"] = st.selectbox("⚙️ Kernel", ["linear", "rbf", "poly", "sigmoid"], help="...")
            params["C"] = st.slider("🔧 Tham số C", 0.1, 10.0, 1.0, help="...")

        if st.button("🚀 Huấn luyện mô hình"):
            with st.spinner("🔄 Đang huấn luyện..."):
                progress_bar = st.progress(0)  # Khởi tạo thanh tiến trình
                model, train_accuracy, val_accuracy, test_accuracy = train_model(
                    custom_model_name, model_name, params, X_train, X_val, X_test, y_train, y_val, y_test, progress_bar
                )
                st.session_state.model = model  # Lưu mô hình vào session_state
            st.success(f"✅ Huấn luyện xong!")
            st.write(f"🎯 **Độ chính xác trên tập train: {train_accuracy:.4f}**")
            st.write(f"🎯 **Độ chính xác trên tập validation: {val_accuracy:.4f}**")
            st.write(f"🎯 **Độ chính xác trên tập test: {test_accuracy:.4f}**")

    with tab2:
        if st.session_state.model is None:
            st.warning("⚠️ Vui lòng huấn luyện mô hình trước khi dự đoán!")
        else:
            option = st.radio("🖼️ Chọn phương thức nhập:", ["📂 Tải ảnh lên", "✏️ Vẽ số"])
            if option == "📂 Tải ảnh lên":
                uploaded_file = st.file_uploader("📤 Tải ảnh số viết tay (PNG, JPG)", type=["png", "jpg", "jpeg"])
                if uploaded_file is not None:
                    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
                    processed_image = preprocess_uploaded_image(image)
                    st.image(image, caption="📷 Ảnh tải lên", use_column_width=True)
                    if st.button("🔮 Dự đoán"):
                        prediction = st.session_state.model.predict(processed_image)[0]
                        probabilities = st.session_state.model.predict_proba(processed_image)[0]
                        st.write(f"🎯 **Dự đoán: {prediction}**")
                        st.write(f"🔢 **Độ tin cậy: {probabilities[prediction] * 100:.2f}%**")
            elif option == "✏️ Vẽ số":
                canvas_result = st_canvas(
                    fill_color="white", stroke_width=15, stroke_color="black",
                    background_color="white", width=280, height=280, drawing_mode="freedraw", key="canvas"
                )
                if st.button("🔮 Dự đoán"):
                    if canvas_result.image_data is not None:
                        processed_canvas = preprocess_canvas_image(canvas_result.image_data)
                        prediction = st.session_state.model.predict(processed_canvas)[0]
                        probabilities = st.session_state.model.predict_proba(processed_canvas)[0]
                        st.write(f"🎯 **Dự đoán: {prediction}**")
                        st.write(f"🔢 **Độ tin cậy: {probabilities[prediction] * 100:.2f}%**")

    with tab3:
        st.header("📊 MLflow Tracking")
        st.write("Xem chi tiết các kết quả đã lưu trong MLflow.")

        runs = mlflow.search_runs(order_by=["start_time desc"])
        if not runs.empty:
            runs["model_custom_name"] = runs["tags.mlflow.runName"]
            model_names = runs["model_custom_name"].dropna().unique().tolist()

            search_model_name = st.text_input("🔍 Nhập tên mô hình để tìm kiếm:", "")
            if search_model_name:
                filtered_runs = runs[runs["model_custom_name"].str.contains(search_model_name, case=False, na=False)]
            else:
                filtered_runs = runs

            if not filtered_runs.empty:
                st.write("### 📜 Danh sách mô hình đã lưu:")
                display_df = filtered_runs[["model_custom_name", "params.model_name", "run_id", "start_time", 
                                           "metrics.train_accuracy", "metrics.val_accuracy", "metrics.test_accuracy"]]
                display_df = display_df.rename(columns={
                    "model_custom_name": "Custom Model Name",
                    "params.model_name": "Model Type"
                })
                st.dataframe(display_df)

                selected_run_id = st.selectbox("📝 Chọn một mô hình để xem chi tiết:", filtered_runs["run_id"].tolist())
                if selected_run_id:
                    run_details = mlflow.get_run(selected_run_id)
                    custom_name = run_details.data.tags.get('mlflow.runName', 'Không có tên')
                    model_type = run_details.data.params.get('model_name', 'Không xác định')
                    st.write(f"### 🔍 Chi tiết mô hình: `{custom_name}`")
                    st.write(f"**📌 Loại mô hình huấn luyện:** {model_type}")
                    st.write("📌 **Tham số:**")
                    for key, value in run_details.data.params.items():
                        if key != 'model_name':
                            st.write(f"- **{key}**: {value}")
                    st.write("📊 **Metric:**")
                    for key, value in run_details.data.metrics.items():
                        st.write(f"- **{key}**: {value}")
                    st.write("📂 **Artifacts:**")
                    if run_details.info.artifact_uri:
                        st.write(f"- **Artifact URI**: {run_details.info.artifact_uri}")
                    else:
                        st.write("- Không có artifacts nào.")
            else:
                st.write("❌ Không tìm thấy mô hình nào.")
        else:
            st.write("⚠️ Không có phiên làm việc nào được ghi lại.")

if __name__ == "__main__":
    create_streamlit_app()
