import streamlit as st
import mlflow
import mlflow.sklearn
import numpy as np
import cv2
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# 📌 Tải và xử lý dữ liệu MNIST từ OpenML
@st.cache_data
def load_data(sample_size=None):
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)
    X = X / 255.0
    if sample_size is not None and sample_size < len(X):
        X, _, y, _ = train_test_split(X, y, train_size=sample_size, random_state=42)
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

# 📌 Huấn luyện mô hình với cross-validation (chỉ giữ mean)
def train_model(custom_model_name, model_name, params, X_train, X_val, X_test, y_train, y_val, y_test, cv_folds):
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Đang khởi tạo mô hình... (0%)")

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

    try:
        with mlflow.start_run(run_name=custom_model_name):
            # Bước 1: Khởi tạo mô hình
            progress_bar.progress(0.1)
            status_text.text("Đang thực hiện cross-validation... (10%)")
            start_time = time.time()

            # Bước 2: Cross-validation (chỉ tính mean)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="accuracy")
            cv_mean = np.mean(cv_scores)
            progress_bar.progress(0.3)
            status_text.text(f"Cross-validation hoàn tất ({cv_folds} folds)... (30%)")

            # Bước 3: Huấn luyện mô hình trên toàn bộ tập train
            model.fit(X_train, y_train)
            train_end_time = time.time()
            train_duration = train_end_time - start_time
            progress_bar.progress(0.5)
            status_text.text("Đã huấn luyện xong... (50%)")

            # Bước 4: Dự đoán trên các tập dữ liệu
            y_train_pred = model.predict(X_train)
            progress_bar.progress(0.6)
            status_text.text("Đang dự đoán trên tập train... (60%)")

            y_val_pred = model.predict(X_val)
            progress_bar.progress(0.7)
            status_text.text("Đang dự đoán trên tập validation... (70%)")

            y_test_pred = model.predict(X_test)
            predict_end_time = time.time()
            predict_duration = predict_end_time - train_end_time
            progress_bar.progress(0.8)
            status_text.text("Đã dự đoán xong... (80%)")

            # Tính toán độ chính xác
            train_accuracy = accuracy_score(y_train, y_train_pred)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            # Bước 5: Ghi log vào MLflow
            status_text.text("Đang ghi log vào MLflow... (90%)")
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("cv_folds", cv_folds)
            mlflow.log_params(params)
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("val_accuracy", val_accuracy)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("cv_mean_accuracy", cv_mean)
            
            input_example = X_train[:1]
            mlflow.sklearn.log_model(model, custom_model_name, input_example=input_example)
            progress_bar.progress(1.0)
            status_text.text("Hoàn tất! (100%)")
    except Exception as e:
        st.error(f"Lỗi trong quá trình huấn luyện: {str(e)}")
        return None, None, None, None, None

    return model, train_accuracy, val_accuracy, test_accuracy, cv_mean

# 📌 Hàm tải mô hình từ MLflow dựa trên custom_model_name
def load_model_from_mlflow(custom_model_name):
    runs = mlflow.search_runs(order_by=["start_time desc"])
    if not runs.empty:
        run = runs[runs["tags.mlflow.runName"] == custom_model_name]
        if not run.empty:
            run_id = run.iloc[0]["run_id"]
            model_uri = f"runs:/{run_id}/{custom_model_name}"
            return mlflow.sklearn.load_model(model_uri)
    return None

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
    
    tab1, tab2, tab3, tab4 = st.tabs(["📓 Lí thuyết", "📋 Huấn luyện", "🔮 Dự đoán", "⚡ MLflow"])
    with tab1:
        algorithm = st.selectbox("Chọn thuật toán:", ["Decision Tree", "SVM"])
        if algorithm == "Decision Tree":
            st.write("##### Decision Tree")
            st.write("###### Các tiêu chí đánh giá phân chia trong Decision Tree")
            st.write("**1. Gini Index (Chỉ số Gini)**")
            st.write("- **Định nghĩa**: Đo lường mức độ 'không thuần khiết' của tập dữ liệu.")
            st.latex(r"Gini = 1 - \sum_{i=1}^{n} p_i^2")
            st.markdown("Với $$( p_i $$) là tỷ lệ của lớp $$( i $$) trong tập dữ liệu.")
    
            st.write("**2. Entropy**")
            st.write("- **Định nghĩa**: Đo lường mức độ hỗn loạn (uncertainty) trong tập dữ liệu.")
            st.latex(r"Entropy = - \sum_{i=1}^{n} p_i \log_2(p_i)")
            st.write("Với $$( p_i $$) là tỷ lệ của lớp $$( i $$).")
    
            st.write("**3. Log Loss (Hàm mất mát Logarit)**")
            st.write("- **Định nghĩa**: Đo lường sai lệch giữa xác suất dự đoán và nhãn thực tế.")
            st.latex(r"Log\ Loss = - \frac{1}{N} \sum_{i=1}^{N} [y_i \log(p_i) + (1 - y_i) \log(1 - p_i)]")
            st.write("Với $$( N $$) là số mẫu, $$( y_i $$) là nhãn thực tế, $$( p_i $$) là xác suất dự đoán.")
        elif algorithm == "SVM":
            st.write("##### Support Vector Machine (SVM)")
            st.write("###### Các kernel trong SVM")
            st.write("**1. Linear Kernel (Kernel Tuyến tính)**")
            st.latex(r"K(x, x') = x \cdot x'")
            x = np.linspace(-2, 2, 100)
            k_linear = x
            fig, ax = plt.subplots(figsize=(3, 2))
            ax.plot(x, k_linear, label="Linear Kernel")
            ax.set_xlabel("x", fontsize=6)
            ax.set_ylabel("K(x, x')", fontsize=6)
            ax.legend(loc='upper left', fontsize=6)
            ax.grid(True)
            st.pyplot(fig)
        
            st.write("**2. RBF Kernel (Radial Basis Function)**")
            st.latex(r"K(x, x') = \exp\left(-\frac{||x - x'||^2}{2\sigma^2}\right)")
            dist = np.linspace(0, 3, 100)
            sigma = 1.0
            k_rbf = np.exp(-dist**2 / (2 * sigma**2))
            fig, ax = plt.subplots(figsize=(3, 2))
            ax.plot(dist, k_rbf, label="RBF Kernel (σ=1)")
            ax.set_xlabel("||x - x'||", fontsize=6)
            ax.set_ylabel("K(x, x')", fontsize=6)
            ax.legend(loc='upper right', fontsize=6)
            ax.grid(True)
            st.pyplot(fig)
    
            st.write("**3. Polynomial Kernel (Kernel Đa thức)**")
            st.latex(r"K(x, x') = (x \cdot x' + c)^d")
            x = np.linspace(-2, 2, 100)
            k_poly_d2 = (x + 1)**2
            k_poly_d3 = (x + 1)**3
            fig, ax = plt.subplots(figsize=(3, 2))
            ax.plot(x, k_poly_d2, label="Poly Kernel (d=2, c=1)")
            ax.plot(x, k_poly_d3, label="Poly Kernel (d=3, c=1)")
            ax.set_xlabel("x", fontsize=6)
            ax.set_ylabel("K(x, x')", fontsize=6)
            ax.legend(loc='upper left', fontsize=6)
            ax.grid(True)
            st.pyplot(fig)
            
            st.write("**4. Sigmoid Kernel**")
            st.latex(r"K(x, x') = \tanh(\alpha \cdot (x \cdot x') + c)")
            x = np.linspace(-2, 2, 100)
            alpha, c = 1.0, 0.0
            k_sigmoid = np.tanh(alpha * x + c)
            fig, ax = plt.subplots(figsize=(3, 2))
            ax.plot(x, k_sigmoid, label="Sigmoid Kernel (α=1, c=0)")
            ax.set_xlabel("x", fontsize=6)
            ax.set_ylabel("K(x, x')", fontsize=6)
            ax.legend(loc='upper left', fontsize=6)
            ax.grid(True)
            st.pyplot(fig)

    with tab2:
        sample_size = st.number_input("**Chọn cỡ mẫu để huấn luyện**", 1000, 70000, 10000, step=1000)
        X, y = load_data(sample_size=sample_size)
        st.write(f"**Số lượng mẫu của bộ dữ liệu: {X.shape[0]}**")
        show_sample_images(X, y)
        
        st.write("**📊 Tỷ lệ dữ liệu**")
        test_size = st.slider("Tỷ lệ Test (%)", min_value=5, max_value=30, value=15, step=5)
        val_size = st.slider("Tỷ lệ Validation (%)", min_value=5, max_value=30, value=15, step=5)
        
        train_size = 100 - test_size
        val_ratio = val_size / train_size
        
        if val_ratio >= 1.0:
            st.error("Tỷ lệ Validation quá lớn so với Train! Vui lòng điều chỉnh lại.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, random_state=42)
        
            data_ratios = pd.DataFrame({
                "Tập dữ liệu": ["Train", "Validation", "Test"],
                "Tỷ lệ (%)": [train_size - val_size, val_size, test_size],
                "Số lượng mẫu": [len(X_train), len(X_val), len(X_test)]
            })
            st.table(data_ratios)

        st.write("**🚀 Huấn luyện mô hình**")
        custom_model_name = st.text_input("Nhập tên mô hình :", "")
        if not custom_model_name:
            custom_model_name = "Default_model"

        model_name = st.selectbox("🔍 Chọn mô hình", ["Decision Tree", "SVM"])
        params = {}

        if model_name == "Decision Tree":
            params["criterion"] = st.selectbox("📏 Tiêu chí phân tách", ["gini", "entropy", "log_loss"])
            params["max_depth"] = st.slider("🌳 Độ sâu tối đa (max_depth)", 1, 30, 15)
            params["min_samples_split"] = st.slider("🔄 Số mẫu tối thiểu để chia nhánh (min_samples_split)", 2, 10, 5)
            params["min_samples_leaf"] = st.slider("🍃 Số mẫu tối thiểu ở lá (min_samples_leaf)", 1, 10, 2)
        elif model_name == "SVM":
            params["kernel"] = st.selectbox("⚙️ Kernel", ["linear", "rbf", "poly", "sigmoid"])
            params["C"] = st.slider("🔧 Tham số C ", 0.1, 10.0, 1.0)

        # Sử dụng slider cho số fold
        cv_folds = st.slider("🔢 Số fold cho Cross-Validation", 3, 10, 5)

        if st.button("🚀 Huấn luyện mô hình"):
            with st.spinner("🔄 Đang khởi tạo huấn luyện..."):
                model, train_accuracy, val_accuracy, test_accuracy, cv_mean = train_model(
                    custom_model_name, model_name, params, X_train, X_val, X_test, y_train, y_val, y_test, cv_folds
                )
            
            if model is not None:
                st.success(f"✅ Huấn luyện xong!")
                st.write(f"🎯 **Độ chính xác trên tập train: {train_accuracy:.4f}**")
                st.write(f"🎯 **Độ chính xác trên tập validation: {val_accuracy:.4f}**")
                st.write(f"🎯 **Độ chính xác trên tập test: {test_accuracy:.4f}**")
                st.write(f"📊 **Cross-Validation ({cv_folds} folds) - Độ chính xác trung bình: {cv_mean:.4f}**")
            else:
                st.error("Huấn luyện thất bại, không có kết quả để hiển thị.")

    with tab3:
        runs = mlflow.search_runs(order_by=["start_time desc"])
        model_names = runs["tags.mlflow.runName"].dropna().unique().tolist() if not runs.empty else ["Không có mô hình nào"]
        
        st.write("**📝 Chọn mô hình để dự đoán**")
        selected_model_name = st.selectbox("Chọn tên mô hình:", model_names)

        option = st.radio("🖼️ Chọn phương thức nhập:", ["📂 Tải ảnh lên", "✏️ Vẽ số"])
        if option == "📂 Tải ảnh lên":
            uploaded_file = st.file_uploader("📤 Tải ảnh số viết tay (PNG, JPG)", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
                processed_image = preprocess_uploaded_image(image)
                st.image(image, caption="📷 Ảnh tải lên", use_column_width=True)
                if st.button("🔮 Dự đoán"):
                    if selected_model_name != "Không có mô hình nào":
                        model = load_model_from_mlflow(selected_model_name)
                        if model is not None:
                            prediction = model.predict(processed_image)[0]
                            probabilities = model.predict_proba(processed_image)[0]
                            st.write(f"🎯 **Dự đoán: {prediction}**")
                            st.write(f"🔢 **Độ tin cậy: {probabilities[prediction] * 100:.2f}%**")
                        else:
                            st.error("Không thể tải mô hình đã chọn!")
                    else:
                        st.error("Vui lòng chọn một mô hình hợp lệ để dự đoán.")
        elif option == "✏️ Vẽ số":
            canvas_result = st_canvas(
                fill_color="white", stroke_width=15, stroke_color="black",
                background_color="white", width=280, height=280, drawing_mode="freedraw", key="canvas"
            )
            if st.button("🔮 Dự đoán"):
                if canvas_result.image_data is not None:
                    processed_canvas = preprocess_canvas_image(canvas_result.image_data)
                    if selected_model_name != "Không có mô hình nào":
                        model = load_model_from_mlflow(selected_model_name)
                        if model is not None:
                            prediction = model.predict(processed_canvas)[0]
                            probabilities = model.predict_proba(processed_canvas)[0]
                            st.write(f"🎯 **Dự đoán: {prediction}**")
                            st.write(f"🔢 **Độ tin cậy: {probabilities[prediction] * 100:.2f}%**")
                        else:
                            st.error("Không thể tải mô hình đã chọn!")
                    else:
                        st.error("Vui lòng chọn một mô hình hợp lệ để dự đoán.")

    with tab4:
        st.write("##### 📊 MLflow Tracking")
        st.write("Xem chi tiết các kết quả đã lưu trong MLflow.")

        runs = mlflow.search_runs(order_by=["start_time desc"])
        if not runs.empty:
            runs["model_custom_name"] = runs["tags.mlflow.runName"]

            if "params.model_name" in runs.columns:
                model_names = runs["params.model_name"].dropna().unique().tolist()
            else:
                model_names = ["Không xác định"]

            search_model_name = st.text_input("🔍 Nhập tên mô hình để tìm kiếm:", "")
            if search_model_name:
                filtered_runs = runs[runs["model_custom_name"].str.contains(search_model_name, case=False, na=False)]
            else:
                filtered_runs = runs

            if not filtered_runs.empty:
                st.write("##### 📜 Danh sách mô hình đã lưu:")
                available_columns = [col for col in ["model_custom_name", "params.model_name", "start_time", 
                                                     "metrics.train_accuracy", "metrics.val_accuracy", 
                                                     "metrics.test_accuracy", "metrics.cv_mean_accuracy"] 
                                     if col in runs.columns]
                display_df = filtered_runs[available_columns]
                
                for col in display_df.columns:
                    if display_df[col].dtype == 'object':
                        display_df[col] = display_df[col].astype(str)
                
                display_df = display_df.rename(columns={
                    "model_custom_name": "Custom Model Name",
                    "params.model_name": "Model Type",
                    "metrics.cv_mean_accuracy": "CV Mean Accuracy"
                })
                st.dataframe(display_df)

                selected_model_name = st.selectbox("📝 Chọn một mô hình để xem chi tiết:", 
                                                   filtered_runs["model_custom_name"].tolist())
                if selected_model_name:
                    selected_run = filtered_runs[filtered_runs["model_custom_name"] == selected_model_name].iloc[0]
                    run_details = mlflow.get_run(selected_run["run_id"])
                    custom_name = run_details.data.tags.get('mlflow.runName', 'Không có tên')
                    model_type = run_details.data.params.get('model_name', 'Không xác định')
                    st.write(f"##### 🔍 Chi tiết mô hình: `{custom_name}`")
                    st.write(f"**📌 Loại mô hình huấn luyện:** {model_type}")

                    st.write("📌 **Tham số:**")
                    for key, value in run_details.data.params.items():
                        if key != 'model_name':
                            st.write(f"- **{key}**: {value}")

                    st.write("📊 **Metric:**")
                    for key, value in run_details.data.metrics.items():
                        st.write(f"- **{key}**: {value}")
            else:
                st.write("❌ Không tìm thấy mô hình nào.")
        else:
            st.write("⚠️ Không có phiên làm việc nào được ghi lại.")

if __name__ == "__main__":
    create_streamlit_app()
