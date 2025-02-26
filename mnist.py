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
def train_model(model_name,params, X_train, X_val, X_test, y_train, y_val, y_test,custom_name=""):
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


    run_name = custom_name if custom_name else f"{model_name}_Classification"
   
    with mlflow.start_run(run_name=run_name):
        # Log các tham số chi tiết của mô hình
        mlflow.log_param("model_name", model_name)
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        
        # Log các metrics
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        
        # Lưu mô hình
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
        # Nhập tên mô hình
        model_custom_name = st.text_input("Nhập tên mô hình để lưu vào MLflow:")
        mlflow.log_param("model_custom_name", model_custom_name)
        # Chọn mô hình
        model_name = st.selectbox("🔍 Chọn mô hình", ["Decision Tree", "SVM"])
        params = {}

        if model_name == "Decision Tree":
            params["criterion"] = st.selectbox("📏 Tiêu chí đánh giá", ["gini", "entropy", "log_loss"],help="""- **Gini impurity** đo lường xác suất một mẫu được chọn ngẫu nhiên từ tập dữ liệu bị phân loại sai 
            nếu nó được gán nhãn ngẫu nhiên theo phân phối của các lớp trong tập dữ liệu.
            \n- **Entropy** đo lường mức độ hỗn loạn hoặc không chắc chắn trong tập dữ liệu. Nó dựa trên khái niệm entropy trong lý thuyết thông tin.
            \n- **Log loss (hay cross-entropy)** đo lường sự khác biệt giữa phân phối xác suất thực tế và phân phối xác suất dự đoán. Nó thường được sử dụng trong các bài toán phân loại xác suất.
            """)
            params["max_depth"] = st.slider("🌳 Độ sâu tối đa (max_depth)", 1, 30, 15,help="""- **max_depth** là tham số giới hạn độ sâu tối đa của cây quyết định. Độ sâu của cây được tính 
            từ nút gốc (root) đến nút lá (leaf) xa nhất.
            \n Nếu (max_depth > 25) quá lớn, cây có thể trở nên phức tạp và dễ bị overfitting (học thuộc dữ liệu huấn luyện nhưng kém hiệu quả trên dữ liệu mới).
            \n Nếu (max_depth < 10) quá nhỏ, cây có thể quá đơn giản và dẫn đến underfitting (không học được đủ thông tin từ dữ liệu).""")
            params["min_samples_split"] = st.slider("🔄 Số mẫu tối thiểu để chia nhánh (min_samples_split)", 2, 10, 5,help="""
            \n- **min_samples_split** là số lượng mẫu tối thiểu cần thiết để chia một nút (node) thành các nút con. Nếu số lượng mẫu tại một nút ít hơn giá trị này, nút đó sẽ không được chia tiếp.
            \n Giá trị lớn hơn (5-10) giúp ngăn chặn việc chia nhánh quá mức, từ đó giảm nguy cơ overfitting.
            \n Giá trị nhỏ hơn (2-4) cho phép cây chia nhánh nhiều hơn, nhưng có thể dẫn đến cây phức tạp hơn.
            
            """)
            params["min_samples_leaf"] = st.slider("🍃 Số mẫu tối thiểu ở lá (min_samples_leaf)", 1, 10, 2,help="""
            \n- **min_samples_leaf** là số lượng mẫu tối thiểu cần thiết tại mỗi nút lá (leaf node). Nếu một phân chia dẫn đến một lá có ít mẫu hơn giá trị này, phân chia đó sẽ không được thực hiện.
            \n Giá trị lớn hơn (5-10) giúp ngăn chặn việc tạo ra các lá quá nhỏ, từ đó giảm nguy cơ overfitting.
            \n Giá trị nhỏ hơn (1-4) cho phép cây tạo ra các lá nhỏ hơn, nhưng có thể dẫn đến cây phức tạp hơn.
            """)

        elif model_name == "SVM":
            params["kernel"] = st.selectbox("⚙️ Kernel", ["linear", "rbf", "poly", "sigmoid"],help="""**Kernel** là một hàm được sử dụng trong SVM để ánh xạ dữ liệu từ không gian đầu vào sang một không gian đặc trưng (feature space) có chiều cao hơn, giúp SVM có thể phân loại dữ liệu phi tuyến tính.
            \n- **Linear Kernel**: một trong những loại kernel đơn giản nhất. Nó được sử dụng khi dữ liệu có thể được phân loại bằng một đường thẳng (hoặc mặt phẳng trong không gian nhiều chiều).
            \n- **RBF Kernel (Radial Basis Function)**: một loại kernel phi tuyến tính, rất phổ biến trong SVM. Nó có khả năng xử lý các mối quan hệ phức tạp giữa các điểm dữ liệu.
            \n- **Polynomial Kernel**: cho phép mô hình hóa các mối quan hệ phi tuyến tính bằng cách sử dụng các đa thức. Tham số bậc của đa thức có thể được điều chỉnh để thay đổi độ phức tạp của mô hình.
            \n- **Sigmoid Kernel**: tương tự như hàm kích hoạt sigmoid trong mạng nơ-ron. Nó có thể được sử dụng để tạo ra các quyết định phi tuyến tính.
            """)
            params["C"] = st.slider("🔧 Tham số C ", 0.1, 10.0, 1.0,help="""\n- **C** là tham số điều chỉnh (regularization parameter) trong SVM, kiểm soát sự đánh đổi giữa việc tạo ra một biên (margin) rộng và việc phân loại chính xác các điểm dữ liệu huấn luyện.
            \n C lớn: Mô hình cố gắng phân loại chính xác tất cả các điểm dữ liệu huấn luyện, có thể dẫn đến overfitting.
            \n C nhỏ: Mô hình cho phép một số điểm dữ liệu bị phân loại sai để tạo ra biên rộng hơn, giúp giảm overfitting.""")
        # Huấn luyện mô hình
        if st.button("🚀 Huấn luyện mô hình"):
            with st.spinner("🔄 Đang huấn luyện..."):
                model, train_accuracy, val_accuracy, test_accuracy = train_model(
                model_name,params, X_train, X_val, X_test, y_train, y_val, y_test,custom_name=model_custom_name
            )
               # Lưu thông tin vào MLFlow
            with mlflow.start_run():
                mlflow.set_tag("model_name", model_custom_name)
                mlflow.log_params(params)
                mlflow.log_metrics({
                    "train_accuracy": train_accuracy,
                    "val_accuracy": val_accuracy,
                    "test_accuracy": test_accuracy
                })
                mlflow.sklearn.log_model(model, "model", registered_model_name=model_custom_name)
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
                    model, train_accuracy, val_accuracy, test_accuracy= train_model(model_name,params, X_train, X_val, X_test, y_train, y_val, y_test,custom_name=model_custom_name)
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

                    model, train_accuracy, val_accuracy, test_accuracy= train_model(model_name,params, X_train, X_val, X_test, y_train, y_val, y_test,custom_name=model_custom_name)
                    prediction = model.predict(processed_canvas)[0]
                    probabilities = model.predict_proba(processed_canvas)[0]

                    st.write(f"🎯 **Dự đoán: {prediction}**")
                    st.write(f"🔢 **Độ tin cậy: {probabilities[prediction] * 100:.2f}%**")

    with tab3:
        st.header("📊 MLflow Tracking")

        # Lấy danh sách các phiên làm việc từ MLflow
        runs = mlflow.search_runs(order_by=["start_time desc"])

        if not runs.empty:
            # Lấy danh sách tên mô hình
            runs["model_custom_name"] = runs["tags.mlflow.runName"]  # Giả sử tên mô hình lưu trong tag `mlflow.runName`
            model_names = runs["model_custom_name"].dropna().unique().tolist()

            # **Tìm kiếm mô hình**
            search_model_name = st.text_input("🔍 Nhập tên mô hình để tìm kiếm:", "")

            if search_model_name:
                filtered_runs = runs[runs["model_custom_name"].str.contains(search_model_name, case=False, na=False)]
            else:
                filtered_runs = runs

            # **Hiển thị danh sách mô hình**
            if not filtered_runs.empty:
                st.write("### 📜 Danh sách mô hình đã lưu:")
                st.dataframe(filtered_runs[["model_custom_name", "run_id", "start_time", "metrics.train_accuracy", "metrics.val_accuracy", "metrics.test_accuracy"]])

                # **Chọn một mô hình để xem chi tiết**
                selected_run_id = st.selectbox("📝 Chọn một mô hình để xem chi tiết:", filtered_runs["run_id"].tolist())

                if selected_run_id:
                    run_details = mlflow.get_run(selected_run_id)
                    st.write(f"### 🔍 Chi tiết mô hình: `{run_details.data.tags.get('mlflow.runName', 'Không có tên')}`")
                    st.write("📌 **Tham số:**")
                    for key, value in run_details.data.params.items():
                        st.write(f"- **{key}**: {value}")

                    st.write("📊 **Metric:**")
                    for key, value in run_details.data.metrics.items():
                        st.write(f"- **{key}**: {value}")

                    # st.write("📂 **Artifacts:**")
                    # if run_details.info.artifact_uri:
                    #     st.write(f"- **Artifact URI**: {run_details.info.artifact_uri}")
                    # else:
                    #     st.write("- Không có artifacts nào.")

            else:
                st.write("❌ Không tìm thấy mô hình nào.")

        else:
            st.write("⚠️ Không có phiên làm việc nào được ghi lại.")
        

if __name__ == "__main__":
    create_streamlit_app()
