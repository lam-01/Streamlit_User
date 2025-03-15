import streamlit as st
import mlflow
import mlflow.sklearn
import numpy as np
import cv2
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import time

# 📌 Tải và xử lý dữ liệu MNIST từ OpenML
@st.cache_data
def load_data(n_samples=None):
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)  # Chuyển nhãn về kiểu số nguyên
    X = X / 255.0  # Chuẩn hóa về [0,1]
    if n_samples is not None and n_samples < len(X):
        indices = np.random.choice(len(X), n_samples, replace=False)
        X = X[indices]
        y = y[indices]
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

# 📌 Huấn luyện mô hình với thanh tiến trình và cross-validation
def train_model(custom_model_name, params, X_train, X_val, X_test, y_train, y_val, y_test, cv_folds):
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Tạo tuple cho hidden_layer_sizes dựa trên số lớp ẩn và số neuron mỗi lớp
    hidden_layer_sizes = tuple([params["neurons_per_layer"]] * params["num_hidden_layers"])

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=params["epochs"],
        activation=params["activation"],
        random_state=42,
        warm_start=True  # Cho phép huấn luyện tiếp tục để mô phỏng tiến trình
    )

    # Huấn luyện mô hình
    try:
        with mlflow.start_run(run_name=custom_model_name):
            # Mô phỏng tiến trình huấn luyện cho Neural Network
            for i in range(params["epochs"]):
                model.max_iter = i + 1  # Tăng số lần lặp từng bước
                model.fit(X_train, y_train)  # Huấn luyện từng epoch
                progress = (i + 1) / params["epochs"]
                progress_bar.progress(progress)
                status_text.text(f"Đang huấn luyện: {int(progress * 100)}%")
                time.sleep(0.1)  # Giả lập thời gian huấn luyện để thấy tiến trình

            # Dự đoán và tính toán độ chính xác
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            y_val_pred = model.predict(X_val)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            # Thực hiện cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds)
            cv_mean_accuracy = np.mean(cv_scores)

            # Ghi log tham số và metric vào MLflow
            mlflow.log_param("model_name", "Neural Network")
            mlflow.log_params(params)  # Ghi toàn bộ tham số
            mlflow.log_param("cv_folds", cv_folds)  # Ghi số lượng fold
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("val_accuracy", val_accuracy)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("cv_mean_accuracy", cv_mean_accuracy)
            mlflow.sklearn.log_model(model, "Neural Network")
    except Exception as e:
        st.error(f"Lỗi trong quá trình huấn luyện: {str(e)}")
        return None, None, None, None, None

    # Xóa thanh tiến trình và trạng thái sau khi hoàn thành
    progress_bar.empty()
    status_text.empty()
    return model, train_accuracy, val_accuracy, test_accuracy, cv_mean_accuracy

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
        st.write("##### Neural Network")
        st.write("""Neural Network là một phương thức phổ biến trong lĩnh vực trí tuệ nhân tạo, được dùng để điều khiển máy tính dự đoán, nhận dạng và xử lý dữ liệu như một bộ não của con người. 
        Bên cạnh đó, quy trình này còn được biết đến với thuật ngữ quen thuộc là “deep learning”, nghĩa là việc vận dụng các nơ-ron hoặc các nút tạo sự liên kết với nhau trong cùng một cấu trúc phân lớp.""")
        st.write("##### 1. Đặc điểm của Neural Network")
        st.write("""- Mạng lưới nơ-ron nhân tạo hoạt động như nơ-ron trong não bộ con người. Trong đó, mỗi nơ-ron là một hàm toán học, có chức năng thu thập và phân loại dữ liệu, thông tin theo cấu trúc chi tiết. 
        \n- Neural Network tương đồng với những phương pháp thống kê theo đồ thị đường cong hoặc phân tích hồi quy. Để giải thích đơn giản nhất, bạn hãy hình dung Neural Network bao hàm các nút mạng liên kết với nhau. 
        \n- Mỗi nút là một tập hợp tri giác, cấu tạo tương tự hàm hồi quy đa tuyến tính, được sắp xếp liên kết với nhau. Các lớp này sẽ thu thập thông tin, sau đó phân loại và phát tín hiệu đầu ra tương ứng.
        """)
        st.write("##### 2. Cấu trúc mạng Neural Network")
        st.write("""- Input Layer (tầng đầu vào): Nằm bên trái của hệ thống, bao gồm dữ liệu thông tin đầu vào. 
        \n- Output Layer (tầng đầu ra): Nằm bên phải của hệ thống, bao gồm dữ liệu thông tin đầu ra. 
        \n- Hidden Layer (tầng ẩn): Nằm ở giữa tầng đầu vào và đầu ra, thể hiện quá trình suy luận và xử lý thôngtin của hệ thống.    
        """)
        st.image("neural_networks.png", caption="Cấu trúc mạng Neural Network", width=500)
        st.write("Ví dụ minh họa với bộ dữ liệu mnist : ")
        st.image("mau.png", caption="Nguồn : https://www.researchgate.net/", width=700)
        st.write("##### 3. Các tham số quan trọng")
        st.write("""**a. Số lớp ẩn (num_hidden_layers)**:
        \n- Đây là số lượng tầng ẩn trong mạng nơ-ron. Nhiều tầng ẩn hơn có thể giúp mô hình học được các đặc trưng phức tạp hơn, nhưng cũng làm tăng độ phức tạp tính toán.
        \n**b. Số neuron mỗi lớp (neurons_per_layer)**:
        \n- Đây là số lượng nơ-ron trong mỗi tầng ẩn. Số lượng nơ-ron ảnh hưởng đến khả năng học các đặc trưng từ dữ liệu.
        \n**c. Epochs**:
        \n- Đây là số lần toàn bộ dữ liệu huấn luyện được sử dụng để cập nhật trọng số của mô hình.""")
        st.latex(r"w = w - \eta \cdot \nabla L(w)")
        st.markdown(r"""
        Trong đó:
            $$w$$ là trọng số.
            $$\eta$$ là tốc độ học (learning rate).
            $$\nabla L(w)$$ là gradient của hàm mất mát (loss function) theo trọng số.
        """)
        st.write("""**d. Hàm kích hoạt (activation)**: 
        \n- Hàm kích hoạt là một hàm toán học được áp dụng cho đầu ra của mỗi nơ-ron trong tầng ẩn. Nó giúp mô hình học được các mối quan hệ phi tuyến giữa các đặc trưng. Các hàm kích hoạt phổ biến bao gồm:""")
        st.write("**ReLU (Rectified Linear Unit)**: Hàm này trả về giá trị đầu vào nếu nó lớn hơn 0, ngược lại trả về 0. ReLU giúp giảm thiểu vấn đề vanishing gradient.")
        st.latex("f(x) = \max(0, x)")
        st.write("**Tanh**: Hàm này trả về giá trị trong khoảng từ -1 đến 1, giúp cải thiện tốc độ hội tụ so với hàm sigmoid.")
        st.latex(r" f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} ")
        st.write("**Logistic (Sigmoid)**: Hàm này trả về giá trị trong khoảng từ 0 đến 1, thường được sử dụng cho các bài toán phân loại nhị phân.")
        st.latex(r"f(x) = \frac{1}{1 + e^{-x}}")

    with tab2:
        # Cho phép nhập số mẫu để huấn luyện
        max_samples = 70000  # Tổng số mẫu trong MNIST
        n_samples = st.number_input(
            "Số lượng mẫu để huấn luyện",
            min_value=1000,
            max_value=max_samples,
            value=10000,
            step=1000,
            help=f"Nhập số lượng mẫu từ 1,000 đến {max_samples} để huấn luyện."
        )
        
        X, y = load_data(n_samples=n_samples)
        st.write(f"**Số lượng mẫu được chọn để huấn luyện: {X.shape[0]}**")
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

        st.write("**🚀 Huấn luyện mô hình Neural Network**")
        custom_model_name = st.text_input("Nhập tên mô hình để lưu vào MLflow:", "MyModel")
        params = {}
        
        params["num_hidden_layers"] = st.slider("Số lớp ẩn", 1, 5, 2, help="Số lượng tầng ẩn trong mạng nơ-ron.")
        params["neurons_per_layer"] = st.slider("Số neuron mỗi lớp", 50, 200, 100, help="Số nơ-ron trong mỗi tầng ẩn.")
        params["epochs"] = st.slider("Epochs", 5, 50, 10, help="Số lần lặp qua toàn bộ dữ liệu huấn luyện.")
        params["activation"] = st.selectbox("Hàm kích hoạt", ["relu", "tanh", "logistic"], help="Hàm kích hoạt cho các nơ-ron.")
        cv_folds = st.slider("Số lượng fold cho Cross-Validation", 2, 10, 5, help="Số lượng fold để đánh giá mô hình bằng cross-validation.")

        if st.button("🚀 Huấn luyện mô hình"):
            with st.spinner("🔄 Đang khởi tạo huấn luyện..."):
                result = train_model(
                    custom_model_name, params, X_train, X_val, X_test, y_train, y_val, y_test, cv_folds
                )
                if result[0] is not None:  # Check if model was returned successfully
                    model, train_accuracy, val_accuracy, test_accuracy, cv_mean_accuracy = result
                    st.success(f"✅ Huấn luyện xong!")
                    st.write(f"🎯 **Độ chính xác trên tập train: {train_accuracy:.4f}**")
                    st.write(f"🎯 **Độ chính xác trên tập validation: {val_accuracy:.4f}**")
                    st.write(f"🎯 **Độ chính xác trên tập test: {test_accuracy:.4f}**")
                    st.write(f"🎯 **Độ chính xác trung bình Cross-Validation: {cv_mean_accuracy:.4f}**")
                else:
                    st.error("Huấn luyện thất bại. Vui lòng kiểm tra lỗi ở trên.")

    with tab3:
        option = st.radio("🖼️ Chọn phương thức nhập:", ["📂 Tải ảnh lên", "✏️ Vẽ số"])
        if option == "📂 Tải ảnh lên":
            uploaded_file = st.file_uploader("📤 Tải ảnh số viết tay (PNG, JPG)", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
                processed_image = preprocess_uploaded_image(image)
                st.image(image, caption="📷 Ảnh tải lên", use_column_width=True)
                if st.button("🔮 Dự đoán"):
                    result = train_model(
                        custom_model_name, params, X_train, X_val, X_test, y_train, y_val, y_test, cv_folds
                    )
                    if result[0] is not None:
                        model, train_accuracy, val_accuracy, test_accuracy, cv_mean_accuracy = result
                        prediction = model.predict(processed_image)[0]
                        probabilities = model.predict_proba(processed_image)[0]
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
                    result = train_model(
                        custom_model_name, params, X_train, X_val, X_test, y_train, y_val, y_test, cv_folds
                    )
                    if result[0] is not None:
                        model, train_accuracy, val_accuracy, test_accuracy, cv_mean_accuracy = result
                        prediction = model.predict(processed_canvas)[0]
                        probabilities = model.predict_proba(processed_canvas)[0]
                        st.write(f"🎯 **Dự đoán: {prediction}**")
                        st.write(f"🔢 **Độ tin cậy: {probabilities[prediction] * 100:.2f}%**")

    with tab4:
        st.header("📊 MLflow Tracking")
        st.write("Xem chi tiết các kết quả đã lưu trong MLflow.")
        
        runs = mlflow.search_runs(order_by=["start_time desc"])
        if not runs.empty:
            # Safely assign 'model_custom_name' from tags, with a fallback
            if "tags.mlflow.runName" in runs.columns:
                runs["model_custom_name"] = runs["tags.mlflow.runName"]
            else:
                runs["model_custom_name"] = "Unnamed Model"  # Default value if tag is missing
            model_names = runs["model_custom_name"].dropna().unique().tolist()
        
            search_model_name = st.text_input("🔍 Nhập tên mô hình để tìm kiếm:", "")
            if search_model_name:
                filtered_runs = runs[runs["model_custom_name"].str.contains(search_model_name, case=False, na=False)]
            else:
                filtered_runs = runs
        
            if not filtered_runs.empty:
                st.write("### 📜 Danh sách mô hình đã lưu:")
                # Define available columns dynamically
                available_columns = [
                    col for col in [
                        "model_custom_name", "params.model_name", "run_id", "start_time",
                        "metrics.train_accuracy", "metrics.val_accuracy", "metrics.test_accuracy",
                        "metrics.cv_mean_accuracy"
                    ] if col in filtered_runs.columns
                ]
                display_df = filtered_runs[available_columns]
                display_df = display_df.rename(columns={
                    "model_custom_name": "Custom Model Name",
                    "params.model_name": "Model Type"
                })
                st.dataframe(display_df)
        
                # Use custom_model_name in selectbox instead of run_id
                selected_model_name = st.selectbox("📝 Chọn một mô hình để xem chi tiết:", model_names)
                if selected_model_name:
                    # Get the run_id corresponding to the selected custom_model_name
                    selected_run = filtered_runs[filtered_runs["model_custom_name"] == selected_model_name].iloc[0]
                    selected_run_id = selected_run["run_id"]
                    
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
                st.write("❌ Không tìm thấy mô hình nào khớp với tìm kiếm.")
        else:
            st.write("⚠️ Không có phiên làm việc nào được ghi lại.")

if __name__ == "__main__":
    create_streamlit_app()

