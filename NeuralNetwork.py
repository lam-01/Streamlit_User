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

    model = MLPClassifier(
        hidden_layer_sizes=(params["hidden_layer_size"],),
        max_iter=params["max_iter"],
        activation=params["activation"],
        solver=params["solver"],
        random_state=42,
        warm_start=True  # Cho phép huấn luyện tiếp tục để mô phỏng tiến trình
    )

    # Huấn luyện mô hình
    with mlflow.start_run(run_name=custom_model_name):
        # Mô phỏng tiến trình huấn luyện cho Neural Network
        for i in range(params["max_iter"]):
            model.max_iter = i + 1  # Tăng số lần lặp từng bước
            model.fit(X_train, y_train)  # Huấn luyện từng epoch
            progress = (i + 1) / params["max_iter"]
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
        cv_std_accuracy = np.std(cv_scores)

        # Ghi log tham số và metric vào MLflow
        mlflow.log_param("model_name", "Neural Network")
        mlflow.log_params(params)  # Ghi toàn bộ tham số
        mlflow.log_param("cv_folds", cv_folds)  # Ghi số lượng fold
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("cv_mean_accuracy", cv_mean_accuracy)
        mlflow.log_metric("cv_std_accuracy", cv_std_accuracy)
        mlflow.sklearn.log_model(model, "Neural Network")
    
    # Xóa thanh tiến trình và trạng thái sau khi hoàn thành
    status_text.text("Hoàn thành huấn luyện!")
    return model, train_accuracy, val_accuracy, test_accuracy, cv_mean_accuracy, cv_std_accuracy

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
        \n- Hidden Layer (tầng ẩn): Nằm ở giữa tầng đầu vào và đầu ra, thể hiện quá trình suy luận và xử lý thông tin của hệ thống.    
        """)
        st.image("neural_networks.png", caption="Cấu trúc mạng Neural Network", width=500)
        st.write("Ví dụ minh họa với bộ dữ liệu mnist : ")
        st.image("mau.png", caption="Nguồn : https://www.researchgate.net/", width=700)
        st.write("##### 3. Các tham số quan trọng")
        st.write("""**a. Kích thước tầng ẩn (hidden_layer_size)**:
        \n- Đây là số lượng nơ-ron trong tầng ẩn của mạng nơ-ron. Tầng ẩn là nơi mà các phép toán phi tuyến được thực hiện, giúp mô hình học được các đặc trưng phức tạp từ dữ liệu. Kích thước của tầng ẩn có thể ảnh hưởng lớn đến khả năng học của mô hình.
        \n**b. Số lần lặp tối đa (max_iter)**:
        \n- Đây là số lần mà thuật toán tối ưu sẽ cập nhật trọng số của mô hình trong quá trình huấn luyện.""")
        st.latex(r"w = w - \eta \cdot \nabla L(w)")
        st.markdown(r"""
        Trong đó:
            $$w$$ là trọng số.
            $$\eta$$ là tốc độ học (learning rate).
            $$\nabla L(w)$$ là gradient của hàm mất mát (loss function) theo trọng số.
        """)
        st.write("""**c. Hàm kích hoạt (activation)**: 
        \n- Hàm kích hoạt là một hàm toán học được áp dụng cho đầu ra của mỗi nơ-ron trong tầng ẩn. Nó giúp mô hình học được các mối quan hệ phi tuyến giữa các đặc trưng. Các hàm kích hoạt phổ biến bao gồm:""")
        st.write("**ReLU (Rectified Linear Unit)**: Hàm này trả về giá trị đầu vào nếu nó lớn hơn 0, ngược lại trả về 0. ReLU giúp giảm thiểu vấn đề vanishing gradient.")
        st.latex("f(x) = \max(0, x)")
        st.write("**Tanh**: Hàm này trả về giá trị trong khoảng từ -1 đến 1, giúp cải thiện tốc độ hội tụ so với hàm sigmoid.")
        st.latex(r" f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} ")
        st.write("**Logistic (Sigmoid)**: Hàm này trả về giá trị trong khoảng từ 0 đến 1, thường được sử dụng cho các bài toán phân loại nhị phân.")
        st.latex(r"f(x) = \frac{1}{1 + e^{-x}}")
        st.write("""**d. Bộ giải tối ưu (solver)**:
        \n- Bộ giải tối ưu là thuật toán được sử dụng để cập nhật trọng số của mô hình trong quá trình huấn luyện. Các bộ giải phổ biến bao gồm:""")
        st.write("**Adam**: Một trong những bộ giải tối ưu phổ biến nhất, kết hợp các ưu điểm của hai bộ giải khác là AdaGrad và RMSProp. Adam tự động điều chỉnh tốc độ học cho từng trọng số.")
        st.write("Bước 1: Tính toán gradient")
        st.latex(r"g_t = \nabla L(w_t)") 
        st.write("Bước 2: Cập nhật các ước lượng trung bình")
        st.latex(r"m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t ] [ v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 ")
        st.write("Bước 3: Điều chỉnh bias")
        st.latex(r"\hat{m}_t = \frac{m_t}{1 - \beta_1^t} ] [ \hat{v}_t = \frac{v_t}{1 - \beta_2^t} ")
        st.write("Bước 4: Cập nhật trọng số")
        st.latex(r"w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t ")
        st.write("**SGD (Stochastic Gradient Descent)**: Một phương pháp đơn giản và hiệu quả, cập nhật trọng số dựa trên một mẫu ngẫu nhiên từ tập dữ liệu. SGD có thể hội tụ nhanh hơn nhưng có thể không ổn định.")

    with tab2:
        # Cho phép chọn số mẫu để huấn luyện
        max_samples = 70000  # Tổng số mẫu trong MNIST
        n_samples = st.slider("Số lượng mẫu để huấn luyện", 1000, max_samples, 10000, step=1000, 
                              help=f"Chọn số lượng mẫu từ 1,000 đến {max_samples} để huấn luyện.")
        
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
        
        params["hidden_layer_size"] = st.slider("Kích thước tầng ẩn", 50, 200, 100, help="Số nơ-ron trong tầng ẩn.")
        params["max_iter"] = st.slider("Số lần lặp tối đa", 5, 50, 10, help="Số lần lặp tối đa để huấn luyện.")
        params["activation"] = st.selectbox("Hàm kích hoạt", ["relu", "tanh", "logistic"], help="Hàm kích hoạt cho các nơ-ron.")
        params["solver"] = st.selectbox("Bộ giải tối ưu", ["adam", "sgd"], help="Bộ giải tối ưu hóa trọng số.")
        cv_folds = st.slider("Số lượng fold cho Cross-Validation", 2, 10, 5, help="Số lượng fold để đánh giá mô hình bằng cross-validation.")

        if st.button("🚀 Huấn luyện mô hình"):
            with st.spinner("🔄 Đang khởi tạo huấn luyện..."):
                model, train_accuracy, val_accuracy, test_accuracy, cv_mean_accuracy, cv_std_accuracy = train_model(
                    custom_model_name, params, X_train, X_val, X_test, y_train, y_val, y_test, cv_folds
                )
            st.success(f"✅ Huấn luyện xong!")
            st.write(f"🎯 **Độ chính xác trên tập train: {train_accuracy:.4f}**")
            st.write(f"🎯 **Độ chính xác trên tập validation: {val_accuracy:.4f}**")
            st.write(f"🎯 **Độ chính xác trên tập test: {test_accuracy:.4f}**")
            st.write(f"🎯 **Độ chính xác trung bình Cross-Validation: {cv_mean_accuracy:.4f} (±{cv_std_accuracy:.4f})**")

    with tab3:
        option = st.radio("🖼️ Chọn phương thức nhập:", ["📂 Tải ảnh lên", "✏️ Vẽ số"])
        if option == "📂 Tải ảnh lên":
            uploaded_file = st.file_uploader("📤 Tải ảnh số viết tay (PNG, JPG)", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
                processed_image = preprocess_uploaded_image(image)
                st.image(image, caption="📷 Ảnh tải lên", use_column_width=True)
                if st.button("🔮 Dự đoán"):
                    model, train_accuracy, val_accuracy, test_accuracy, cv_mean_accuracy, cv_std_accuracy = train_model(
                        custom_model_name, params, X_train, X_val, X_test, y_train, y_val, y_test, cv_folds
                    )
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
                    model, train_accuracy, val_accuracy, test_accuracy, cv_mean_accuracy, cv_std_accuracy = train_model(
                        custom_model_name, params, X_train, X_val, X_test, y_train, y_val, y_test, cv_folds
                    )
                    prediction = model.predict(processed_canvas)[0]
                    probabilities = model.predict_proba(processed_canvas)[0]
                    st.write(f"🎯 **Dự đoán: {prediction}**")
                    st.write(f"🔢 **Độ tin cậy: {probabilities[prediction] * 100:.2f}%**")

    with tab4:
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
                                           "metrics.train_accuracy", "metrics.val_accuracy", "metrics.test_accuracy",
                                           "metrics.cv_mean_accuracy", "metrics.cv_std_accuracy"]]
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
