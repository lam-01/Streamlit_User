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
import seaborn as sns
from sklearn.neural_network import MLPClassifier
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

# 📌 Huấn luyện mô hình với thanh tiến trình
def train_model(custom_model_name, model_name, params, X_train, X_val, X_test, y_train, y_val, y_test):
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Đang huấn luyện...")

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
    elif model_name == "Neural Network":
        model = MLPClassifier(
            hidden_layer_sizes=(params["hidden_layer_size"],),
            max_iter=params["max_iter"],
            activation=params["activation"],
            solver=params["solver"],
            learning_rate_init=params["learning_rate"],
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
    else:
        raise ValueError("Invalid model selected!")

    try:
        with mlflow.start_run(run_name=custom_model_name):
            start_time = time.time()
            model.fit(X_train, y_train)
            end_time = time.time()
            
            progress_bar.progress(1.0)
            # status_text.text(f"Đã hoàn tất huấn luyện trong {end_time - start_time:.2f} giây!")

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            y_val_pred = model.predict(X_val)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            mlflow.log_param("model_name", model_name)
            mlflow.log_params(params)
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("val_accuracy", val_accuracy)
            mlflow.log_metric("test_accuracy", test_accuracy)
            # mlflow.log_metric("training_time", end_time - start_time)
            
            input_example = X_train[:1]
            mlflow.sklearn.log_model(model, model_name, input_example=input_example)
    except Exception as e:
        st.error(f"Lỗi trong quá trình huấn luyện: {str(e)}")
        return None, None, None, None

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
    
    tab1, tab2, tab3 ,tab4 = st.tabs(["📓 Lí thuyết","📋 Huấn luyện", "🔮 Dự đoán", "⚡ MLflow"])
    with tab1:
        algorithm =st.selectbox("Chọn thuật toán:", ["Neural Network", "Decision Tree","SVM"])
        # Nội dung cho Neural Network
        if algorithm == "Neural Network":
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
            st.image("neural_networks.png",caption="Cấu trúc mạng Neural Network",width=500)
            st.write("Ví dụ minh họa với bộ dữ liệu mnist : ")
            st.image("mau.png",caption="Nguồn : https://www.researchgate.net/",width=700)
            st.write("##### 3. Các tham số quan trọng")
            st.write("""
            **a. Kích thước tầng ẩn (hidden_layer_size)**:
            \n- Đây là số lượng nơ-ron trong tầng ẩn của mạng nơ-ron. Tầng ẩn là nơi mà các phép toán phi tuyến được thực hiện, giúp mô hình học được các đặc trưng phức tạp từ dữ liệu. Kích thước của tầng ẩn có thể ảnh hưởng lớn đến khả năng học của mô hình
            \n **b. Số lần lặp tối đa (max_iter)**:
            \n- Đây là số lần mà thuật toán tối ưu sẽ cập nhật trọng số của mô hình trong quá trình huấn luyện .""")
            st.latex(r"w = w - \eta \cdot \nabla L(w)")
            st.markdown(r"""
            Trong đó:
                $$w$$ là trọng số.
                $$\eta$$ là tốc độ học (learning rate).
                $$\nabla L(w)$$ là gradient của hàm mất mát (loss function) theo trọng số.
            """)
            st.write("""
            **c. Hàm kích hoạt (activation)**: 
            \n- Hàm kích hoạt là một hàm toán học được áp dụng cho đầu ra của mỗi nơ-ron trong tầng ẩn. Nó giúp mô hình học được các mối quan hệ phi tuyến giữa các đặc trưng. Các hàm kích hoạt phổ biến bao gồm:""")
            st.write("**ReLU (Rectified Linear Unit)**: Hàm này trả về giá trị đầu vào nếu nó lớn hơn 0, ngược lại trả về 0. ReLU giúp giảm thiểu vấn đề vanishing gradient.")
            st.latex("f(x) = \max(0, x)")
            st.write("**Tanh**: Hàm này trả về giá trị trong khoảng từ -1 đến 1, giúp cải thiện tốc độ hội tụ so với hàm sigmoid.")
            st.latex(r" f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} ")
            st.write("**Logistic (Sigmoid)**: Hàm này trả về giá trị trong khoảng từ 0 đến 1, thường được sử dụng cho các bài toán phân loại nhị phân.")
            st.latex(r"f(x) = \frac{1}{1 + e^{-x}}")
            st.write("""
            **d. Bộ giải tối ưu (solver)**:
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
            st.write("""
            **e. Tốc độ học (learning_rate)**:
            \n- Tốc độ học là một tham số điều chỉnh mức độ mà trọng số của mô hình được cập nhật trong mỗi lần lặp. Tốc độ học quá cao có thể dẫn đến việc mô hình không hội tụ, trong khi tốc độ học quá thấp có thể làm cho quá trình huấn luyện trở nên chậm.
            """)
        elif algorithm == "Decision Tree":
            st.write("")
            
            
        elif algorithm == "SVM":
            st.write("")
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
        custom_model_name = st.text_input("Nhập tên mô hình :")
        if not custom_model_name:
            custom_model_name = "Default_model"

        model_name = st.selectbox("🔍 Chọn mô hình", ["Decision Tree", "SVM", "Neural Network"])
        params = {}

        if model_name == "Decision Tree":
            params["criterion"] = st.selectbox("📏 Tiêu chí phân tách", ["gini", "entropy", "log_loss"])
            params["max_depth"] = st.slider("🌳 Độ sâu tối đa (max_depth)", 1, 30, 15)
            params["min_samples_split"] = st.slider("🔄 Số mẫu tối thiểu để chia nhánh (min_samples_split)", 2, 10, 5)
            params["min_samples_leaf"] = st.slider("🍃 Số mẫu tối thiểu ở lá (min_samples_leaf)", 1, 10, 2)
        elif model_name == "SVM":
            params["kernel"] = st.selectbox("⚙️ Kernel", ["linear", "rbf", "poly", "sigmoid"])
            params["C"] = st.slider("🔧 Tham số C ", 0.1, 10.0, 1.0)
        elif model_name == "Neural Network":
            params["hidden_layer_size"] = st.slider("Kích thước tầng ẩn", 10, 100, 50, help="Số nơ-ron trong tầng ẩn.")
            params["max_iter"] = st.slider("Số lần lặp tối đa", 5, 20, 10, help="Số lần lặp tối đa để huấn luyện.")
            params["activation"] = st.selectbox("Hàm kích hoạt", ["relu", "tanh", "logistic"], help="Hàm kích hoạt cho các nơ-ron.")
            params["solver"] = st.selectbox("Bộ giải tối ưu", ["adam", "sgd"], help="Bộ giải tối ưu hóa trọng số.")
            params["learning_rate"] = st.slider("Tốc độ học", 0.0001, 0.01, 0.001, format="%.4f", help="Tốc độ học ban đầu.")

        if st.button("🚀 Huấn luyện mô hình"):
            with st.spinner("🔄 Đang khởi tạo huấn luyện..."):
                model, train_accuracy, val_accuracy, test_accuracy = train_model(
                    custom_model_name, model_name, params, X_train, X_val, X_test, y_train, y_val, y_test
                )
            
            if model is not None:
                st.success(f"✅ Huấn luyện xong!")
                st.write(f"🎯 **Độ chính xác trên tập train: {train_accuracy:.4f}**")
                st.write(f"🎯 **Độ chính xác trên tập validation: {val_accuracy:.4f}**")
                st.write(f"🎯 **Độ chính xác trên tập test: {test_accuracy:.4f}**")
            else:
                st.error("Huấn luyện thất bại, không có kết quả để hiển thị.")

    with tab3:
        option = st.radio("🖼️ Chọn phương thức nhập:", ["📂 Tải ảnh lên", "✏️ Vẽ số"])
        if option == "📂 Tải ảnh lên":
            uploaded_file = st.file_uploader("📤 Tải ảnh số viết tay (PNG, JPG)", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
                processed_image = preprocess_uploaded_image(image)
                st.image(image, caption="📷 Ảnh tải lên", use_column_width=True)
                if st.button("🔮 Dự đoán"):
                    model, train_accuracy, val_accuracy, test_accuracy = train_model(
                        custom_model_name, model_name, params, X_train, X_val, X_test, y_train, y_val, y_test
                    )
                    if model is not None:
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
                    model, train_accuracy, val_accuracy, test_accuracy = train_model(
                        custom_model_name, model_name, params, X_train, X_val, X_test, y_train, y_val, y_test
                    )
                    if model is not None:
                        prediction = model.predict(processed_canvas)[0]
                        probabilities = model.predict_proba(processed_canvas)[0]
                        st.write(f"🎯 **Dự đoán: {prediction}**")
                        st.write(f"🔢 **Độ tin cậy: {probabilities[prediction] * 100:.2f}%**")

    with tab4:
        st.subheader("📊 MLflow Tracking")
    
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
                                                     "metrics.train_accuracy", "metrics.val_accuracy", "metrics.test_accuracy"] 
                                     if col in runs.columns]
                display_df = filtered_runs[available_columns]
                
                for col in display_df.columns:
                    if display_df[col].dtype == 'object':
                        display_df[col] = display_df[col].astype(str)
                
                display_df = display_df.rename(columns={
                    "model_custom_name": "Custom Model Name",
                    "params.model_name": "Model Type"
                })
                st.dataframe(display_df)
    
                # Thay đổi từ run_id sang model_custom_name
                selected_model_name = st.selectbox("📝 Chọn một mô hình để xem chi tiết:", 
                                                   filtered_runs["model_custom_name"].tolist())
                if selected_model_name:
                    # Lấy run_id tương ứng với custom_model_name được chọn
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
