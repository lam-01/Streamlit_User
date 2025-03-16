import streamlit as st
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import mlflow
import mlflow.keras
from datetime import datetime

# Hàm xây dựng model NN
def create_model():
    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# Tải và xử lý dữ liệu MNIST
@st.cache_data
def load_and_prepare_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    return x_train, y_train, x_test, y_test

# Hàm chọn 1% dữ liệu cho mỗi class
def select_initial_data(x_train, y_train, percentage=0.01):
    labeled_idx = []
    for i in range(10):
        class_idx = np.where(y_train == i)[0]
        n_samples = int(len(class_idx) * percentage)
        selected_idx = np.random.choice(class_idx, n_samples, replace=False)
        labeled_idx.extend(selected_idx)
    
    x_labeled = x_train[labeled_idx]
    y_labeled = y_train[labeled_idx]
    unlabeled_idx = [i for i in range(len(x_train)) if i not in labeled_idx]
    x_unlabeled = x_train[unlabeled_idx]
    
    return x_labeled, y_labeled, x_unlabeled, unlabeled_idx

# Thuật toán Pseudo Labelling với MLflow tracking
def pseudo_labeling_with_mlflow(x_labeled, y_labeled, x_unlabeled, x_test, y_test, threshold, max_iterations):
    with mlflow.start_run(run_name=f"Pseudo_Labeling_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        model = create_model()
        
        # Log parameters
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("max_iterations", max_iterations)
        mlflow.log_param("initial_labeled_percentage", 0.01)
        
        x_train_current = x_labeled.copy()
        y_train_current = y_labeled.copy()
        remaining_unlabeled = x_unlabeled.copy()
        
        for iteration in range(max_iterations):
            history = model.fit(x_train_current, y_train_current,
                              epochs=5,
                              batch_size=32,
                              verbose=0,
                              validation_data=(x_test, y_test))
            
            mlflow.log_metric("train_accuracy", history.history['accuracy'][-1], step=iteration)
            mlflow.log_metric("val_accuracy", history.history['val_loss'][-1], step=iteration)
            
            predictions = model.predict(remaining_unlabeled, verbose=0)
            max_probs = np.max(predictions, axis=1)
            pseudo_labels = np.argmax(predictions, axis=1)
            
            confident_idx = np.where(max_probs >= threshold)[0]
            
            if len(confident_idx) == 0:
                break
                
            x_train_current = np.concatenate([x_train_current, remaining_unlabeled[confident_idx]])
            y_train_current = np.concatenate([y_train_current, pseudo_labels[confident_idx]])
            remaining_unlabeled = np.delete(remaining_unlabeled, confident_idx, axis=0)
            
            mlflow.log_metric("labeled_samples", len(confident_idx), step=iteration)
            
            st.write(f"Iteration {iteration + 1}: Đã gán nhãn cho {len(confident_idx)} mẫu")
            
            if len(remaining_unlabeled) == 0:
                break
        
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.keras.log_model(model, "final_model")
        
    return model

# Giao diện Streamlit
def main():
    # Tạo 4 tab
    tab1, tab2, tab3, tab4 = st.tabs(["Giới thiệu", "Huấn luyện", "Dự đoán", "MLflow Tracking"])
    
    # Tab 1: Giới thiệu
    with tab1:
        st.write("### Giới thiệu về ứng dụng")
        st.write("""
        Đây là ứng dụng thực hiện thuật toán **Pseudo Labelling** trên tập dữ liệu MNIST sử dụng Neural Network.
        - **Tab Huấn luyện**: Chạy thuật toán Pseudo Labelling và theo dõi quá trình.
        - **Tab Dự đoán**: Sử dụng model đã huấn luyện để dự đoán trên dữ liệu mới.
        - **Tab MLflow Tracking**: Xem lịch sử huấn luyện và chi tiết các run trong MLflow.
        """)
        st.write("Tập dữ liệu MNIST gồm 70,000 ảnh chữ số viết tay (0-9), kích thước 28x28 pixel.")
    
    # Tab 2: Huấn luyện
    with tab2:
        x_train, y_train, x_test, y_test = load_and_prepare_data()
        
        st.write("### Huấn luyện mô hình Pseudo Labelling")
        threshold = st.slider("Ngưỡng tin cậy", 0.5, 0.99, 0.95, 0.01)
        max_iterations = st.slider("Số vòng lặp tối đa", 1, 20, 5)
        
        if st.button("Chạy Pseudo Labelling"):
            x_labeled, y_labeled, x_unlabeled, _ = select_initial_data(x_train, y_train)
            
            st.write("Kích thước tập dữ liệu:")
            st.write(f"Tập labeled ban đầu: {len(x_labeled)} mẫu")
            st.write(f"Tập unlabeled: {len(x_unlabeled)} mẫu")
            
            with st.spinner("Đang huấn luyện..."):
                model = pseudo_labeling_with_mlflow(
                    x_labeled, y_labeled, x_unlabeled,
                    x_test, y_test, threshold, max_iterations
                )
                st.session_state['model'] = model  # Lưu model vào session_state
            st.success("Hoàn thành huấn luyện!")
    
    # Tab 3: Dự đoán
    with tab3:
        st.write("### Dự đoán chữ số")
        if 'model' not in st.session_state:
            st.warning("Vui lòng huấn luyện mô hình trước ở tab Huấn luyện!")
        else:
            uploaded_file = st.file_uploader("Tải lên ảnh chữ số (28x28)", type=['png', 'jpg'])
            if uploaded_file is not None:
                from PIL import Image
                img = Image.open(uploaded_file).convert('L')  # Chuyển sang grayscale
                img = img.resize((28, 28))  # Resize về 28x28
                img_array = np.array(img) / 255.0  # Chuẩn hóa
                
                st.image(img, caption="Ảnh đã tải lên", width=100)
                
                model = st.session_state['model']
                prediction = model.predict(np.expand_dims(img_array, axis=0))
                predicted_digit = np.argmax(prediction)
                confidence = np.max(prediction)
                
                st.write(f"**Dự đoán**: Chữ số {predicted_digit}")
                st.write(f"**Độ tin cậy**: {confidence:.4f}")
    
    # Tab 4: MLflow Tracking
    with tab4:
        st.write("##### 📊 MLflow Tracking")
        st.write("Xem chi tiết các kết quả đã lưu trong MLflow.")
        
        runs = mlflow.search_runs(order_by=["start_time desc"])
        if not runs.empty:
            if "tags.mlflow.runName" in runs.columns:
                runs["model_custom_name"] = runs["tags.mlflow.runName"]
            else:
                runs["model_custom_name"] = "Unnamed Model"
            model_names = runs["model_custom_name"].dropna().unique().tolist()
        
            search_model_name = st.text_input("🔍 Nhập tên mô hình để tìm kiếm:", "")
            if search_model_name:
                filtered_runs = runs[runs["model_custom_name"].str.contains(search_model_name, case=False, na=False)]
            else:
                filtered_runs = runs
        
            if not filtered_runs.empty:
                st.write("##### 📜 Danh sách mô hình đã lưu:")
                available_columns = [
                    col for col in [
                        "model_custom_name", "start_time",
                        "metrics.train_accuracy", "metrics.val_accuracy", "metrics.test_accuracy",
                        "metrics.labeled_samples"
                    ] if col in filtered_runs.columns
                ]
                display_df = filtered_runs[available_columns]
                display_df = display_df.rename(columns={
                    "model_custom_name": "Custom Model Name",
                })
                st.dataframe(display_df)
        
                selected_model_name = st.selectbox("📝 Chọn một mô hình để xem chi tiết:", model_names)
                if selected_model_name:
                    selected_run = filtered_runs[filtered_runs["model_custom_name"] == selected_model_name].iloc[0]
                    selected_run_id = selected_run["run_id"]
                    
                    run_details = mlflow.get_run(selected_run_id)
                    custom_name = run_details.data.tags.get('mlflow.runName', 'Không có tên')
                    st.write(f"##### 🔍 Chi tiết mô hình: `{custom_name}`")
        
                    st.write("📌 **Tham số:**")
                    for key, value in run_details.data.params.items():
                        st.write(f"- **{key}**: {value}")
        
                    st.write("📊 **Metric:**")
                    for key, value in run_details.data.metrics.items():
                        st.write(f"- **{key}**: {value}")
            else:
                st.write("❌ Không tìm thấy mô hình nào khớp với tìm kiếm.")
        else:
            st.write("⚠️ Không có phiên làm việc nào được ghi lại.")

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")  # Thay đổi nếu cần
    main()