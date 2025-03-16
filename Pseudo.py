import streamlit as st
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import mlflow
import mlflow.keras
import cv2
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import time
import pandas as pd

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

# Tải và xử lý dữ liệu MNIST với tỉ lệ train/test tùy chỉnh
@st.cache_data
def load_data(train_split=0.8):
    (x_full, y_full), _ = keras.datasets.mnist.load_data()
    x_full = x_full.astype('float32') / 255
    
    total_samples = len(x_full)
    train_size = int(total_samples * train_split)
    
    indices = np.random.permutation(total_samples)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    x_train = x_full[train_indices]
    y_train = y_full[train_indices]
    x_test = x_full[test_indices]
    y_test = y_full[test_indices]
    
    return x_train, y_train, x_test, y_test

# Chọn dữ liệu labeled ban đầu với tỉ lệ tùy chỉnh
def select_initial_data(x_train, y_train, percentage):
    labeled_idx = []
    for i in range(10):
        class_idx = np.where(y_train == i)[0]
        n_samples = max(1, int(len(class_idx) * percentage))
        selected_idx = np.random.choice(class_idx, n_samples, replace=False)
        labeled_idx.extend(selected_idx)
    
    x_labeled = x_train[labeled_idx]
    y_labeled = y_train[labeled_idx]
    unlabeled_idx = [i for i in range(len(x_train)) if i not in labeled_idx]
    x_unlabeled = x_train[unlabeled_idx]
    
    return x_labeled, y_labeled, x_unlabeled, unlabeled_idx

# Thuật toán Pseudo Labelling với MLflow và hiển thị chi tiết
def pseudo_labeling_with_mlflow(x_labeled, y_labeled, x_unlabeled, x_test, y_test, threshold, max_iterations, custom_model_name, show_details=False):
    if show_details:
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_container = st.empty()
        log_text = ""
    else:
        log_text = ""
    
    with mlflow.start_run(run_name=custom_model_name):
        model = create_model()
        
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("max_iterations", max_iterations)
        mlflow.log_param("initial_labeled_percentage", percentage * 100)
        
        x_train_current = x_labeled.copy()
        y_train_current = y_labeled.copy()
        remaining_unlabeled = x_unlabeled.copy()
        
        log_text += "✅ **Bước 0**: Chia tập train/test hoàn tất.\n"
        if show_details:
            log_container.text(log_text)
            progress_bar.progress(0.1)
            status_text.text("Đang khởi tạo mô hình... (10%)")
        
        log_text += f"✅ **Bước 1**: Đã chọn {len(x_labeled)} mẫu làm tập labeled ban đầu ({percentage*100:.1f}% mỗi class).\n"
        if show_details:
            log_container.text(log_text)
        
        for iteration in range(max_iterations):
            log_text += f"🔄 **Bước 2 (Iteration {iteration+1})**: Huấn luyện model với {len(x_train_current)} mẫu.\n"
            if show_details:
                log_container.text(log_text)
            history = model.fit(x_train_current, y_train_current,
                              epochs=5,
                              batch_size=32,
                              verbose=0,
                              validation_data=(x_test, y_test))
            
            train_acc = history.history['accuracy'][-1]
            val_acc = history.history['val_accuracy'][-1]
            mlflow.log_metric("train_accuracy", train_acc, step=iteration)
            mlflow.log_metric("val_accuracy", val_acc, step=iteration)
            log_text += f"📊 Độ chính xác train: {train_acc:.4f}, validation: {val_acc:.4f}\n"
            if show_details:
                log_container.text(log_text)
            
            log_text += f"🔮 **Bước 3 (Iteration {iteration+1})**: Dự đoán nhãn cho {len(remaining_unlabeled)} mẫu unlabeled.\n"
            if show_details:
                log_container.text(log_text)
            predictions = model.predict(remaining_unlabeled, verbose=0)
            max_probs = np.max(predictions, axis=1)
            pseudo_labels = np.argmax(predictions, axis=1)
            
            confident_idx = np.where(max_probs >= threshold)[0]
            log_text += f"📌 **Bước 4 (Iteration {iteration+1})**: Gán nhãn giả cho {len(confident_idx)} mẫu với ngưỡng {threshold}.\n"
            if show_details:
                log_container.text(log_text)
            
            if show_details:
                progress_bar.progress(0.5 + 0.4 * (iteration + 1) / max_iterations)
                status_text.text(f"Iteration {iteration + 1}: Đã gán nhãn cho {len(confident_idx)} mẫu ({int(50 + 40 * (iteration + 1) / max_iterations)}%)")
            
            if len(confident_idx) == 0:
                log_text += "⛔ Không còn mẫu nào vượt ngưỡng. Dừng thuật toán.\n"
                if show_details:
                    log_container.text(log_text)
                break
                
            x_train_current = np.concatenate([x_train_current, remaining_unlabeled[confident_idx]])
            y_train_current = np.concatenate([y_train_current, pseudo_labels[confident_idx]])
            remaining_unlabeled = np.delete(remaining_unlabeled, confident_idx, axis=0)
            mlflow.log_metric("labeled_samples", len(confident_idx), step=iteration)
            log_text += f"🔄 **Bước 5 (Iteration {iteration+1})**: Tập huấn luyện mới có {len(x_train_current)} mẫu.\n"
            if show_details:
                log_container.text(log_text)
            
            if len(remaining_unlabeled) == 0:
                log_text += "✅ Đã gán nhãn hết dữ liệu unlabeled. Dừng thuật toán.\n"
                if show_details:
                    log_container.text(log_text)
                break
        
        if show_details:
            progress_bar.progress(0.9)
            status_text.text("Đang đánh giá trên test set... (90%)")
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.keras.log_model(model, "final_model")
        log_text += f"✅ **Đánh giá cuối**: Độ chính xác trên test set: {test_accuracy:.4f}\n"
        if show_details:
            log_container.text(log_text)
            progress_bar.progress(1.0)
            status_text.text("Hoàn tất! (100%)")
        
    return model, test_accuracy, log_text

# Xử lý ảnh tải lên
def preprocess_uploaded_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    image = image / 255.0
    return image.reshape(1, 28, 28)

# Xử lý ảnh từ canvas
def preprocess_canvas_image(canvas):
    image = np.array(canvas)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    image = cv2.bitwise_not(image)
    image = cv2.resize(image, (28, 28))
    image = image / 255.0
    return image.reshape(1, 28, 28)

# Hiển thị mẫu dữ liệu
def show_sample_images(X, y):
    st.write("**🖼️ Một vài mẫu dữ liệu từ MNIST**")
    fig, axes = plt.subplots(1, 10, figsize=(15, 3))
    for digit in range(10):
        idx = np.where(y == digit)[0][0]
        ax = axes[digit]
        ax.imshow(X[idx], cmap='gray')
        ax.set_title(f"{digit}")
        ax.axis('off')
    st.pyplot(fig)

# Giao diện Streamlit
def create_streamlit_app():
    st.title("🔢 Pseudo Labelling trên MNIST với Neural Network")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📓 Giới thiệu", "📋 Huấn luyện", "🔮 Dự đoán", "⚡ MLflow"])
    
    # Tab 1: Giới thiệu
    with tab1:
        st.write("##### Pseudo Labelling với Neural Network")
        st.write(""" 
        Ứng dụng này thực hiện thuật toán **Pseudo Labelling** trên tập dữ liệu MNIST:
        - (0) Chia tập train/test.
        - (1) Lấy tỉ lệ dữ liệu có nhãn ban đầu (mặc định 1% mỗi class).
        - (2) Huấn luyện NN trên tập dữ liệu labeled.
        - (3) Dự đoán nhãn cho dữ liệu unlabeled.
        - (4) Gán nhãn giả dựa trên ngưỡng tin cậy.
        - (5) Lặp lại từ bước (2) với tập dữ liệu mới.
        Quá trình dừng khi gán hết nhãn hoặc sau số vòng lặp tối đa.
        """)
        x_train, y_train, _, _ = load_data()
        show_sample_images(x_train, y_train)
    
    # Tab 2: Huấn luyện
    with tab2:
        st.write("##### Chia tập dữ liệu")
        
        train_split = st.slider("Tỉ lệ dữ liệu train/test", 0.5, 0.95, 0.8, 0.05,
                                help="Chọn tỉ lệ dữ liệu dùng để huấn luyện (phần còn lại là test).")
        x_train, y_train, x_test, y_test = load_data(train_split)
        
        labeled_percentage = st.slider("Tỉ lệ dữ liệu labeled ban đầu (%)", 0.1, 10.0, 1.0, 0.1,
                                      help="Chọn phần trăm dữ liệu có nhãn ban đầu trong tập train.")
        
        # Chia dữ liệu labeled và unlabeled ngay sau khi chọn tỉ lệ
        global percentage
        percentage = labeled_percentage / 100
        x_labeled, y_labeled, x_unlabeled, _ = select_initial_data(x_train, y_train, percentage)
        
        # Tạo và hiển thị bảng dữ liệu
        total_samples = len(x_train) + len(x_test)
        data = {
            "Tập dữ liệu": ["Tập train", "Tập test", "Tập labeled ban đầu", "Tập unlabeled"],
            "Số mẫu": [len(x_train), len(x_test), len(x_labeled), len(x_unlabeled)],
            "Tỷ lệ (%)": [
                f"{len(x_train)/total_samples*100:.1f}%",
                f"{len(x_test)/total_samples*100:.1f}%",
                f"{len(x_labeled)/len(x_train)*100:.1f}% của train",
                f"{len(x_unlabeled)/len(x_train)*100:.1f}% của train"
            ]
        }
        df = pd.DataFrame(data)
        st.write("**Kích thước tập dữ liệu sau khi chia:**")
        st.table(df)
        st.write("##### Huấn luyện mô hình Pseudo Labelling")
        custom_model_name = st.text_input("Nhập tên mô hình:")
        if not custom_model_name:
            custom_model_name = "Default_model"
        threshold = st.slider("Ngưỡng tin cậy", 0.5, 0.99, 0.95, 0.01)
        max_iterations = st.slider("Số vòng lặp tối đa", 1, 20, 5)
        
        with st.expander("Tùy chọn hiển thị chi tiết", expanded=False):
            show_details = st.checkbox("Hiển thị chi tiết quá trình huấn luyện", value=False)
        
        if st.button("🚀 Chạy Pseudo Labelling"):
            with st.spinner("🔄 Đang khởi tạo huấn luyện..."):
                model, test_accuracy, log_text = pseudo_labeling_with_mlflow(
                    x_labeled, y_labeled, x_unlabeled, x_test, y_test,
                    threshold, max_iterations, custom_model_name, show_details
                )
                st.session_state['model'] = model
            
            st.success(f"✅ Huấn luyện xong! Độ chính xác trên test: {test_accuracy:.4f}")
            
            show_details = st.checkbox("Hiển thị chi tiết quá trình huấn luyện", value=False)
    
    # Tab 3: Dự đoán
    with tab3:
        st.write("**🔮 Dự đoán chữ số**")
        if 'model' not in st.session_state:
            st.warning("Vui lòng huấn luyện mô hình trước ở tab Huấn luyện!")
        else:
            option = st.radio("🖼️ Chọn phương thức nhập:", ["📂 Tải ảnh lên", "✏️ Vẽ số"])
            
            if option == "📂 Tải ảnh lên":
                uploaded_file = st.file_uploader("📤 Tải ảnh số viết tay (PNG, JPG)", type=["png", "jpg", "jpeg"])
                if uploaded_file is not None:
                    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
                    processed_image = preprocess_uploaded_image(image)
                    st.image(image, caption="📷 Ảnh tải lên", width=200)
                    
                    if st.button("🔮 Dự đoán"):
                        model = st.session_state['model']
                        prediction = model.predict(processed_image)
                        predicted_digit = np.argmax(prediction)
                        confidence = np.max(prediction)
                        st.write(f"🎯 **Dự đoán: {predicted_digit}**")
                        st.write(f"🔢 **Độ tin cậy: {confidence * 100:.2f}%**")
            
            elif option == "✏️ Vẽ số":
                canvas_result = st_canvas(
                    fill_color="white", stroke_width=15, stroke_color="black",
                    background_color="white", width=280, height=280, drawing_mode="freedraw", key="canvas"
                )
                if st.button("🔮 Dự đoán"):
                    if canvas_result.image_data is not None:
                        processed_canvas = preprocess_canvas_image(canvas_result.image_data)
                        model = st.session_state['model']
                        prediction = model.predict(processed_canvas)
                        predicted_digit = np.argmax(prediction)
                        confidence = np.max(prediction)
                        st.write(f"🎯 **Dự đoán: {predicted_digit}**")
                        st.write(f"🔢 **Độ tin cậy: {confidence * 100:.2f}%**")
    
    # Tab 4: MLflow Tracking
    with tab4:
        st.header("📊 MLflow Tracking")
        st.write("Xem chi tiết các kết quả đã lưu trong MLflow.")
        
        runs = mlflow.search_runs(order_by=["start_time desc"])
        if not runs.empty:
            runs["model_custom_name"] = runs["tags.mlflow.runName"]
            
            search_model_name = st.text_input("🔍 Nhập tên mô hình để tìm kiếm:", "")
            if search_model_name:
                filtered_runs = runs[runs["model_custom_name"].str.contains(search_model_name, case=False, na=False)]
            else:
                filtered_runs = runs
            
            if not filtered_runs.empty:
                st.write("### 📜 Danh sách mô hình đã lưu:")
                available_columns = [col for col in [
                    "model_custom_name", "start_time",
                    "metrics.train_accuracy", "metrics.val_accuracy", "metrics.test_accuracy",
                    "metrics.labeled_samples"
                ] if col in filtered_runs.columns]
                display_df = filtered_runs[available_columns]
                display_df = display_df.rename(columns={"model_custom_name": "Custom Model Name"})
                st.dataframe(display_df)
                
                selected_model_name = st.selectbox("📝 Chọn một mô hình để xem chi tiết:",
                                                  filtered_runs["model_custom_name"].tolist())
                if selected_model_name:
                    selected_run = filtered_runs[filtered_runs["model_custom_name"] == selected_model_name].iloc[0]
                    run_details = mlflow.get_run(selected_run["run_id"])
                    custom_name = run_details.data.tags.get('mlflow.runName', 'Không có tên')
                    st.write(f"### 🔍 Chi tiết mô hình: `{custom_name}`")
                    
                    st.write("📌 **Tham số:**")
                    for key, value in run_details.data.params.items():
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
