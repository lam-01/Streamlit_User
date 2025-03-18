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
from sklearn.model_selection import train_test_split, KFold

# Hàm xây dựng model NN với tham số tùy chỉnh
def create_model(num_hidden_layers, neurons_per_layer, activation, learning_rate):
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    
    for _ in range(num_hidden_layers):
        model.add(layers.Dense(neurons_per_layer, activation=activation))
        model.add(layers.Dropout(0.2))
    
    model.add(layers.Dense(10, activation='softmax'))
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# Tải và chia dữ liệu với train/val/test
@st.cache_data
def load_data(train_split=0.7, val_split=0.15, sample_size=70000):
    (x_full, y_full), _ = keras.datasets.mnist.load_data()
    x_full = x_full.astype('float32') / 255
    
    # Giới hạn số lượng mẫu theo sample_size
    if sample_size < len(x_full):
        indices = np.random.permutation(len(x_full))[:sample_size]
        x_full = x_full[indices]
        y_full = y_full[indices]
    
    total_samples = len(x_full)
    train_size = int(total_samples * train_split)
    val_size = int(total_samples * val_split)
    test_size = total_samples - train_size - val_size
    
    indices = np.random.permutation(total_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    x_train = x_full[train_indices]
    y_train = y_full[train_indices]
    x_val = x_full[val_indices]
    y_val = y_full[val_indices]
    x_test = x_full[test_indices]
    y_test = y_full[test_indices]
    
    return x_train, y_train, x_val, y_val, x_test, y_test

# Chọn dữ liệu labeled ban đầu
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

# Thuật toán Pseudo Labelling với hiển thị kết quả mỗi iteration
def pseudo_labeling_with_mlflow(x_labeled, y_labeled, x_unlabeled, x_val, y_val, x_test, y_test, 
                              params, custom_model_name, cv_folds=5):
    progress_bar = st.progress(0)
    status_text = st.empty()
    result_container = st.empty()
    
    with mlflow.start_run(run_name=custom_model_name):
        mlflow.log_params(params)
        
        x_train_current = x_labeled.copy()
        y_train_current = y_labeled.copy()
        remaining_unlabeled = x_unlabeled.copy()
        
        progress_bar.progress(0.1)
        status_text.text("Đang khởi tạo mô hình... (10%)")
        
        for iteration in range(params["max_iterations"]):
            # Cross-validation
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(x_train_current)):
                x_cv_train = x_train_current[train_idx]
                y_cv_train = y_train_current[train_idx]
                x_cv_val = x_train_current[val_idx]
                y_cv_val = y_train_current[val_idx]
                
                model = create_model(params["num_hidden_layers"], 
                                   params["neurons_per_layer"],
                                   params["activation"],
                                   params["learning_rate"])
                
                model.fit(x_cv_train, y_cv_train,
                         epochs=params["epochs"],
                         batch_size=32,
                         verbose=0)
                
                val_acc = model.evaluate(x_cv_val, y_cv_val, verbose=0)[1]
                cv_scores.append(val_acc)
            
            # Huấn luyện trên toàn bộ dữ liệu hiện tại
            model = create_model(params["num_hidden_layers"], 
                               params["neurons_per_layer"],
                               params["activation"],
                               params["learning_rate"])
            
            history = model.fit(x_train_current, y_train_current,
                              epochs=params["epochs"],
                              batch_size=32,
                              verbose=0,
                              validation_data=(x_val, y_val))
            
            train_acc = history.history['accuracy'][-1]
            val_acc = history.history['val_accuracy'][-1]
            cv_mean_acc = np.mean(cv_scores)
            
            # Dự đoán trên unlabeled
            predictions = model.predict(remaining_unlabeled, verbose=0)
            max_probs = np.max(predictions, axis=1)
            pseudo_labels = np.argmax(predictions, axis=1)
            
            confident_idx = np.where(max_probs >= params["threshold"])[0]
            
            mlflow.log_metric("train_accuracy", train_acc, step=iteration)
            mlflow.log_metric("val_accuracy", val_acc, step=iteration)
            mlflow.log_metric("cv_mean_accuracy", cv_mean_acc, step=iteration)
            mlflow.log_metric("labeled_samples", len(confident_idx), step=iteration)
            
            # Hiển thị kết quả sau mỗi iteration
            with result_container.container():
                st.write(f"**Iteration {iteration + 1}:**")
                st.write(f"- Số mẫu được gán nhãn: {len(confident_idx)}")
                st.write(f"- Train accuracy: {train_acc:.4f}")
                st.write(f"- Validation accuracy: {val_acc:.4f}")
                st.write(f"- CV mean accuracy: {cv_mean_acc:.4f}")
                
                if len(confident_idx) > 0:
                    st.write("**Hình ảnh mẫu được gán nhãn:**")
                    fig, axes = plt.subplots(1, min(5, len(confident_idx)), figsize=(15, 3))
                    for i, ax in enumerate(axes if len(confident_idx) > 1 else [axes]):
                        idx = confident_idx[i]
                        ax.imshow(remaining_unlabeled[idx], cmap='gray')
                        ax.set_title(f"Nhãn: {pseudo_labels[idx]}")
                        ax.axis('off')
                    st.pyplot(fig)
            
            progress_bar.progress(0.5 + 0.4 * (iteration + 1) / params["max_iterations"])
            status_text.text(f"Iteration {iteration + 1}: Đã gán nhãn cho {len(confident_idx)} mẫu ({int(50 + 40 * (iteration + 1) / params['max_iterations'])}%)")
            
            if len(confident_idx) == 0:
                break
                
            x_train_current = np.concatenate([x_train_current, remaining_unlabeled[confident_idx]])
            y_train_current = np.concatenate([y_train_current, pseudo_labels[confident_idx]])
            remaining_unlabeled = np.delete(remaining_unlabeled, confident_idx, axis=0)
            
            if len(remaining_unlabeled) == 0:
                break
        
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.keras.log_model(model, "final_model")
        
        progress_bar.progress(1.0)
        status_text.text("Hoàn tất! (100%)")
        
    return model, test_accuracy

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
    
    with tab1:
        st.write("##### Pseudo Labelling với Neural Network")
        st.write(""" 
        **Pseudo Labelling** là một kỹ thuật học bán giám sát sử dụng dữ liệu có nhãn và không nhãn để cải thiện hiệu suất mô hình.
        \n Các bước chính:
        1. Chia dữ liệu thành train/val/test
        2. Lấy một phần nhỏ dữ liệu có nhãn ban đầu
        3. Huấn luyện NN và dự đoán nhãn cho dữ liệu không nhãn
        4. Gán nhãn giả cho các mẫu có độ tin cậy cao
        5. Lặp lại với tập dữ liệu mở rộng
        """)
    
    with tab2:
        sample_size = st.number_input("**Chọn cỡ mẫu để huấn luyện**", 1000, 70000, 10000, step=1000)
        x_train, y_train, x_val, y_val, _, _ = load_data(sample_size=sample_size)
        st.write(f"**Số lượng mẫu của bộ dữ liệu: {len(x_train) + len(x_val)}**")
        show_sample_images(x_train, y_train)
        
        st.write("##### Chia tập dữ liệu")
        train_split = st.slider("Tỉ lệ dữ liệu train", 0.5, 0.9, 0.7, 0.05)
        val_split = st.slider("Tỉ lệ dữ liệu validation", 0.05, 0.3, 0.15, 0.05)
        test_split = 1 - train_split - val_split
        if test_split < 0:
            st.error("Tổng tỉ lệ vượt quá 100%! Vui lòng điều chỉnh lại.")
            return
        
        x_train, y_train, x_val, y_val, x_test, y_test = load_data(train_split, val_split, sample_size)
        labeled_percentage = st.slider("Tỉ lệ dữ liệu labeled ban đầu (%)", 0.1, 10.0, 1.0, 0.1)
        percentage = labeled_percentage / 100
        x_labeled, y_labeled, x_unlabeled, _ = select_initial_data(x_train, y_train, percentage)
        
        data = {
            "Tập dữ liệu": ["Tập train", "Tập validation", "Tập test", "Tập labeled ban đầu", "Tập unlabeled"],
            "Số mẫu": [len(x_train), len(x_val), len(x_test), len(x_labeled), len(x_unlabeled)],
            "Tỷ lệ (%)": [
                f"{train_split*100:.1f}%",
                f"{val_split*100:.1f}%",
                f"{test_split*100:.1f}%",
                f"{len(x_labeled)/len(x_train)*100:.1f}% của train",
                f"{len(x_unlabeled)/len(x_train)*100:.1f}% của train"
            ]
        }
        df = pd.DataFrame(data)
        st.write("**Kích thước tập dữ liệu sau khi chia:**")
        st.table(df)
        
        st.write("##### Huấn luyện mô hình Pseudo Labelling")
        custom_model_name = st.text_input("Nhập tên mô hình:", "Default_model")
        params = {
            "threshold": st.slider("Ngưỡng tin cậy", 0.5, 0.99, 0.95, 0.01),
            "max_iterations": st.slider("Số vòng lặp tối đa", 1, 20, 5),
            "num_hidden_layers": st.slider("Số lớp ẩn", 1, 5, 2),
            "neurons_per_layer": st.slider("Số neuron mỗi lớp", 50, 200, 100),
            "epochs": st.slider("Epochs", 5, 50, 10),
            "activation": st.selectbox("Hàm kích hoạt", ["relu", "tanh", "sigmoid"]),
            "learning_rate": st.slider("Tốc độ học (learning rate)", 0.0001, 0.1, 0.001),
            "initial_labeled_percentage": labeled_percentage
        }
        st.session_state.cv_folds = st.slider("Số lượng fold cho Cross-Validation", 2, 10, 5)
        
        if st.button("🚀 Chạy Pseudo Labelling"):
            with st.spinner("🔄 Đang khởi tạo huấn luyện..."):
                model, test_accuracy = pseudo_labeling_with_mlflow(
                    x_labeled, y_labeled, x_unlabeled, x_val, y_val, x_test, y_test,
                    params, custom_model_name, st.session_state.cv_folds
                )
                st.session_state['model'] = model
            
            st.success(f"✅ Huấn luyện xong! Độ chính xác trên test: {test_accuracy:.4f}")
    
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
    
    with tab4:
        st.write("##### MLflow Tracking")
        
        runs = mlflow.search_runs(order_by=["start_time desc"])
        if not runs.empty:
            runs["model_custom_name"] = runs["tags.mlflow.runName"]
            
            search_model_name = st.text_input("🔍 Nhập tên mô hình để tìm kiếm:", "")
            if search_model_name:
                filtered_runs = runs[runs["model_custom_name"].str.contains(search_model_name, case=False, na=False)]
            else:
                filtered_runs = runs
            
            if not filtered_runs.empty:
                st.write("##### 📜 Danh sách mô hình đã lưu:")
                available_columns = [col for col in [
                    "model_custom_name", "start_time",
                    "metrics.train_accuracy", "metrics.val_accuracy", "metrics.test_accuracy",
                    "metrics.cv_mean_accuracy", "metrics.labeled_samples"
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
                    st.write(f"##### 🔍 Chi tiết mô hình: `{custom_name}`")
                    
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
    mlflow.set_tracking_uri("http://localhost:5000")
    create_streamlit_app()
