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
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Hàm xây dựng model NN với tham số tùy chỉnh
def create_model(num_hidden_layers=2, neurons_per_layer=128, activation='relu', learning_rate=0.001):
    model = keras.Sequential()
    model.add(layers.Input(shape=(784,)))
    
    for _ in range(num_hidden_layers):
        model.add(layers.Dense(neurons_per_layer, activation=activation))
        model.add(layers.Dropout(0.2))
    
    model.add(layers.Dense(10, activation='softmax'))
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# Tải và xử lý dữ liệu MNIST từ OpenML
@st.cache_data
def load_data(sample_size=None):
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)
    X = X / 255.0
    if sample_size is not None and sample_size < len(X):
        X, _, y, _ = train_test_split(X, y, train_size=sample_size, random_state=42)
    return X, y

# Chọn dữ liệu labeled ban đầu với tỉ lệ chính xác
def select_initial_data(x_train, y_train, percentage):
    total_labeled_samples = int(len(x_train) * percentage)
    n_classes = 10
    min_samples_per_class = 1
    remaining_samples = total_labeled_samples - min_samples_per_class * n_classes
    
    labeled_idx = []
    for i in range(n_classes):
        class_idx = np.where(y_train == i)[0]
        initial_samples = np.random.choice(class_idx, min_samples_per_class, replace=False)
        labeled_idx.extend(initial_samples)
    
    if remaining_samples > 0:
        remaining_idx = [i for i in range(len(x_train)) if i not in labeled_idx]
        x_remaining = x_train[remaining_idx]
        y_remaining = y_train[remaining_idx]
        extra_idx = np.random.choice(len(x_remaining), remaining_samples, replace=False)
        labeled_idx.extend([remaining_idx[i] for i in extra_idx])
    
    x_labeled = x_train[labeled_idx]
    y_labeled = y_train[labeled_idx]
    unlabeled_idx = [i for i in range(len(x_train)) if i not in labeled_idx]
    x_unlabeled = x_train[unlabeled_idx]
    
    return x_labeled, y_labeled, x_unlabeled, unlabeled_idx

# Hiển thị mẫu dữ liệu được gán nhãn giả
def show_pseudo_labeled_samples(model, samples, predictions, n_samples=5):
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 3, 4))
    
    if len(samples) <= n_samples:
        selected_indices = np.arange(len(samples))
    else:
        selected_indices = np.random.choice(len(samples), n_samples, replace=False)
    
    for i, idx in enumerate(selected_indices):
        axes[0, i].imshow(samples[idx].reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        
        pred_idx = np.argmax(predictions[idx])
        confidence = np.max(predictions[idx])
        axes[1, i].axis('off')
        axes[1, i].text(0.5, 0.5, f"{pred_idx}\n{confidence:.2f}", 
                      ha='center', va='center',
                      color='green' if confidence > 0.9 else 'blue')
    
    plt.tight_layout()
    return fig

# Thuật toán Pseudo Labelling với MLflow
def pseudo_labeling_with_mlflow(x_labeled, y_labeled, x_unlabeled, x_val, y_val, x_test, y_test, 
                               threshold, max_iterations, custom_model_name, model_params):
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.empty()
    metrics_container = st.empty()
    samples_container = st.empty()
    
    with mlflow.start_run(run_name=custom_model_name):
        model = create_model(
            num_hidden_layers=model_params['num_hidden_layers'],
            neurons_per_layer=model_params['neurons_per_layer'],
            activation=model_params['activation'],
            learning_rate=model_params['learning_rate']
        )
        
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("max_iterations", max_iterations)
        mlflow.log_param("initial_labeled_percentage", percentage * 100)
        for key, value in model_params.items():
            mlflow.log_param(key, value)
        
        x_train_current = x_labeled.copy()
        y_train_current = y_labeled.copy()
        remaining_unlabeled = x_unlabeled.copy()
        
        metrics_history = {
            'iteration': [],
            'labeled_samples_count': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'test_accuracy': []
        }
        
        metrics_history['iteration'].append(0)
        metrics_history['labeled_samples_count'].append(len(x_labeled))
        metrics_history['train_accuracy'].append(0)
        metrics_history['val_accuracy'].append(0)
        metrics_history['test_accuracy'].append(0)
        
        progress_bar.progress(0)
        status_text.text("Khởi tạo mô hình... (0%)")
        
        total_steps = max_iterations * 2
        current_step = 0
        
        for iteration in range(max_iterations):
            current_step += 1
            progress = min(100, int((current_step / total_steps) * 100))
            progress_bar.progress(progress)
            status_text.text(f"Iteration {iteration + 1}: Đang huấn luyện... ({progress}%)")
            
            history = model.fit(
                x_train_current, y_train_current,
                epochs=model_params['epochs'],
                batch_size=32,
                verbose=0,
                validation_data=(x_val, y_val)
            )
            
            train_acc = history.history['accuracy'][-1]
            val_acc = history.history['val_accuracy'][-1]
            test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
            
            mlflow.log_metric("train_accuracy", train_acc, step=iteration)
            mlflow.log_metric("val_accuracy", val_acc, step=iteration)
            mlflow.log_metric("test_accuracy", test_acc, step=iteration)
            
            current_step += 1
            progress = min(100, int((current_step / total_steps) * 100))
            progress_bar.progress(progress)
            status_text.text(f"Iteration {iteration + 1}: Đang gán nhãn... ({progress}%)")
            
            if len(remaining_unlabeled) > 0:
                predictions = model.predict(remaining_unlabeled, verbose=0)
                max_probs = np.max(predictions, axis=1)
                pseudo_labels = np.argmax(predictions, axis=1)
                
                confident_idx = np.where(max_probs >= threshold)[0]
                
                if len(confident_idx) > 0:
                    fig = show_pseudo_labeled_samples(
                        model, 
                        remaining_unlabeled[confident_idx], 
                        predictions[confident_idx]
                    )
                    samples_container.pyplot(fig)
                    
                    x_train_current = np.concatenate([x_train_current, remaining_unlabeled[confident_idx]])
                    y_train_current = np.concatenate([y_train_current, pseudo_labels[confident_idx]])
                    remaining_unlabeled = np.delete(remaining_unlabeled, confident_idx, axis=0)
                    mlflow.log_metric("labeled_samples", len(confident_idx), step=iteration)
                else:
                    break
            else:
                break
            
            metrics_history['iteration'].append(iteration + 1)
            metrics_history['labeled_samples_count'].append(len(x_train_current))
            metrics_history['train_accuracy'].append(train_acc)
            metrics_history['val_accuracy'].append(val_acc)
            metrics_history['test_accuracy'].append(test_acc)
            
            with results_container.container():
                st.markdown(f"### Iteration {iteration + 1} kết thúc:")
                st.write(f"- Số mẫu labeled hiện tại: {len(x_train_current)}")
                st.write(f"- Số mẫu unlabeled còn lại: {len(remaining_unlabeled)}")
                st.write(f"- Độ chính xác train: {train_acc:.4f}")
                st.write(f"- Độ chính xác validation: {val_acc:.4f}")
                st.write(f"- Độ chính xác test: {test_acc:.4f}")
            
            metrics_df = pd.DataFrame(metrics_history)
            metrics_container.dataframe(metrics_df)
        
        progress_bar.progress(100)
        status_text.text("Hoàn tất huấn luyện! (100%)")
        
        final_test_loss, final_test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        mlflow.log_metric("final_test_accuracy", final_test_accuracy)
        mlflow.keras.log_model(model, "final_model")
        
    return model, final_test_accuracy, metrics_history

# Xử lý ảnh tải lên
def preprocess_uploaded_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    image = image / 255.0
    return image.reshape(1, 784)

# Xử lý ảnh từ canvas
def preprocess_canvas_image(canvas):
    image = np.array(canvas)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    image = cv2.bitwise_not(image)
    image = cv2.resize(image, (28, 28))
    image = image / 255.0
    return image.reshape(1, 784)

# Hiển thị mẫu dữ liệu
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

# Giao diện Streamlit
def create_streamlit_app():
    st.title("🔢 Pseudo Labelling trên MNIST với Neural Network")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📓 Giới thiệu", "📋 Huấn luyện", "🔮 Dự đoán", "⚡ MLflow"])
    
    with tab1:
        st.write("##### Pseudo Labelling với Neural Network")
        st.write(""" 
        **Pseudo Labelling** là một kỹ thuật học bán giám sát nhằm tận dụng cả dữ liệu có nhãn và không nhãn để cải thiện hiệu suất mô hình.
        """)
    
    with tab2:
        st.write("##### Chuẩn bị dữ liệu")
        
        sample_size = st.number_input("**Chọn cỡ mẫu để huấn luyện**", 1000, 70000, 10000, step=1000)
        X, y = load_data(sample_size=sample_size)
        st.write(f"**Số lượng mẫu của bộ dữ liệu: {X.shape[0]}**")
        
        show_sample_images(X, y)
        
        st.write("##### Chia tập dữ liệu")
        
        train_size = st.slider("Tỷ lệ Train (%)", min_value=50, max_value=90, value=70, step=5)
        val_size = st.slider("Tỷ lệ Validation (%)", min_value=5, max_value=30, value=15, step=5)
        test_size = st.slider("Tỷ lệ Test (%)", min_value=5, max_value=30, value=15, step=5)
        
        total_ratio = train_size + val_size + test_size
        if total_ratio != 100:
            st.error(f"Tổng tỉ lệ phải bằng 100%, hiện tại là {total_ratio}%!")
            return
        
        # Chia tập test trước
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
        
        # Chia tập validation và train từ phần còn lại
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, 
                                                         test_size=val_size/(train_size + val_size), 
                                                         random_state=42)
        
        labeled_percentage = st.slider("Tỉ lệ dữ liệu labeled ban đầu (%)", 0.1, 10.0, 1.0, 0.1)
        
        global percentage
        percentage = labeled_percentage / 100
        x_labeled, y_labeled, x_unlabeled, _ = select_initial_data(X_train, y_train, percentage)
        
        total_samples = len(X)
        data = {
            "Tập dữ liệu": ["Tổng mẫu", "Tập train", "Tập validation", "Tập test", "Tập labeled ban đầu", "Tập unlabeled"],
            "Số mẫu": [len(X), len(X_train), len(X_val), len(X_test), len(x_labeled), len(x_unlabeled)],
            "Tỷ lệ (%)": [
                "100%",
                f"{len(X_train)/total_samples*100:.1f}%",
                f"{len(X_val)/total_samples*100:.1f}%",
                f"{len(X_test)/total_samples*100:.1f}%",
                f"{len(x_labeled)/len(X_train)*100:.1f}% của train",
                f"{len(x_unlabeled)/len(X_train)*100:.1f}% của train"
            ]
        }
        df = pd.DataFrame(data)
        st.write("**Kích thước tập dữ liệu sau khi chia:**")
        st.table(df)
        
        st.write("##### Thiết lập tham số Neural Network")
        params = {}
        params["num_hidden_layers"] = st.slider("Số lớp ẩn", 1, 5, 2)
        params["neurons_per_layer"] = st.slider("Số neuron mỗi lớp", 50, 200, 100)
        params["epochs"] = st.slider("Epochs", 5, 50, 10)
        params["activation"] = st.selectbox("Hàm kích hoạt", ["relu", "tanh", "sigmoid"])
        params["learning_rate"] = st.slider("Tốc độ học (learning rate)", 0.0001, 0.1, 0.001, format="%.4f")
        
        st.write("##### Huấn luyện mô hình Pseudo Labelling")
        custom_model_name = st.text_input("Nhập tên mô hình:", f"PseudoLabel_Model_{int(time.time())}")
        threshold = st.slider("Ngưỡng tin cậy", 0.5, 0.99, 0.95, 0.01)
        max_iterations = st.slider("Số vòng lặp tối đa", 1, 20, 5)
        
        if st.button("🚀 Chạy Pseudo Labelling"):
            with st.spinner("🔄 Đang khởi tạo huấn luyện..."):
                model, test_accuracy, metrics_history = pseudo_labeling_with_mlflow(
                    x_labeled, y_labeled, x_unlabeled, X_val, y_val, X_test, y_test,
                    threshold, max_iterations, custom_model_name, params
                )
                st.session_state['model'] = model
            
            st.success(f"✅ Huấn luyện xong! Độ chính xác cuối cùng trên test: {test_accuracy:.4f}")
    
    with tab3:
        st.write("**🔮 Dự đoán chữ số**")
        
        runs = mlflow.search_runs(order_by=["start_time desc"])
        model_options = ["Mô hình vừa huấn luyện (nếu có)"]
        if not runs.empty:
            runs["model_custom_name"] = runs["tags.mlflow.runName"]
            model_options.extend(runs["model_custom_name"].tolist())
        
        selected_model_name = st.selectbox("Chọn mô hình để dự đoán:", model_options)
        
        if selected_model_name == "Mô hình vừa huấn luyện (nếu có)" and 'model' not in st.session_state:
            st.warning("Vui lòng huấn luyện mô hình trước ở tab Huấn luyện!")
        else:
            if selected_model_name != "Mô hình vừa huấn luyện (nếu có)":
                selected_run = runs[runs["model_custom_name"] == selected_model_name].iloc[0]
                model_path = f"runs:/{selected_run['run_id']}/final_model"
                model = mlflow.keras.load_model(model_path)
                st.session_state['model'] = model
            
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
                    "metrics.labeled_samples", "metrics.final_test_accuracy"
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
    create_streamlit_app()
