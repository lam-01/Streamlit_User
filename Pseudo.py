import streamlit as st
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import mlflow
import mlflow.keras
import cv2
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_model_selection import train_test_split

# Khai báo biến percentage toàn cục
percentage = 0.01  # Giá trị mặc định 1%

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
    # Validate epochs
    if model_params['epochs'] <= 0:
        st.error("Số epochs phải lớn hơn 0!")
        return None, 0, {}

    # Validate input data
    if len(x_labeled) == 0 or len(y_labeled) == 0:
        st.error("Tập dữ liệu labeled ban đầu rỗng!")
        return None, 0, {}
    if len(x_unlabeled) == 0:
        st.warning("Tập dữ liệu unlabeled rỗng, sẽ chỉ huấn luyện trên dữ liệu labeled.")
    if len(x_val) == 0 or len(y_val) == 0:
        st.error("Tập dữ liệu validation rỗng!")
        return None, 0, {}
    if len(x_test) == 0 or len(y_test) == 0:
        st.error("Tập dữ liệu test rỗng!")
        return None, 0, {}

    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.empty()
    metrics_container = st.empty()
    samples_container = st.empty()
    metrics_plot = st.empty()  # Container for the plot
    
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
        
        # Lists to store per-epoch accuracy for plotting
        train_acc_history = []
        val_acc_history = []
        
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
            
            try:
                history = model.fit(
                    x_train_current, y_train_current,
                    epochs=model_params['epochs'],
                    batch_size=32,
                    verbose=0,
                    validation_data=(x_val, y_val)
                )
                # Collect per-epoch accuracy
                train_acc_history.extend(history.history['accuracy'])
                val_acc_history.extend(history.history['val_accuracy'])
                train_acc = history.history['accuracy'][-1]
                val_acc = history.history['val_accuracy'][-1]
            except Exception as e:
                st.error(f"Lỗi trong quá trình huấn luyện: {str(e)}")
                return None, 0, metrics_history
            
            test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
            
            mlflow.log_metric("train_accuracy", train_acc, step=iteration)
            mlflow.log_metric("val_accuracy", val_acc, step=iteration)
            mlflow.log_metric("test_accuracy", test_acc, step=iteration)
            
            # Display results after each iteration
            with results_container.container():
                st.markdown(f"### Iteration {iteration + 1} kết thúc:")
                st.write(f"- Số mẫu labeled hiện tại: {len(x_train_current)}")
                st.write(f"- Số mẫu unlabeled còn lại: {len(remaining_unlabeled)}")
                st.write(f"- Độ chính xác train: {train_acc:.4f}")
                st.write(f"- Độ chính xác validation: {val_acc:.4f}")
                st.write(f"- Độ chính xác test: {test_acc:.4f}")
            
            # Plot training progress
            if train_acc_history and len(train_acc_history) > 0 and (iteration + 1) % max(1, max_iterations // 5) == 0:
                fig, ax = plt.subplots(figsize=(10, 4))
                epochs_range = list(range(1, len(train_acc_history) + 1))
                ax.plot(epochs_range, train_acc_history, 'b-', label='Train Accuracy')
                ax.plot(epochs_range, val_acc_history, 'r-', label='Validation Accuracy')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Accuracy')
                ax.set_title('Training Progress')
                ax.legend()
                ax.grid(True)
                metrics_plot.pyplot(fig)
                plt.close(fig)
            
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
            
            metrics_df = pd.DataFrame(metrics_history)
            metrics_container.dataframe(metrics_df)
        
        progress_bar.progress(100)
        status_text.text("Hoàn tất huấn luyện! (100%)")
        
        final_test_loss, final_test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        mlflow.log_metric("final_test_accuracy", final_test_accuracy)
        mlflow.keras.log_model(model, "final_model")
        
        # Final results display
        with results_container.container():
            st.markdown("### Kết quả cuối cùng:")
            st.write(f"- Độ chính xác train cuối cùng: {train_acc:.4f}")
            st.write(f"- Độ chính xác validation cuối cùng: {val_acc:.4f}")
            st.write(f"- Độ chính xác test cuối cùng: {final_test_accuracy:.4f}")
        
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

# Visualize mạng nơ-ron với kết quả dự đoán
def visualize_neural_network_prediction(model, input_image, predicted_label):
    try:
        hidden_layer_sizes = []
        for layer in model.layers:
            if isinstance(layer, layers.Dense) and layer != model.layers[-1]:
                hidden_layer_sizes.append(layer.units)
        output_layer_size = model.layers[-1].units
        input_layer_size = 784
        layer_sizes = [input_layer_size] + hidden_layer_sizes + [output_layer_size]
        num_layers = len(layer_sizes)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 3]})

        ax1.imshow(input_image.reshape(28, 28), cmap='gray')
        ax1.set_title("Input Image")
        ax1.axis('off')

        pos = {}
        layer_names = ['Input'] + [f'Hidden {i+1}' for i in range(len(hidden_layer_sizes))] + ['Output']

        for layer_idx, layer_size in enumerate(layer_sizes):
            for neuron_idx in range(layer_size):
                if layer_size > 20 and layer_idx == 0:
                    if neuron_idx < 10 or neuron_idx >= layer_size - 10:
                        pos[(layer_idx, neuron_idx)] = (layer_idx, neuron_idx / layer_size)
                    elif neuron_idx == 10:
                        pos[('dots', layer_idx)] = (layer_idx, 0.5)
                else:
                    pos[(layer_idx, neuron_idx)] = (layer_idx, neuron_idx / (layer_size - 1) if layer_size > 1 else 0.5)

        for layer_idx, layer_size in enumerate(layer_sizes):
            for neuron_idx in range(layer_size):
                if layer_size > 20 and layer_idx == 0 and neuron_idx >= 10 and neuron_idx < layer_size - 10:
                    continue
                
                x, y = pos[(layer_idx, neuron_idx)]
                circle = Circle((x, y), 0.05, color='white', ec='black')
                ax2.add_patch(circle)
                
                if layer_idx == num_layers - 1:
                    ax2.text(x + 0.2, y, f"{neuron_idx}", fontsize=12, color='white')
                
                if layer_idx == num_layers - 1 and neuron_idx == predicted_label:
                    square = Rectangle((x - 0.07, y - 0.07), 0.14, 0.14, fill=False, edgecolor='yellow', linewidth=2)
                    ax2.add_patch(square)

        if ('dots', 0) in pos:
            x, y = pos[('dots', 0)]
            ax2.text(x, y, "...", fontsize=12, color='white', ha='center', va='center')

        for layer_idx in range(len(layer_sizes) - 1):
            current_layer_size = layer_sizes[layer_idx]
            next_layer_size = layer_sizes[layer_idx + 1]

            if layer_idx == 0 and current_layer_size > 20:
                neuron_indices_1 = list(range(5)) + list(range(current_layer_size - 5, current_layer_size))
            else:
                neuron_indices_1 = range(current_layer_size)

            if layer_idx == len(layer_sizes) - 2:
                neuron_indices_2 = [predicted_label]
            else:
                if next_layer_size > 10:
                    neuron_indices_2 = list(range(5)) + list(range(next_layer_size - 5, next_layer_size))
                else:
                    neuron_indices_2 = range(next_layer_size)

            for idx1, neuron1 in enumerate(neuron_indices_1):
                for idx2, neuron2 in enumerate(neuron_indices_2):
                    x1, y1 = pos[(layer_idx, neuron1)]
                    x2, y2 = pos[(layer_idx + 1, neuron2)]
                    color = plt.cm.coolwarm(idx2 / max(len(neuron_indices_2), 1))
                    ax2.plot([x1, x2], [y1, y2], color=color, alpha=0.5, linewidth=1)

        ax2.set_xlim(-0.5, num_layers - 0.5)
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_xticks(range(num_layers))
        ax2.set_xticklabels(layer_names)
        ax2.set_yticks([])
        ax2.set_title(f"Neural Network Prediction: {predicted_label}")
        ax2.set_facecolor('black')

        return fig
    except AttributeError as e:
        st.error(f"Lỗi khi trực quan hóa mạng nơ-ron: {str(e)}")
        return None

# Giao diện Streamlit
def create_streamlit_app():
    st.title("🔢 Pseudo Labelling trên MNIST với Neural Network")
    
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    
    tab1, tab2, tab3, tab4 = st.tabs(["📓 Giới thiệu", "📋 Huấn luyện", "🔮 Dự đoán", "⚡ MLflow"])
    
    with tab1:
        st.write("##### Pseudo Labelling với Neural Network")
        st.write(""" 
            **Pseudo Labelling** là một kỹ thuật học bán giám sát (semi-supervised learning) nhằm tận dụng cả dữ liệu có nhãn (labeled data) và dữ liệu không nhãn (unlabeled data) để cải thiện hiệu suất của mô hình học máy, đặc biệt khi lượng dữ liệu có nhãn ban đầu rất hạn chế. Phương pháp này dựa trên ý tưởng sử dụng mô hình để dự đoán nhãn cho dữ liệu không nhãn, sau đó chọn các dự đoán có độ tin cậy cao để bổ sung vào tập dữ liệu có nhãn, từ đó huấn luyện lại mô hình.
            \n **Cơ chế hoạt động**
            \n Phương pháp Pseudo Labelling với Neural Network bao gồm các bước chính sau:
            # ... (rest of the introduction text remains unchanged) ...
             """)
        st.image("lb.png", caption="Sơ đồ chi tiết quy trình Pseudo Labelling với MNIST")

    with tab2:
        st.write("##### Tùy chọn mẫu dữ liệu")
        
        sample_size = st.number_input("**Chọn cỡ mẫu để huấn luyện**", 1000, 70000, 10000, step=1000)
        X, y = load_data(sample_size=sample_size)
        st.write(f"**Số lượng mẫu của bộ dữ liệu: {X.shape[0]}**")
        
        show_sample_images(X, y)
        
        st.write("##### Chia tập dữ liệu")
        
        test_size = st.slider("Tỷ lệ Test (%)", min_value=5, max_value=50, value=15, step=5)
        val_size = st.slider("Tỷ lệ Validation (%)", min_value=5, max_value=50, value=15, step=5)
        
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
        val_ratio = val_size / (100 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, 
                                                         test_size=val_ratio, 
                                                         random_state=42)
        
        labeled_percentage = st.slider("Tỉ lệ dữ liệu labeled ban đầu (%)", 0.1, 10.0, 1.0, 0.1)
        
        global percentage
        percentage = labeled_percentage / 100
        x_labeled, y_labeled, x_unlabeled, _ = select_initial_data(X_train, y_train, percentage)
        
        total_samples = len(X)
        data = {
            "Tập dữ liệu": ["Tập train", "Tập validation", "Tập test", "Tập labeled ban đầu", "Tập unlabeled"],
            "Số mẫu": [len(X_train), len(X_val), len(X_test), len(x_labeled), len(x_unlabeled)],
            "Tỷ lệ (%)": [
                f"{len(X_train)/total_samples*100:.1f}%",
                f"{len(X_val)/total_samples*100:.1f}%",
                f"{len(X_test)/total_samples*100:.1f}%",
                f"{len(x_labeled)/len(X_train)*100:.1f}% của train",
                f"{len(x_unlabeled)/len(X_train)*100:.1f}% của train"
            ]
        }
        df = pd.DataFrame(data)
        st.write("**Kích thước tập dữ liệu :**")
        st.table(df)
        
        st.write("##### Thiết lập tham số Neural Network")
        params = {}
        params["num_hidden_layers"] = st.slider("Số lớp ẩn", 1, 5, 2)
        params["neurons_per_layer"] = st.slider("Số neuron mỗi lớp", 50, 200, 128)
        params["epochs"] = st.slider("Epochs", 1, 50, 10)
        params["activation"] = st.selectbox("Hàm kích hoạt", ["relu", "tanh", "sigmoid"])
        params["learning_rate"] = st.slider("Tốc độ học (learning rate)", 0.0001, 0.1, 0.001, format="%.4f")
        
        st.write("##### Huấn luyện mô hình Pseudo Labelling")
        custom_model_name = st.text_input("Nhập tên mô hình :", "")
        if not custom_model_name:
            custom_model_name = "Default_model"
        
        threshold = st.slider("Ngưỡng tin cậy", 0.5, 0.99, 0.95, 0.01)
        max_iterations = st.slider("Số vòng lặp tối đa", 1, 20, 5)
        
        if st.button("🚀 Chạy Pseudo Labelling"):
            with st.spinner("🔄 Đang khởi tạo huấn luyện..."):
                model, test_accuracy, metrics_history = pseudo_labeling_with_mlflow(
                    x_labeled, y_labeled, x_unlabeled, X_val, y_val, X_test, y_test,
                    threshold, max_iterations, custom_model_name, params
                )
                if model is not None:
                    st.session_state.trained_models[custom_model_name] = model
                    st.success(f"✅ Huấn luyện xong! Độ chính xác cuối cùng trên test: {test_accuracy:.4f}")
                    
                    # Plot final training progress
                    if 'train_accuracy' in metrics_history and 'val_accuracy' in metrics_history:
                        fig, ax = plt.subplots(figsize=(10, 4))
                        iterations_range = metrics_history['iteration']
                        ax.plot(iterations_range, metrics_history['train_accuracy'], 'b-', label='Train Accuracy')
                        ax.plot(iterations_range, metrics_history['val_accuracy'], 'r-', label='Validation Accuracy')
                        ax.plot(iterations_range, metrics_history['test_accuracy'], 'g-', label='Test Accuracy')
                        ax.set_xlabel('Iteration')
                        ax.set_ylabel('Accuracy')
                        ax.set_title('Final Training Progress Across Iterations')
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)
                        plt.close(fig)
                else:
                    st.error("Huấn luyện thất bại! Vui lòng kiểm tra thông báo lỗi ở trên.")
    
    with tab3:
        st.write("**🔮 Dự đoán chữ số**")
        
        if 'trained_models' not in st.session_state or not st.session_state.trained_models:
            st.warning("⚠️ Vui lòng huấn luyện ít nhất một mô hình trước khi dự đoán!")
        else:
            model_names = list(st.session_state.trained_models.keys())
            selected_model_name = st.selectbox("📝 Chọn mô hình để dự đoán:", model_names)
            selected_model = st.session_state.trained_models[selected_model_name]
            
            show_visualization = st.checkbox("Hiển thị biểu đồ mạng nơ-ron", value=True)
            
            option = st.radio("🖼️ Chọn phương thức nhập:", ["📂 Tải ảnh lên", "✏️ Vẽ số"])
            
            if option == "📂 Tải ảnh lên":
                uploaded_file = st.file_uploader("📤 Tải ảnh số viết tay (PNG, JPG)", type=["png", "jpg", "jpeg"])
                if uploaded_file is not None:
                    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
                    processed_image = preprocess_uploaded_image(image)
                    st.image(image, caption="📷 Ảnh tải lên", width=200)
                    
                    if st.button("🔮 Dự đoán"):
                        if selected_model is None:
                            st.error("Mô hình không hợp lệ! Vui lòng chọn lại mô hình.")
                        else:
                            prediction = selected_model.predict(processed_image)
                            predicted_digit = np.argmax(prediction)
                            confidence = np.max(prediction)
                            st.write(f"🎯 **Dự đoán: {predicted_digit}**")
                            st.write(f"🔢 **Độ tin cậy: {confidence * 100:.2f}%**")
                            if show_visualization:
                                fig = visualize_neural_network_prediction(selected_model, processed_image[0], predicted_digit)
                                if fig is not None:
                                    st.pyplot(fig)
            
            elif option == "✏️ Vẽ số":
                st.write("Vẽ chữ số của bạn dưới đây:")
                canvas_result = st_canvas(
                    fill_color="white",
                    stroke_width=15,
                    stroke_color="black",
                    background_color="white",
                    width=280,
                    height=280,
                    drawing_mode="freedraw",
                    key="prediction_canvas"
                )
                if st.button("🔮 Dự đoán"):
                    if canvas_result.image_data is not None:
                        if selected_model is None:
                            st.error("Mô hình không hợp lệ! Vui lòng chọn lại mô hình.")
                        else:
                            processed_canvas = preprocess_canvas_image(canvas_result.image_data)
                            prediction = selected_model.predict(processed_canvas)
                            predicted_digit = np.argmax(prediction)
                            confidence = np.max(prediction)
                            st.write(f"🎯 **Dự đoán: {predicted_digit}**")
                            st.write(f"🔢 **Độ tin cậy: {confidence * 100:.2f}%**")
                            if show_visualization:
                                fig = visualize_neural_network_prediction(selected_model, processed_canvas[0], predicted_digit)
                                if fig is not None:
                                    st.pyplot(fig)
                    else:
                        st.warning("Vui lòng vẽ một chữ số trước khi dự đoán!")
    
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
                    selected_run = runs[runs["model_custom_name"] == selected_model_name].iloc[0]
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
