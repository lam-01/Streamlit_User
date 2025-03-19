import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import mlflow
import mlflow.sklearn

# Hàm lấy 1% dữ liệu cho mỗi class
def get_initial_labeled_data(X, y, percentage=0.01):
    X_labeled = []
    y_labeled = []
    for digit in range(10):
        digit_indices = np.where(y == digit)[0]
        n_samples = int(len(digit_indices) * percentage)
        selected_indices = np.random.choice(digit_indices, n_samples, replace=False)
        X_labeled.append(X[selected_indices])
        y_labeled.append(y[selected_indices])
    return np.vstack(X_labeled), np.hstack(y_labeled)

# Hàm thực hiện Pseudo Labeling
def pseudo_labeling(model, X_unlabeled, threshold=0.95):
    probs = model.predict_proba(X_unlabeled)
    max_probs = np.max(probs, axis=1)
    pseudo_labels = np.argmax(probs, axis=1)
    confident_indices = np.where(max_probs >= threshold)[0]
    return X_unlabeled[confident_indices], pseudo_labels[confident_indices], len(confident_indices)

# Hàm huấn luyện với Pseudo Labeling
def train_with_pseudo_labeling(custom_model_name, params, X_train, y_train, X_test, y_test, cv_folds, max_iterations=5, threshold=0.95):
    # Bước 0 & 1: Lấy 1% dữ liệu ban đầu làm tập labeled
    X_labeled, y_labeled = get_initial_labeled_data(X_train, y_train, percentage=0.01)
    X_unlabeled = np.delete(X_train, np.where(np.isin(y_train, y_labeled)), axis=0)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    hidden_layer_sizes = tuple([params["neurons_per_layer"]] * params["num_hidden_layers"])
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=1,
        activation=params["activation"],
        learning_rate_init=params["learning_rate"],
        solver='adam',
        alpha=0.0001,
        random_state=42,
        warm_start=True
    )
    
    iteration = 0
    while iteration < max_iterations and len(X_unlabeled) > 0:
        iteration += 1
        st.write(f"🔄 **Vòng lặp Pseudo Labeling {iteration}/{max_iterations}**")
        
        # Bước 2: Huấn luyện mô hình trên tập labeled
        with mlflow.start_run(run_name=f"{custom_model_name}_iter_{iteration}"):
            for epoch in range(params["epochs"]):
                model.fit(X_labeled, y_labeled)
                progress = (epoch + 1) / params["epochs"]
                progress_bar.progress(progress)
                status_text.text(f"Đang huấn luyện vòng {iteration}: {int(progress * 100)}%")
            
            # Đánh giá trên tập test
            y_test_pred = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            st.write(f"🎯 Độ chính xác trên tập test sau vòng {iteration}: {test_accuracy:.4f}")
            
            # Log vào MLflow
            mlflow.log_param("iteration", iteration)
            mlflow.log_params(params)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.sklearn.log_model(model, f"model_iter_{iteration}")
        
        # Bước 3 & 4: Dự đoán và gán nhãn giả cho tập unlabeled
        X_pseudo, y_pseudo, n_confident = pseudo_labeling(model, X_unlabeled, threshold)
        st.write(f"✅ Gán nhãn giả cho {n_confident} mẫu với ngưỡng {threshold}")
        
        # Bước 5: Cập nhật tập labeled và loại bỏ dữ liệu đã gán khỏi unlabeled
        if n_confident > 0:
            X_labeled = np.vstack([X_labeled, X_pseudo])
            y_labeled = np.hstack([y_labeled, y_pseudo])
            X_unlabeled = np.delete(X_unlabeled, np.where(np.isin(X_unlabeled, X_pseudo).all(axis=1)), axis=0)
        else:
            st.write("⚠️ Không có mẫu nào đạt ngưỡng tin cậy. Dừng quá trình.")
            break
    
    progress_bar.empty()
    status_text.empty()
    return model, test_accuracy

# Cập nhật giao diện Streamlit
def update_streamlit_app():
    st.title("🔢 Phân loại chữ số viết tay với Pseudo Labeling")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📓 Lí thuyết", "📋 Huấn luyện", "🔮 Dự đoán", "⚡ MLflow", "🔄 Pseudo Labeling"])
    
    # Các tab cũ giữ nguyên, chỉ thêm tab mới
    with tab5:
        st.write("##### 🔄 Pseudo Labeling trên MNIST")
        st.write("Quy trình: (1) Lấy 1% dữ liệu mỗi class làm tập labeled ban đầu; (2) Huấn luyện NN; (3) Dự đoán nhãn giả cho dữ liệu unlabeled; (4) Gán nhãn với ngưỡng tin cậy; (5) Lặp lại.")
        
        max_samples = 70000
        n_samples = st.number_input(
            "Số lượng mẫu để huấn luyện", min_value=1000, max_value=max_samples, value=9000, step=1000, key="pseudo_n_samples"
        )
        
        X, y = load_data(n_samples=n_samples)
        st.write(f"**Số lượng mẫu được chọn: {X.shape[0]}**")
        show_sample_images(X, y)
        
        test_size = st.slider("Tỷ lệ Test (%)", min_value=5, max_value=30, value=15, step=5, key="pseudo_test_size")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
        
        st.write("**🚀 Cấu hình Pseudo Labeling**")
        custom_model_name = st.text_input("Nhập tên mô hình để lưu vào MLflow:", "Pseudo_Model", key="pseudo_model_name")
        params = {}
        params["num_hidden_layers"] = st.slider("Số lớp ẩn", 1, 2, 1, key="pseudo_layers")
        params["neurons_per_layer"] = st.slider("Số neuron mỗi lớp", 20, 100, 50, key="pseudo_neurons")
        params["epochs"] = st.slider("Epochs mỗi vòng", 5, 50, 10, key="pseudo_epochs")
        params["activation"] = st.selectbox("Hàm kích hoạt", ["relu", "tanh", "logistic"], key="pseudo_activation")
        params["learning_rate"] = st.slider("Tốc độ học", 0.0001, 0.1, 0.001, key="pseudo_lr")
        cv_folds = st.slider("Số fold CV", 2, 5, 3, key="pseudo_cv")
        threshold = st.slider("Ngưỡng tin cậy (threshold)", 0.8, 0.99, 0.95, step=0.01, key="pseudo_threshold")
        max_iterations = st.slider("Số vòng lặp tối đa", 1, 10, 5, key="pseudo_iterations")
        
        if st.button("🚀 Bắt đầu Pseudo Labeling", key="pseudo_start"):
            with st.spinner("🔄 Đang thực hiện Pseudo Labeling..."):
                model, test_accuracy = train_with_pseudo_labeling(
                    custom_model_name, params, X_train, y_train, X_test, y_test, cv_folds, max_iterations, threshold
                )
                st.session_state.trained_models[custom_model_name] = model
                st.success(f"✅ Hoàn tất Pseudo Labeling! Độ chính xác cuối cùng trên tập test: {test_accuracy:.4f}")

# Cập nhật hàm main
if __name__ == "__main__":
    update_streamlit_app()
