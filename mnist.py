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

# Tạo selectbox để chọn dự án
option = st.sidebar.selectbox(
    "📌 Chọn một dự án để thực hiện:",
    ["Phân tích Titanic", "MNIST"]
)

# Hiển thị nội dung tương ứng với lựa chọn
if option == "MNIST":
# 📌 Tải và xử lý dữ liệu MNIST từ OpenML
    @st.cache_data
    def load_data():
        mnist = fetch_openml("mnist_784", version=1, as_frame=False)
        X, y = mnist.data, mnist.target.astype(int)  # Chuyển nhãn về kiểu số nguyên
        X = X / 255.0  # Chuẩn hóa về [0,1]
        return X, y

    # 📌 Chia dữ liệu thành train, validation, và test
    def split_data(X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
        """
        Chia dữ liệu thành 3 tập: train, validation, và test.

        """
        # Chia tập train và tập tạm (temp)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, train_size=train_size, random_state=random_state
        )
        
        # Chia tập tạm thành validation và test
        val_ratio = val_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, train_size=val_ratio, random_state=random_state
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    # 📌 Huấn luyện mô hình
    def train_model(model_name, X_train, X_val, X_test, y_train, y_val, y_test):
        if model_name == "Decision Tree":
            model = DecisionTreeClassifier(
                    max_depth=15,           
                    min_samples_split=5,    
                    min_samples_leaf=2,     
                    random_state=42
                
            )
        elif model_name == "SVM":
            model = SVC(kernel="linear", probability=True)
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
        
        # Lưu mô hình vào MLFlow
        with mlflow.start_run(run_name="MNIST_Classification"):
            mlflow.log_param("model_name", model_name)
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("val_accuracy", val_accuracy)
            mlflow.log_metric("test_accuracy", test_accuracy)
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

        
    # 📌 So sánh các số có tính tương đồng
    def show_similar_digits(X, y):
        st.write("**🔍 So sánh các số tương đồng và dễ gây nhầm lẫn**")
        # Cho phép người dùng chọn số tham chiếu
        reference_number = st.selectbox("Chọn số tham chiếu (0-9):", options=list(range(10)))
        
        # Lấy tất cả các mẫu của số tham chiếu
        reference_indices = np.where(y == reference_number)[0]
        
        # Chọn ngẫu nhiên một mẫu từ số tham chiếu
        reference_idx = np.random.choice(reference_indices)
        reference_image = X[reference_idx].reshape(1, -1)
        
        # Tính toán cosine similarity giữa ảnh tham chiếu và toàn bộ tập dữ liệu
        similarities = cosine_similarity(reference_image, X)[0]
        
        # Lọc ra các ảnh không phải là số tham chiếu
        non_reference_indices = np.where(y != reference_number)[0]
        non_reference_similarities = similarities[non_reference_indices]
        
        # Lấy chỉ số của 5 ảnh có độ tương đồng cao nhất (không phải số tham chiếu)
        top_similar_indices = non_reference_indices[np.argsort(non_reference_similarities)[-5:][::-1]]
        
        # Hiển thị ảnh tham chiếu và các ảnh tương đồng
        st.write(f"Số tham chiếu: {reference_number}")
        fig, axes = plt.subplots(1, 6, figsize=(15, 3))
        
        # Hiển thị ảnh tham chiếu
        axes[0].imshow(reference_image.reshape(28, 28), cmap='gray')
        axes[0].set_title(f"Tham chiếu: {reference_number}")
        axes[0].axis('off')
        
        # Hiển thị các ảnh tương đồng
        for i, idx in enumerate(top_similar_indices, start=1):
            axes[i].imshow(X[idx].reshape(28, 28), cmap='gray')
            axes[i].set_title(f"Giống: {similarities[idx]:.2f}\nSố: {y[idx]}")
            axes[i].axis('off')
        
        st.pyplot(fig)
        
        # Giải thích sự nhầm lẫn
        st.write(f"**❗Số {reference_number}** có thể bị nhầm lẫn với các số trên do hình dạng tương tự, chẳng hạn như nét cong, độ dày nét, hoặc cấu trúc chung.")


    # Hàm xử lý ảnh theo lựa chọn của người dùng
    def preprocess_image(image, methods):
        processed_image = image.copy()

        if "Gaussian Blur" in methods:
            processed_image = cv2.GaussianBlur(processed_image, (3, 3), 0)

        if "Sobel" in methods:
            sobelx = cv2.Sobel(processed_image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(processed_image, cv2.CV_64F, 0, 1, ksize=3)
            processed_image = np.sqrt(sobelx**2 + sobely**2)
            processed_image = np.uint8(processed_image / processed_image.max() * 255)

        if "Binarization" in methods:
            _, processed_image = cv2.threshold(processed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return processed_image
    def display_mlflow_experiments():
        try:
            # Lấy danh sách các thí nghiệm từ MLflow
            experiments = mlflow.list_experiments()
            
            if experiments:
                st.write("#### Danh sách thí nghiệm")
                experiment_data = []
                for exp in experiments:
                    experiment_data.append({
                        "Experiment ID": exp.experiment_id,
                        "Experiment Name": exp.name,
                        "Artifact Location": exp.artifact_location
                    })
                st.dataframe(pd.DataFrame(experiment_data))
                
                # Chọn thí nghiệm để xem chi tiết
                selected_exp_id = st.selectbox(
                    "🔍 Chọn thí nghiệm để xem chi tiết",
                    options=[exp.experiment_id for exp in experiments]
                )
                
                # Lấy danh sách runs trong thí nghiệm đã chọn
                runs = mlflow.search_runs(selected_exp_id)
                if not runs.empty:
                    st.write("#### Danh sách runs")
                    st.dataframe(runs)
                    
                    # Chọn run để xem chi tiết
                    selected_run_id = st.selectbox(
                        "🔍 Chọn run để xem chi tiết",
                        options=runs["run_id"]
                    )
                    
                    # Hiển thị chi tiết run
                    run = mlflow.get_run(selected_run_id)
                    st.write("##### Thông tin run")
                    st.write(f"**Run ID:** {run.info.run_id}")
                    st.write(f"**Experiment ID:** {run.info.experiment_id}")
                    st.write(f"**Start Time:** {run.info.start_time}")
                    
                    # Hiển thị metrics
                    st.write("##### Metrics")
                    st.json(run.data.metrics)
                    
                    # Hiển thị params
                    st.write("##### Params")
                    st.json(run.data.params)
                    
                    # Hiển thị artifacts
                    artifacts = mlflow.list_artifacts(selected_run_id)
                    if artifacts:
                        st.write("##### Artifacts")
                        for artifact in artifacts:
                            st.write(f"- {artifact.path}")
                    else:
                        st.write("Không có artifacts nào.")
                else:
                    st.warning("Không có runs nào trong thí nghiệm này.")
            else:
                st.warning("Không có thí nghiệm nào được tìm thấy.")
        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi lấy danh sách thí nghiệm: {e}")

    # 📌 Giao diện Streamlit
    def main():
        st.title("🔢 Phân loại chữ số viết tay")
        
        # # Load dữ liệu
        X, y = load_data()
        # Tạo các tab
        tab1, tab2 ,tab3= st.tabs(["📋 Huấn luyện", "🔮 Dự đoán","⚡ Mlflow"])
        with tab1:
            st.write(f"**Số lượng mẫu của bộ dữ liệu MNIST : {X.shape[0]}**")
            # Hiển thị mẫu dữ liệu và phân phối dữ liệu
            show_sample_images(X, y)

            # So sánh các số tương đồng
            show_similar_digits(X, y)

            # Cho phép người dùng chọn 2 số và hiển thị sự tương đồng
            st.write("**🔍 So sánh sự tương đồng giữa 2 số**")
            st.write("- Các cặp số có sự tương đồng : 0 và 6, 3 và 8, 5 và 6, 4 và 9 , 3 và 5,... ")
            col1, col2 = st.columns(2)
            with col1:
                num1 = st.selectbox("Chọn số thứ nhất", options=list(range(10)), index=0)
            with col2:
                num2 = st.selectbox("Chọn số thứ hai", options=list(range(10)), index=1)

            if st.button("🔍 Tính toán sự tương đồng"):
                # Lấy các mẫu đầu tiên của hai số được chọn
                idx1 = np.where(y == num1)[0][0]
                idx2 = np.where(y == num2)[0][0]

                # Lấy ảnh của hai số
                image1 = X[idx1].reshape(1, -1)
                image2 = X[idx2].reshape(1, -1)

                # Tính toán cosine similarity
                similarity = cosine_similarity(image1, image2)[0][0]

                # Hiển thị kết quả
                st.write(f"🔢 Sự tương đồng giữa số {num1} và số {num2}: **{similarity:.4f}**")

                # Hiển thị ảnh của hai số
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(image1.reshape(28, 28), cmap='gray')
                axes[0].set_title(f"Số {num1}")
                axes[0].axis('off')
                axes[1].imshow(image2.reshape(28, 28), cmap='gray')
                axes[1].set_title(f"Số {num2}")
                axes[1].axis('off')
                st.pyplot(fig)
            
                    # Thêm phần chọn kỹ thuật xử lý ảnh
            st.write("**🖼️ Xử lý ảnh**")
            st.markdown("""
    - **Sobel Operator** giúp làm nổi bật các cạnh trong ảnh bằng cách phát hiện sự thay đổi cường độ pixel.  
    - **Gaussian Blur** làm mịn ảnh và giảm nhiễu không mong muốn, giúp cải thiện chất lượng hình ảnh trước khi xử lý tiếp theo.  
    """)
            
            # Chọn kỹ thuật xử lý ảnh
            selected_preprocessing = st.multiselect(
            "Chọn kỹ thuật xử lý dữ liệu:",
            options=["Sobel", "Gaussian Blur"],
            default=["Sobel", "Gaussian Blur"]
    )
            
            # Chọn một ảnh ngẫu nhiên từ tập dữ liệu
            random_idx = np.random.choice(len(X))
            original_image = X[random_idx].reshape(28, 28)
            
            processed_image = preprocess_image(original_image, selected_preprocessing) if selected_preprocessing else original_image

            col1, col2 = st.columns(2)

        
            with col1:
                st.image(original_image, width=50, caption=f"Số {y[random_idx]}", use_container_width=True, clamp=True)

            with col2:
                st.image(processed_image, width=50, caption="Ảnh sau xử lý", use_container_width=True, clamp=True)
            
            
                # Chọn tỷ lệ dữ liệu huấn luyện, validation, và test
            train_size = st.slider("Tỷ lệ huấn luyện (%)", min_value=50, max_value=90, value=70, step=5)
            val_size = st.slider("Tỷ lệ validation (%)", min_value=5, max_value=30, value=15, step=5)
            test_size = 100 - train_size - val_size
            
            # Kiểm tra tỷ lệ hợp lệ
            if test_size <= 0:
                st.error("Tỷ lệ không hợp lệ! Vui lòng điều chỉnh lại.")
                return
            
            # Load dữ liệu
            X, y = load_data()
            # Lưu tỷ lệ dữ liệu vào session state
            st.session_state.train_size = train_size
            st.session_state.val_size = val_size
            st.session_state.test_size = test_size
            # Chia dữ liệu
            X_train, X_val, X_test, y_train, y_val, y_test = split_data(
                X, y, train_size=train_size/100, val_size=val_size/100, test_size=test_size/100)

            data_ratios = pd.DataFrame({
            "Tập dữ liệu": ["Train", "Validation", "Test"],
            "Tỷ lệ (%)": [st.session_state.train_size, st.session_state.val_size, st.session_state.test_size]
            })

            # Hiển thị bảng
            st.write("**📊 Tỷ lệ dữ liệu**")
            st.table(data_ratios)

            # Hiển thị số lượng mẫu
            st.write(f"🧮 Số lượng mẫu train: {len(X_train)}")
            st.write(f"🧮 Số lượng mẫu validation: {len(X_val)}")
            st.write(f"🧮 Số lượng mẫu test: {len(X_test)}")



            st.write("**🚀 Huấn luyện mô hình**")
            model_name = st.selectbox("🔍 Chọn mô hình", ["Decision Tree", "SVM"])
            # Huấn luyện mô hình
            if st.button("🚀 Huấn luyện mô hình"):
                with st.spinner("🔄 Đang huấn luyện..."):
                    model, train_accuracy, val_accuracy, test_accuracy = train_model(
                    model_name, X_train, X_val, X_test, y_train, y_val, y_test
                )
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
                        model, _ = train_model(model_name, X_train, X_val, X_test, y_train, y_val, y_test)
                        prediction = model.predict(processed_image)[0]
                        probabilities = model.predict_proba(processed_image)[0]

                        st.write(f"🎯 **Dự đoán: {prediction}**")
                        prob_df = pd.DataFrame({"Chữ số": list(range(10)), "Xác suất (%)": probabilities * 100})
                        st.bar_chart(prob_df.set_index("Chữ số"))

                        # 📌 Hiển thị các số có thể bị nhầm lẫn
                        top_similar_numbers = get_top_similar_numbers(probabilities, top_n=3)
                        st.write("🤔 **Số có thể bị nhầm lẫn:**")
                        for num, prob in top_similar_numbers:
                            st.write(f"🔹 {num} ({prob*100:.2f}%)")

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

                        model, _ = train_model(model_name, X_train, X_val, y_train, y_val)
                        prediction = model.predict(processed_canvas)[0]
                        probabilities = model.predict_proba(processed_canvas)[0]

                        st.write(f"🎯 **Dự đoán: {prediction}**")
                        prob_df = pd.DataFrame({"Chữ số": list(range(10)), "Xác suất (%)": probabilities * 100})
                        st.bar_chart(prob_df.set_index("Chữ số"))

                        # 📌 Hiển thị các số có thể bị nhầm lẫn
                        top_similar_numbers = get_top_similar_numbers(probabilities, top_n=3)
                        st.write("🤔 **Số có thể bị nhầm lẫn:**")
                        for num, prob in top_similar_numbers:
                            st.write(f"🔹 {num} ({prob*100:.2f}%)")

        with tab3:
            st.write("### 📊 Tracking MLflow")
            
            try:
                # Lấy danh sách thí nghiệm từ MLflow
                experiments = mlflow.search_experiments()
                
                if experiments:
                    st.write("##### Danh sách thí nghiệm")
                    experiment_data = []
                    for exp in experiments:
                        experiment_data.append({
                            "Experiment ID": exp.experiment_id,
                            "Experiment Name": exp.name,
                            "Artifact Location": exp.artifact_location
                        })
                    st.dataframe(pd.DataFrame(experiment_data))
                    
                    # Chọn thí nghiệm để xem chi tiết
                    selected_exp_id = st.selectbox(
                        "🔍 Chọn thí nghiệm để xem chi tiết",
                        options=[exp.experiment_id for exp in experiments]
                    )
                    
                    # Lấy danh sách runs trong thí nghiệm đã chọn
                    runs = mlflow.search_runs(selected_exp_id)
                    if not runs.empty:
                        st.write("##### Danh sách runs")
                        st.dataframe(runs)
                        
                        # Chọn run để xem chi tiết
                        selected_run_id = st.selectbox(
                            "🔍 Chọn run để xem chi tiết",
                            options=runs["run_id"]
                        )
                        
                        # Hiển thị chi tiết run
                        run = mlflow.get_run(selected_run_id)
                        # st.write("##### Thông tin run")
                        # st.write(f"**Run ID:** {run.info.run_id}")
                        # st.write(f"**Experiment ID:** {run.info.experiment_id}")
                        # st.write(f"**Start Time:** {run.info.start_time}")
                        
                        # Hiển thị metrics
                        st.write("##### Metrics")
                        st.json(run.data.metrics)
                        
                        # Hiển thị params
                        st.write("##### Params")
                        st.json(run.data.params)
                        
                        # Hiển thị artifacts
                        artifacts = mlflow.list_artifacts(selected_run_id)
                        if artifacts:
                            st.write("##### Artifacts")
                            for artifact in artifacts:
                                st.write(f"- {artifact.path}")
                    else:
                        st.warning("Không có runs nào trong thí nghiệm này.")
                else:
                    st.warning("Không có thí nghiệm nào được tìm thấy.")
            except Exception as e:
                st.error(f"Đã xảy ra lỗi khi lấy danh sách thí nghiệm: {e}")


elif option == "Phân tích Titanic":
    st.write("🚢 **Bạn đã chọn dự án: Phân tích dữ liệu Titanic!**")
    # Thêm code phân tích dữ liệu Titanic tại đây
if __name__ == "__main__":
    main()
