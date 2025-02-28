import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_openml
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Tiêu đề ứng dụng
st.title("Phân cụm dữ liệu MNIST với K-means và DBSCAN")
st.write("Ứng dụng này thực hiện phân cụm trên tập dữ liệu chữ số viết tay MNIST")

# Tạo các tab
tab1, tab2, tab3 = st.tabs(["Tổng quan lý thuyết", "Phân cụm ", "MLFlow"])

# Tab 1:  Tiền xử lý
with tab1:
    st.write("##### Lí thuyết")


# Tab 2: Phân cụm và Đánh giá
with tab2:
    st.write("##### Tùy chọn số lượng dữ liệu ")
    
    # Tùy chọn số lượng dữ liệu
    sample_size = st.slider("Số lượng mẫu", 1000, 70000, 7000, key="sample_size_tab1")
    
    # Tải dữ liệu MNIST
    @st.cache_data
    def load_mnist(sample_size):
        logger.info(f"Đang tải dữ liệu MNIST với kích thước mẫu {sample_size}")
        
        # Tải dữ liệu từ OpenML
        mnist = fetch_openml('mnist_784', version=1, parser='auto')
        X = mnist.data.astype('float32')
        y = mnist.target.astype('int')
        
        # Lấy mẫu ngẫu nhiên
        if sample_size < X.shape[0]:
            indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sampled = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
            y_sampled = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
        else:
            X_sampled = X
            y_sampled = y
        
        logger.info(f"Đã tải xong dữ liệu MNIST: {X_sampled.shape}")
        st.text(f"Số lượng mẫu : {X_sampled.shape[0]} mẫu với {X_sampled.shape[1]} chiều")
        
        return X_sampled, y_sampled
    
    # Tải dữ liệu
    X, y = load_mnist(sample_size)
    
    # Hiển thị một số ảnh từ tập dữ liệu
    def display_random_images(X, n_samples=10):
        st.write("##### Hiển thị một số ảnh từ tập dữ liệu")
        
        # Tạo lưới để hiển thị ảnh
        fig, axes = plt.subplots(1, n_samples, figsize=(15, 2))
        
        # Chọn ngẫu nhiên các chỉ số
        indices = np.random.choice(X.shape[0], n_samples, replace=False)
        
        # Hiển thị mỗi ảnh
        for i, idx in enumerate(indices):
            img = X.iloc[idx].values.reshape(28, 28)
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
        
        st.pyplot(fig)
    
    # Hiển thị ảnh
    display_random_images(X)
    st.write("##### Phân cụm và Đánh giá")
    
    # Tùy chọn thuật toán
    algorithm = st.selectbox("Thuật toán phân cụm", ["K-means", "DBSCAN"], key="algorithm_tab2")
    
    if algorithm == "K-means":
        n_clusters = st.slider("Số lượng cụm (k)", 2, 20, 10, key="n_clusters_tab2")
        max_iter = st.slider("Số lần lặp tối đa", 100, 1000, 300, key="max_iter_tab2")
    elif algorithm == "DBSCAN":
        eps = st.slider("Epsilon (bán kính vùng lân cận)", 0.1, 20.0, 5.0, key="eps_tab2")
        min_samples = st.slider("Số lượng mẫu tối thiểu", 2, 100, 5, key="min_samples_tab2")
    
    # Nút "Phân cụm"
    if st.button("Phân cụm"):
        if algorithm == "K-means":
            # Thực hiện K-means
            def run_kmeans(X, n_clusters, max_iter):
                logger.info(f"Thực hiện K-means với {n_clusters} cụm")
                st.text(f"Số cụm : {n_clusters} cụm")
                
                start_time = time.time()
                
                # Khởi tạo và thực hiện K-means
                kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X)
                
                elapsed_time = time.time() - start_time
                logger.info(f"K-means hoàn thành trong {elapsed_time:.2f} giây")
                
                return clusters, kmeans
            
            clusters, model = run_kmeans(X, n_clusters, max_iter)
        
        elif algorithm == "DBSCAN":
            # Thực hiện DBSCAN
            def run_dbscan(X, eps, min_samples):
                logger.info(f"Thực hiện DBSCAN với eps={eps}, min_samples={min_samples}")
                st.text(f"eps={eps}, min_samples={min_samples}")
                
                start_time = time.time()
                
                # Khởi tạo và thực hiện DBSCAN
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                clusters = dbscan.fit_predict(X)
                
                elapsed_time = time.time() - start_time
                
                # Xác định số lượng cụm (không tính điểm nhiễu -1)
                n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
                n_noise = list(clusters).count(-1)
                
                logger.info(f"DBSCAN hoàn thành trong {elapsed_time:.2f} giây. Tìm thấy {n_clusters} cụm và {n_noise} điểm nhiễu")
                st.text(f"Tìm thấy {n_clusters} cụm và {n_noise} điểm nhiễu")
                
                return clusters, dbscan
            
            clusters, model = run_dbscan(X, eps, min_samples)
        
        # Đánh giá kết quả phân cụm
        def evaluate_clustering(X, clusters):
            results = {}
            
            # Kiểm tra nếu có điểm nhiễu
            if -1 in clusters:
                st.warning("DBSCAN tìm thấy điểm nhiễu. Loại bỏ điểm nhiễu để tính toán chỉ số đánh giá.")
                
                # Loại bỏ điểm nhiễu
                valid_indices = clusters != -1
                X_valid = X[valid_indices]
                clusters_valid = clusters[valid_indices]
                
                # Kiểm tra số lượng cụm hợp lệ
                unique_clusters = set(clusters_valid)
                if len(unique_clusters) > 1:
                    # Silhouette Score
                    results["Silhouette Score"] = silhouette_score(X_valid, clusters_valid)
                    
                    # Calinski-Harabasz Index
                    results["Calinski-Harabasz Index"] = calinski_harabasz_score(X_valid, clusters_valid)
                else:
                    st.warning("Không thể tính chỉ số đánh giá do chỉ có một cụm sau khi loại bỏ điểm nhiễu.")
            else:
                # Nếu không có điểm nhiễu, tính toán chỉ số bình thường
                unique_clusters = set(clusters)
                if len(unique_clusters) > 1:
                    # Silhouette Score
                    results["Silhouette Score"] = silhouette_score(X, clusters)
                    
                    # Calinski-Harabasz Index
                    results["Calinski-Harabasz Index"] = calinski_harabasz_score(X, clusters)
                else:
                    st.warning("Không thể tính chỉ số đánh giá do chỉ có một cụm.")
            
            return results
        
        # Tính toán kết quả đánh giá
        evaluation_results = evaluate_clustering(X, clusters)
        
        # Hiển thị kết quả đánh giá
        st.markdown("##### Kết quả đánh giá phân cụm")
        if evaluation_results and isinstance(evaluation_results, dict):
            for metric, value in evaluation_results.items():
                st.write(f"{metric}: {value:.4f}")
        else:
            st.warning("Không có kết quả đánh giá nào được tính toán.")
        # Hàm trực quan hóa kết quả phân cụm
        def visualize_clusters(X, clusters, y_true=None, algorithm_name=""):
            st.write(f"##### Kết quả phân cụm {algorithm_name}")
            
            # Sử dụng PCA để giảm chiều dữ liệu
            if X.shape[1] > 2:
                st.text("Sử dụng PCA để hiển thị trong không gian 2D")
                pca = PCA(n_components=2)
                X_2d = pca.fit_transform(X)
            else:
                X_2d = X
            
            # Hiển thị biểu đồ phân cụm
            fig, ax = plt.subplots(1, 2, figsize=(16, 6))
            
            # Vẽ phân cụm theo nhóm
            scatter = ax[0].scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, cmap='viridis', alpha=0.5)
            ax[0].set_title(f'Phân cụm {algorithm_name}')
            
            # Thêm legend nếu DBSCAN (để hiển thị điểm nhiễu)
            if algorithm_name == "DBSCAN" and -1 in clusters:
                unique_clusters = np.unique(clusters)
                if len(unique_clusters) <= 20:  # Giới hạn số lượng nhãn hiển thị
                    legend_labels = [f'Cụm {i}' for i in unique_clusters]
                    legend_labels[0] = 'Nhiễu' if unique_clusters[0] == -1 else legend_labels[0]
                    ax[0].legend(handles=scatter.legend_elements()[0], labels=legend_labels, loc="upper right")
            
            # Vẽ phân cụm theo số thật (nếu có)
            if y_true is not None:
                ax[1].scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap='tab10', alpha=0.5)
                ax[1].set_title('Phân loại thực tế (digit)')
            else:
                ax[1].axis('off')
            
            st.pyplot(fig)

        # Hàm hiển thị thông tin chi tiết về từng cụm dưới dạng bảng
        def display_cluster_info(X, clusters):
            st.subheader("Thông tin chi tiết về từng cụm")
            
            # Lấy các cụm duy nhất
            unique_clusters = np.unique(clusters)
            
            # Tạo một danh sách để lưu thông tin từng cụm
            cluster_info = []
            
            for cluster_id in unique_clusters:
                cluster_name = f"Cụm {cluster_id}" if cluster_id != -1 else "Điểm nhiễu (cụm -1)"
                
                # Lấy các chỉ số của các điểm trong cụm này
                indices = np.where(clusters == cluster_id)[0]
                
                # Tính số lượng điểm dữ liệu trong cụm
                n_samples = len(indices)
                
                # Thêm thông tin vào danh sách
                cluster_info.append({
                    "Tên cụm": cluster_name,
                    "Số lượng điểm dữ liệu": n_samples
                })
            
            # Tạo DataFrame từ danh sách thông tin
            cluster_df = pd.DataFrame(cluster_info)
            
            # Hiển thị bảng thông tin
            st.dataframe(cluster_df)

        # Hàm hiển thị một số ảnh từ mỗi cụm
        def display_cluster_examples(X, clusters, n_clusters=10, n_samples=5):
            st.subheader("Hiển thị một số ảnh từ mỗi cụm")
            
            # Tùy chọn số lượng cụm hiển thị
            n_clusters = st.slider("Chọn số lượng cụm hiển thị", 1, 20, 10, key="n_clusters_display")
            
            # Tùy chọn số lượng ảnh hiển thị từ mỗi cụm
            n_samples = st.slider("Chọn số lượng ảnh hiển thị từ mỗi cụm", 1, 10, 5, key="n_samples_display")
            
            # Lấy các cụm duy nhất
            unique_clusters = np.unique(clusters)
            n_unique = min(len(unique_clusters), n_clusters)
            
            # Chỉ hiển thị n_clusters đầu tiên
            display_clusters = unique_clusters[:n_clusters]
            
            # Đối với mỗi cụm, hiển thị một số ảnh mẫu
            for cluster_id in display_clusters:
                cluster_name = f"Cụm {cluster_id}" if cluster_id != -1 else "Điểm nhiễu (cụm -1)"
                st.write(f"### {cluster_name}")
                
                # Lấy các chỉ số của các điểm trong cụm này
                indices = np.where(clusters == cluster_id)[0]
                
                # Nếu không có đủ mẫu trong cụm, hiển thị tất cả
                n_to_display = min(len(indices), n_samples)
                
                if n_to_display > 0:
                    # Chọn ngẫu nhiên các mẫu để hiển thị
                    display_indices = np.random.choice(indices, n_to_display, replace=False)
                    
                    # Tạo lưới để hiển thị ảnh
                    fig, axes = plt.subplots(1, n_to_display, figsize=(15, 2))
                    
                    # Trường hợp chỉ có 1 ảnh
                    if n_to_display == 1:
                        img = X[display_indices[0]].reshape(28, 28)
                        axes.imshow(img, cmap='gray')
                        axes.axis('off')
                    else:
                        # Hiển thị mỗi ảnh
                        for i, idx in enumerate(display_indices):
                            img = X.iloc[idx].values.reshape(28, 28)
                            axes[i].imshow(img, cmap='gray')
                            axes[i].axis('off')
                    
                    st.pyplot(fig)
                else:
                    st.write("Không có mẫu nào trong cụm này")

        # Gọi các hàm để hiển thị kết quả
        visualize_clusters(X, clusters, y, algorithm)
        display_cluster_info(X, clusters)
        display_cluster_examples(X, clusters)

        st.write("##### Lưu kết quả phân cụm vào MLFlow")
        user_name = st.text_input("Nhập tên của bạn để lưu kết quả phân cụm", key="user_name_tab2")
        
        if user_name:
            if st.button("Lưu kết quả phân cụm vào MLFlow"):
                if 'model' in locals() and 'clusters' in locals():
                    with mlflow.start_run(run_name=f"Clustering_{user_name}"):
                        # Log phương pháp phân cụm
                        mlflow.log_param("Algorithm", algorithm)
                        
                        # Log thông tin cụ thể về phương pháp phân cụm
                        if algorithm == "K-means":
                            mlflow.log_param("n_clusters", n_clusters)
                            mlflow.log_param("max_iter", max_iter)
                        elif algorithm == "DBSCAN":
                            mlflow.log_param("eps", eps)
                            mlflow.log_param("min_samples", min_samples)
                            mlflow.log_param("n_clusters", len(set(clusters)) - (1 if -1 in clusters else 0))
                            mlflow.log_param("n_noise", list(clusters).count(-1))
                        
                        # Log chỉ số đánh giá
                        if evaluation_results:
                            for metric, value in evaluation_results.items():
                                mlflow.log_metric(metric, value)
                        
                        # Log mô hình
                        mlflow.sklearn.log_model(model, "model")
                        
                        st.success(f"Kết quả phân cụm đã được lưu vào MLFlow với tên {user_name}.")
                else:
                    st.warning("Vui lòng thực hiện phân cụm trước khi lưu kết quả.")
            

with tab3:
    st.header("📊 MLflow Tracking")

    # # Kết nối đến MLflow
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Đảm bảo MLflow server đang chạy

    # Tìm kiếm mô hình theo tên
    search_name = st.text_input("🔍 Nhập tên mô hình để tìm kiếm:", "")

    # Lấy danh sách các phiên làm việc từ MLflow
    if search_name:
        runs = mlflow.search_runs(filter_string=f"tags.mlflow.runName LIKE '%{search_name}%'", order_by=["start_time desc"])
    else:
        runs = mlflow.search_runs(order_by=["start_time desc"])

    if not runs.empty:
        # Hiển thị danh sách các mô hình
        st.write("### 📜 Danh sách mô hình đã lưu:")
        st.dataframe(runs[["tags.mlflow.runName", "run_id"]])

        # Chọn một mô hình để xem chi tiết
        selected_run_id = st.selectbox("📝 Chọn một mô hình để xem chi tiết:", runs["run_id"].tolist())

        if selected_run_id:
            # Lấy thông tin chi tiết về run được chọn
            run_details = mlflow.get_run(selected_run_id)
            st.write(f"### 🔍 Chi tiết mô hình: `{run_details.data.tags.get('mlflow.runName', 'Không có tên')}`")
            st.write("**🟢 Trạng thái:**", run_details.info.status)
            st.write("**⏳ Thời gian bắt đầu:**", run_details.info.start_time)
            st.write("**🏁 Thời gian kết thúc:**", run_details.info.end_time)

            # Hiển thị tham số
            st.write("📌 **Tham số:**")
            for key, value in run_details.data.params.items():
                st.write(f"- **{key}**: {value}")

            # Hiển thị metric
            st.write("📊 **Metric:**")
            for key, value in run_details.data.metrics.items():
                st.write(f"- **{key}**: {value}")

            # Hiển thị artifacts (nếu có)
            st.write("📂 **Artifacts:**")
            if run_details.info.artifact_uri:
                st.write(f"- **Artifact URI**: {run_details.info.artifact_uri}")
                # Tải mô hình từ artifact
                if st.button("Tải mô hình", key=f"load_{selected_run_id}"):
                    model = mlflow.sklearn.load_model(f"runs:/{selected_run_id}/model")
                    st.success(f"Đã tải mô hình {run_details.data.tags.get('mlflow.runName', 'Không có tên')} thành công!")
                    st.write(f"Thông tin mô hình: {model}")
            else:
                st.write("- Không có artifacts nào.")

    else:
        st.warning("⚠️ Không tìm thấy mô hình nào.")
