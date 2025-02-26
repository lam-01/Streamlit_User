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

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Tiêu đề ứng dụng
st.title("Phân cụm dữ liệu MNIST với K-means và DBSCAN")
st.write("Ứng dụng này thực hiện phân cụm trên tập dữ liệu chữ số viết tay MNIST")

# Tạo các tab
tab1, tab2, tab3 = st.tabs(["Tiền xử lý dữ liệu", "Phân cụm và Đánh giá", "MLFlow"])

# Tab 1:  Tiền xử lý
with tab1:
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

    
    # Tùy chọn PCA
    st.write("##### Giảm chiều dữ liệu PCA")
    use_pca = st.checkbox("Sử dụng PCA để giảm chiều", True, key="use_pca_tab1")
    if use_pca:
        n_components = st.slider("Số lượng thành phần PCA", 2, 50, 20, key="n_components_tab1")
        
        # Áp dụng PCA
        def apply_pca(X, n_components):
            logger.info(f"Áp dụng PCA với {n_components} thành phần")
            
            # Chuẩn hóa dữ liệu
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Áp dụng PCA
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)
            
            # Tính tỷ lệ phương sai được giải thích
            variance_ratio = np.sum(pca.explained_variance_ratio_)
            
            logger.info(f"PCA hoàn thành. Giải thích {variance_ratio:.2f} phương sai")
            st.text(f"PCA giảm chiều từ {X.shape[1]} xuống {n_components} thành phần")
            st.text(f"Tỷ lệ phương sai : {variance_ratio:.2f}")
            
            return X_pca, pca
        
        X_processed, pca_model = apply_pca(X, n_components)
    else:
        X_processed = X

# Tab 2: Phân cụm và Đánh giá
with tab2:
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
                
                return clusters, kmeans
            
            clusters, model = run_kmeans(X_processed, n_clusters, max_iter)
        
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
            
            clusters, model = run_dbscan(X_processed, eps, min_samples)
        
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
        evaluation_results = evaluate_clustering(X_processed, clusters)
        
        # Hiển thị kết quả đánh giá
        st.markdown("##### Kết quả đánh giá phân cụm",help="""**Calinski-Harabasz Index (CH Index)**, còn được gọi là **Variance Ratio Criterion**, là một chỉ số đánh giá chất lượng của các cụm trong phân cụm.
        Nó đo lường sự phân tách giữa các cụm và sự đồng nhất bên trong các cụm .
        \n- Giá trị của CH Index càng cao thì chất lượng phân cụm càng tốt. Điều này có nghĩa là các cụm được phân tách rõ ràng và các điểm trong cùng một cụm gần nhau hơn.
        \n- Nếu CH Index thấp, điều này có thể chỉ ra rằng các cụm không được phân tách tốt hoặc có thể có quá ít cụm.
        \n **Silhouette Score** là một chỉ số đánh giá chất lượng của các cụm trong phân cụm. Nó đo lường mức độ tương đồng của 
        một điểm với các điểm trong cùng một cụm so với các điểm trong cụm khác.
        \n- Gần 1: Điểm nằm gần các điểm trong cùng một cụm và xa các điểm trong cụm khác, cho thấy phân cụm tốt.
        \n- Gần 0: Điểm nằm ở ranh giới giữa hai cụm, cho thấy phân cụm không rõ ràng.
        \n- Gần -1: Điểm có thể đã được phân cụm sai, nằm gần các điểm trong cụm khác hơn là trong cụm của nó.
        """)
        if evaluation_results and isinstance(evaluation_results, dict):
            for metric, value in evaluation_results.items():
                st.write(f"{metric}: {value:.4f}")
        else:
            st.warning("Không có kết quả đánh giá nào được tính toán.")
        
        # Trực quan hóa kết quả
        def visualize_clusters(X, clusters, y_true=None, algorithm_name=""):
            st.write(f"##### Kết quả phân cụm {algorithm_name}")
            
            # Nếu dữ liệu có nhiều hơn 2 chiều, sử dụng PCA để trực quan hóa
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
        
        # Hiển thị kết quả
        visualize_clusters(X_processed, clusters, y, algorithm)
        
        # Hiển thị một số ảnh từ mỗi cụm
        def display_cluster_examples(X, clusters, n_clusters=10, n_samples=5):
            st.subheader("Hiển thị một số ảnh từ mỗi cụm")
            
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
        
        # Hiển thị ảnh từ mỗi cụm
        display_cluster_examples(X, clusters)

# Tab 3: Theo dõi với MLFlow
with tab3:
    st.write("")
