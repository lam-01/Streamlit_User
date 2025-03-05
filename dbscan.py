import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from openml import datasets
import datetime
import os
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
from sklearn.datasets import fetch_openml

# Set up MLflow
def setup_mlflow():
    mlflow_tracking_uri = "./mlruns"
    if not os.path.exists(mlflow_tracking_uri):
        os.makedirs(mlflow_tracking_uri)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    return MlflowClient()

# Tải bộ dữ liệu MNIST từ OpenML
@st.cache_data
def load_mnist_data():
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X = mnist.data.astype('float32').values
    y = mnist.target.astype('int')
    return X, y

# Preprocess data
@st.cache_data
def preprocess_data(X, sample_size=5000):
    np.random.seed(42)
    indices = np.random.choice(X.shape[0], sample_size, replace=False)
    X_sample = X[indices]
    X_original = X_sample.copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    return X_sample, X_scaled, X_pca, indices, X_original, pca 

# Thực hiện phân cụm K-Means
def run_kmeans(X_scaled, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    return kmeans

# Thực hiện phân cụm DBSCAN
def run_dbscan(X_scaled, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X_scaled)
    return dbscan

# Hàm log mô hình vào MLFlow
def log_model(model, model_name, params, metrics, cluster_images, experiment_name="MNIST_Clustering_Experiment"):
    mlflow.set_experiment(experiment_name)  # Thí nghiệm duy nhất
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_param("model_name", model_name)
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        mlflow.sklearn.log_model(model, "model")
        for cluster_id, images in cluster_images.items():
            if images is not None and len(images) > 0:
                for i, img_data in enumerate(images[:5]):
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.imshow(img_data.reshape(28, 28), cmap='gray')
                    plt.axis('off')
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    mlflow.log_image(Image.open(buf), f"cluster_{cluster_id}_sample_{i}.png")
                    plt.close(fig)
        return run.info.run_id

# Trực quan hóa kết quả phân cụm
def visualize_clusters(X_pca, labels, model_type, centroids=None):
    fig = px.scatter(
        x=X_pca[:, 0], 
        y=X_pca[:, 1], 
        color=[str(label) for label in labels],
        title=f"{model_type} Clustering Results (PCA Visualization)",
        labels={"x": "PCA Component 1", "y": "PCA Component 2"},
        color_discrete_sequence=px.colors.qualitative.G10
    )
    if centroids is not None and model_type == 'K-means':
        pca = PCA(n_components=2)
        pca.fit(X_pca)
        centroids_pca = pca.transform(centroids)
        fig.add_trace(
            go.Scatter(
                x=centroids_pca[:, 0],
                y=centroids_pca[:, 1],
                mode='markers',
                marker=dict(symbol='x', color='black', size=10, line=dict(width=2)),
                name='Centroids'
            )
        )
    return fig

# Nhận các ví dụ về chữ số bằng cụm
def get_digit_examples_by_cluster(X_original, cluster_labels):
    examples_by_cluster = {}
    unique_labels = np.unique(cluster_labels)
    for label in unique_labels:
        if label == -1:
            continue
        cluster_indices = np.where(cluster_labels == label)[0]
        cluster_samples = X_original[cluster_indices]
        examples_by_cluster[label] = cluster_samples
    return examples_by_cluster

# Danh sách thí nghiệm
def list_experiments(client):
    return client.search_experiments()

# Liệt kê tất cả các lần chạy cho một thử nghiệm
def list_runs(client, experiment_id):
    return client.search_runs(experiment_id)

# Tìm kiếm các mô hình theo tên
def search_models(client, query, experiment_id):
    runs = client.search_runs(
        experiment_id,
        filter_string=f"params.model_name LIKE '%{query}%'"
    )
    return runs

# Nhận chi tiết mô hình
def get_model_details(client, run_id):
    run = client.get_run(run_id)
    return run

# Ứng dụng Streamlit chính
def main():
    st.title("MNIST Clustering ")
    
    client = setup_mlflow()
    
    tab1, tab2, tab3 = st.tabs(["Tổng quan ", "Phân cụm ", "MLFlow"])

    with tab1:
        try:
            X, y = load_mnist_data()
            st.subheader("🔹 Một vài mẫu dữ liệu từ MNIST")
            if len(X) == 0 or len(y) == 0:
                st.error("Dữ liệu MNIST trống. Vui lòng kiểm tra lại hàm tải dữ liệu.")
            else:
                cols = st.columns(5)
                for i, col in enumerate(cols):
                    idx = np.random.randint(0, len(X))
                    with col:
                        fig, ax = plt.subplots(figsize=(3, 3))
                        ax.imshow(X[idx].reshape(28, 28), cmap='gray')
                        ax.set_title(f"Digit: {y[idx]}")
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close(fig)
        except Exception as e:
            st.error(f"Error loading MNIST data: {e}")
            st.error(f"Chi tiết lỗi: {str(e)}")

    with tab2:
        st.header("Run Clustering Algorithms")
        
        try:
            X, y = load_mnist_data()
            st.success(f"Bộ dữ liệu MNIST được tải thành công với {X.shape[0]} mẫu.")
            
            sample_size = st.slider("Chọn cỡ mẫu để phân cụm", 
                                    min_value=1000, 
                                    max_value=10000, 
                                    value=5000, 
                                    step=1000)
            
            X_sample, X_scaled, X_pca, indices, X_original, pca = preprocess_data(X, sample_size)
            st.success(f"Số lượng mẫu: {sample_size} mẫu.")
            
            # Nhập tên mô hình (model_name) thay vì tên experiment
            model_name_input = st.text_input("Nhập tên mô hình:", "My_Model")
            if not model_name_input:
                model_name_input = "My_Model"
            st.write(f"Tên mô hình hiện tại: {model_name_input}")
            
            selected_tab = st.selectbox("Chọn thuật toán phân cụm", ["K-means", "DBSCAN"])

            if selected_tab == "K-means":
                st.subheader("K-means Clustering")
                n_clusters = st.slider("Số cụm (k)", min_value=5, max_value=20, value=10)
                
                if st.button("Run K-means"):
                    with st.spinner("Chạy phân cụm K-Means ..."):
                        kmeans_model = run_kmeans(X_scaled, n_clusters)
                        kmeans_labels = kmeans_model.labels_
                        
                        if len(np.unique(kmeans_labels)) > 1:
                            silhouette = silhouette_score(X_scaled, kmeans_labels)
                            calinski = calinski_harabasz_score(X_scaled, kmeans_labels)
                        else:
                            silhouette = 0
                            calinski = 0
                        st.markdown("Các số liệu phân cụm", help="""**Silhouette Score** đo lường mức độ tương đồng của một điểm với các điểm trong cùng một cụm so với các điểm trong cụm khác.
                        \n- Giá trị của Silhouette Score nằm trong khoảng từ -1 đến 1:
                        \n +Gần 1: Điểm nằm gần các điểm trong cùng một cụm và xa các điểm trong cụm khác, cho thấy phân cụm tốt.
                        \n +Gần 0: Điểm nằm ở ranh giới giữa hai cụm, cho thấy phân cụm không rõ ràng.
                        \n +Gần -1: Điểm có thể đã được phân cụm sai, nằm gần các điểm trong cụm khác hơn là trong cụm của nó.
                        \n
                        \n **Calinski-Harabasz Score** đo lường sự phân tách giữa các cụm và sự đồng nhất bên trong các cụm.
                        \n- Giá trị của Calinski-Harabasz Score càng cao thì chất lượng phân cụm càng tốt.
                        """)
                        st.write(f"Silhouette Score: {silhouette:.4f}")
                        st.write(f"Calinski-Harabasz Score: {calinski:.4f}")
                        
                        digit_examples = get_digit_examples_by_cluster(X_original, kmeans_labels)
                        
                        params = {
                            "algorithm": "KMeans",
                            "n_clusters": n_clusters,
                            "sample_size": sample_size
                        }
                        metrics = {
                            "silhouette_score": silhouette,
                            "calinski_harabasz_score": calinski
                        }
                        
                        # Sử dụng tên mô hình do người dùng nhập
                        run_id = log_model(kmeans_model, model_name_input, params, metrics, digit_examples)
                        st.success(f"Mô hình K-means được lưu vào MLflow với run ID: {run_id}")
                        
                        st.subheader("Các chữ số mẫu từ mỗi cụm")
                        for cluster_idx in range(min(n_clusters)):
                            if cluster_idx in digit_examples:
                                st.write(f"Cluster {cluster_idx}")
                                cols = st.columns(5)
                                for i, col in enumerate(cols):
                                    if i < len(digit_examples[cluster_idx]):
                                        with col:
                                            fig, ax = plt.subplots(figsize=(2, 2))
                                            ax.imshow(digit_examples[cluster_idx][i].reshape(28, 28), cmap='gray')
                                            ax.axis('off')
                                            st.pyplot(fig)
                                            plt.close(fig)

            elif selected_tab == "DBSCAN":
                st.subheader("Phân cụm DBSCAN")
                eps = st.slider("Epsilon", min_value=0.1, max_value=10.0, value=5.0, step=0.1, help="""**Epsilon** : Bán kính để xác định khu vực lân cận của một điểm.
                \n- Nếu một điểm có đủ số lượng hàng xóm (≥ min_samples) trong phạm vi eps, nó sẽ trở thành core point và giúp tạo cụm.
                \n- Giá trị eps càng lớn(6-10), thì cụm càng rộng và số lượng cụm giảm xuống.
                \n- Nếu eps quá nhỏ(0.1-2), thuật toán có thể tạo quá nhiều cụm nhỏ hoặc không tìm thấy cụm nào.
                 """)
                min_samples = st.slider("Min Samples", min_value=2, max_value=50, value=10, help="""**MinPts** : Số lượng điểm tối thiểu cần thiết để một khu vực được coi là đủ mật độ.
                \n- Nếu min_samples nhỏ(2-5), các cụm có thể dễ dàng hình thành, ngay cả với dữ liệu nhiễu.
                \n- Nếu min_samples lớn(>30), thuật toán có thể khó nhận diện cụm nhỏ và có thể đánh dấu nhiều điểm là nhiễu.
                """)
                
                if st.button("Run DBSCAN"):
                    with st.spinner("Chạy phân cụm DBSCAN ..."):
                        dbscan_model = run_dbscan(X_scaled, eps, min_samples)
                        dbscan_labels = dbscan_model.labels_
                        
                        n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
                        if n_clusters > 1:
                            silhouette = silhouette_score(X_scaled, dbscan_labels)
                            calinski = calinski_harabasz_score(X_scaled, dbscan_labels)
                        else:
                            silhouette = 0
                            calinski = 0
                        
                        st.subheader("Kết quả phân cụm")
                        st.write(f"Số lượng cụm được tìm thấy: {n_clusters}")
                        noise_points = np.sum(dbscan_labels == -1)
                        st.write(f"Số điểm nhiễu: {noise_points} ({noise_points / len(dbscan_labels) * 100:.2f}%)")
                        
                        st.markdown("Các số liệu phân cụm",help="""**Silhouette Score** đo lường mức độ tương đồng của một điểm với các điểm trong cùng một cụm so với các điểm trong cụm khác.
                        \n- Giá trị của Silhouette Score nằm trong khoảng từ -1 đến 1:
                        \n +Gần 1: Điểm nằm gần các điểm trong cùng một cụm và xa các điểm trong cụm khác, cho thấy phân cụm tốt.
                        \n +Gần 0: Điểm nằm ở ranh giới giữa hai cụm, cho thấy phân cụm không rõ ràng.
                        \n +Gần -1: Điểm có thể đã được phân cụm sai, nằm gần các điểm trong cụm khác hơn là trong cụm của nó.
                        \n
                        \n **Calinski-Harabasz Score** đo lường sự phân tách giữa các cụm và sự đồng nhất bên trong các cụm.
                        \n- Giá trị của Calinski-Harabasz Score càng cao thì chất lượng phân cụm càng tốt.
                        """)
                        st.write(f"Silhouette Score: {silhouette:.4f}")
                        st.write(f"Calinski-Harabasz Score: {calinski:.4f}")
                        
                        digit_examples = get_digit_examples_by_cluster(X_original, dbscan_labels)
                        
                        params = {
                            "algorithm": "DBSCAN",
                            "eps": eps,
                            "min_samples": min_samples,
                            "sample_size": sample_size
                        }
                        metrics = {
                            "silhouette_score": silhouette,
                            "calinski_harabasz_score": calinski,
                            "n_clusters": n_clusters,
                            "noise_percentage": noise_points / len(dbscan_labels) * 100
                        }
                        
                        # Sử dụng tên mô hình do người dùng nhập
                        run_id = log_model(dbscan_model, model_name_input, params, metrics, digit_examples)
                        
                        st.subheader("Các chữ số mẫu từ mỗi cụm")
                        unique_labels = sorted(set(dbscan_labels))
                        if -1 in unique_labels:
                            unique_labels.remove(-1)
                            
                        for cluster_idx in unique_labels:
                            if cluster_idx in digit_examples:
                                st.write(f"Cụm {cluster_idx}")
                                cols = st.columns(5)
                                for i, col in enumerate(cols):
                                    if i < len(digit_examples[cluster_idx]):
                                        with col:
                                            fig, ax = plt.subplots(figsize=(2, 2))
                                            ax.imshow(digit_examples[cluster_idx][i].reshape(28, 28), cmap='gray')
                                            ax.axis('off')
                                            st.pyplot(fig)
                                            plt.close(fig)
                        
                        if -1 in dbscan_labels:
                            noise_indices = np.where(dbscan_labels == -1)[0]
                            if len(noise_indices) > 0:
                                st.write("Điểm nhiễu mẫu")
                                cols = st.columns(5)
                                for i, col in enumerate(cols):
                                    if i < min(5, len(noise_indices)):
                                        idx = noise_indices[i]
                                        with col:
                                            fig, ax = plt.subplots(figsize=(2, 2))
                                            ax.imshow(X_original[idx].reshape(28, 28), cmap='gray')
                                            ax.axis('off')
                                            st.pyplot(fig)
                                            plt.close(fig)
        
        except Exception as e:
            st.error(f"Error: {e}")
            st.error(f"Error details: {str(e)}")

    with tab3:
        st.header("MLflow Tracking")
        client = setup_mlflow()
        
        experiments = list_experiments(client)
        if not experiments:
            st.warning("Không tìm thấy thí nghiệm nào trong MLflow. Vui lòng chạy một số thuật toán phân cụm trước!")
        else:
            # Chỉ sử dụng thí nghiệm "MNIST_Clustering_Experiment"
            selected_exp = next((exp for exp in experiments if exp.name == "MNIST_Clustering_Experiment"), None)
            if not selected_exp:
                st.warning("Thí nghiệm 'MNIST_Clustering_Experiment' chưa tồn tại. Chạy phân cụm để tạo!")
            else:
                runs = list_runs(client, selected_exp.experiment_id)
                if not runs:
                    st.warning("Không tìm thấy run nào trong thí nghiệm này!")
                else:
                    search_query = st.text_input("Tìm kiếm theo tên mô hình", "")
                    
                    run_data = []
                    for run in runs:
                        model_name = run.data.params.get("model_name", "Unknown")
                        algorithm = run.data.params.get("algorithm", "Unknown")
                        if algorithm == "KMeans":
                            main_param = f"n_clusters={run.data.params.get('n_clusters', 'N/A')}"
                        elif algorithm == "DBSCAN":
                            main_param = f"eps={run.data.params.get('eps', 'N/A')}, min_samples={run.data.params.get('min_samples', 'N/A')}"
                        else:
                            main_param = "N/A"
                        silhouette = run.data.metrics.get("silhouette_score", "N/A")
                        calinski = run.data.metrics.get("calinski_harabasz_score", "N/A")
                        run_data.append({
                            "Model Name": model_name,
                            "Algorithm": algorithm,
                            "Main Parameters": main_param,
                            "Silhouette Score": silhouette if silhouette != "N/A" else "N/A",
                            "Calinski-Harabasz Score": calinski if calinski != "N/A" else "N/A",
                            "Run ID": run.info.run_id,
                            "Start Time": run.info.start_time
                        })
                    
                    run_df = pd.DataFrame(run_data)
                    run_df["Start Time"] = pd.to_datetime(run_df["Start Time"]).dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    if search_query:
                        filtered_df = run_df[run_df["Model Name"].str.contains(search_query, case=False, na=False)]
                    else:
                        filtered_df = run_df
                    
                    st.dataframe(filtered_df.drop("Run ID", axis=1), use_container_width=True)
                    
                    selected_model = st.selectbox(
                        "Chọn một mô hình để xem chi tiết",
                        options=filtered_df["Model Name"].unique()
                    )
                    
                    if selected_model:
                        selected_run_id = filtered_df[filtered_df["Model Name"] == selected_model]["Run ID"].iloc[0]
                        run_details = get_model_details(client, selected_run_id)
                        
                        st.write("**Parameters:**")
                        params_df = pd.DataFrame(list(run_details.data.params.items()), columns=["Parameter", "Value"])
                        st.dataframe(params_df)
                        
                        st.write("**Metrics:**")
                        metrics_df = pd.DataFrame(list(run_details.data.metrics.items()), columns=["Metric", "Value"])
                        st.dataframe(metrics_df)
                        
                        try:
                            model_uri = f"runs:/{selected_run_id}/model"
                            model = mlflow.sklearn.load_model(model_uri)
                            if run_details.data.params.get("algorithm") == "KMeans":
                                st.write("**Cluster Centers Shape:**", model.cluster_centers_.shape)
                                st.write("**Iterations:**", model.n_iter_)
                                st.subheader("Cluster Centers Visualization")
                                cols = st.columns(5)
                                for i, col in enumerate(cols):
                                    if i < min(5, model.cluster_centers_.shape[0]):
                                        with col:
                                            fig, ax = plt.subplots(figsize=(3, 3))
                                            ax.imshow(model.cluster_centers_[i].reshape(28, 28), cmap='gray')
                                            ax.set_title(f"Cluster {i}")
                                            ax.axis('off')
                                            st.pyplot(fig)
                                            plt.close(fig)
                            elif run_details.data.params.get("algorithm") == "DBSCAN":
                                st.write("**Core Samples:**", len(model.core_sample_indices_))
                        except Exception as e:
                            st.error(f"Không thể tải mô hình: {e}")

if __name__ == "__main__":
    main()
