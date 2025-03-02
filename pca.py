import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import os
import time

# Thiết lập logging với MLflow
mlflow.set_tracking_uri("file:./mlruns")

# Tải dữ liệu MNIST từ OpenML
@st.cache_data
def load_mnist_data():
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    return X, y

# Lấy mẫu dữ liệu ngẫu nhiên
def sample_data(X, y, sample_size):
    if sample_size > len(X):
        sample_size = len(X)
    indices = np.random.choice(len(X), sample_size, replace=False)
    return X[indices], y[indices]

# Hàm giảm chiều bằng PCA
def apply_pca(X, n_components):
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    explained_variance = np.sum(pca.explained_variance_ratio_)
    return X_reduced, explained_variance

# Hàm giảm chiều bằng t-SNE
def apply_tsne(X, n_components, perplexity, learning_rate):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
    X_reduced = tsne.fit_transform(X)
    return X_reduced

# Vẽ biểu đồ phân tán
def plot_scatter(X_reduced, y, title):
    df = pd.DataFrame({
        'x': X_reduced[:, 0],
        'y': X_reduced[:, 1],
        'label': y
    })
    fig = px.scatter(df, x='x', y='y', color='label', title=title,
                     labels={'x': 'Component 1', 'y': 'Component 2'})
    return fig

def main():
    st.title("Giảm Chiều Dữ Liệu MNIST với PCA và t-SNE")
    
    # Tạo các tab
    tab1, tab2, tab3 = st.tabs(["Tổng quan", "Phương pháp PCA và t-SNE", "MLflow"])

    # Tab 1: Tổng quan
    with tab1:
        X, y = load_mnist_data()
        # Hiển thị nhiều ảnh mẫu
        st.subheader("Một số ảnh mẫu từ tập dữ liệu MNIST")
        
        # Số lượng ảnh muốn hiển thị
        num_samples = st.slider("Chọn số lượng ảnh mẫu", 1, 20, 5)
        
        # Tạo các cột để hiển thị ảnh
        cols = st.columns(5)  # Hiển thị tối đa 5 ảnh trên một hàng
        
        for i in range(num_samples):
            with cols[i % 5]:  # Chia ảnh vào các cột
                st.image(X[i].reshape(28, 28), caption=f"Chữ số {y[i]}", width=100)

        st.subheader("🔹Thuật toán giảm chiều dữ liệu")
        st.write("##### 1. PCA (Principal Component Analysis)")
        st.write("""- PCA là một phương pháp giảm chiều dữ liệu (dimensionality reduction) tương đối hiệu quả dựa trên phép phân tích
         suy biến (singular decomposition) mà ở đó chúng 
         ta sẽ chiếu các điểm dữ liệu trong không gian cao chiều xuống một số ít
         những véc tơ thành phần chính trong không gian thấp chiều
         mà đồng thời vẫn bảo toàn tối đa độ biến động của dữ liệu sau biến đổi. Ưu điểm của PCA đó là
         sử dụng tất cả các biến đầu vào nên phương pháp này không bỏ sót những biến quan trọng.""")
        st.write("- Các bước thực hiện PCA :")
        st.image("p1.png")
        st.image("p2.png")
        st.write("##### 2. t-SNE (t-Distributed Stochastic Neighbor Embedding) ")
        st.write("""-  t-SNE là xác định một hàm phân phối xác suất chung dựa trên Gaussian cho các điểm dữ liệu chiều cao, xác định một hàm phân phối xác suất 
        chung dựa trên phân phối t cho các điểm dữ liệu chiều thấp và sau đó sắp xếp lại dữ liệu chiều thấp điểm để giảm độ chênh lệch (về KL phân kì) giữa hai lần phân bố. """)
        st.write("Các bước thực hiện t-SNE :")
        st.write("""+ Bước 1: t-SNE mô hình hóa một điểm được chọn làm lân cận của một điểm khác ở cả chiều cao hơn và chiều thấp hơn. Nó bắt đầu bằng cách tính toán độ tương đồng từng cặp giữa
        tất cả các điểm dữ liệu trong không gian chiều cao bằng cách sử dụng hạt nhân Gaussian. Các điểm xa nhau có xác suất được chọn thấp hơn các điểm gần nhau. """)
        st.write("+ Bước 2: Sau đó, thuật toán sẽ cố gắng ánh xạ các điểm dữ liệu có chiều cao hơn vào không gian có chiều thấp hơn trong khi vẫn bảo toàn các điểm tương đồng theo từng cặp.")  
        st.write("""+ Bước 3: Nó đạt được bằng cách giảm thiểu sự phân kỳ giữa phân phối xác suất chiều cao ban đầu và chiều thấp ban đầu. Thuật toán sử dụng gradient descent để giảm thiểu sự 
        phân kỳ. Nhúng chiều thấp được tối ưu hóa đến trạng thái ổn định.""")

    # Tab 2: Phương pháp PCA và t-SNE
    with tab2:
        st.header("Phương pháp PCA và t-SNE")

        # Tải dữ liệu
        X, y = load_mnist_data()

        # Tùy chọn mẫu dữ liệu
        st.subheader("Tùy chọn mẫu dữ liệu")
        sample_size = st.slider("Chọn kích thước mẫu dữ liệu", 100, 10000, 1000, key="sample_size_tab2")
        X_sample, y_sample = sample_data(X, y, sample_size)
        st.write(f"Kích thước dữ liệu sau khi lấy mẫu: {X_sample.shape}")

        # Nhập tên thí nghiệm
        experiment_name = st.text_input("Nhập tên thí nghiệm")
        if not experiment_name:
            experiment_name = "Default_Model"
        mlflow.set_experiment(experiment_name)

        # Chọn phương pháp giảm chiều
        method = st.selectbox("Chọn phương pháp giảm chiều", ["PCA", "t-SNE"], key="method_tab2")

        # Tham số cho từng phương pháp
        if method == "PCA":
            n_components = st.slider("Số lượng thành phần PCA", 2, 50, 2, key="n_components_tab2")
        elif method == "t-SNE":
            n_components = 2
            perplexity = st.slider("Perplexity", 5, 50, 30, key="perplexity_tab2",help="""**Perplexity** : Tham số kiểm soát số lượng điểm lân cận mà t-SNE xem xét khi xây dựng phân phối xác suất trong không gian nhiều chiều. 
            \n- Perplexity thấp (5-10), tập trung vào các điểm lân cận gần nhất, tạo ra các cụm nhỏ và chi tiết hơn , có thể dẫn đến việc phân tách quá mức.
            \n- Perplexity cao (40-50), xem xét nhiều điểm lân cận hơn,tạo ra các cụm lớn hơn và tổng quát hơn, có thể làm mất đi các chi tiết nhỏ trong dữ liệu.
            """)
            learning_rate = st.slider("Learning Rate", 10, 1000, 200, key="learning_rate_tsne",help=""" **learning_rate** : Tốc độ học của thuật toán t-SNE, kiểm soát cách thuật toán cập nhật các điểm trong không gian chiều thấp.
            \n- Learning rate thấp (10-100):Thuật toán học chậm hơn, có thể dẫn đến kết quả không ổn định hoặc không hội tụ.
            \n- Learning rate cao (500-1000):Thuật toán học nhanh hơn, có thể dẫn đến việc các điểm "nhảy" quá mức, làm mất đi cấu trúc cụm.
            """)

        # Nút bấm giảm chiều
        if st.button("Giảm chiều", key="reduce_button_tab2"):
            with st.spinner("Đang thực hiện giảm chiều..."):
                start_time = time.time()
                if method == "PCA":
                    with mlflow.start_run(run_name=f"PCA_{n_components}_components"):
                        X_reduced, explained_variance = apply_pca(X_sample, n_components)
                        st.write(f"Tỷ lệ phương sai giải thích: {explained_variance:.4f}")
                        
                        # Logging với MLflow
                        mlflow.log_param("method", "PCA")
                        mlflow.log_param("n_components", n_components)
                        mlflow.log_param("sample_size", sample_size)
                        mlflow.log_metric("explained_variance", explained_variance)
                        
                        # Visual hóa
                        fig = plot_scatter(X_reduced, y_sample, f"PCA - {n_components} Components")
                        st.plotly_chart(fig)

                elif method == "t-SNE":
                    with mlflow.start_run(run_name=f"t-SNE_perplexity_{perplexity}"):
                        X_reduced = apply_tsne(X_sample, n_components, perplexity, learning_rate)
                        
                        # Logging với MLflow
                        mlflow.log_param("method", "t-SNE")
                        mlflow.log_param("n_components", n_components)
                        mlflow.log_param("perplexity", perplexity)
                        mlflow.log_param("sample_size", sample_size)
                        mlflow.log_param("learning_rate", learning_rate)
                        
                        # Visual hóa
                        fig = plot_scatter(X_reduced, y_sample, f"t-SNE - Perplexity: {perplexity}")
                        st.plotly_chart(fig)
                
                # Đo thời gian thực thi
                execution_time = time.time() - start_time
                # st.write(f"Thời gian thực thi: {execution_time:.2f} giây")
                mlflow.log_metric("execution_time", execution_time)
            
            time.sleep(1)
            st.success(f"Đã hoàn thành giảm chiều và lưu vào thí nghiệm '{experiment_name}'!")

    # Tab 3: MLflow
    with tab3:
        st.header("MLflow Tracking")
        st.write("Chọn một thí nghiệm và một kết quả để xem chi tiết.")

        # Lấy danh sách experiment
        experiments = mlflow.search_experiments()
        experiment_names = [exp.name for exp in experiments]
        experiment_dict = {exp.name: exp.experiment_id for exp in experiments}

        if not experiment_names:
            st.write("Chưa có thí nghiệm nào được lưu.")
        else:
            # Chọn thí nghiệm
            selected_experiment = st.selectbox("Chọn thí nghiệm", experiment_names, key="select_exp_tab3")
            selected_exp_id = experiment_dict[selected_experiment]

            # Tìm kiếm theo từ khóa trong experiment đã chọn
            search_query = st.text_input("Tìm kiếm trong thí nghiệm (theo phương pháp, ...)", "", key="search_tab3")
            runs = mlflow.search_runs(experiment_ids=[selected_exp_id])

            if not runs.empty:
                # Thêm cột tên thí nghiệm
                runs['experiment_name'] = selected_experiment
                
                # Lọc dữ liệu dựa trên từ khóa
                if search_query:
                    runs = runs[runs.apply(lambda row: search_query.lower() in str(row).lower(), axis=1)]
                
                st.write(f"Tìm thấy {len(runs)} kết quả trong thí nghiệm '{selected_experiment}'.")
                # Hiển thị danh sách run
                available_columns = [col for col in ['run_id', 'experiment_name', 'start_time', 'params.method', 
                                                    'params.n_components', 'params.perplexity', 'params.sample_size', 
                                                    'metrics.explained_variance', 'params.learning_rate', 
                                                    'metrics.execution_time'] if col in runs.columns]
                st.dataframe(runs[available_columns])

                # Chọn một run để xem chi tiết
                run_ids = runs['run_id'].tolist()
                selected_run_id = st.selectbox("Chọn một kết quả (run) để xem chi tiết", run_ids, key="select_run_tab3")
                
                if selected_run_id:
                    selected_run = runs[runs['run_id'] == selected_run_id].iloc[0]
                    st.subheader(f"Chi tiết của Run ID: {selected_run_id}")
                    
                    # Hiển thị thông tin chung
                    st.write("**Thông tin chung:**")
                    general_info = {
                        'Run ID': selected_run['run_id'],
                        'Experiment Name': selected_run['experiment_name']
                    }
                    for key, value in general_info.items():
                        if pd.notna(value):
                            st.write(f"{key}: {value}")

                    # Hiển thị thông tin liên quan đến phương pháp
                    st.write("**Thông tin liên quan đến phương pháp:**")
                    method = selected_run.get('params.method', 'N/A')
                    if method == "PCA":
                        method_info = {
                            'Phương pháp': method,
                            'Số lượng thành phần (n_components)': selected_run.get('params.n_components', 'N/A'),
                            'Tỷ lệ phương sai giải thích (explained_variance)': selected_run.get('metrics.explained_variance', 'N/A'),
                            'Kích thước mẫu (sample_size)': selected_run.get('params.sample_size', 'N/A'),
                        }
                    elif method == "t-SNE":
                        method_info = {
                            'Phương pháp': method,
                            'Perplexity': selected_run.get('params.perplexity', 'N/A'),
                            'Learning Rate': selected_run.get('params.learning_rate', 'N/A'),
                            'Số lượng thành phần (n_components)': selected_run.get('params.n_components', 'N/A'),
                            'Kích thước mẫu (sample_size)': selected_run.get('params.sample_size', 'N/A'),
                        }
                    else:
                        method_info = {'Phương pháp': 'Không xác định'}

                    for key, value in method_info.items():
                        if pd.notna(value):
                            st.write(f"{key}: {value}")

            else:
                st.write(f"Chưa có run nào trong thí nghiệm '{selected_experiment}'.")

if __name__ == "__main__":
    main()
