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
# Disable MLflow autologging to prevent unintended runs
mlflow.sklearn.autolog(disable=True)

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

# Hàm giảm chiều bằng PCA với tiến trình
def apply_pca(X, n_components, progress_bar):
    progress_bar.progress(20)  # 20% sau khi bắt đầu PCA
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    explained_variance = np.sum(pca.explained_variance_ratio_)
    progress_bar.progress(60)  # 60% sau khi giảm chiều xong
    return X_reduced, explained_variance

# Hàm giảm chiều bằng t-SNE với tiến trình
def apply_tsne(X, n_components, progress_bar):
    progress_bar.progress(20)  # 20% sau khi bắt đầu t-SNE
    tsne = TSNE(n_components=n_components, random_state=42)
    X_reduced = tsne.fit_transform(X)
    progress_bar.progress(60)  # 60% sau khi giảm chiều xong
    return X_reduced

# Vẽ biểu đồ phân tán (2D hoặc 3D)
def plot_scatter(X_reduced, y, title):
    if X_reduced.shape[1] == 2:
        df = pd.DataFrame({
            'x': X_reduced[:, 0],
            'y': X_reduced[:, 1],
            'label': y
        })
        fig = px.scatter(df, x='x', y='y', color='label', title=title,
                         labels={'x': 'Component 1', 'y': 'Component 2'})
        return fig
    elif X_reduced.shape[1] == 3:
        df = pd.DataFrame({
            'x': X_reduced[:, 0],
            'y': X_reduced[:, 1],
            'z': X_reduced[:, 2],
            'label': y
        })
        fig = px.scatter_3d(df, x='x', y='y', z='z', color='label', title=title,
                            labels={'x': 'Component 1', 'y': 'Component 2', 'z': 'Component 3'})
        return fig
    return None

def main():
    st.title("Giảm Chiều Dữ Liệu MNIST với PCA và t-SNE")
    
    if 'run_in_progress' not in st.session_state:
        st.session_state.run_in_progress = False
    if 'last_run_id' not in st.session_state:
        st.session_state.last_run_id = None

    tab1, tab2, tab3 = st.tabs(["Tổng quan", "PCA và t-SNE", "MLflow"])

    with tab1:
        algorithm =st.selectbox("Chọn thuật toán:", ["t-SNE","PCA"])
        

        if algorithm == "t-SNE":
            st.write("##### Thuật toán t-SNE")
            st.write("**t-SNE (T-distributed Stochastic Neighbor Embedding)** là một kỹ thuật giảm kích thước phi tuyến không giám sát để khám phá dữ liệu và trực quan hóa dữ liệu chiều cao. Giảm kích thước phi tuyến tính có nghĩa là thuật toán cho phép chúng ta tách dữ liệu không thể phân tách bằng đường thẳng.")
            st.write("**Nguyên lí hoạt động**")
            st.write("- 💠**Bước 1**:  t-SNE mô hình hóa một điểm được chọn làm hàng xóm của một điểm khác ở cả chiều cao hơn và thấp hơn. Nó bắt đầu bằng cách tính toán sự tương đồng theo cặp giữa tất cả các điểm dữ liệu trong không gian chiều cao bằng cách sử dụng hạt nhân Gaussian.")
            st.image("tnse.png",caption="Nguồn : https://statquest.org")
            st.write("Sử dụng phân phối chuẩn nếu các điểm cách xa nhau, chúng có ít điểm giống nhau, và nếu chúng gần nhau, chúng có nhiều điểm giống nhau")
            st.image("tnse2.png",caption="Nguồn : https://statquest.org")
            st.image("tnse3.png",caption="Nguồn : https://statquest.org")
            st.write("Lặp lại thao tác này cho tất cả các điểm nằm trong phạm vi đã xác định trước đó")
            st.image("tnse4.png",caption="Nguồn : https://statquest.org")
            st.write("Sau khi tính toán các khoảng cách được biểu diễn trên phân phối chuẩn, chuyển đổi chúng thành một tập hợp xác suất của tất cả các điểm (cchia tỷ lệ chúng sao cho tất cả các giá trị có tổng bằng 1). Điều này cung cấp cho chúng ta một tập hợp xác suất cho tất cả các điểm ở đó")
            st.image("tnse5.png",caption="Nguồn : https://statquest.org")

            st.write("- 💠**Bước 2** : Sau đó, thuật toán cố gắng ánh xạ các điểm dữ liệu chiều cao hơn vào không gian chiều thấp hơn trong khi vẫn giữ nguyên các điểm tương đồng theo cặp.")
            st.image("tnse6.png",caption="Nguồn : https://statquest.org")
            st.write("Khi t-SNE chuyển đổi dữ liệu từ không gian cao chiều xuống không gian thấp chiều (2D hoặc 3D), nó không sử dụng phân phối Gaussian nữa mà thay vào đó dùng phân phối t hay còn gọi là phân phối Cauchy")
            st.write("a. Tính toán tất cả các khoảng cách",caption="Nguồn : https://statquest.org")
            st.image("tnse7.png")
            st.write("b. Danh sách các điểm tương đồng",caption="Nguồn : https://statquest.org")
            st.image("tnse8.png")
            st.write("c. Tính toán tập xác suất  trong không gian có chiều thấp")
            st.image("tnse9.png",caption="Nguồn : https://statquest.org")
            st.write("- 💠**Bước 3**: Nó đạt được bằng cách giảm thiểu sự phân kỳ giữa phân phối xác suất chiều cao và chiều thấp hơn ban đầu. Thuật toán sử dụng độ dốc gradient để giảm thiểu sự phân kỳ. Việc nhúng chiều thấp hơn được tối ưu hóa ở trạng thái ổn định.")
            st.write("Làm cho tập hợp các xác suất từ không gian chiều thấp phản ánh càng sát càng tốt các xác suất từ không gian chiều cao giúp hai cấu trúc tập hợp này phải giống nhau.")
            st.image("tnse10.png",caption="Nguồn : https://statquest.org")
            st.write("Trong thuật toán t-SNE, để so sánh hai phân phối xác suất giữa không gian cao chiều (trước khi giảm chiều) và không gian thấp chiều (sau khi giảm chiều), ta sử dụng phân kỳ Kullback-Leibler (KL Divergence).")
            st.image("tnse11.png",caption="Nguồn : https://statquest.org")
        elif algorithm == "PCA":
            st.write("")
       
    with tab2:
        X, y = load_mnist_data()
        st.write("##### Một số ảnh mẫu từ tập dữ liệu MNIST")
        num_samples = 5  
        cols = st.columns(5)
        for i in range(num_samples):
            with cols[i % 5]:
                fig, ax = plt.subplots()
                ax.imshow(X[i].reshape(28, 28), cmap="gray")
                ax.axis("off")
                st.pyplot(fig)
                st.caption(f"Chữ số {y[i]}")

        st.write("##### Tùy chọn mẫu dữ liệu")
        sample_size = st.number_input("Chọn cỡ mẫu để phân cụm", min_value=1000, max_value=70000, value=5000, step=1000)
        X_sample, y_sample = sample_data(X, y, sample_size)
        st.write(f"Kích thước dữ liệu sau khi lấy mẫu: {X_sample.shape}")

        model_name = st.text_input("Nhập tên mô hình:")
        if not model_name:
            model_name= "Default_Model"
        method = st.selectbox("Chọn phương pháp giảm chiều", ["PCA", "t-SNE"], key="method_tab2")

        if method == "PCA":
            n_components = st.slider("Số lượng thành phần PCA", 2, 50, 2, key="n_components_tab2")
        elif method == "t-SNE":
            n_components = st.slider("Số lượng thành phần t-SNE", 2, 10, 2, key="n_components_tsne")

        if st.button("Giảm chiều", key="reduce_button_tab2"):
            if not st.session_state.run_in_progress:  
                st.session_state.run_in_progress = True
                with st.spinner("Đang thực hiện giảm chiều..."):
                    progress_bar = st.progress(0)
                    
                    if mlflow.active_run():
                        mlflow.end_run()
                    
                    progress_bar.progress(10)
                    start_time = time.time()
                    mlflow.set_experiment("MNIST_Dimensionality_Reduction")
                    
                    try:
                        with mlflow.start_run(run_name=model_name) as run:
                            st.session_state.last_run_id = run.info.run_id
                            if method == "PCA":
                                X_reduced, explained_variance = apply_pca(X_sample, n_components, progress_bar)
                                mlflow.log_param("method", "PCA")
                                mlflow.log_param("n_components", n_components)
                                mlflow.log_param("sample_size", sample_size)
                                mlflow.log_param("model_name", model_name)
                                mlflow.log_metric("explained_variance", explained_variance)
                                if n_components in [2, 3]:
                                    fig = plot_scatter(X_reduced, y_sample, f"PCA - {n_components} Components")
                                    progress_bar.progress(80)
                                    if fig is not None:
                                        st.plotly_chart(fig, use_container_width=True)
                                    st.write(f"Tỷ lệ phương sai giải thích: {explained_variance:.4f}")
                                else:
                                    progress_bar.progress(80)
                                    st.write(f"Kết quả PCA có {n_components} chiều, không thể visual hóa trực tiếp ở 2D hoặc 3D. Tỷ lệ phương sai giải thích: {explained_variance:.4f}")

                            elif method == "t-SNE":
                                X_reduced = apply_tsne(X_sample, n_components, progress_bar)
                                mlflow.log_param("method", "t-SNE")
                                mlflow.log_param("n_components", n_components)
                                mlflow.log_param("sample_size", sample_size)
                                mlflow.log_param("model_name", model_name)
                                if n_components in [2, 3]:
                                    fig = plot_scatter(X_reduced, y_sample, f"t-SNE - {n_components} Components")
                                    progress_bar.progress(80)
                                    if fig is not None:
                                        st.plotly_chart(fig, use_container_width=True)
                                else:
                                    progress_bar.progress(80)
                                    st.write(f"Kết quả t-SNE có {n_components} chiều, không thể visual hóa trực tiếp ở 2D hoặc 3D.")
                            
                            execution_time = time.time() - start_time
                            mlflow.log_metric("execution_time", execution_time)
                            progress_bar.progress(100)
                    except Exception as e:
                        if mlflow.active_run():
                            mlflow.end_run()
                    
                    # Đảm bảo chạy được kết thúc
                    if mlflow.active_run():
                        mlflow.end_run()
                
                time.sleep(1)
                # st.success(f"Đã hoàn thành giảm chiều và lưu vào thí nghiệm 'MNIST_Dimensionality_Reduction' với tên mô hình '{model_name}'!")
                st.session_state.run_in_progress = False

    with tab3:
        st.subheader("MLflow Tracking")

        if mlflow.active_run():
            mlflow.end_run()

        experiments = mlflow.search_experiments()
        experiment_dict = {exp.name: exp.experiment_id for exp in experiments}
        selected_exp_id = experiment_dict.get("MNIST_Dimensionality_Reduction")

        if not selected_exp_id:
            st.write("Chưa có thí nghiệm 'MNIST_Dimensionality_Reduction'. Vui lòng giảm chiều dữ liệu trước.")
        else:
            search_query = st.text_input("Tìm kiếm theo tên mô hình", "", key="search_tab3")
            runs = mlflow.search_runs(experiment_ids=[selected_exp_id])

            if not runs.empty:
                runs['experiment_name'] = "MNIST_Dimensionality_Reduction"
                if search_query:
                    runs = runs[runs['params.model_name'].str.contains(search_query, case=False, na=False)]
                
                available_columns = ['params.model_name', 'start_time', 'params.method', 
                                    'params.n_components', 'params.sample_size', 'metrics.explained_variance']
                display_columns = [col for col in available_columns if col in runs.columns]
                display_df = runs[display_columns].rename(columns={'params.model_name': 'Model Name'})
                st.dataframe(display_df)

                model_names = runs['params.model_name'].unique().tolist()
                selected_model_name = st.selectbox("Chọn một mô hình để xem chi tiết", model_names, key="select_model_tab3")
                
                if selected_model_name:
                    selected_run = runs[runs['params.model_name'] == selected_model_name].iloc[0]
                    st.write(f"##### Chi tiết của Model Name: {selected_model_name}")
                    
                    st.write("**Thông tin chung:**")
                    general_info = {
                        'Model Name': selected_run.get('params.model_name', 'N/A'),
                        'Start Time': selected_run.get('start_time', 'N/A')
                        # 'Execution Time (s)': selected_run.get('metrics.execution_time', 'N/A'),  # Không nhận thức và cố định
                    }
                    for key, value in general_info.items():
                        if pd.notna(value):
                            st.write(f"{key}: {value}")

                    st.write("**Thông tin liên quan đến phương pháp:**")
                    method = selected_run.get('params.method', 'N/A')
                    if method == "PCA":
                        method_info = {
                            'Phương pháp': method,
                            'Số lượng thành phần (n_components)': selected_run.get('params.n_components', 'N/A'),
                            'Tỷ lệ phương sai giải thích': selected_run.get('metrics.explained_variance', 'N/A'),
                            'Kích thước mẫu': selected_run.get('params.sample_size', 'N/A'),  # Fixed spacing
                        }
                    elif method == "t-SNE":
                        method_info = {
                            'Phương pháp': method,
                            'Số lượng thành phần (n_components)': selected_run.get('params.n_components', 'N/A'),
                            'Kích thước mẫu': selected_run.get('params.sample_size', 'N/A'),  # Fixed spacing
                        }
                    else:
                        method_info = {'Phương pháp': 'Không xác định'}

                    for key, value in method_info.items():
                        if pd.notna(value):
                            st.write(f"{key}: {value}")
            else:
                st.write("Chưa có run nào trong thí nghiệm 'MNIST_Dimensionality_Reduction'.")

if __name__ == "__main__":
    main()
