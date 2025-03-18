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
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.stats import linregress

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
        algorithm =st.selectbox("Chọn thuật toán:", ["PCA","t-SNE"])
        
        if algorithm == "PCA":
            st.write("##### Thuật toán PCA")
            st.write("""**PCA (Principal Component Analysis)** là một phương pháp giảm chiều dữ liệu tuyến tính, tìm ra các thành phần chính (principal components) để chiếu dữ liệu từ không gian chiều cao xuống không gian chiều thấp hơn mà vẫn giữ tối đa thông tin (phương sai).""")
            st.write("**Các bước thực hiện PCA** :")
        
            # Tạo dữ liệu giả lập 2D
            st.write("🔹Minh họa PCA trên dữ liệu giả lập 2D")
            st.write("Chúng ta sẽ sử dụng một tập dữ liệu 2D giả lập với 300 điểm, phân bố theo dạng elip nghiêng.")
        
            # Tạo dữ liệu giả lập
            np.random.seed(42)
            n_samples = 300
            cov = [[1, 0.8], [0.8, 1]]  # Ma trận hiệp phương sai với tương quan cao
            X_sim = np.random.multivariate_normal(mean=[0, 0], cov=cov, size=n_samples)
            y_sim = (X_sim[:, 0] + X_sim[:, 1] > 0).astype(int)
        
            # Bước 1: Chuẩn hóa dữ liệu
            st.write("- **Bước 1: Chuẩn hóa dữ liệu**")
            st.write("Đồng nhất hóa thang đo và mức độ biến thiên của các biến số, nhằm loại bỏ sự thiên lệch do khác biệt về đơn vị hoặc phạm vi giá trị, thực hiện bằng công thức Z-score:")
            st.latex(r"""
            X' = \frac{X - \mu}{\sigma}
            """)
            st.write("Trong đó:")
            st.latex(r"""
            \begin{aligned}
            &X: \text{Giá trị gốc của dữ liệu} \\
            &\mu: \text{Trung bình của mỗi chiều}, \quad \mu = \frac{1}{n} \sum_{i=1}^{n} X_i \\
            &\sigma: \text{Độ lệch chuẩn của mỗi chiều}, \quad \sigma = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (X_i - \mu)^2}
            \end{aligned}
            """)

            # Chuẩn hóa dữ liệu
            X_mean = X_sim.mean(axis=0)
            X_std = X_sim.std(axis=0)
            X_std[X_std == 0] = 1e-10  # Tránh chia cho 0
            X_normalized = (X_sim - X_mean) / X_std
        
            # Vẽ dữ liệu trước và sau chuẩn hóa
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].scatter(X_sim[:, 0], X_sim[:, 1], c=y_sim, cmap="viridis", alpha=0.6)
            ax[0].set_title("Dữ liệu gốc")
            ax[0].set_xlabel("X")
            ax[0].set_ylabel("Y")
            ax[0].grid(True)
        
            ax[1].scatter(X_normalized[:, 0], X_normalized[:, 1], c=y_sim, cmap="viridis", alpha=0.6)
            ax[1].set_title("Dữ liệu sau chuẩn hóa")
            ax[1].set_xlabel("X (chuẩn hóa)")
            ax[1].set_ylabel("Y (chuẩn hóa)")
            ax[1].grid(True)
            plt.tight_layout()
            st.pyplot(fig)
            st.write("Dữ liệu gốc có phân bố elip nghiêng. Sau chuẩn hóa hình dạng phân bố không thay đổi.")
        
            # Bước 2: Tính ma trận hiệp phương sai (Biểu đồ phân tán với đường hồi quy)
            st.write("- **Bước 2: Tính ma trận hiệp phương sai (Covariance Matrix)**")
            st.write("Ma trận hiệp phương sai biểu diễn mức độ tương quan giữa các biến:")
            st.latex(r"""
            \Sigma = \frac{1}{n-1} X^T X
            """)
            st.write("Trong đó:")
            st.latex(r"""
            \begin{aligned}
            &X: \text{Ma trận dữ liệu đã chuẩn hóa} \, (n \times d, \text{với } n \text{ là số mẫu, } d \text{ là số chiều}) \\
            &X^T: \text{Ma trận chuyển vị của } X \\
            &\Sigma_{ij}: \text{Phần tử tại hàng } i, \text{ cột } j \text{ là hiệp phương sai giữa chiều } i \text{ và chiều } j \\
            &\quad \Sigma_{ij} = \frac{1}{n-1} \sum_{k=1}^{n} (X_{ki} - \mu_i)(X_{kj} - \mu_j)
            \end{aligned}
            """)
            st.write("Nếu hai biến có hiệp phương sai lớn, chúng có xu hướng thay đổi cùng nhau. Để minh họa, chúng ta vẽ biểu đồ phân tán của hai chiều với đường hồi quy tuyến tính, phản ánh mức độ tương quan.")
        
            # Tính ma trận hiệp phương sai
            covariance_matrix = np.cov(X_normalized.T)
        
            # Tính đường hồi quy tuyến tính
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(X_normalized[:, 0], X_normalized[:, 1])
            line = slope * X_normalized[:, 0] + intercept
        
            # Vẽ biểu đồ phân tán với đường hồi quy
            fig, ax = plt.subplots(figsize=(3, 2))
            ax.scatter(X_normalized[:, 0], X_normalized[:, 1], c=y_sim, cmap="viridis", alpha=0.6)
            ax.plot(X_normalized[:, 0], line, color="red", linestyle="--", label=f"Đường hồi quy (R² = {r_value**2:.2f})")
            ax.set_title("Phân tán và đường hồi quy giữa X và Y",fontsize=6)
            ax.set_xlabel("X (chuẩn hóa)",fontsize=6)
            ax.set_ylabel("Y (chuẩn hóa)",fontsize=6)
            ax.tick_params(axis='both', labelsize=6)  # Giảm kích thước chữ trên các dấu tick
            ax.grid(True)
            ax.legend(fontsize=6) 
            st.pyplot(fig)
            st.write(f"Đường hồi quy (màu đỏ) cho thấy mức độ tương quan giữa X và Y, với hệ số R² = {r_value**2:.2f}. Ma trận hiệp phương sai sẽ có giá trị ngoài đường chéo (khoảng {covariance_matrix[0, 1]:.2f}) phản ánh tương quan này.")
        
            # Bước 3: Tính toán giá trị riêng và vector riêng
            st.write("- **Bước 3: Tính toán giá trị riêng và vector riêng**")
            st.write("Giải phương trình eigenvalue decomposition:")
            st.latex(r"""
            \Sigma v = \lambda v
            """)
            st.write("Trong đó:")
            st.latex(r"""
            \begin{aligned}
            &\Sigma: \text{Ma trận hiệp phương sai} \\
            &v: \text{Vector riêng (hướng của thành phần chính, là vector đơn vị, } ||v|| = 1\text{)} \\
            &\lambda: \text{Giá trị riêng (số thực, thể hiện phương sai theo hướng } v\text{)}
            \end{aligned}
            """)
            st.markdown("Phương trình này được giải bằng phân rã giá trị riêng (eigen decomposition), tìm tất cả $$( (\lambda, v) $$) sao cho phương trình thỏa mãn.")
        
            # Tính giá trị riêng và vector riêng
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
        
            # Vẽ dữ liệu với vector riêng
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=X_normalized[:, 0], y=X_normalized[:, 1], mode="markers",
                                    marker=dict(color=y_sim, colorscale="Viridis", size=8, opacity=0.6),
                                    name="Dữ liệu"))
            
            # Thêm vector riêng
            scale = 2
            for i in range(2):
                fig.add_shape(type="line", x0=0, y0=0, x1=eigenvectors[0, i] * scale * np.sqrt(eigenvalues[i]),
                            y1=eigenvectors[1, i] * scale * np.sqrt(eigenvalues[i]),
                            line=dict(color="red", width=3))
                fig.add_annotation(x=eigenvectors[0, i] * scale * np.sqrt(eigenvalues[i]),
                                y=eigenvectors[1, i] * scale * np.sqrt(eigenvalues[i]),
                                text=f"PC{i+1}", showarrow=False)
        
            fig.update_layout(title="Dữ liệu với vector riêng (PC1, PC2)",
                            xaxis_title="X (chuẩn hóa)", yaxis_title="Y (chuẩn hóa)",
                            showlegend=True)
            st.plotly_chart(fig)
            st.write(f"Giá trị riêng: PC1 = {eigenvalues[0]:.2f}, PC2 = {eigenvalues[1]:.2f}. Vector riêng (PC1, PC2) là các hướng chính, thể hiện độ biến thiên lớn nhất.")
        
            # Bước 4: Chọn số lượng thành phần chính
            st.write("- **Bước 4: Chọn số lượng thành phần chính**")
            st.write("Chọn số thành phần chính dựa trên tỷ lệ phương sai tích lũy:")
            st.latex(r"""
            \text{Explained Variance Ratio}_i = \frac{\lambda_i}{\sum_{j=1}^{d} \lambda_j}
            """)
            st.latex(r"""
            \text{Cumulative Explained Variance} = \sum_{i=1}^{k} \frac{\lambda_i}{\sum_{j=1}^{d} \lambda_j}
            """)
            st.write("Trong đó:")
            st.latex(r"""
            \begin{aligned}
            &\lambda_i: \text{Giá trị riêng của thành phần thứ } i \\
            &d: \text{Tổng số chiều của dữ liệu} \\
            &k: \text{Số thành phần chính được chọn}
            \end{aligned}
            """)
            st.write("Thường chọn \( k \) sao cho tổng phương sai đạt 85-95%.")
        
            # Tính tỷ lệ phương sai tích lũy
            explained_variance_ratio = eigenvalues / eigenvalues.sum()
            cumulative_variance = np.cumsum(explained_variance_ratio)
        
            # Vẽ biểu đồ tỷ lệ phương sai tích lũy
            fig = px.bar(x=["PC1", "PC2"], y=explained_variance_ratio,
                        title="Tỷ lệ phương sai giải thích bởi từng thành phần chính",
                        labels={'x': 'Thành phần chính', 'y': 'Tỷ lệ phương sai'})
            st.plotly_chart(fig)
            st.write(f"PC1 giải thích {explained_variance_ratio[0]*100:.2f}% phương sai, PC2 giải thích {explained_variance_ratio[1]*100:.2f}%. Tổng cộng: {cumulative_variance[-1]*100:.2f}%. Trong ví dụ này, chúng ta chọn cả 2 thành phần chính để trực quan hóa.")
        
            # Bước 5: Biến đổi dữ liệu sang không gian mới
            st.write("- **Bước 5: Biến đổi dữ liệu sang không gian mới**")
            st.write("Chuyển dữ liệu sang hệ tọa độ mới bằng cách nhân với ma trận chứa các vector riêng:")
            st.latex(r"""
            X_{\text{new}} = X V_k
            """)
            st.write("Trong đó:")
            st.latex(r"""
            \begin{aligned}
            &X: \text{Ma trận dữ liệu đã chuẩn hóa} \, (n \times d) \\
            &V_k: \text{Ma trận chứa } k \text{ vector riêng đầu tiên} \, (d \times k), \text{ với các cột là vector riêng} \\
            &X_{\text{new}}: \text{Ma trận dữ liệu sau khi giảm chiều} \, (n \times k)
            \end{aligned}
            """)
        
            # Chiếu dữ liệu lên không gian mới
            X_new = np.dot(X_normalized, eigenvectors)
        
            # Vẽ dữ liệu trong không gian mới
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=X_new[:, 0], y=X_new[:, 1], mode="markers",
                                    marker=dict(color=y_sim, colorscale="Viridis", size=8, opacity=0.6),
                                    name="Dữ liệu"))
            fig.update_layout(title="Dữ liệu trong không gian mới (PC1, PC2)",
                            xaxis_title="PC1", yaxis_title="PC2",
                            showlegend=True)
            st.plotly_chart(fig)
            st.write("Dữ liệu được chiếu lên không gian mới, với trục tọa độ là các thành phần chính PC1 và PC2. PC1 (trục X) là hướng có độ biến thiên lớn nhất.")
        elif algorithm == "t-SNE":
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
       
    with tab2:
        X, y = load_mnist_data()
        st.write("**🖼️ Một vài mẫu dữ liệu từ MNIST**")
        num_samples = 10  
        cols = st.columns(10)
        for i in range(num_samples):
            with cols[i % 10]:
                fig, ax = plt.subplots()
                ax.imshow(X[i].reshape(28, 28), cmap="gray")
                ax.axis("off")
                st.pyplot(fig)
                st.caption(f"{y[i]}")

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
