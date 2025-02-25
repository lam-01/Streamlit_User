import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn
import streamlit as st
import plotly.express as px
from sklearn.metrics import mean_squared_error, r2_score
import os

class TitanicAnalyzer:
    def __init__(self):
        self.data = None
        self.model = None
        self.scaler = None  # Sẽ khởi tạo sau khi có dữ liệu
        self.poly = None
        self.feature_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
        self.is_fitted = False
        self.sex_male = 0  # Default value
        self.sex_female = 1  # Default value
        self.embarked_C = 0  # Default value
        self.embarked_Q = 1  # Default value
        self.embarked_S = 2  # Default value
    
    def load_and_preprocess(self, data_path):
        """Đọc và tiền xử lý dữ liệu với MLflow"""
        try:
            mlflow.start_run()
            st.write("##### **📚Tiền xử lý dữ liệu**")
            
            # Đọc dữ liệu
            st.write("**1. Đọc dữ liệu**")
            self.data = pd.read_csv(data_path)
            mlflow.log_param("initial_data_shape", self.data.shape)
            st.write("Dữ liệu ban đầu:", self.data.head())
            
            # Xử lý missing values
            st.write("**2. Xử lý giá trị bị thiếu**")
            st.write("- Các cột dữ liệu bị thiếu: Age, Cabin, Embarked")
            missing_values_before = self.data.isnull().sum()
            st.write("Số lượng dữ liệu bị thiếu : ")
            st.dataframe(missing_values_before.to_frame().T)

            # Chọn phương pháp xử lý giá trị bị thiếu
            missing_value_strategy = st.selectbox(
                "## Chọn phương pháp ", ["Điền giá trị trung bình mean", "Điền giá trị trung vị median", "Điền giá trị xuất hiện nhiều nhất mode", "Xóa hàng chứa dữ liệu thiếu drop"], index=0
            )

            # Hàm xử lý dữ liệu bị thiếu
            def preprocess_data(df, missing_value_strategy):
                df = df.dropna(subset=['Survived'])  # Bỏ các hàng có giá trị thiếu ở cột mục tiêu

                # Xác định cột số và cột phân loại
                num_cols = df.select_dtypes(include=['number']).columns
                cat_cols = df.select_dtypes(exclude=['number']).columns

                # Xử lý giá trị thiếu cho cột số
                if missing_value_strategy == 'mean':
                    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
                elif missing_value_strategy == 'median':
                    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
                elif missing_value_strategy == 'mode':
                    for col in num_cols:
                        if not df[col].mode().dropna().empty:
                            df[col] = df[col].fillna(df[col].mode()[0])
                # Luôn xử lý giá trị thiếu cho Cabin và Embarked
                df['Cabin'] = df['Cabin'].fillna("Unknown")  # Điền "Unknown" cho Cabin
                if not df['Embarked'].mode().dropna().empty:
                    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Điền mode() cho Embarked

                if missing_value_strategy == 'drop':
                    df.dropna(inplace=True)  # Nếu chọn "drop", xóa hàng còn thiếu

                return df  # Trả về dataframe đã xử lý

            # Gọi hàm xử lý dữ liệu bị thiếu
            self.data = preprocess_data(self.data, missing_value_strategy)

            # Kiểm tra số lượng dữ liệu bị thiếu sau khi xử lý
            missing_values_after = self.data.isnull().sum().sum()
            mlflow.log_metric("missing_values_before", missing_values_before.sum())  # Chuyển thành số tổng
            mlflow.log_metric("missing_values_after", missing_values_after)
            st.write("Số lượng giá trị bị thiếu sau xử lý:")
            st.dataframe(self.data.isnull().sum().to_frame().T)


            # Xóa các cột không cần thiết
            st.write("**3. Xóa các cột không cần thiết**")
            st.write("""
            - **Name**: Tên hành khách không ảnh hưởng trực tiếp đến khả năng sống sót.
            - **Ticket**: Số vé là một chuỗi ký tự không mang ý nghĩa rõ ràng đối với mô hình dự đoán.
            - **Cabin**: Dữ liệu bị thiếu quá nhiều, rất nhiều hành khách không có thông tin về cabin.
            """)

            # Cho phép người dùng chọn cột để xóa
            columns_to_drop = st.multiselect(
                "Chọn cột để xóa:",
                self.data.columns.tolist(),  
                default=['Name', 'Ticket', 'Cabin']  # Gợi ý mặc định
            )
            # Loại bỏ cột PassengerId
            if 'PassengerId' in self.data.columns:
                self.data = self.data.drop(columns=['PassengerId'])
    
            # Xóa các cột được chọn
            self.data.drop(columns=columns_to_drop, inplace=True, errors='ignore')

            # Hiển thị thông tin sau khi xóa cột
            st.write("Dữ liệu sau khi xóa các cột không cần thiết:")
            st.dataframe(self.data.head())

            
            st.write("**4. Mã hóa biến phân loại**")
            st.write("**Mã hóa cột Sex:**")
            sex_male = st.number_input("Nhập giá trị mã hóa cho 'male':", value=0, key="sex_male")
            sex_female = st.number_input("Nhập giá trị mã hóa cho 'female':", value=1, key="sex_female")

            # Kiểm tra xem giá trị mã hóa có trùng nhau không
            if sex_male == sex_female:
                st.error("Giá trị mã hóa cho 'male' và 'female' không được trùng nhau!")
            else:
                # Mã hóa cột 'Sex'
                if 'Sex' in self.data.columns:
                    self.data['Sex'] = self.data['Sex'].map({'male': sex_male, 'female': sex_female})
                    st.write(f"Đã mã hóa 'male' thành {sex_male} và 'female' thành {sex_female}.")

            # Cho phép người dùng nhập giá trị mã hóa cho 'Embarked'
            st.write("**Mã hóa cột Embarked:**")
            embarked_C = st.number_input("Nhập giá trị mã hóa cho 'C':", value=0, key="embarked_C")
            embarked_Q = st.number_input("Nhập giá trị mã hóa cho 'Q':", value=1, key="embarked_Q")
            embarked_S = st.number_input("Nhập giá trị mã hóa cho 'S':", value=2, key="embarked_S")

            # Kiểm tra xem giá trị mã hóa có trùng nhau không
            embarked_values = [embarked_C, embarked_Q, embarked_S]
            if len(embarked_values) != len(set(embarked_values)):
                st.error("Giá trị mã hóa cho 'C', 'Q', và 'S' không được trùng nhau!")
            else:
                # Điền giá trị thiếu cho 'Embarked' và mã hóa
                if 'Embarked' in self.data.columns:
                    self.data['Embarked'] = self.data['Embarked'].fillna('Unknown')

                    # Mã hóa cột 'Embarked'
                    embarked_mapping = {'C': embarked_C, 'Q': embarked_Q, 'S': embarked_S}
                    self.data['Embarked'] = self.data['Embarked'].map(lambda x: embarked_mapping.get(x, -1))
                    st.write(f"Đã mã hóa 'C' thành {embarked_C}, 'Q' thành {embarked_Q}, và 'S' thành {embarked_S}.")
                    # Hiển thị dữ liệu sau khi mã hóa
                    st.write("Dữ liệu sau khi mã hóa:")
                    st.dataframe(self.data.head())


            
            mlflow.end_run()
            return self.data
            
        except Exception as e:
            st.error(f"Lỗi khi tiền xử lý dữ liệu: {str(e)}")
            mlflow.end_run(status='FAILED')
            return None


def create_streamlit_app():
    st.title("Titanic 🚢")
    
    # Sử dụng st.tabs để tạo thanh menu
    tab1, tab2, tab3 = st.tabs([ "🔍 Xử lý và Huấn luyện ","🪄 Dự đoán", "🚀 MLflow"])
    analyzer = TitanicAnalyzer()
        
    with tab1:
        data_path = "titanic.csv"  # Đường dẫn cố định
        analyzer = TitanicAnalyzer()
        data = analyzer.load_and_preprocess(data_path)
        total_samples = len(data)

        # Cho phép người dùng chọn tỷ lệ chia dữ liệu
        st.write("##### 📊 Chọn tỷ lệ chia dữ liệu")
        test_size = st.slider("Tập kiểm tra (Test)", 0.1, 0.3, 0.2)  # Chọn Test trước
        train_valid_size = 1 - test_size  # Phần còn lại dành cho Train + Validation

        valid_size = st.slider("Tỷ lệ Validation trong tập Train", 0.1, 0.4, 0.2)  # Tỉ lệ của Valid trong Train
        train_size = 1 - valid_size  # Tập Train chính xác

        # Tính số mẫu
        test_samples = int(test_size * total_samples)
        train_valid_samples = total_samples - test_samples  # Train + Valid
        valid_samples = int(valid_size * train_valid_samples)
        train_samples = train_valid_samples - valid_samples  # Phần còn lại cho Train

        # Tạo DataFrame hiển thị kết quả
        split_df = pd.DataFrame({
            "Tập dữ liệu": ["Train", "Validation", "Test"],
            "Tỷ lệ (%)": [f"{train_samples / total_samples * 100:.2f}", 
                        f"{valid_samples / total_samples * 100:.2f}", 
                        f"{test_samples / total_samples * 100:.2f}"],
            "Số lượng mẫu": [train_samples, valid_samples, test_samples]
        })

        # Hiển thị bảng kết quả
        st.write("📋 **Tỷ lệ chia dữ liệu và số lượng mẫu:**")
        st.table(split_df)

        # Hiển thị kết quả trong Streamlit
        st.write("##### 📊 **Huấn luyện mô hình hồi quy**")
        # Lựa chọn mô hình
        regression_type = st.radio("Chọn loại hồi quy:", ["Multiple Regression", "Polynomial Regression"])

        # Chọn bậc của Polynomial Regression (chỉ hiển thị nếu chọn Polynomial)
        degree = None
        if regression_type == "Polynomial Regression":
            degree = st.slider("Chọn bậc của hồi quy đa thức:", min_value=2, max_value=5, value=2)

        # Load dữ liệu và chia train/valid/test
        X = data.drop(columns=["Survived"])
        y = data["Survived"]

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, random_state=42)
        X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, train_size=valid_size / (valid_size + test_size), random_state=42)
        
        imputer = SimpleImputer(strategy='mean')  # Điền giá trị thiếu bằng giá trị trung bình
        X_train = imputer.fit_transform(X_train)
        X_valid = imputer.transform(X_valid)
        X_test = imputer.transform(X_test)

        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        X_test_scaled = scaler.transform(X_test)

        # Lưu scaler vào session_state
        st.session_state["scaler"] = scaler

        with mlflow.start_run():
            # Chọn mô hình dựa trên loại hồi quy
            if regression_type == "Polynomial Regression":
                poly = PolynomialFeatures(degree=degree)
                X_train_poly = poly.fit_transform(X_train_scaled)
                X_valid_poly = poly.transform(X_valid_scaled)
                X_test_poly = poly.transform(X_test_scaled)

                model = LinearRegression()
                model.fit(X_train_poly, y_train)

                y_pred_train = model.predict(X_train_poly)
                y_pred_valid = model.predict(X_valid_poly)
                y_pred_test = model.predict(X_test_poly)

            else:  # Multiple Regression
                model = LinearRegression()
                model.fit(X_train_scaled, y_train)

                y_pred_train = model.predict(X_train_scaled)
                y_pred_valid = model.predict(X_valid_scaled)
                y_pred_test = model.predict(X_test_scaled)

            # Lưu mô hình vào session_state
            st.session_state["model"] = model
            if regression_type == "Polynomial Regression":
                st.session_state["poly"] = poly

            # Tính toán metrics
            mse_train = mean_squared_error(y_train, y_pred_train)
            mse_valid = mean_squared_error(y_valid, y_pred_valid)
            mse_test = mean_squared_error(y_test, y_pred_test)

            r2_train = r2_score(y_train, y_pred_train)
            r2_valid = r2_score(y_valid, y_pred_valid)
            r2_test = r2_score(y_test, y_pred_test)

            # Cross-validation
            # Thêm thanh trượt để chọn số lượng folds cho cross-validation
            cv_folds = st.slider("Chọn số lượng folds cho Cross-Validation:", min_value=2, max_value=10, value=5, step=1)
            y_pred_cv = cross_val_predict(model, X_train_scaled, y_train, cv=cv_folds)
            mse_cv = mean_squared_error(y_train, y_pred_cv)

            # Ghi log vào MLflow
            mlflow.log_metrics({
                "train_mse": mse_train,
                "valid_mse": mse_valid,
                "test_mse": mse_test,
                "cv_mse": mse_cv
            })

            st.write(f"**Loại hồi quy đang sử dụng:** {regression_type}")
            
            results_df = pd.DataFrame({
                "Metric": ["MSE (Train)", "MSE (Validation)", "MSE (Test)", "MSE (Cross-Validation)"],
                "Value": [mse_train, mse_valid, mse_test, mse_cv]
            })
            if st.button("Huấn luyện mô hình"):
                st.write("**📌 Kết quả đánh giá mô hình:**")
                st.table(results_df)
    with tab2 :             
            # Prediction interface
            st.subheader("Giao diện dự đoán")
# Kiểm tra nếu mô hình đã huấn luyện trước khi dự đoán
            if 'model' in st.session_state and 'scaler' in st.session_state:
                analyzer.model = st.session_state['model']
                analyzer.scaler = st.session_state['scaler']
                if regression_type == "Polynomial Regression":
                    analyzer.poly = st.session_state['poly']
                analyzer.is_fitted = True
            else:
                st.error("Vui lòng huấn luyện mô hình trước khi dự đoán!")

            if analyzer.is_fitted:
                col1, col2 = st.columns(2)

                with col1:
                    pclass = st.selectbox("Passenger Class", [1, 2, 3])
                    age = st.number_input("Age", 0, 100, 30)
                    sex = st.selectbox("Sex", ["male", "female"])
                
                with col2:
                    sibsp = st.number_input("Siblings", 0, 10, 0)
                    parch = st.number_input("Parents/Children", 0, 10, 0)
                    fare = st.number_input("Fare", 0.0, 500.0, 32.0)
                    embarked = st.selectbox("Port of Embarkation", ['C', 'Q', 'S'])
                
                if st.button("Dự đoán"):

                    # Tạo DataFrame đầu vào
                    input_data = pd.DataFrame({
                        'Pclass': [pclass],
                        'Age': [age],
                        'SibSp': [sibsp],
                        'Parch': [parch],
                        'Fare': [fare],
                        'Sex': [analyzer.sex_male if sex == "male" else analyzer.sex_female],  # Sử dụng giá trị mã hóa do người dùng nhập
                        'Embarked': [analyzer.embarked_C if embarked == "C" else 
                                        analyzer.embarked_Q if embarked == "Q" else 
                                        analyzer.embarked_S]  
                    })
                    # Kiểm tra xem đối tượng có thuộc tập dữ liệu gốc hay không
                    exists_in_data = False
                    if analyzer.data is not None:
                        exists_in_data = any((analyzer.data['Pclass'] == pclass) & 
                                            (analyzer.data['Age'] == age) & 
                                            (analyzer.data['SibSp'] == sibsp) & 
                                            (analyzer.data['Parch'] == parch) & 
                                            (analyzer.data['Fare'] == fare) & 
                                            (analyzer.data['Sex'] == (analyzer.sex_male if sex == "male" else analyzer.sex_female)) & 
                                            (analyzer.data['Embarked'] == (analyzer.embarked_C if embarked == "C" else 
                                                                            analyzer.embarked_Q if embarked == "Q" else 
                                                                            analyzer.embarked_S)))

                    # Scale dữ liệu đầu vào
                    input_scaled = st.session_state['scaler'].transform(input_data)
                    
                    # Kiểm tra xem mô hình có sử dụng PolynomialFeatures không
                    if regression_type == "Polynomial Regression":
                        input_transformed = st.session_state['poly'].transform(input_scaled)
                    else:
                        input_transformed = input_scaled

                    # Dự đoán
                    prediction = st.session_state['model'].predict(input_transformed)[0]
                    
                    # Hiển thị kết quả
                    st.success(f"Dự đoán : {'Survived' if prediction == 1 else 'Not Survived'}")
                     # Hiển thị thông tin về việc đối tượng có thuộc tập dữ liệu gốc hay không
                    if exists_in_data:
                        st.info("Đối tượng này có tồn tại trong tập dữ liệu gốc.")
                    else:
                        st.warning("Đối tượng này không có trong tập dữ liệu gốc.")

    with tab3:
            st.header("📊 MLflow Tracking")

            # Hiển thị thông tin về các phiên làm việc
            if st.button("Xem các phiên làm việc"):
                # Lấy danh sách các phiên làm việc
                runs = mlflow.search_runs(order_by=["start_time desc"])
                if not runs.empty:
                    st.write("### Danh sách các phiên làm việc:")
                    st.dataframe(runs[["run_id", "experiment_id", "start_time", "status", "metrics.train_mse", "metrics.valid_mse", "metrics.test_mse"]])

                    # Hiển thị thông tin chi tiết cho từng phiên làm việc
                    selected_run_id = st.selectbox("Chọn một phiên làm việc để xem chi tiết:", runs['run_id'].tolist())
                    if selected_run_id:
                        run_details = mlflow.get_run(selected_run_id)
                        # st.write("### Thông tin chi tiết cho phiên làm việc:", selected_run_id)
                        # st.write("**Trạng thái:**", run_details.info.status)
                        # st.write("**Thời gian bắt đầu:**", run_details.info.start_time)
                        # st.write("**Thời gian kết thúc:**", run_details.info.end_time)
                        st.write("**Tham số:**")
                        for key, value in run_details.data.params.items():
                            st.write(f"- **{key}**: {value}")
                        st.write("**Metric:**")
                        for key, value in run_details.data.metrics.items():
                            st.write(f"- **{key}**: {value}")
                        st.write("**Artifacts:**")
                        # if run_details.info.artifact_uri:
                        #     st.write(f"- **Artifact URI**: {run_details.info.artifact_uri}")
                        # else:
                        #     st.write("- Không có artifacts nào.")

                else:
                    st.write("Không có phiên làm việc nào được ghi lại.")

            # Hiển thị MLflow Tracking UI trong iframe
            mlflow_url = "http://localhost:5000"  # Thay đổi nếu chạy trên server khác
            st.markdown(f"[Mở MLflow Tracking UI]({mlflow_url})", unsafe_allow_html=True)


if __name__ == "__main__":
    create_streamlit_app()
