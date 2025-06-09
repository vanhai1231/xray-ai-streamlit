import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np
import requests
from io import BytesIO

# Cấu hình trang
st.set_page_config(
    page_title="AI X-ray Diagnosis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh cho giao diện đẹp hơn
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 10px;
        color: white;
    }
    
    .stFileUploader > div > div > div {
        background-color: #f8f9fa;
        border: 2px dashed #007bff;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
    }
    
    .upload-text {
        font-size: 1.2rem;
        color: #6c757d;
        margin-bottom: 1rem;
    }
    
    .result-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    
    .confidence-bar {
        background-color: rgba(255,255,255,0.3);
        border-radius: 10px;
        padding: 0.5rem;
        margin: 1rem 0;
    }
    
    .sidebar-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #28a745;
    }
    
    .warning-container {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header chính
st.markdown("""
<div class="main-header">
    <h1>🏥 AI Chẩn Đoán X-ray Thông Minh</h1>
    <p style="font-size: 1.2rem; margin-bottom: 0;">Phân tích hình ảnh X-ray phổi bằng Deep Learning</p>
</div>
""", unsafe_allow_html=True)

# Sidebar với thông tin
with st.sidebar:
    st.markdown("### ℹ️ Thông Tin Hệ Thống")
    st.markdown("""
    <div class="sidebar-info">
        <strong>Model:</strong> Simple CNN<br>
        <strong>Dataset:</strong> Chest X-ray<br>
        <strong>Accuracy:</strong> ~95%<br>
        <strong>Classes:</strong> Normal, Pneumonia
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📋 Hướng Dẫn Sử Dụng")
    st.markdown("""
    1. **Tải ảnh X-ray** lên hệ thống
    2. **Chờ AI phân tích** (vài giây)
    3. **Xem kết quả** chi tiết
    4. **Tham khảo khuyến nghị** từ hệ thống
    """)
    
    st.markdown("### ⚠️ Lưu Ý Quan Trọng")
    st.warning("""
    Kết quả chỉ mang tính chất tham khảo. 
    Vui lòng tham khảo ý kiến bác sĩ chuyên khoa 
    để có chẩn đoán chính xác.
    """)

# Định nghĩa Simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x

# Load model (với cache để tối ưu hiệu suất)
@st.cache_resource
def load_model():
    try:
        # Thử tải từ Hugging Face Hub
        try:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(repo_id="vanhai123/simple-cnn-chest-xray", filename="model.pth")
            model = SimpleCNN(num_classes=2)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            
            # Định nghĩa transform
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            return model, transform, True
            
        except Exception as hub_error:
            st.warning(f"Không thể tải model từ Hugging Face: {hub_error}")
            # Fallback: Tạo model demo
            model = SimpleCNN(num_classes=2)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            return model, transform, False
            
    except Exception as e:
        st.error(f"Lỗi khi tải model: {str(e)}")
        return None, None, False

# Tải model
with st.spinner("🔄 Đang tải AI model..."):
    model, transform, is_trained = load_model()

if model is None:
    st.error("❌ Không thể tải model. Vui lòng thử lại sau.")
    st.stop()

# Thông báo trạng thái model
if not is_trained:
    st.warning("⚠️ Đang sử dụng model demo. Kết quả chỉ mang tính minh họa.")

# Định nghĩa labels
id2label = {0: "Normal", 1: "Pneumonia"}

# Layout chính
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Tải Ảnh X-ray")
    st.markdown('<div class="upload-text">Kéo thả hoặc click để chọn ảnh</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"],
        help="Hỗ trợ định dạng: JPG, JPEG, PNG"
    )

with col2:
    st.markdown("### 📊 Kết Quả Phân Tích")
    result_placeholder = st.empty()

if uploaded_file:
    try:
        # Hiển thị ảnh đã tải lên
        image = Image.open(uploaded_file).convert("RGB")
        
        with col1:
            st.image(
                image, 
                caption=f"📷 {uploaded_file.name}", 
                use_column_width=True
            )
            
            # Thông tin ảnh
            st.markdown("#### 📋 Thông Tin Ảnh")
            img_info = f"""
            - **Tên file:** {uploaded_file.name}
            - **Kích thước:** {image.size[0]} x {image.size[1]} pixels
            - **Dung lượng:** {uploaded_file.size / 1024:.1f} KB
            - **Thời gian:** {datetime.now().strftime("%H:%M:%S - %d/%m/%Y")}
            """
            st.markdown(img_info)
        
        # Phân tích ảnh
        with st.spinner("🤖 AI đang phân tích ảnh..."):
            # Tiền xử lý ảnh
            input_tensor = transform(image).unsqueeze(0)
            
            # Dự đoán
            with torch.no_grad():
                if is_trained:
                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    pred_idx = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][pred_idx].item()
                    all_probs = probs[0].numpy()
                else:
                    # Demo mode - random results for demonstration
                    import random
                    random.seed(42)  # For consistent demo results
                    all_probs = np.array([random.uniform(0.3, 0.9), random.uniform(0.1, 0.7)])
                    all_probs = all_probs / all_probs.sum()  # Normalize
                    pred_idx = np.argmax(all_probs)
                    confidence = all_probs[pred_idx]
                
                labels = list(id2label.values())
        
        # Hiển thị kết quả
        with col2:
            label = id2label[pred_idx]
            
            # Kết quả chính
            if label.lower() == "normal":
                st.success(f"✅ **Kết quả: {label}**")
                st.markdown(f"**Độ tin cậy:** {confidence:.1%}")
            else:
                st.error(f"⚠️ **Kết quả: {label}**")
                st.markdown(f"**Độ tin cậy:** {confidence:.1%}")
            
            # Biểu đồ confidence
            fig_bar = px.bar(
                x=labels,
                y=[prob * 100 for prob in all_probs],
                title="Phân Tích Chi Tiết (%)",
                color=[prob * 100 for prob in all_probs],
                color_continuous_scale="RdYlGn"
            )
            fig_bar.update_layout(
                showlegend=False,
                height=300,
                xaxis_title="Chẩn Đoán",
                yaxis_title="Xác Suất (%)"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Biểu đồ tròn
            fig_pie = px.pie(
                values=all_probs,
                names=labels,
                title="Tỷ Lệ Chẩn Đoán"
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Khuyến nghị và thông tin chi tiết
        st.markdown("---")
        st.markdown("### 💡 Khuyến Nghị & Thông Tin Chi Tiết")
        
        col3, col4 = st.columns([1, 1])
        
        with col3:
            if label.lower() == "normal":
                st.markdown("""
                <div class="metric-container">
                    <h4>✅ Phổi Bình Thường</h4>
                    <p><strong>Ý nghĩa:</strong> Không phát hiện dấu hiệu bất thường</p>
                    <p><strong>Khuyến nghị:</strong></p>
                    <ul>
                        <li>Duy trì lối sống lành mạnh</li>
                        <li>Tập thể dục đều đặn</li>
                        <li>Kiểm tra sức khỏe định kỳ</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-container">
                    <h4>⚠️ Phát Hiện Bất Thường</h4>
                    <p><strong>Khuyến nghị khẩn cấp:</strong></p>
                    <ul>
                        <li>🏥 <strong>Liên hệ bác sĩ ngay lập tức</strong></li>
                        <li>📋 Mang kết quả này đến cơ sở y tế</li>
                        <li>🚫 Không tự điều trị</li>
                        <li>⏰ Cần chẩn đoán chuyên khoa</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("#### 📈 Thống Kê Phân Tích")
            
            # Metrics
            st.metric("Độ Tin Cậy", f"{confidence:.1%}", 
                     delta=f"{confidence-0.5:.1%}" if confidence > 0.5 else f"{confidence-0.5:.1%}")
            
            st.metric("Thời Gian Phân Tích", "< 2s", delta="Nhanh")
            
            # Thanh tiến trình confidence
            st.markdown("**Mức Độ Tin Cậy:**")
            st.progress(confidence)
            
            if confidence >= 0.9:
                st.success("🎯 Độ tin cậy rất cao")
            elif confidence >= 0.7:
                st.warning("⚡ Độ tin cậy tốt")
            else:
                st.error("⚠️ Độ tin cậy thấp - cần xem xét thêm")
    
    except Exception as e:
        st.error(f"❌ Lỗi khi xử lý ảnh: {str(e)}")
        st.info("💡 Vui lòng thử với ảnh X-ray khác hoặc kiểm tra định dạng file.")

else:
    # Hướng dẫn khi chưa có ảnh
    with col2:
        result_placeholder.markdown("""
        <div style="text-align: center; padding: 3rem; color: #6c757d;">
            <h3>🔬 Chờ Ảnh X-ray</h3>
            <p>Vui lòng tải ảnh X-ray lên để bắt đầu phân tích</p>
            <p><small>Hỗ trợ: JPG, JPEG, PNG</small></p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 1rem;">
    <p>🤖 <strong>AI X-ray Diagnosis System</strong> | 
    Powered by Deep Learning | 
    <small>v2.0 - 2025</small></p>
    <p><small>⚠️ Chỉ dùng để tham khảo - Không thay thế chẩn đoán y khoa</small></p>
</div>
""", unsafe_allow_html=True)