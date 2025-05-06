Helmet Detection Using Faster R-CNN
Overview
Helmet Detection là một dự án sử dụng mô hình Faster R-CNN để nhận diện mũ bảo hiểm trong ảnh. Dự án được xây dựng với mục tiêu phát hiện và phân loại hai lớp chính: helmet (có mũ bảo hiểm) và no-helmet (không có mũ bảo hiểm). Đây là một ứng dụng hữu ích trong việc giám sát an toàn giao thông, đảm bảo người tham gia giao thông tuân thủ quy định đội mũ bảo hiểm.
Dự án sử dụng dữ liệu theo định dạng PASCAL VOC và được huấn luyện trên framework PyTorch. Mô hình Faster R-CNN được triển khai với backbone VGG16 và bao gồm các thành phần chính như Region Proposal Network (RPN) và ROI Head để phát hiện và phân loại đối tượng.

Project Structure
Cấu trúc thư mục của dự án được tổ chức như sau:
Helmet-Detection/
│
├── Config/
│   ├── __init__.py
│   └── config.py                # Cấu hình tham số cho mô hình
│
├── Dataset_Management/
│   ├── __init__.py
│   └── data_management.py       # Quản lý và tổ chức dữ liệu theo định dạng PASCAL VOC
│
├── Dataset_Loader/
│   ├── __init__.py
│   └── voc_dataset.py           # Tải và xử lý dữ liệu PASCAL VOC
│
├── Inference/
│   ├── __init__.py
│   ├── example.png              # Ảnh ví dụ cho suy luận
│   └── helmet_inference.py      # Script suy luận để phát hiện mũ bảo hiểm
│
├── Model/
│   ├── __init__.py
│   └── faster_rcnn.py           # Triển khai mô hình Faster R-CNN
│
├── Training/
│   ├── __init__.py
│   └── train.py                 # Script huấn luyện mô hình
│
├── requirements.txt             # Danh sách các thư viện cần thiết
└── README.md                    # Tài liệu hướng dẫn


Features

Phát hiện đối tượng: Sử dụng Faster R-CNN để phát hiện mũ bảo hiểm trong ảnh.
Phân loại: Phân loại đối tượng thành hai lớp: helmet và no-helmet.
Tăng cường dữ liệu: Áp dụng lật ngang ảnh trong quá trình huấn luyện để cải thiện hiệu suất mô hình.
Tùy chỉnh tham số: Các tham số mô hình và huấn luyện được cấu hình trong Config/config.py.


Requirements
Để chạy dự án, bạn cần cài đặt các thư viện sau (được liệt kê trong requirements.txt):

torch==1.13.1
torchvision==0.14.1
pillow==9.0.1
tqdm==4.64.0

Cài đặt các thư viện bằng lệnh:
pip install -r requirements.txt


Dataset
Dự án sử dụng dữ liệu theo định dạng PASCAL VOC, với cấu trúc thư mục như sau:

JPEGImages/: Chứa các file ảnh (.jpg hoặc .png).
Annotations/: Chứa các file annotation (.xml) tương ứng với từng ảnh.

Chuẩn bị dữ liệu

Đặt dữ liệu huấn luyện vào thư mục /kaggle/working/helmet-detector/train.
Đặt dữ liệu kiểm tra vào thư mục /kaggle/working/helmet-detector/test.
Script data_management.py sẽ tự động tổ chức các file ảnh và annotation vào đúng thư mục JPEGImages và Annotations.


Usage
1. Huấn luyện mô hình
Để huấn luyện mô hình, chạy file train.py trong thư mục Training:
python Training/train.py


Đường dẫn dữ liệu huấn luyện được cấu hình trong train.py (mặc định: /kaggle/working/helmet-detector/train).
Mô hình sau khi huấn luyện sẽ được lưu tại checkpoints/faster_rcnn_helmet.pth.

2. Suy luận (Inference)
Để thực hiện suy luận trên ảnh mới, sử dụng file helmet_inference.py trong thư mục Inference:
python Inference/helmet_inference.py


Đảm bảo mô hình đã huấn luyện (faster_rcnn_helmet.pth) được tải trước khi suy luận.
Kết quả suy luận sẽ hiển thị các hộp giới hạn (bounding boxes) và nhãn (helmet hoặc no-helmet) trên ảnh.


Model Architecture
Dự án sử dụng Faster R-CNN với các thành phần chính:

Backbone: VGG16 (các lớp trước max-pooling cuối cùng).
Region Proposal Network (RPN): Tạo các đề xuất vùng (region proposals) tiềm năng chứa đối tượng.
ROI Head: Phân loại và tinh chỉnh các vùng đề xuất thành các hộp giới hạn cuối cùng.

Chi tiết triển khai mô hình nằm trong file Model/faster_rcnn.py.

Results
Mô hình đạt hiệu suất tốt trong việc phát hiện và phân loại mũ bảo hiểm. Một ví dụ kết quả suy luận có thể được xem trong file Inference/example.png.

Future Improvements

Thêm các kỹ thuật tăng cường dữ liệu khác (xoay ảnh, thay đổi độ sáng, v.v.).
Tối ưu hóa mô hình để chạy trên các thiết bị nhúng (embedded devices).
Hỗ trợ nhận diện thời gian thực (real-time detection) trên video.


Contributing
Mọi đóng góp đều được hoan nghênh! Nếu bạn muốn cải thiện dự án, hãy:

Fork repository này.
Tạo branch mới (git checkout -b feature/YourFeature).
Commit thay đổi của bạn (git commit -m 'Add some feature').
Push lên branch (git push origin feature/YourFeature).
Tạo Pull Request.


License
Dự án được phát hành dưới MIT License.

Contact
Nếu bạn có câu hỏi hoặc cần hỗ trợ, hãy liên hệ qua email: [your-email@example.com].
