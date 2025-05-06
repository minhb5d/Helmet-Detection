# Helmet Detection Using Faster R-CNN

**1.Overview**
- Helmet Detection là một dự án sử dụng mô hình Faster R-CNN để nhận diện mũ bảo hiểm trong ảnh. Dự án được xây dựng với mục tiêu phát hiện và phân loại hai lớp chính: helmet (có mũ bảo hiểm) và no-helmet (không có mũ bảo hiểm). Đây là một ứng dụng hữu ích trong việc giám sát an toàn giao thông, đảm bảo người tham gia giao thông tuân thủ quy định đội mũ bảo hiểm.
Dự án sử dụng dữ liệu theo định dạng PASCAL VOC và được huấn luyện trên framework PyTorch. Mô hình Faster R-CNN được triển khai với backbone VGG16 và bao gồm các thành phần chính như Region Proposal Network (RPN) và ROI Head để phát hiện và phân loại đối tượng.

**2.Features**
- Phát hiện đối tượng: Sử dụng Faster R-CNN để phát hiện mũ bảo hiểm trong ảnh.
- Phân loại: Phân loại đối tượng thành hai lớp: helmet và no-helmet.
- Tùy chỉnh tham số: Các tham số mô hình và huấn luyện được cấu hình trong Config/config.py.

**3.Requirements**
- Để chạy dự án, bạn cần cài đặt các thư viện trong file requirements.txt
- Cài đặt các thư viện bằng lệnh: ***pip install -r requirements.txt***

**4.Dataset**
- Dự án sử dụng dữ liệu theo định dạng PASCAL VOC, với cấu trúc thư mục như sau:
  - JPEGImages/: Chứa các file ảnh (.jpg hoặc .png).
  - Annotations/: Chứa các file annotation (.xml) tương ứng với từng ảnh.
- Dataset gồm khoảng 4720 ảnh đã được tiền xử lý và gán nhãn qua Roboflow
- Phân chia tập dữ liệu: 80%-10%-10% tương ứng với từng tập train test valid
- Số lượng label:
   - helmet: 23332 
   - no-helmet: 18539

**5.Train**
- Đường dẫn dữ liệu huấn luyện được cấu hình trong train.py 
- Mô hình sau khi huấn luyện sẽ được lưu tại ***checkpoints/faster_rcnn_helmet.pth***.

**6.Inference**
- Để thực hiện dự đoán trên ảnh mới, sử dụng file helmet_inference.py trong thư mục Inference: ***python Inference/helmet_inference.py***
- Đảm bảo mô hình đã huấn luyện (faster_rcnn_helmet.pth) được tải trước khi dự đoán.
- Kết quả dự đoán sẽ hiển thị các hộp giới hạn (bounding boxes) và nhãn (helmet hoặc no-helmet) trên ảnh.

**7.Model Architecture**
- Dự án sử dụng Faster R-CNN với các thành phần chính:
  - Backbone: VGG16 (các lớp trước max-pooling cuối cùng).
  - Region Proposal Network (RPN): Tạo các đề xuất vùng (region proposals) tiềm năng chứa đối tượng.
  - ROI Head: Phân loại và tinh chỉnh các vùng đề xuất thành các hộp giới hạn cuối cùng.
- Chi tiết triển khai mô hình nằm trong file ***Model/faster_rcnn.py***.

**8.Results**
- Mô hình đạt hiệu suất tốt trong việc phát hiện và phân loại mũ bảo hiểm. Một ví dụ kết quả dự đoán có thể được xem trong file ***Inference/example.png***.





