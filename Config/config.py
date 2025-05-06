model_params = {
    'backbone_out_channels': 512,  # Số kênh đầu ra của backbone (VGG16), ảnh hưởng đến kích thước đặc trưng đầu vào của RPN
    'scales': [128, 256, 512],    # Danh sách các kích thước neo (anchor scales) để tạo anchor boxes, đại diện cho diện tích neo
    'aspect_ratios': [0.5, 1.0, 2.0],  # Danh sách tỷ lệ khung hình của anchor boxes (width/height), ảnh hưởng đến hình dạng neo
    'rpn_bg_threshold': 0.3,       # Ngưỡng IOU thấp nhất để phân loại anchor là nền (background), dưới ngưỡng này được coi là nền
    'rpn_fg_threshold': 0.7,       # Ngưỡng IOU cao nhất để phân loại anchor là đối tượng chính (foreground), trên ngưỡng này được coi là đối tượng
    'rpn_nms_threshold': 0.7,      # Ngưỡng IOU cho Non-Maximum Suppression (NMS) trong RPN, loại bỏ các đề xuất chồng lấp
    'rpn_batch_size': 256,         # Kích thước batch của anchor được lấy mẫu trong RPN để tính loss
    'rpn_pos_fraction': 0.5,       # Tỷ lệ anchor tích cực (positive) trong batch RPN, còn lại là anchor tiêu cực
    'rpn_train_topk': 2000,        # Số lượng đề xuất hàng đầu (top-k) được giữ lại trong giai đoạn huấn luyện
    'rpn_test_topk': 1000,         # Số lượng đề xuất hàng đầu (top-k) được giữ lại trong giai đoạn kiểm tra
    'rpn_train_prenms_topk': 12000,# Số lượng đề xuất tối đa trước khi áp dụng NMS trong huấn luyện
    'rpn_test_prenms_topk': 6000,  # Số lượng đề xuất tối đa trước khi áp dụng NMS trong kiểm tra
    'roi_batch_size': 128,         # Kích thước batch của các vùng đề xuất (proposals) được lấy mẫu trong ROI Head
    'roi_pos_fraction': 0.25,      # Tỷ lệ vùng đề xuất tích cực (positive) trong batch ROI Head
    'roi_iou_threshold': 0.5,      # Ngưỡng IOU để gán vùng đề xuất với ground truth, quyết định vùng đề xuất là đối tượng hay nền
    'roi_low_bg_iou': 0.0,         # Ngưỡng IOU thấp nhất để phân loại vùng đề xuất là nền (nếu tăng lên 0.1, sẽ tạo khó khăn hơn cho phân loại nền)
    'roi_nms_threshold': 0.5,      # Ngưỡng IOU cho NMS trong ROI Head, loại bỏ các vùng đề xuất chồng lấp
    'roi_topk_detections': 100,    # Số lượng phát hiện tối đa (top-k) được giữ lại sau NMS trong suy luận
    'roi_score_threshold': 0.05,   # Ngưỡng điểm số (score) tối thiểu để giữ lại phát hiện trong suy luận
    'roi_pool_size': 7,            # Kích thước lưới (grid size) của ROI Pooling, ảnh hưởng đến kích thước đặc trưng đầu ra
    'fc_inner_dim': 1024,          # Kích thước chiều nội tại của các lớp fully connected trong ROI Head
    'min_im_size': 600,            # Kích thước nhỏ nhất của hình ảnh đầu vào sau khi resize (chiều ngắn nhất)
    'max_im_size': 1000            # Kích thước lớn nhất của hình ảnh đầu vào sau khi resize (chiều dài nhất)
}
