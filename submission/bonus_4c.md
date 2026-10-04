# Bonus — 4C và bài tập về nhà 1: tập val có nói thật không?

Link notebook đã chạy: ttps://www.kaggle.com/code/anhquanjerryus/track-4-lab-3

Môi trường: Kaggle, Tesla T4, `ultralytics==8.4.171`, YOLO26n-pose, 40 epoch, imgsz 640, seed 0.
Mọi con số dưới đây lấy từ output của ô 4C và ô "Bài tập về nhà 1" trong `lab_2d_perception_student.ipynb`.

## 1. Thí nghiệm

- **Hai model**, chỉ khác nhau ở `flip_idx` dùng cho augmentation lật ngang:
  - *giải phẫu*: `[0, 1, 2, 3, 7, 6, 5, 4, 10, 11, 8, 9]` (hoán đổi từng cặp `left_*` ↔ `right_*`);
  - *đồng nhất*: `[0, 1, …, 11]` như YAML gốc của tiger-pose.
- **Hai tập val**: val gốc (53 ảnh) và val lật gương (cùng 53 ảnh lật ngang, nhãn chuyển theo quy ước giải phẫu) để giả lập hổ quay trái.
- Dữ liệu gốc: train 210/0 và val 53/0 (quay phải / quay trái), tức **không có con hổ nào quay trái**.

## 2. Kết quả 4C (σ = 1/12 cho mọi keypoint, mặc định của Ultralytics)

| Model | Pose mAP50-95 — val gốc | Pose mAP50-95 — val lật gương | Thay đổi |
|---|---:|---:|---:|
| `flip_idx` giải phẫu | 0.436 | 0.425 | −0.011 |
| `flip_idx` đồng nhất | 0.415 | 0.275 | −0.140 (≈ −34%) |

Chi tiết của model đồng nhất:

| Tập val | Box mAP50-95 | Pose P | Pose R | Pose mAP50 | Pose mAP50-95 |
|---|---:|---:|---:|---:|---:|
| val gốc | 0.908 | 0.999 | 1.000 | 0.995 | 0.415 |
| val lật gương | 0.882 | 0.923 | 0.925 | 0.877 | 0.275 |

## 3. Metric nào đã che lỗi?

**Pose mAP trên val gốc.** Trên tập này model đồng nhất chỉ kém model giải phẫu 0.021 (0.415 so với 0.436), Pose mAP50 của cả hai
đều là 0.995 và Box mAP50-95 gần như bằng nhau (0.908 so với 0.914). Nhìn riêng bảng val gốc thì không có dấu hiệu nào của bug.

Lý do: mọi ảnh train và val đều có hổ quay phải, nên chân phía camera luôn mang nhãn `right_*`. Với `flip_idx` đồng nhất, ảnh lật
trong lúc train vẫn gọi chân phía camera là "right". Model vì thế học quy tắc "chân gần camera = right" thay vì trái/phải giải phẫu,
và quy tắc sai đó khớp với toàn bộ 53 ảnh val.

Val lật gương phá vỡ sự trùng hợp này: model đồng nhất tụt 0.140 điểm mAP50-95, còn model giải phẫu gần như không đổi (−0.011).
Box mAP của model đồng nhất chỉ giảm nhẹ (0.908 → 0.882), nghĩa là nó vẫn tìm đúng con hổ; thứ sai là **tên** trái/phải của keypoint.

## 4. Bài tập về nhà 1 — σ tự ước lượng cho từng keypoint

Cách ước lượng: `σ_i = sqrt(mean(d_i² / area))`, với `d_i` là sai số của model giải phẫu trên 53 ảnh val (mục 4B) và `area` là diện
tích object. Khai báo qua `kpt_oks_sigmas` trong YAML rồi chấm lại, không train lại.

| Nhóm keypoint | σ ước lượng | So với 1/12 ≈ 0.083 |
|---|---|---|
| nose, head, withers, tail_base | 0.034 – 0.045 | chặt hơn khoảng 2 lần |
| front wrist | 0.102 – 0.120 | lỏng hơn một chút |
| hind hock | 0.163 – 0.172 | lỏng hơn khoảng 2 lần |
| 4 paw | 0.250 – 0.290 | lỏng hơn khoảng 3 lần |

Pose mAP50-95:

| Model | val gốc, σ = 1/12 | val gốc, σ ước lượng | val lật, σ = 1/12 | val lật, σ ước lượng |
|---|---:|---:|---:|---:|
| `flip_idx` giải phẫu | 0.436 | 0.794 | 0.425 | 0.763 |
| `flip_idx` đồng nhất | 0.415 | 0.767 | 0.275 | 0.616 |

Nhận xét:

1. **Mọi con số tăng vọt dù model không đổi** (0.436 → 0.794). σ được ước lượng trên chính tập đem chấm, nên keypoint nào model sai
   nhiều thì được nới lỏng nhiều. Đây là hệ quả của cách ước lượng, không phải model tốt lên, và không so sánh được với mAP ở mục 2.
2. **σ lỏng ở chân che bớt lỗi `flip_idx`.** Trên val lật gương, Pose mAP50 của model đồng nhất quay về 0.995 (từ 0.877) và mức tụt
   tương đối của mAP50-95 giảm từ 34% xuống 20% (0.767 → 0.616). Lỗi đảo trái/phải nằm đúng ở các keypoint chân, là nhóm vừa được nới σ.
3. **mAP50-95 vẫn thấy lỗi.** Khoảng cách tuyệt đối giữa hai model trên val lật gương gần như giữ nguyên: 0.150 với σ = 1/12 và
   0.147 với σ ước lượng.

**Giới hạn.** σ đúng nghĩa đo độ lệch giữa nhiều người gán nhãn cùng một ảnh (cách COCO làm). Ở đây tôi không có nhãn lặp nên dùng
sai số của model làm xấp xỉ; con số này trộn lẫn độ mơ hồ của keypoint với điểm yếu của model, và có tính vòng lặp như nhận xét 1.

## 5. Tôi sẽ thiết kế tập val thế nào

1. **Đủ cả hai hướng quay.** Tối thiểu luôn chấm kèm một bản val lật gương; tốt hơn là thu thêm ảnh hổ quay trái thật.
2. **Báo cáo theo nhóm keypoint**, tách trái/phải khỏi trục giữa, kèm tỉ lệ ảnh "đổi nhãn trái ↔ phải thì OKS cao hơn" (9/53 ở mục
   4B). Một con số mAP gộp không chỉ ra kiểu lỗi này.
3. **Chia train/val theo video, không theo frame.** Các frame liền nhau của cùng một đoạn phim gần như giống hệt nhau, nên val hiện
   tại đo khả năng nhớ nhiều hơn khả năng tổng quát.
4. **Không lấy mAP50 làm metric chính** cho keypoint, và không dùng σ ước lượng từ sai số model để so với kết quả của người khác.
