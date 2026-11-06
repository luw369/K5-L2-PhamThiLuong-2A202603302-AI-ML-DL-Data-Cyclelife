# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:** 3.13.16 / 2.11.0+cu130 / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` 

**Thay đổi so với notebook nguồn:** Không 

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468 ; `class_name`= cab ; `rank`=1 ; `score`= 0.510915 ; `taxonomy_name`= ImageNet-1K

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
=> Record mô tả toàn bộ bức ảnh vì task được xác định là image_classification, không có thông tin bounding box hay tọa độ của từng vật thể.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
=> Danh sách lớp được xác định bởi taxonomy và checkpoint đi kèm, không phải do model tự nghĩ ra. Ở record này thì taxonomy là ImageNet-1K.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
=> Vì mỗi trường cung cấp một loại thông tin khác nhau. class_id là mã định danh của lớp, class_name là tên tương ứng giúp con người dễ đọc và hiểu, còn taxonomy_name cho biết lớp đó thuộc hệ thống phân loại nào

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
=> Guideline cần quy định tiêu chí chọn một lớp đại diện, chẳng hạn chủ thể chính, mức độ nổi bật hoặc diện tích chiếm trong ảnh, đồng thời hướng dẫn xử lý khi có nhiều chủ thể ngang nhau

6. Vì sao không được lấy `score` của model làm ground truth?
=> Vì score chỉ thể hiện mức độ tự tin của model đối với dự đoán, không đảm bảo dự đoán

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`= bus ; `score`= 0.912558 ; `bbox_xyxy`=[ 93.17, 187.95, 223.01, 320.91] ; `bbox_width`= 129.84 ; `bbox_height`= 132.96
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
=> Góc trên-trái ở tọa độ (93.17, 187.95), góc dưới-phải ở tọa độ (223.01, 320.91), chiều rộng 129.84px, chiều cao 132.96px.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
=> Ngưỡng 0.2 có 17 prediction; ngưỡng 0.6 có 6 prediction

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
=> Khi hạ ngưỡng confidence, mô hình tạo ra nhiều prediction hơn, có thể xuất hiện các hộp dự đoán sai, trùng lặp hoặc không rõ đối tượng. Vì vậy, reviewer phải kiểm tra nhiều bounding box hơn, xác minh đối tượng có thực sự tồn tại hay không, loại bỏ các prediction sai và điều chỉnh lại nhãn hoặc vị trí hộp nếu cần thiết

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
=> Bounding box phải bao quanh toàn bộ phần đối tượng nhìn thấy được, không bao gồm các vật thể xung quanh. Nếu đối tượng bị che khuất, chỉ vẽ box theo phần đối tượng có thể quan sát được, không tự suy đoán phần bị che.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
=> Bowl bị cắt mép. Em sẽ đưa trường hợp này vào guideline nếu đây là tình huống thường gặp và có thể thống nhất cách gán nhãn. Quy tắc là chỉ vẽ bounding box bao quanh phần chiếc bát thực sự nhìn thấy được. Nếu trường hợp che khuất quá nhiều khiến không thể xác định chính xác đối tượng hoặc không thống nhất được cách gán nhãn, em sẽ escalation lên Lab Coach để được hướng dẫn, tránh việc mỗi người gán nhãn theo một cách khác nhau.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`= kitchen-001 ; `class_name`= person ; `score`= 0.899318 ; số điểm polygon=`polygon_point_count`= 348 ; 3–5 điểm đầu của `polygon_xy`= [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
=> Box là hình chữ nhật bao vật thể; polygon mô tả biên và hình dạng của instance sát hơn.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
=> instance_id nhận diện một đối tượng riêng trong output lab. Nó không định danh lớp như class ID, cũng không theo dõi cùng đối tượng qua nhiều khung hình như tracking ID.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
=> Vẽ polygon sát biên phần nhìn thấy của một instance; không gộp hai vật cùng lớp thành một mask

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
=> Rau ở mép trái ảnh bị khung hình cắt. Nếu guideline chưa nói mask chỉ bao phần nhìn thấy hay xử lý phần ngoài khung thế nào, nên escalation để thống nhất trước

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một lớp hoặc các lớp cho toàn ảnh, theo taxonomy và guideline | Top-1 của ảnh traffic là cab (score 0.510915). Cần kiểm tra lớp này có đúng tiêu chí chủ đề chính của guideline hay không; prediction không tự xác nhận ground truth | Chọn lớp theo tiêu chí chủ đề chính; hỏi khi ảnh mơ hồ hoặc guideline chưa quy định | Đối chiếu nhãn với toàn ảnh và guideline; không dùng model score làm điểm chất lượng |
| Phát hiện vật thể | Một class và một box [x_min, y_min, x_max, y_max] cho mỗi vật thể | kitchen có 17 prediction ở ngưỡng 0.20, 11 ở 0.35, 6 ở 0.60. Ảnh có nhiều vật nhỏ gần nhau; lò nướng ở mép trái bị cắt | Gán lớp và box theo quy tắc; đánh dấu ca bị che/cắt hoặc không rõ | Kiểm tra box trùng/sai, vật thể bị bỏ sót, độ sát box và các ca bị cắt |
| Instance segmentation | Một class và polygon/mask riêng cho mỗi instance | Output kitchen có 11 instance ở ngưỡng 0.35. Cây/lá bị cắt ở mép trái; các dụng cụ bếp treo gần nhau khiến một số ranh giới khó tách | Vẽ biên theo guideline cho từng instance; escalation nếu không rõ ranh giới hoặc quy tắc với vật thể bị cắt | Kiểm tra mask có theo biên, đúng instance và không gộp vật thể; yêu cầu rework khi cần |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
=> Chỉ sử dụng ảnh và dữ liệu được phép cho bài lab; không tải lên hoặc đưa vào báo cáo dữ liệu cá nhân, nội bộ hay nhạy cảm

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
=> Nếu gặp dữ liệu ngoài phạm vi, dừng xử lý và báo Lab Coach/giảng viên phụ trách

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
