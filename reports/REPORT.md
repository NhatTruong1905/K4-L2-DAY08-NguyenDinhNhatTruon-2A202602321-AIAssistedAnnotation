# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Nguyễn Đinh Nhật Trường

Công cụ gán nhãn đã dùng: CVAT

---

## 1. Dữ liệu và cách chia tập

Tập chưa gán nhãn (pool) và tập kiểm thử (test set) được phân chia theo trục thời gian (temporal split) kèm theo một vùng đệm (buffer zone) ở giữa, thay vì chia ngẫu nhiên (random split).

Nguyên nhân cốt lõi là do dữ liệu được thu thập từ một camera giám sát cố định tại một vị trí bên đường. Trong video liên tục này, một phương tiện khi lưu thông qua tầm nhìn của camera sẽ tồn tại trong nhiều khung hình liên tiếp kéo dài vài giây. Nếu phân chia ngẫu nhiên:
- Cùng một chiếc xe (với cùng góc nhìn, điều kiện ánh sáng và bối cảnh) sẽ xuất hiện đồng thời ở cả tập huấn luyện (pool được chọn) và tập kiểm thử (test set).
- Hiện tượng rò rỉ dữ liệu theo thời gian (temporal data leakage) sẽ xảy ra, khiến mô hình chỉ đơn thuần "học vẹt" hoặc ghi nhớ đặc trưng của những chiếc xe cụ thể đó thay vì học được khả năng khái quát hóa (generalization) cho các phương tiện chưa từng thấy.

Hệ quả là các số đo đánh giá trên tập kiểm thử (như AP50, Precision, Recall) sẽ bị **lệch lạc nghiêm trọng theo chiều hướng lạc quan giả tạo (over-optimistic)**. Điểm số đo đạc sẽ rất cao và đẹp mắt trên giấy tờ, nhưng khi triển khai thực tế ở các thời điểm hoặc bối cảnh khác thì mô hình sẽ hoạt động kém hiệu quả. Việc chia theo trục thời gian và tạo vùng đệm thời gian cách biệt đảm bảo tính độc lập khách quan của tập kiểm thử.

---

## 2. Mô hình khởi đầu lạnh (cold start)

Số đo của mô hình khởi đầu lạnh (Vòng 0) được trích xuất từ [reports/rounds_table.md](file:///d:/K4-DAY08-NguyenDinhNhatTruon-2A202602321/reports/rounds_table.md) và [outputs/metrics_round0.json](file:///d:/K4-DAY08-NguyenDinhNhatTruon-2A202602321/outputs/metrics_round0.json):

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Dựa vào ảnh trực quan hóa [outputs/compare_round0.jpg](file:///d:/K4-DAY08-NguyenDinhNhatTruon-2A202602321/outputs/compare_round0.jpg) và số liệu độ phủ (recall) theo kích thước xe:
- **Xe kích thước nhỏ (`R small = 0.182`)**: Độ phủ cực kỳ thấp, mô hình bỏ sót tới hơn 81% số xe nhỏ. Đây chủ yếu là các xe chạy ở khoảng cách xa, trong bối cảnh ban đêm chỉ lộ ra hai chấm đèn đỏ hoặc vệt sáng nhỏ mờ, thân xe chìm hoàn toàn vào màn đêm. Mô hình khởi đầu lạnh (vốn học từ tập COCO với ảnh ban ngày rõ nét) gần như không thể nhận biết được các xe này.
- **Xe kích thước vừa (`R medium = 0.547`) và lớn (`R large = 0.561`)**: Đạt độ phủ trên 54% - 56%. Mô hình nhận diện tương đối tốt các xe ở cự ly gần và trung bình khi thân xe và đèn xe nhìn thấy rõ ràng. Tuy nhiên, một số xe bị cắt ở viền ảnh hoặc xe tải lớn có hình dạng đặc thù vẫn bị vẽ khung lệch hoặc bỏ sót.

**Trường hợp cần rà soát lại nhãn tham chiếu:**
Nhãn của tập kiểm thử (`data/test/labels`) cũng được sinh tự động bởi mô hình mạnh hơn (pseudo-labels), chưa qua kiểm duyệt thủ công từng khung hình bởi con người. Trong ảnh ban đêm, nhiều vệt ánh sáng đèn pha phản chiếu trên mặt đường ướt hoặc các biển báo giao thông phát sáng dễ bị gán nhầm thành box xe (False Positive của nhãn tham chiếu). Nếu mô hình khởi đầu lạnh không dự đoán box vào vệt sáng đó, nó sẽ bị tính phạt là False Negative một cách oan uổng. Do đó, cần có chuyên viên người thật rà soát lại nhãn tham chiếu trước khi vội vàng kết luận mô hình dự đoán sai.

---

## 3. Chiến lược chọn mẫu

Chiến lược chọn mẫu học chủ động sử dụng công thức tính điểm tổng hợp:
$$\text{score} = W_U \cdot U + W_A \cdot A + W_D \cdot D$$

Ý nghĩa các thành phần:
- **$U$ (Uncertainty, trọng số $W_U = 0.5$)**: Đo lường mức độ bất định/phân vân của mô hình trên toàn ảnh (dựa trên entropy hoặc độ chênh lệch confidence của các dự đoán). Trọng số 0.5 thể hiện đây là yếu tố quan trọng nhất.
- **$A$ (Ambiguity / Density of uncertain boxes, trọng số $W_A = 0.3$)**: Đo lường tỷ lệ các bounding box mà mô hình dự đoán ở mức độ lưỡng lự (confidence nằm trong khoảng không chắc chắn, cần người can thiệp).
- **$D$ (Diversity / Temporal distance, trọng số $W_D = 0.2$)**: Đảm bảo tính đa dạng và phân bổ đều của các khung hình theo trục thời gian trong toàn bộ video.
- **Vai trò của `MIN_GAP_S` (2.0 giây)**: Vì camera đặt cố định, hai ảnh chụp cách nhau dưới 2 giây có góc nhìn, bối cảnh và vị trí các xe gần như trùng lặp hoàn toàn. `MIN_GAP_S` đóng vai trò bộ lọc chống dư thừa (redundancy filter), buộc thuật toán phải bỏ qua các ảnh quá sát nhau về thời gian dù có điểm bất định cao, giúp tiết kiệm tối đa công sức gán nhãn của con người.

**Chứng minh từ [reports/SELECTION.md](file:///d:/K4-DAY08-NguyenDinhNhatTruon-2A202602321/reports/SELECTION.md) và [outputs/selection_round1.csv](file:///d:/K4-DAY08-NguyenDinhNhatTruon-2A202602321/outputs/selection_round1.csv):**
- Ba frame được chọn trong lô 12 ảnh:
  1. `frame_0182.jpg` (hạng 1, điểm 0.9591, thời điểm 72.8s): $U = 0.9182, A = 1.0$, có tới 18/28 box ambiguous.
  2. `frame_0099.jpg` (hạng 8, điểm 0.9063, thời điểm 39.6s): $U = 0.9460, A = 0.7778$, có 14/29 box ambiguous.
  3. `frame_0107.jpg` (hạng 14, điểm 0.8876, thời điểm 42.8s): $U = 0.8752, A = 0.8333$, có 15/33 box ambiguous, cách frame_0099 hơn 3.2 giây.
- Một frame khác bị loại trừ dù điểm rất cao: `frame_0372.jpg` (hạng 6, điểm 0.9101, thời điểm 148.8s). Điểm của frame này cao hơn nhiều frame được chọn (như frame_0187 hạng 10 hay frame_0107 hạng 14), nhưng bị loại vì thời điểm 148.8s của nó chỉ cách `frame_0369.jpg` (hạng 2, t=147.6s) đúng 1.2 giây (dưới ngưỡng 2 giây). Việc loại bỏ này ngăn chặn lãng phí tài nguyên gán nhãn vào cảnh trùng.

**Điểm bất định có chứng minh ảnh đó sẽ cải thiện mô hình không?**
Không. Điểm bất định cao chỉ phản ánh việc mô hình đang lúng túng hoặc phân vân trước các đặc trưng của ảnh đó. Nếu nguyên nhân gây bất định là do ảnh bị nhòe chuyển động quá nặng, ánh đèn pha rọi thẳng gây chói lóa, hoặc bối cảnh chứa nhiều đối tượng nhiễu không theo quy chuẩn, thì việc nạp các ảnh này vào huấn luyện thậm chí có thể gây hại (overfitting vào nhiễu, làm giảm độ khái quát hóa trên tập kiểm thử).

---

## 4. Các vòng học chủ động (active learning)

Bảng tổng hợp từ [reports/rounds_table.md](file:///d:/K4-DAY08-NguyenDinhNhatTruon-2A202602321/reports/rounds_table.md):

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 338 | 0.836 | +0.064 | 1.000 | 0.189 | 0.317 | 0.000 | 0.179 | 0.561 |

**1. Mức độ sửa nhãn gợi ý (từ [outputs/round1_diff.md](file:///d:/K4-DAY08-NguyenDinhNhatTruon-2A202602321/outputs/round1_diff.md)):**
Trong lô 12 ảnh đầu tiên, mô hình pre-label ban đầu chỉ đề xuất 169 box. Sau khi được kiểm duyệt và gán nhãn thủ công trên CVAT, tổng số box chuẩn xác đạt **338 box** (tăng gấp đôi). Chi tiết các thao tác:
- **Giữ nguyên (`accepted`)**: 137 box (chiếm 81% số box gợi ý hợp lệ).
- **Chỉnh sửa viền (`edited`)**: 13 box (chủ yếu kéo dài viền ôm khít thân xe và đèn xe).
- **Xóa bỏ (`deleted`)**: 19 box (loại bỏ các box ảo dự đoán vào vệt đèn phản chiếu hoặc khung trùng lặp).
- **Thêm mới (`added`)**: 188 box (bổ sung toàn bộ các xe ở xa hoặc xe chìm trong bóng tối bị AI bỏ sót).

**2. Biến thiên AP50 và sự thay đổi giữa các nhóm xe:**
- **AP50**: Tăng từ `0.771` lên `0.836` (+0.064, tức tăng 6.4% so với cold start). Điều này chứng minh 12 ảnh gán nhãn chất lượng cao đã giúp mô hình thích nghi mạnh mẽ với miền dữ liệu ban đêm (domain adaptation).
- **Độ chính xác (Precision@0.25)**: Tăng vọt từ `0.925` lên tuyệt đối **`1.000`** (không còn bất kỳ False Positive nào trên toàn bộ 20 ảnh test).
- **Nhóm xe lớn (`R large`)**: Giữ vững độ phủ `0.561` nhưng ranh giới bounding box khít và chuẩn hơn.
- **Nhóm xe nhỏ (`R small`) và vừa (`R medium`)**: Tại ngưỡng tin cậy cố định 0.25, độ phủ giảm (R small về 0.000, R medium về 0.179). Sau khi được huấn luyện với các nhãn làm sạch (xóa các vệt sáng phản chiếu), mô hình trở nên cực kỳ cẩn trọng và khắt khe: nó triệt để từ chối đưa ra dự đoán ở những vùng mập mờ, chỉ phát hiện khi có bằng chứng chắc chắn về hình khối thân xe.

**3. Phân tích hình ảnh [outputs/compare_round0.jpg](file:///d:/K4-DAY08-NguyenDinhNhatTruon-2A202602321/outputs/compare_round0.jpg) và [outputs/compare_round1.jpg](file:///d:/K4-DAY08-NguyenDinhNhatTruon-2A202602321/outputs/compare_round1.jpg):**
Sau fine-tune Vòng 1, các khung chữ nhật ảo bị gán nhầm trên mặt đường ướt hoặc cột đèn phản chiếu ở Vòng 0 đã biến mất hoàn toàn. Các xe chạy ở làn giữa và làn gần được phát hiện với bounding box gọn gàng, ôm trọn thân xe theo đúng chuẩn.

**4. Đối chiếu ba nguồn quan sát:**
- *Quan sát độc lập bằng mắt ([reports/BLIND_SCAN.md](file:///d:/K4-DAY08-NguyenDinhNhatTruon-2A202602321/reports/BLIND_SCAN.md))*: Khi chưa xem gợi ý của AI trên `frame_0099.jpg`, mắt người đếm được 21 xe, xác định rõ 2 khu vực khó: xe ở xa chỉ thấy một bóng đèn đỏ nhỏ mờ, và xe ở sát mép bị mất nét viền thân xe trong bóng tối.
- *Nhật ký sửa nhãn ([reports/REVIEW_LOG.csv](file:///d:/K4-DAY08-NguyenDinhNhatTruon-2A202602321/reports/REVIEW_LOG.csv))*: Ghi nhận 3 can thiệp tiêu biểu: thêm box cho xe ở xa có đèn hậu đỏ (`added` trên frame_0099.jpg), xóa vệt đèn phản chiếu giữa đường (`deleted` trên frame_0107.jpg), và kéo lại chiều cao box ôm khít xe ở mép phải màn hình (`edited` trên frame_0187.jpg).
- *Kết quả sau huấn luyện*: Mô hình học được chính xác sự phân biệt này: triệt tiêu hoàn toàn lỗi nhận nhầm vệt đèn (Precision đạt 1.0), tuy nhiên vẫn chưa tự tin bắt được các xe nhỏ xíu ở xa chỉ có 1 chấm đèn.

**Mô tả ca khó theo guideline:**
Trường hợp xe ở xa có kích thước nhỏ hơn 16 pixel (chỉ còn 2 chấm đèn li ti). Theo [GUIDELINE_LABEL.md](file:///d:/K4-DAY08-NguyenDinhNhatTruon-2A202602321/GUIDELINE_LABEL.md), các box này được phép gán hoặc bỏ qua tùy ý và không tính điểm phạt trong tập kiểm thử. Việc phân định giữa 2 chấm đèn xe ở xa với các đốm sáng phản chiếu ven đường là ranh giới mong manh nhất trong bài toán gán nhãn ban đêm.

---

## 5. Kết luận và giới hạn

**1. Kết luận và quyết định:**
Sau vòng 1, AP50 tăng từ `0.771` lên `0.836` (+0.064) và Precision đạt tuyệt đối `1.000`. Tuy nhiên, Recall tổng thể bị giảm tại ngưỡng conf=0.25 (mô hình quá thận trọng với xe nhỏ).
- **Quyết định: TIẾP TỤC VÒNG 2.**
- **Lý do**: Mô hình đã giải quyết xuất sắc bài toán chống dương tính giả (FP = 0), nhưng cần được bổ sung thêm dữ liệu huấn luyện ở cự ly xa và trung bình để cải thiện Recall mà không làm suy giảm Precision.

**2. Đề xuất hai ca còn yếu cho vòng sau:**
1. *Ca 1: Xe ở khoảng cách xa trong bóng tối (chỉ lộ đèn hậu và vệt mờ thân xe)*: Cần thêm các ảnh có xe chạy ở làn xa nhưng vẫn còn nhận diện được đường nét cơ bản quanh cụm đèn. Chi phí rà nhãn ở nhóm này cao vì người gắn nhãn phải phóng to và căn chỉnh thủ công; cần lưu ý nguy cơ trùng cảnh nếu chọn các frame liên tiếp nhau dưới 2 giây.
2. *Ca 2: Xe bị cắt ở viền ảnh hoặc bị xe phía trước che khuất một phần (occlusion / boundary truncation)*: Cần chọn các khung hình có xe đang tiến vào hoặc rời khỏi góc khung hình, tuân thủ đúng quy tắc chỉ vẽ box cho phần thân xe nhìn thấy được.

**3. Giới hạn của bài toán và ảnh hưởng đến kết luận:**
- Tập kiểm thử chỉ có 20 ảnh (với 403 box tham chiếu). Kích thước tập test nhỏ đồng nghĩa với việc phương sai đánh giá (evaluation variance) còn lớn; sự xuất hiện hay biến mất của một vài box trên tập test có thể làm dao động đáng kể chỉ số Recall.
- Quy tắc bỏ qua xe dưới 16 pixel và việc nhãn tham chiếu test được sinh tự động bằng mô hình (chưa qua chuẩn hóa 100% bởi con người) khiến cho một số trường hợp mô hình thực sự phát hiện đúng nhưng lại không khớp với nhãn máy, hoặc ngược lại. Do đó, các chỉ số số học chỉ mang tính tham chiếu định hướng, không thể xem là thước đo chân lý tuyệt đối.

**4. Hành động nếu AP50 giảm:**
Nếu ở vòng tiếp theo AP50 bị suy giảm, tôi sẽ thực hiện các bước kiểm tra sau trước khi tiếp tục huấn luyện:
1. Mở file `outputs/roundX_diff.md` và `REVIEW_LOG.csv` để rà soát chất lượng nhãn người đã sửa: kiểm tra xem có trường hợp gán nhãn thiếu nhất quán (ví dụ cùng một loại đốm sáng mà ảnh này khoanh, ảnh khác lại bỏ) làm mô hình bị học nhiễu hay không.
2. Kiểm tra lại danh sách các ảnh được chọn để đảm bảo khoảng cách thời gian `min_gap_s >= 2.0s`, tránh tình trạng dữ liệu huấn luyện bị overfit cục bộ vào một cụm thời gian duy nhất.
3. Rà soát lại việc điều chỉnh bounding box ở các góc mép ảnh để đảm bảo không vi phạm các điều khoản trong [GUIDELINE_LABEL.md](file:///d:/K4-DAY08-NguyenDinhNhatTruon-2A202602321/GUIDELINE_LABEL.md).
