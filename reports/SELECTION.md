# Vì sao chọn lô này?

Nếu chỉ có ngân sách rà 5 ảnh trong số 50 dòng đứng đầu `outputs/selection_round1.csv`, tôi chọn:
1. `frame_0182.jpg` (hạng 1, điểm 0.959, thời điểm 72.8s)
2. `frame_0369.jpg` (hạng 2, điểm 0.932, thời điểm 147.6s)
3. `frame_0380.jpg` (hạng 3, điểm 0.917, thời điểm 152.0s)
4. `frame_0326.jpg` (hạng 4, điểm 0.916, thời điểm 130.4s)
5. `frame_0331.jpg` (hạng 5, điểm 0.915, thời điểm 132.4s)

Năm ảnh này đứng đầu danh sách về độ bất định và phân bổ ở các cụm thời gian khác nhau. Về quyết định xét ảnh gần trùng: `frame_0187.jpg` (hạng 10, điểm 0.899, thời điểm 74.8s) chỉ cách `frame_0182.jpg` đúng 2.0 giây, cảnh vật và vị trí các xe gần như giống hệt nhau. Do đó, nếu ngân sách chỉ có 5 ảnh, tôi giữ lại `frame_0182.jpg` và không chọn `frame_0187.jpg` để tránh lãng phí công sức vào dữ liệu dư thừa, ưu tiên chọn các frame ở mốc thời gian khác để tăng tính đa dạng.

Trong 12 ảnh AI đã chọn, tôi phân tích ba ảnh: `frame_0182.jpg`, `frame_0099.jpg` và `frame_0107.jpg`. Cả ba đều có điểm tổng hợp rất cao (lần lượt là 0.959, 0.906 và 0.888). Bằng chứng từ CSV và ảnh contact sheet `outputs/selection_round1.jpg` cho thấy đây là những khung hình có mật độ xe dày đặc (từ 28 đến 33 box) nhưng AI có từ 14 đến 18 khung chưa chắc chắn (cột `n_ambiguous`), độ bất định `U` đều vượt trên 0.87.

Về trường hợp điểm cao nhưng không được chọn: `frame_0372.jpg` đứng hạng 6 với điểm 0.910, cao hơn nhiều ảnh trong lô được chọn (như `frame_0187.jpg` hạng 10 hay `frame_0107.jpg` hạng 14), nhưng AI đã bỏ qua vì thời điểm t=148.8s của nó chỉ cách `frame_0369.jpg` (hạng 2, t=147.6s) có 1.2 giây (dưới ngưỡng giãn cách thời gian tối thiểu 2 giây). Hai ảnh gần như cùng một cảnh, nên nếu chọn cả hai thì tốn công gán nhãn mà mô hình ít học thêm được thông tin mới.

Điểm số cao chỉ phản ánh rằng AI đang ở trạng thái phân vân (độ bất định cao). Việc lựa chọn theo tiêu chí này chưa chứng minh được rằng sau khi sửa xong nhãn cho các ảnh này thì chất lượng mô hình tổng thể chắc chắn sẽ tốt hơn trên tập kiểm thử độc lập.
