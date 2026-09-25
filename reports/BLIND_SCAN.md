# Quét độc lập trước khi xem pre-label

Frame: `frame_0099.jpg`

Số xe nhìn thấy bằng mắt: `21`

Hai vị trí dễ bị AI bỏ sót hoặc vẽ sai, kèm mô tả xe:
1. **Xe phía trước ở khoảng cách xa:** Chỉ thấy 1 bóng đèn đỏ nhỏ mờ ở xa, toàn bộ thân xe chìm trong bóng tối không thấy rõ biên dạng, AI rất dễ bỏ sót hoặc nhận diện sai.
2. **Xe ở sát viền mép ảnh:** Xe đã mất bóng ở đường viền (không rõ khung viền thân xe, chìm vào nền tối), rất khó xác định là xe hay vật thể khác ven đường, AI dễ bỏ sót hoặc xác định sai vùng bounding box.

Chạy `python3 tools/lock_blind.py` ngay sau khi điền. Sau đó giữ file này nguyên vẹn.
