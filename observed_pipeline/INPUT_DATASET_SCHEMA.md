# Cấu trúc dữ liệu đầu vào

Pipeline nhận hai tệp CSV được mã hóa UTF-8. Tên tệp có thể thay đổi; truyền đúng đường dẫn bằng `--attempts-csv` và `--ap-events-csv` khi chạy.

## 1. Bảng lượt thi

Mỗi dòng là một lượt thi. Các cột đánh dấu **Bắt buộc** phải có trong tệp.

| Cột | Kiểu dữ liệu | Yêu cầu | Ý nghĩa / định dạng |
| --- | --- | --- | --- |
| `attempt_key` | chuỗi | Bắt buộc | Mã duy nhất của lượt thi. |
| `student_key` | chuỗi | Bắt buộc | Mã tài khoản học sinh; dùng để ghép lịch sử và sự kiện AP. |
| `assessment_key` | chuỗi | Bắt buộc | Mã bài thi hoặc bộ đề. |
| `typeExam` | chuỗi | Bắt buộc | Kỹ năng/loại bài thi, ví dụ `READING`, `LISTENING`, `WRITING`, `SPEAKING`, `SAT`. Giá trị được chuẩn hóa thành chữ hoa. |
| `mode` | chuỗi | Bắt buộc | Chế độ làm bài, ví dụ `FULL_TEST` hoặc `PASSAGE_PRACTICE`. |
| `difficulty_level` | chuỗi | Bắt buộc | Mức độ bài thi, ví dụ `Easy`, `Medium`, `Hard`. |
| `started_at` | thời điểm | Bắt buộc | Thời điểm bắt đầu, khuyến nghị ISO 8601 có múi giờ, ví dụ `2026-10-01T09:00:00+07:00`. |
| `submitted_at` | thời điểm | Bắt buộc | Thời điểm nộp bài, cùng định dạng với `started_at`. |
| `assessment_duration_minutes` | số | Bắt buộc | Thời lượng quy định của bài thi, tính bằng phút. |
| `real_score` | số | Bắt buộc | Điểm thực tế dùng làm mục tiêu hồi quy và xác định đạt mục tiêu. |
| `target_score` | số | Bắt buộc | Điểm mục tiêu đã biết trước lượt thi. |
| `duration_seconds` | số | Tùy chọn | Thời gian làm bài thực tế, tính bằng giây. |
| `total_score` | số | Tùy chọn | Tổng điểm thô của lượt thi. |
| `max_score` | số | Tùy chọn | Điểm thô tối đa của lượt thi. |
| `ielts_band_score` | số | Tùy chọn | Band điểm hệ thống ghi nhận cho lượt thi. |

Các cột tùy chọn có thể để trống hoặc bỏ khỏi tệp. Những cột khác ngoài danh sách trên không được dùng trong bảng phân tích đầu ra.

## 2. Bảng sự kiện AP

Mỗi dòng là một sự kiện AP. Tất cả các cột dưới đây là **Bắt buộc**.

| Cột | Kiểu dữ liệu | Yêu cầu | Ý nghĩa / định dạng |
| --- | --- | --- | --- |
| `student_key` | chuỗi | Bắt buộc | Mã tài khoản phát sinh sự kiện; phải cùng hệ mã với bảng lượt thi. |
| `event_at` | thời điểm | Bắt buộc | Thời điểm sự kiện, định dạng ISO 8601 có múi giờ. |
| `event_type` | chuỗi | Bắt buộc | Loại sự kiện AP, chẳng hạn `IMPROVEMENT` hoặc loại được ghi trong nguồn dữ liệu. |
| `scope` | chuỗi | Bắt buộc | Phạm vi ghi nhận; dòng `OVERALL` được dùng để đếm và tính lượng AP. |
| `amount` | số | Bắt buộc | Lượng AP của sự kiện. |
| `reference_type` | chuỗi | Bắt buộc | Loại đối tượng tham chiếu; ví dụ `Attempt`, `AttemptImprovement`, `ShortVideoView`, `ShortVideoCompletion`. |
| `reference_key` | chuỗi | Bắt buộc | Mã đối tượng được tham chiếu, nếu có. |

## Quy tắc dữ liệu

- Hai bảng dùng chung `student_key` để nối dữ liệu tài khoản.
- `attempt_key` không được trùng. `started_at` và `submitted_at` cần có giá trị thời gian hợp lệ.
- Thời điểm nên có múi giờ để việc sắp xếp và xác định lịch sử trước lượt thi nhất quán.
- Các cột điểm và thời lượng phải chứa giá trị số hoặc ô trống.
- Những cột feature lịch sử/AP như `prior_same_skill_attempt_count` và `prior_ap_events_30d` là đầu ra tính từ hai bảng đầu vào, không phải cột bắt buộc của tệp nguồn.

## Lệnh chạy

```bash
python3 research/data_mining/observed_pipeline/run_pipeline.py \
  --attempts-csv /path/to/attempts.csv \
  --ap-events-csv /path/to/ap_events.csv \
  --output-dir /path/to/pipeline_output
```
