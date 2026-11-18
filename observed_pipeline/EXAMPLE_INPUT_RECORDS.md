# Ví dụ bản ghi đầu vào

Tài liệu này chỉ minh họa cấu trúc và cách các dòng dữ liệu liên kết với nhau. Các giá trị bên dưới là ví dụ, không phải dữ liệu thực nghiệm và không dùng làm input để chạy pipeline.

## Bảng lượt thi (`attempts.csv`)

Mười dòng ví dụ dưới đây minh họa nhiều lượt của cùng một học sinh, nhiều kỹ năng, độ khó và điểm mục tiêu khác nhau.

| attempt_key | student_key | assessment_key | typeExam | mode | difficulty_level | started_at | submitted_at | assessment_duration_minutes | real_score | target_score |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| ATT-DEMO-001 | STU-DEMO-01 | ASM-READ-01 | READING | FULL_TEST | Medium | 2026-01-05T08:00:00+07:00 | 2026-01-05T09:00:00+07:00 | 60 | 5.5 | 6.0 |
| ATT-DEMO-002 | STU-DEMO-01 | ASM-LIST-01 | LISTENING | FULL_TEST | Easy | 2026-01-08T08:00:00+07:00 | 2026-01-08T08:40:00+07:00 | 40 | 6.0 | 6.0 |
| ATT-DEMO-003 | STU-DEMO-02 | ASM-WRIT-01 | WRITING | FULL_TEST | Hard | 2026-01-09T13:00:00+07:00 | 2026-01-09T14:00:00+07:00 | 60 | 6.5 | 7.0 |
| ATT-DEMO-004 | STU-DEMO-01 | ASM-READ-02 | READING | PASSAGE_PRACTICE | Medium | 2026-01-12T08:00:00+07:00 | 2026-01-12T08:25:00+07:00 | 25 | 6.0 | 6.0 |
| ATT-DEMO-005 | STU-DEMO-02 | ASM-SPEAK-01 | SPEAKING | FULL_TEST | Medium | 2026-01-14T15:00:00+07:00 | 2026-01-14T15:15:00+07:00 | 15 | 6.5 | 6.5 |
| ATT-DEMO-006 | STU-DEMO-01 | ASM-LIST-02 | LISTENING | FULL_TEST | Hard | 2026-01-17T09:00:00+07:00 | 2026-01-17T09:40:00+07:00 | 40 | 5.5 | 6.0 |
| ATT-DEMO-007 | STU-DEMO-03 | ASM-READ-03 | READING | FULL_TEST | Easy | 2026-01-18T10:00:00+07:00 | 2026-01-18T11:00:00+07:00 | 60 | 7.0 | 6.5 |
| ATT-DEMO-008 | STU-DEMO-02 | ASM-WRIT-02 | WRITING | FULL_TEST | Medium | 2026-01-20T13:00:00+07:00 | 2026-01-20T14:00:00+07:00 | 60 | 7.0 | 7.0 |
| ATT-DEMO-009 | STU-DEMO-01 | ASM-READ-04 | READING | FULL_TEST | Hard | 2026-01-23T08:00:00+07:00 | 2026-01-23T09:00:00+07:00 | 60 | 6.5 | 6.0 |
| ATT-DEMO-010 | STU-DEMO-03 | ASM-SPEAK-01 | SPEAKING | FULL_TEST | Easy | 2026-01-25T15:00:00+07:00 | 2026-01-25T15:15:00+07:00 | 15 | 6.0 | 6.5 |

Các cột tùy chọn như `duration_seconds`, `total_score`, `max_score` và `ielts_band_score` được lược khỏi bảng để dễ đọc; chúng có thể được thêm vào CSV thật theo [schema đầu vào](INPUT_DATASET_SCHEMA.md).

## Bảng sự kiện AP (`ap_events.csv`)

| student_key | event_at | event_type | scope | amount | reference_type | reference_key |
| --- | --- | --- | --- | ---: | --- | --- |
| STU-DEMO-01 | 2026-01-06T10:00:00+07:00 | ATTEMPT_COMPLETED | OVERALL | 10 | Attempt | ATT-DEMO-001 |
| STU-DEMO-01 | 2026-01-07T10:00:00+07:00 | IMPROVEMENT | OVERALL | 15 | AttemptImprovement | ATT-DEMO-001 |
| STU-DEMO-01 | 2026-01-08T09:00:00+07:00 | ATTEMPT_COMPLETED | OVERALL | 10 | Attempt | ATT-DEMO-002 |
| STU-DEMO-01 | 2026-01-13T10:00:00+07:00 | ATTEMPT_COMPLETED | OVERALL | 10 | Attempt | ATT-DEMO-004 |
| STU-DEMO-01 | 2026-01-18T10:00:00+07:00 | IMPROVEMENT | OVERALL | 15 | AttemptImprovement | ATT-DEMO-006 |
| STU-DEMO-02 | 2026-01-10T10:00:00+07:00 | ATTEMPT_COMPLETED | OVERALL | 10 | Attempt | ATT-DEMO-003 |
| STU-DEMO-02 | 2026-01-15T10:00:00+07:00 | IMPROVEMENT | OVERALL | 15 | AttemptImprovement | ATT-DEMO-005 |
| STU-DEMO-02 | 2026-01-21T10:00:00+07:00 | ATTEMPT_COMPLETED | OVERALL | 10 | Attempt | ATT-DEMO-008 |
| STU-DEMO-03 | 2026-01-19T10:00:00+07:00 | ATTEMPT_COMPLETED | OVERALL | 10 | Attempt | ATT-DEMO-007 |
| STU-DEMO-03 | 2026-01-26T10:00:00+07:00 | IMPROVEMENT | OVERALL | 15 | AttemptImprovement | ATT-DEMO-010 |

Ví dụ: khi xử lý `ATT-DEMO-009`, các lượt thi và sự kiện AP của `STU-DEMO-01` có thời điểm hoàn thành trước `2026-01-23T08:00:00+07:00` có thể được dùng làm lịch sử. Những sự kiện xảy ra sau mốc bắt đầu đó không được tính cho lượt này.
