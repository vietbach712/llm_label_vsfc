# LLM gán nhãn → huấn luyện mô hình nhỏ trên UIT-VSFC

Project hai thành viên nghiên cứu khả năng dùng nhãn do LLM tạo ra để huấn luyện mô hình phân loại cảm xúc tiếng Việt. Nhóm so sánh **TF-IDF + Logistic Regression** và **PhoBERT** khi học từ nhãn người hoặc nhãn LLM, trên cùng tập test có nhãn người.

Teacher là **`gpt-6-luna`**, gọi qua OpenAI API theo cấu hình nhóm sử dụng. Mỗi câu được gán một trong ba nhãn `negative`, `neutral`, `positive`. Student học từ các nhãn rời rạc này; thí nghiệm không sử dụng xác suất/logit của teacher hay hàm mất mát KL để distill.

**Kết quả nổi bật:** teacher đạt macro-F1 **0.7946** trên test. Với 4.000 mẫu train, PhoBERT học nhãn LLM đạt **0.7842 ± 0.0078**, gần mức học nhãn người **0.7840 ± 0.0176**. TF-IDF học nhãn LLM đạt **0.6485**, thấp hơn mức học nhãn người **0.6715**. Chưa có kết quả student học nhãn LLM trong các cấu hình đã thử vượt teacher theo macro-F1.

README tổng hợp bộ kết quả được đối chiếu ngày **17/10/2026**. Điểm số dùng thang 0–1; `±` là **độ lệch chuẩn mẫu giữa các seed**, không phải khoảng tin cậy. Khi chỉ có một seed, chỉ báo cáo điểm, không gán độ lệch chuẩn bằng 0.

## 1. Câu hỏi nghiên cứu

1. Nhãn LLM đồng thuận với nhãn người đến mức nào? Student bị ảnh hưởng ra sao khi thay nhãn người bằng nhãn LLM?
2. Cần khoảng bao nhiêu nhãn LLM để đạt điểm của student học từ 1.000 nhãn người?
3. Student học từ 4.000 nhãn LLM có vượt teacher LLM trên cùng tập test không?

Thí nghiệm bổ sung khảo sát hai cách tận dụng nhãn: **lọc các câu mà hai prompt đồng ý** (`llm_agree`) và **kết hợp 500 nhãn người với nhãn LLM** (`mix500`).

## 2. Dữ liệu và quy tắc đánh giá

Nhóm dùng bài toán phân loại **cảm xúc** của [UIT-VSFC trên Hugging Face](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback), một bộ phản hồi sinh viên tiếng Việt. Nhãn chủ đề của bộ dữ liệu không được sử dụng trong project này.

Tập con cố định có **5.300 câu**. Phân bố dưới đây được đối chiếu từ `support` trong [báo cáo theo lớp](results/teacher_classification_report.txt):

| Tập | Negative | Neutral | Positive | Tổng | Vai trò |
| --- | ---: | ---: | ---: | ---: | --- |
| Train | 1.864 | 160 | 1.976 | 4.000 | Pool để lấy mẫu huấn luyện student |
| Dev | 134 | 14 | 152 | 300 | Thử và chọn prompt teacher |
| Test | 445 | 53 | 502 | 1.000 | Đánh giá teacher và toàn bộ student bằng nhãn người |

Neutral chỉ chiếm **4,00% train**, **4,67% dev** và **5,30% test**. Vì vậy, chỉ số chính là **macro-F1**, trung bình F1 của ba lớp với trọng số ngang nhau. Accuracy và Cohen’s kappa được báo cáo bổ sung.

Thiết kế lấy dữ liệu quy định seed 42, lấy train/dev/test tương ứng từ các split gốc. File `data/data.csv` đã tạo được giữ cố định trong suốt project; mọi thành viên dùng chung file này.

Các quy tắc thực nghiệm:

- Chọn prompt bằng **macro-F1 trên dev**, không dùng test để chọn hoặc sửa prompt.
- Với cùng seed và cỡ mẫu, `human` và `llm` dùng cùng các câu train; các tập mẫu nhỏ được lấy lồng trong các tập lớn theo thiết kế trong `common.py`.
- Test luôn dùng **nhãn người**. Nhãn LLM của test chỉ phục vụ đo teacher, không được dùng để huấn luyện student.
- Nhãn người là chuẩn tham chiếu khi tính điểm, nhưng một số trường hợp vẫn có thể mơ hồ hoặc cần rà soát.
- Thí nghiệm `llm_agree` chọn một tập con khác sau khi lọc; cần xét cả thay đổi thành phần dữ liệu khi diễn giải kết quả.

## 3. Teacher và lựa chọn prompt

### 3.1. Cấu hình và đầu ra

| Thành phần | Giá trị nhóm sử dụng |
| --- | --- |
| API | OpenAI |
| Base URL | `https://api.openai.com/v1` |
| Model | `gpt-6-luna` |
| API key | Biến môi trường `LLM_API_KEY` |
| Prompt | `v1`, `v2` |
| File nhãn | `data/llm_v1.csv`, `data/llm_v2.csv` |
| File bàn giao cho student | `data/llm_final.csv`, lấy từ v2 |

Theo nhật ký A2, mỗi prompt đã gán đủ **5.300 câu**, `INVALID = 0`. Các câu trả lời được cache trong `llm_cache/` để dùng lại. Chỉ kết quả của teacher Luna được dùng trong README này.

Trong thiết kế ban đầu, v1 là hướng dẫn phân loại cơ bản; v2 bổ sung quy ước gán nhãn và ví dụ, bao gồm cách xử lý câu góp ý. Đây là mô tả ý tưởng của prompt. **Nguyên văn prompt cuối cùng phải lấy từ các file prompt/`label_llm.py` đã dùng để tạo nhãn**, không suy ra lại từ ví dụ hoặc điểm số. Bộ tài liệu kết quả dùng để biên soạn README chưa kèm nguyên văn các file đó.

### 3.2. Kết quả dev dùng để chọn prompt

| Prompt | Số câu | INVALID | Accuracy | Macro-F1 | Kappa |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 | 300 | 0 | 0.8433 | 0.7135 | 0.7279 |
| **v2** | **300** | **0** | **0.9100** | **0.7578** | **0.8350** |

Nhóm chọn **v2** vì macro-F1 dev tăng **0.0443**, tương đương **4,43 điểm phần trăm**. Bảng được đối chiếu với nhật ký A1; kết quả dev của v2 cũng khớp `teacher.csv`.

Nhật ký A2 còn ghi nhận v1 đạt macro-F1 **0.7460 trên train** và **0.7254 trên test** (accuracy lần lượt 0.8775, 0.8580; kappa 0.7799, 0.7459). Đây là kết quả mô tả sau khi chạy, không phải căn cứ chọn prompt.

### 3.3. Chất lượng teacher v2

Nguồn: [teacher.csv](results/teacher.csv), [a3_report.txt](results/a3_report.txt).

| Tập | Số câu | INVALID | Accuracy | Macro-F1 | Kappa |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 4.000 | 0 | 0.9395 | 0.8318 | 0.8882 |
| Dev | 300 | 0 | 0.9100 | 0.7578 | 0.8350 |
| Test | 1.000 | 0 | 0.9190 | 0.7946 | 0.8521 |

Trên train, LLM khớp nhãn người ở **3.758/4.000 câu**; còn **242 câu bất đồng**. Trên test, teacher đúng theo nhãn tham chiếu ở **919/1.000 câu**.

Kết quả theo lớp trên test:

| Lớp | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Negative | 0.919 | 0.964 | 0.941 | 445 |
| Neutral | 0.510 | 0.491 | 0.500 | 53 |
| Positive | 0.963 | 0.924 | 0.943 | 502 |

**Neutral là điểm yếu rõ nhất:** F1 chỉ đạt 0.584 trên train, 0.400 trên dev và 0.500 trên test. Trong 53 câu neutral của test, teacher dự đoán đúng 26 câu, nhầm 14 câu thành negative và 13 câu thành positive.

![Ma trận nhầm lẫn của teacher trên train](figs/confusion_train.png)

![Ma trận nhầm lẫn của teacher trên test](figs/confusion_test.png)

Hàng là nhãn người, cột là nhãn LLM. Hai ma trận khớp số mẫu và chỉ số trong báo cáo teacher.

## 4. Student và phạm vi thí nghiệm đã hoàn thành

### 4.1. Các nguồn nhãn

| `source` | Cách tạo tập huấn luyện |
| --- | --- |
| `human` | Dùng nhãn người của các câu train được chọn |
| `llm` | Dùng nhãn teacher v2 của cùng các câu |
| `llm_agree` | Chỉ lấy các câu mà v1 và v2 cho cùng nhãn, rồi dùng nhãn đó để train |
| `mix500` | Trong tổng số n câu, 500 câu dùng nhãn người, n − 500 câu dùng nhãn LLM |

Đồng ý giữa hai prompt không bảo đảm nhãn đúng. Hai prompt dùng cùng một teacher nên có thể cùng mắc một lỗi. `mix500` giữ nguyên tổng số mẫu n và thay nguồn nhãn của 500 câu, không cộng thêm 500 câu ngoài tập n.

### 4.2. Số lượt chạy được xác nhận từ CSV

| Mô hình / nhóm thí nghiệm | Cỡ mẫu | Seed | Số lượt |
| --- | --- | --- | ---: |
| TF-IDF, `human` và `llm` | 250, 500, 1.000, 2.000, 4.000 | 0, 1, 2, 3, 4 | 50 |
| TF-IDF, `llm_agree` | 250, 500, 1.000, 2.000 | 0, 1, 2, 3, 4 | 20 |
| TF-IDF, `mix500` | 1.000, 2.000, 4.000 | 0, 1, 2, 3, 4 | 15 |
| PhoBERT, `human` và `llm` | 500, 1.000, 4.000 | 0, 1, 2 | 18 |
| PhoBERT mở rộng, `human` và `llm` | 250, 2.000 | 0 | 4 |
| PhoBERT, `llm_agree` và `mix500` | 1.000, 2.000 | 0 | 4 |
| **Tổng** | | | **111** |

Có **85 dòng** trong [tfidf.csv](results/tfidf.csv) và **26 dòng** trong [phobert_B.csv](results/phobert_B.csv). Không trùng khóa `(model, source, n_target, seed)`; `n` bằng `n_target` ở mọi dòng.

Seed 1 đã được Việt Bách chạy và gộp vào `phobert_B.csv`, theo nhật ký. Vì vậy bộ kết quả này không cần một bản sao seed 1 trong `phobert_A.csv`.

[check_tfidf.csv](results/check_tfidf.csv) và [check_phobert.csv](results/check_phobert.csv) là lượt kiểm tra ban đầu, **không cộng vào 111 lượt chính và bổ sung**. Các điểm kiểm tra khớp cấu hình tương ứng trong kết quả chính. Ở n = 500, seed 0, PhoBERT đạt 0.6051 và TF-IDF đạt 0.6070; không áp đặt giả định PhoBERT phải tốt hơn ở mọi cỡ mẫu.

### 4.3. Thiết lập mô hình và mức độ tái lập

Theo mã mẫu trong tài liệu hướng dẫn, TF-IDF dùng unigram/bigram, `sublinear_tf=True`, `min_df=1`; Logistic Regression dùng `C=4.0`, `class_weight="balanced"`, `max_iter=2000`. PhoBERT dùng `vinai/phobert-base`, đầu vào tách từ, 4 epoch, learning rate `2e-5`, batch size 32 và max length 128; optimizer AdamW, weight decay 0.01, warmup 10% và gradient clipping 1.0.

**Đây là cấu hình tham chiếu từ tài liệu hướng dẫn, chưa phải bản ghi cấu hình đã xác minh của từng lượt chạy.** CSV và nhật ký xác nhận mô hình, nguồn nhãn, cỡ mẫu, seed và điểm số nhưng không lưu đầy đủ các siêu tham số này. Cần đối chiếu mã `.py`/notebook thực tế trước khi khẳng định tái lập chính xác. Đặc biệt, chưa có bản khóa phiên bản thư viện, revision model, log thiết bị hay log epoch trong bộ tài liệu đối chiếu.

Các cột `minutes` trong CSV PhoBERT chỉ phản ánh thời gian do script ghi cho từng lượt. Project chưa có phép đo đầy đủ để kết luận về chi phí hoặc tốc độ teacher so với student.

## 5. Kết quả thí nghiệm chính

Trung bình và độ lệch chuẩn được tính từ các điểm macro-F1 đã lưu trong CSV (mỗi điểm đã làm tròn 4 chữ số). Cột chênh lệch là **LLM trừ nhãn người** trên thang 0–1.

### 5.1. TF-IDF + Logistic Regression

| Số mẫu train | Số seed / nguồn | Nhãn người | Nhãn LLM | LLM − người |
| --- | --- | --- | --- | --- |
| 250 | 5 | 0.5733 ± 0.0122 | 0.5699 ± 0.0138 | -0.0034 |
| 500 | 5 | 0.5990 ± 0.0135 | 0.5907 ± 0.0207 | -0.0083 |
| 1,000 | 5 | 0.6209 ± 0.0133 | 0.6008 ± 0.0152 | -0.0201 |
| 2,000 | 5 | 0.6381 ± 0.0075 | 0.6170 ± 0.0159 | -0.0211 |
| 4,000 | 5 | 0.6715 ± 0.0000 | 0.6485 ± 0.0000 | -0.0230 |

Ở 4.000 mẫu, học nhãn LLM thấp hơn nhãn người **0.0230 macro-F1**, tức **2,30 điểm phần trăm**. Cả năm seed cho điểm giống nhau ở cỡ mẫu này. Điều này phù hợp với việc dùng toàn bộ pool và mô hình gần như không đổi theo thứ tự mẫu; độ lệch chuẩn bằng 0 trong file không có nghĩa mô hình không có bất định trên dữ liệu mới.

![Learning curve của TF-IDF](figs/curve_tfidf_lr.png)

### 5.2. PhoBERT

| Số mẫu train | Số seed / nguồn | Nhãn người | Nhãn LLM | LLM − người |
| --- | --- | --- | --- | --- |
| 250 | 1 | 0.5594 | 0.5591 | -0.0003 |
| 500 | 3 | 0.5958 ± 0.0087 | 0.5929 ± 0.0178 | -0.0028 |
| 1,000 | 3 | 0.6188 ± 0.0027 | 0.6185 ± 0.0027 | -0.0002 |
| 2,000 | 1 | 0.6807 | 0.7708 | +0.0901 |
| 4,000 | 3 | 0.7840 ± 0.0176 | 0.7842 ± 0.0078 | +0.0002 |

Ở 4.000 mẫu, trung bình `llm` và `human` gần bằng nhau: **0.784200 so với 0.784033**, chênh khoảng **0.00017**. Đây chưa phải bằng chứng nhãn LLM tốt hơn. Chênh lệch theo từng seed còn đổi dấu: seed 0 là +0.0065, seed 1 là −0.0116 và seed 2 là +0.0056.

Ở 2.000 mẫu, `llm` đạt 0.7708 và `human` đạt 0.6807, nhưng cả hai chỉ có **seed 0**. Cần thêm seed trước khi khái quát mức chênh này.

![Learning curve của PhoBERT](figs/curve_phobert.png)

Trên biểu đồ, điểm tròn rỗng biểu thị cấu hình chỉ có một seed. Các đường nối giúp quan sát xu hướng giữa các cỡ mẫu đã thử; chúng không bổ sung các thí nghiệm mới ở khoảng giữa.

### 5.3. Trả lời ba câu hỏi nghiên cứu

**Câu 1 — Nhãn LLM ảnh hưởng thế nào?** Teacher khớp 93,95% nhãn người trên pool train, kappa 0.8882. Ở 4.000 mẫu, TF-IDF học nhãn LLM giảm 2,30 điểm phần trăm macro-F1 so với học nhãn người; PhoBERT cho trung bình gần bằng nhau trong ba seed. Tác động của nguồn nhãn phụ thuộc vào mô hình student, không thể suy ra chỉ từ accuracy của teacher.

**Câu 2 — Cần bao nhiêu nhãn LLM để đạt mức 1.000 nhãn người?**

| Mô hình | Mốc human ở n = 1.000 | Ước lượng nội suy trong summary | Cỡ mẫu LLM đã chạy nhỏ nhất đạt mốc trung bình |
| --- | ---: | --- | --- |
| TF-IDF | 0.62088 | Khoảng 2.177 mẫu, khoảng 2,18 lần | 4.000 mẫu, đạt 0.64850 |
| PhoBERT | 0.61877 | Khoảng 1.001 mẫu, xấp xỉ 1 lần | 2.000 mẫu, đạt 0.77080 với seed 0 |

`make_report.py` nội suy tuyến tính điểm số theo **log(số mẫu)** giữa hai mức đã thử. Các giá trị 2.177 và 1.001 **không phải cỡ mẫu đã được huấn luyện trực tiếp**. Đặc biệt, PhoBERT tại 1.000 mẫu LLM đã đạt 0.61853, chỉ thấp hơn mốc human khoảng 0.00023. Vì vậy, diễn giải phù hợp là **xấp xỉ 1.000 nhãn trong thí nghiệm này**, không phải chính xác cần thêm một nhãn. Nội suy PhoBERT còn ghép điểm ba seed ở n = 1.000 với điểm một seed ở n = 2.000, nên chỉ có giá trị mô tả.

**Câu 3 — Student có vượt teacher không?**

| Mô hình học từ 4.000 nhãn LLM | Macro-F1 trung bình | Teacher | Student − teacher |
| --- | ---: | ---: | ---: |
| TF-IDF | 0.6485 | 0.7946 | −0.1461 |
| PhoBERT | 0.7842 | 0.7946 | −0.0104 |

Cả hai chưa vượt teacher trong kết quả hiện có. PhoBERT gần teacher hơn, cách **1,04 điểm phần trăm macro-F1**. Các so sánh này là mô tả thực nghiệm, chưa phải kiểm định ý nghĩa thống kê.

Nguồn tổng hợp: [summary.md](results/summary.md). README giữ thêm chữ số và số seed để giải thích các chênh lệch bị làm tròn thành `+0.000` hoặc `−0.000` trong summary.

## 6. Thí nghiệm bổ sung B4

### 6.1. TF-IDF: năm seed cho mỗi cấu hình

| Số mẫu train | LLM | LLM agree | Mix 500 |
| --- | --- | --- | --- |
| 250 | 0.5699 ± 0.0138 | 0.5707 ± 0.0082 | — |
| 500 | 0.5907 ± 0.0207 | 0.5885 ± 0.0157 | — |
| 1,000 | 0.6008 ± 0.0152 | 0.5996 ± 0.0129 | 0.6120 ± 0.0159 |
| 2,000 | 0.6170 ± 0.0159 | 0.6226 ± 0.0134 | 0.6247 ± 0.0142 |
| 4,000 | 0.6485 ± 0.0000 | — | 0.6518 ± 0.0130 |

Lọc đồng thuận không cải thiện đều ở mọi cỡ mẫu. Ở n = 2.000, `llm_agree` cao hơn `llm` khoảng **0.0056** macro-F1; ở n = 1.000 thì thấp hơn một chút. `mix500` đạt **0.6518** ở n = 4.000, tăng **0.0033** so với `llm`, nhưng vẫn thấp hơn `human` 0.6715. Mức tăng trung bình này nhỏ so với độ dao động của `mix500` giữa các seed.

Dấu `—` nghĩa là không có kết quả cho cấu hình đó. CSV không có `llm_agree` ở 4.000 mẫu. Mã mẫu bỏ qua cỡ mẫu lớn hơn pool đồng thuận, nhưng tài liệu hiện có không ghi chính xác số câu đồng thuận nên không suy đoán con số đó. `mix500` chỉ được thử với n > 500.

### 6.2. PhoBERT: so sánh cùng seed 0

| Số mẫu train | Human | LLM | LLM agree | Mix 500 |
| --- | --- | --- | --- | --- |
| 1,000 | 0.6217 | 0.6210 | 0.6192 | 0.6211 |
| 2,000 | 0.6807 | 0.7708 | 0.7781 | 0.7590 |

Bảng này dùng **seed 0 cho cả baseline và B4** để so sánh cùng điều kiện, thay vì lấy baseline trung bình ba seed. Ở n = 2.000, `llm_agree` tăng 0.0073 so với `llm`, còn `mix500` giảm 0.0118. Chỉ có một seed nên chưa thể kết luận các thay đổi này ổn định.

**Vì sao summary B3 và B4 giống nhau?** Phần bảng và ba câu hỏi trong mã `make_report.py` mẫu chỉ so sánh `human` với `llm`. Thêm `llm_agree`/`mix500` không đổi các dòng đó. Kết quả B4 vẫn có trong CSV và được thể hiện ở bảng riêng này cùng learning curve.

## 7. Phân tích 60 câu bất đồng

Nhóm phân loại thủ công **60 câu trong 242 câu train mà teacher khác nhãn người**. Theo thiết kế A3, mẫu bất đồng được lấy với `random_state=0`. File đã gán mã khớp 60 ID và nội dung của file mẫu; không có ID trùng hoặc mã trống.

Nguồn: [llm_errors_tagged.csv](results/llm_errors_tagged.csv).

| Mã | Số câu | Tỉ lệ trong 60 câu | Diễn giải |
| --- | --- | --- | --- |
| KHAC | 29 | 48.33% | Các bất đồng còn lại, cần giải thích theo từng câu |
| TRUNG_TINH | 23 | 38.33% | Ranh giới giữa mô tả sự việc và đánh giá cảm xúc |
| NGUOI_SAI | 3 | 5.00% | Nghi vấn nhãn người chưa phù hợp |
| HON_HOP | 3 | 5.00% | Có cả ý khen và ý chê |
| GOP_Y | 2 | 3.33% | Khác cách diễn giải câu góp ý |
| MIA_MAI | 0 | 0.00% | Mỉa mai hoặc nói ngược |

Đây là tỉ lệ **trong mẫu 60 câu bất đồng**, không phải tỉ lệ lỗi của toàn bộ 5.300 câu. Nhóm `KHAC` lớn nhất, tiếp theo là ranh giới trung tính. Với kết quả cuối này, chưa có cơ sở nói câu góp ý là nguồn lỗi phổ biến nhất.

### Bốn ví dụ tiêu biểu

| ID | Câu phản hồi | Người | LLM | Mã | Nhận xét |
| --- | --- | --- | --- | --- | --- |
| `train_01874` | giảng viên rất đảm bảo giờ lên lớp . | neutral | positive | TRUNG_TINH | Câu mô tả việc bảo đảm giờ lên lớp cũng có thể được đọc như một lời khen. Đây là bất đồng về ranh giới trung tính/tích cực. |
| `train_02466` | mong cô sẽ dạy các môn liên quan trực tiếp tới lập trình như csharp , java . | positive | negative | KHAC | Mong tiếp tục học với cô có thể thể hiện sự yêu thích. Nhãn negative có thể đến từ việc diễn giải lời mong muốn thành yêu cầu cải thiện; đây là một giả thuyết khi đọc câu. |
| `train_01063` | thầy dạy nhiệt tình , tốt nhưng không phù hợp với tôi . | neutral | negative | HON_HOP | Vừa khen giảng viên nhiệt tình, tốt, vừa nói không phù hợp với bản thân. Câu thể hiện hai chiều đánh giá. |
| `train_01890` | giảng dạy không tập trung vào chuyên môn . | positive | negative | NGUOI_SAI | Nội dung mang sắc thái phàn nàn nhưng nhãn gốc là positive. Đánh dấu để rà soát; chưa sửa nhãn gốc và chưa coi đây là lỗi đã được xác minh. |

Các nhận xét trên là diễn giải định tính của nhóm khi đọc câu; không phải lời giải thích được teacher trả về. Ba trường hợp `NGUOI_SAI` là **nghi vấn cần kiểm tra**, chưa đủ để kết luận nhãn gốc sai hoặc đổi nhãn tham chiếu.

File gán mã hiện chỉ có năm cột và chưa có cột `note`. Vì 29 câu thuộc `KHAC`, lần rà soát tiếp theo nên bổ sung giải thích từng câu và thống nhất tiêu chí giữa hai người. README giữ nguyên các mã đã chốt, không tái phân loại để làm thay đổi bảng kết quả.

## 8. Các file chính

| File / thư mục | Vai trò |
| --- | --- |
| `common.py` | Đọc/gộp dữ liệu, lấy mẫu, tính điểm và ghi kết quả |
| `prepare_data.py` | Tạo tập con cố định ban đầu |
| `label_llm.py` và các file prompt | Gán nhãn teacher, dùng cache |
| `analyze_labels.py` | Đánh giá teacher, vẽ confusion matrix, xuất mẫu bất đồng |
| `train_tfidf.py` | Huấn luyện TF-IDF + Logistic Regression |
| `train_phobert.py` | Huấn luyện PhoBERT |
| `make_report.py` | Tổng hợp CSV, tạo learning curve và summary |
| `data/data.csv` | Dữ liệu cố định: `id`, `text`, `human_label`, `split` |
| `data/llm_v1.csv`, `data/llm_v2.csv`, `data/llm_final.csv` | Nhãn teacher theo ID |
| `llm_cache/` | Câu trả lời đã cache |
| `results/` | Kết quả teacher, student, phân tích lỗi và summary |
| `figs/` | Confusion matrix và learning curve |

Các CSV student có `model`, `source`, `n_target`, `n`, `seed`, `acc`, `macro_f1`, `kappa`; PhoBERT có thêm `minutes`. Mỗi dòng tương ứng một cấu hình huấn luyện và đánh giá.

## 9. Cài đặt và chạy lại

### 9.1. Môi trường

Nhóm dùng Python 3.10 trên Windows cho phần teacher/TF-IDF và Google Colab cho PhoBERT. Hướng dẫn gốc chọn GPU T4; log thiết bị của các lượt chạy chưa nằm trong bộ kết quả này.

Clone repo từ nút **Code** trên GitHub, mở terminal tại thư mục repo, rồi chạy trong Windows CMD:

```bat
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Nếu đã có `.venv`, chỉ cần kích hoạt. Giữ file `data/data.csv` được cung cấp trong repo; `prepare_data.py` là bước tạo dữ liệu một lần ở đầu project, không chạy lại trên bộ dữ liệu đã chốt.

### 9.2. Tái tạo báo cáo từ CSV có sẵn

Đây là cách nhanh nhất để xem lại thí nghiệm, không cần gọi API hoặc huấn luyện lại:

```bat
python make_report.py
```

Đầu ra:

- `results/summary.md`
- `figs/curve_tfidf_lr.png`
- `figs/curve_phobert.png`

Mã mẫu đọc các file `results/tfidf*.csv` và `results/phobert*.csv`, vì vậy không đặt bản sao/backup cùng tiền tố trong thư mục này. File `check_*.csv` không thuộc mẫu đọc đó.

Các hình đi kèm README được vẽ lại từ cùng CSV và phân biệt điểm một seed với điểm nhiều seed. Nếu chạy lại script gốc, cách trình bày hình có thể khác; số liệu CSV và số seed mới là căn cứ diễn giải.

Có thể kiểm tra cấu hình trùng bằng lệnh chỉ đọc dữ liệu:

```bat
python -c "import glob,pandas as pd; fs=sorted(glob.glob('results/tfidf*.csv')+glob.glob('results/phobert*.csv')); assert fs,'Khong tim thay ket qua student'; d=pd.concat([pd.read_csv(f) for f in fs],ignore_index=True); k=['model','source','n_target','seed']; print('So dong:',len(d)); print('Cau hinh trung:',d.duplicated(k).sum()); print(d.groupby(['model','source','n_target']).seed.nunique().to_string())"
```

### 9.3. Đánh giá lại teacher từ nhãn đã lưu

```bat
python analyze_labels.py --llm_file data/llm_final.csv
```

Lệnh đọc nhãn đã có để tính điểm, không gọi LLM. Nó ghi lại `teacher.csv`, các confusion matrix và file mẫu `llm_errors_sample.csv`. Kết quả phân loại thủ công được giữ riêng trong `llm_errors_tagged.csv`.

### 9.4. Huấn luyện lại student

Các lệnh dưới đây dùng giao diện script trong hướng dẫn dự án. Trước khi chạy, đối chiếu siêu tham số với bản code/notebook đã tạo ra kết quả công bố. Chúng ghi vào **`runs/reproduction/`** để tách lần chạy mới khỏi kết quả đã lưu trong `results/`.

Chỉ bắt đầu một lần tái lập mới khi hai file output dưới đây chưa tồn tại. Nếu chạy bị ngắt, kiểm tra các cấu hình đã hoàn thành và chỉ chạy phần còn thiếu: script ghi nối thêm, không tự tránh dòng trùng.

TF-IDF chính và bổ sung, trên CMD:

```bat
python train_tfidf.py --sources human llm --sizes 250 500 1000 2000 4000 --seeds 0 1 2 3 4 --out runs/reproduction/tfidf.csv
python train_tfidf.py --llm_file data/llm_final.csv --llm_file2 data/llm_v1.csv --sources llm_agree mix500 --sizes 250 500 1000 2000 4000 --seeds 0 1 2 3 4 --out runs/reproduction/tfidf.csv
```

PhoBERT, sau khi notebook Colab đã vào đúng thư mục repo và cài môi trường tương ứng:

```python
!nvidia-smi
!python train_phobert.py --sources human llm --sizes 500 1000 4000 --seeds 0 1 2 --out runs/reproduction/phobert_B.csv
!python train_phobert.py --sources human llm --sizes 250 2000 --seeds 0 --out runs/reproduction/phobert_B.csv
!python train_phobert.py --llm_file2 data/llm_v1.csv --sources llm_agree mix500 --sizes 1000 2000 --seeds 0 --out runs/reproduction/phobert_B.csv
```

`nvidia-smi` kiểm tra GPU. Script PhoBERT mẫu tạo `data/data_seg.csv` khi cần tách từ; cần giữ cùng cách tách từ giữa các lượt. Kết quả trên Colab cần lưu ở Drive hoặc tải về sau khi chạy.

`make_report.py` gốc chỉ đọc `results/`, không tự đọc `runs/reproduction/`. Để lập báo cáo cho lần tái lập, dùng một bản repo riêng và đặt **chỉ các CSV của lần chạy đó** vào `results/`, cùng `teacher.csv` tương ứng. Không gộp hai lần chạy cùng cấu hình như thể chúng là các seed độc lập.

### 9.5. Gán nhãn lại khi cần tạo một lần chạy teacher mới

Với bộ kết quả hiện tại, dùng nhãn/cache có sẵn. Khi cần tái tạo teacher, giữ đúng model, dữ liệu và hai prompt của lần chạy đã chốt. Trên CMD mới, key được đặt qua biến môi trường:

```bat
set "LLM_API_KEY=KEY_CUA_BAN"
```

Thay giá trị minh họa bằng key riêng trên máy; không lưu key vào code, README hay Git. Quy trình chọn prompt trong một thí nghiệm mới là:

```bat
python label_llm.py --prompt v1 --split dev
python label_llm.py --prompt v2 --split dev
```

Sau khi chọn bằng dev và khóa prompt, gán train/test:

```bat
python label_llm.py --prompt v1 --split train test
python label_llm.py --prompt v2 --split train test
```

Project này chọn v2 nên file bàn giao là bản sao `llm_v2.csv` thành `llm_final.csv`. Cache phải thuộc đúng model/prompt của lần chạy; chỉ gọi API cho câu còn thiếu. Cần đọc phiên bản `label_llm.py` thực tế để xác nhận cơ chế nhận diện cache trước khi chạy lại. Model/API có thể thay đổi theo thời gian nên việc gọi lại không bảo đảm nhãn giống từng câu.

## 10. Hạn chế và hướng cải thiện

- **Phạm vi nhỏ:** một tập con của một dataset tiếng Việt và một teacher; chưa thể khái quát sang lĩnh vực khác.
- **Lớp neutral ít:** chỉ có 14 mẫu dev và 53 mẫu test, nên đánh giá lớp này dễ biến động. Accuracy cao vẫn có thể đi cùng F1 neutral thấp.
- **Số seed không đồng đều:** PhoBERT chính có ba seed, nhưng các điểm 250/2.000 và B4 chỉ có một seed. Cần bổ sung seed để đánh giá độ ổn định.
- **Chưa kiểm định thống kê:** độ lệch chuẩn giữa seed chỉ mô tả biến động; không chứng minh hai phương án tương đương hay một phương án tốt hơn có ý nghĩa thống kê.
- **Nội suy còn thô:** các mốc 1.001/2.177 chưa được huấn luyện trực tiếp và phụ thuộc giả định nội suy theo log số mẫu.
- **Phân tích lỗi còn chủ quan:** 60 câu bất đồng chưa có đo mức đồng thuận giữa người phân loại lỗi; nhóm `KHAC` còn rộng và thiếu ghi chú từng câu.
- **Thiếu metadata tái lập:** chưa có nguyên văn prompt cuối, bản khóa thư viện, đầy đủ cấu hình thực chạy, log GPU/epoch trong bộ tài liệu tổng hợp. Token, chi phí API và tổng thời gian teacher cũng chưa được ghi đầy đủ, nên chưa có kết luận định lượng về tiết kiệm chi phí.
- **Khác tập mẫu khi lọc:** hiệu quả `llm_agree` có thể đến từ cả chất lượng nhãn lẫn thay đổi độ khó/phân bố câu được chọn.

Các hướng tiếp theo là rà soát ranh giới neutral/góp ý, bổ sung ghi chú lỗi, chạy thêm seed cho PhoBERT, lưu dự đoán test để phân tích lỗi student và đo chi phí thực tế. Mọi điều chỉnh prompt hoặc siêu tham số tiếp theo cần dùng dev hoặc một quy trình validation được xác định trước; tập test hiện tại đã được quan sát trong báo cáo này.

## 11. Đóng góp

| Thành viên | Công việc được ghi nhận |
| --- | --- |
| **Thái Bình (A)** | Thử và chọn prompt; gán nhãn teacher; bàn giao nhãn v2; đánh giá teacher; phân loại 60 câu bất đồng và tổng hợp phần phân tích lỗi |
| **Việt Bách (B)** | Tạo repo và dữ liệu ban đầu; thiết lập TF-IDF/PhoBERT; chạy thí nghiệm chính, bao gồm seed 1; chạy B4; tổng hợp kết quả student bằng `make_report.py` |

Phân công phản ánh nhật ký thực tế. Seed 1 nằm trong kết quả của B, dù kế hoạch ban đầu dự kiến giao A hỗ trợ chạy.

## 12. Nguồn dữ liệu và tài liệu tham khảo

- [UIT-VSFC dataset card](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback). Nguyen và cộng sự, *UIT-VSFC: Vietnamese Students’ Feedback Corpus for Sentiment Analysis*, KSE 2018. DOI: `10.1109/KSE.2018.8573337`.
- [PhoBERT model card](https://huggingface.co/vinai/phobert-base). Dat Quoc Nguyen và Anh Tuan Nguyen, *PhoBERT: Pre-trained language models for Vietnamese*, Findings of EMNLP 2020; [arXiv:2003.00744](https://arxiv.org/abs/2003.00744).
- Nguồn kết quả: các CSV trong `results/`, `a3_report.txt`, `teacher_classification_report.txt`, hai confusion matrix và nhật ký thực nghiệm của nhóm.
- Quy trình tham chiếu: tài liệu “Hướng dẫn project LLM gán nhãn → train mô hình nhỏ (2 người, 3 ngày)”. Mặc định trong tài liệu được phân biệt với thông tin đã xác nhận từ kết quả thực chạy.

Dataset, mô hình tiền huấn luyện và mã nguồn phụ thuộc giữ điều kiện sử dụng của nguồn tương ứng. README này không tự gán giấy phép cho repo.
