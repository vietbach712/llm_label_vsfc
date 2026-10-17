# Kết quả

Chỉ số: macro_f1, chấm trên tập test nhãn người.

## Câu 1: Nhãn LLM tệ hơn nhãn người bao nhiêu?

- Trên pool train (4000 câu): LLM khớp nhãn người acc = 0.940, macro-F1 = 0.832, kappa = 0.888, INVALID = 0.

### phobert

| n | nhãn người | nhãn LLM | chênh lệch |
|---|---|---|---|
| 250 | 0.559 | 0.559 | -0.000 |
| 500 | 0.596 | 0.593 | -0.003 |
| 1000 | 0.619 | 0.619 | -0.000 |
| 2000 | 0.681 | 0.771 | +0.090 |
| 4000 | 0.784 | 0.784 | +0.000 |

- Câu 2 (phobert): 1.000 nhãn người cho 0.619; nhãn LLM cần ≈ 1001 mẫu.
- Câu 3 (phobert): student học 4000 nhãn LLM = 0.784, teacher = 0.795 → student CHƯA vượt teacher (-0.010).

### tfidf_lr

| n | nhãn người | nhãn LLM | chênh lệch |
|---|---|---|---|
| 250 | 0.573 | 0.570 | -0.003 |
| 500 | 0.599 | 0.591 | -0.008 |
| 1000 | 0.621 | 0.601 | -0.020 |
| 2000 | 0.638 | 0.617 | -0.021 |
| 4000 | 0.671 | 0.648 | -0.023 |

- Câu 2 (tfidf_lr): 1.000 nhãn người cho 0.621; nhãn LLM cần ≈ 2177 mẫu.
- Câu 3 (tfidf_lr): student học 4000 nhãn LLM = 0.648, teacher = 0.795 → student CHƯA vượt teacher (-0.146).
