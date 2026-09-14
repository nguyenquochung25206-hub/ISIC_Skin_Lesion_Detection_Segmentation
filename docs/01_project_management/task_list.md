# Danh sách công việc + thành viên

Trạng thái: `Chưa bắt đầu` | `Đang làm` | `Hoàn thành`

## TV1 — Quản lý project + README + báo cáo tổng hợp

| #  | Công việc                                     | File liên quan                     | Trạng thái |
| -- | --------------------------------------------- | ---------------------------------- | ---------- |
| 1  | Khởi tạo cấu trúc thư mục project             | (toàn bộ)                          | Hoàn thành |
| 2  | Viết README.md                                | `README.md`                        | Hoàn thành |
| 3  | Viết CHANGELOG.md, LICENSE                    | `CHANGELOG.md`, `LICENSE`          | Hoàn thành |
| 4  | Liệt kê thư viện, cấu hình .gitignore         | `requirements.txt`, `.gitignore`   | Hoàn thành |
| 5  | Lập kế hoạch tổng thể                         | `project_plan.md`                  | Hoàn thành |
| 6  | Lập danh sách công việc                       | `task_list.md`                     | Hoàn thành |
| 7  | Lập timeline                                  | `timeline.md`                      | Hoàn thành |
| 8  | Xác định yêu cầu chính đề tài                 | `project_requirements.md`          | Hoàn thành |
| 9  | Tổng hợp kết luận cuối project                | `docs/06_results/conclusion.md`    | Hoàn thành |
| 10 | Tổng hợp báo cáo kết quả cuối cùng (cùng TV7) | `results/reports/final_results.md` | Hoàn thành |

## TV2 — Dataset ISIC + tiền xử lý

| # | Công việc                            | File liên quan                                                              | Trạng thái |
| - | ------------------------------------ | --------------------------------------------------------------------------- | ---------- |
| 1 | Nghiên cứu xử lý ảnh y tế            | `docs/03_research/medical_image_processing.md`                              | Hoàn thành |
| 2 | Tìm hiểu dataset ISIC                | `docs/03_research/isic_dataset.md`                                          | Hoàn thành |
| 3 | Thu thập/tổ chức ảnh và mask gốc     | `data/input/`                                                               | Hoàn thành |
| 4 | Viết module tiền xử lý ảnh           | `src/preprocessing/image_preprocessor.py`                                   | Hoàn thành |
| 5 | Viết module data augmentation        | `src/preprocessing/data_augmentation.py`                                    | Hoàn thành |
| 6 | Khảo sát dataset (script + notebook) | `experiments/explore_dataset.py`, `notebooks/01_explore_isic_dataset.ipynb` | Hoàn thành |
| 7 | Notebook thử nghiệm tiền xử lý       | `notebooks/02_preprocessing.ipynb`                                          | Hoàn thành |
| 8 | Test tiền xử lý                      | `tests/test_preprocessing.py`                                               | Hoàn thành |

## TV3 — Faster R-CNN

| # | Công việc                         | File liên quan                                   | Trạng thái |
| - | --------------------------------- | ------------------------------------------------ | ---------- |
| 1 | Nghiên cứu lý thuyết Faster R-CNN | `docs/03_research/faster_rcnn.md`                | Hoàn thành |
| 2 | Sơ đồ pipeline Faster R-CNN       | `docs/04_design/diagrams/detection_pipeline.png` | Hoàn thành |
| 3 | Xây dựng mô hình Faster R-CNN     | `src/detection/faster_rcnn.py`                   | Hoàn thành |
| 4 | Huấn luyện mô hình                | `src/detection/train.py`                         | Hoàn thành |
| 5 | Dự đoán bounding box              | `src/detection/predict.py`                       | Hoàn thành |
| 6 | Script chạy train detection       | `experiments/train_detection.py`                 | Hoàn thành |
| 7 | Notebook thử nghiệm Faster R-CNN  | `notebooks/03_faster_rcnn.ipynb`                 | Hoàn thành |
| 8 | Phân tích kết quả detection       | `docs/06_results/detection_results.md`           | Hoàn thành |

## TV4 — Đánh giá Detection + visualization Detection

| # | Công việc                              | File liên quan                                                       | Trạng thái |
| - | -------------------------------------- | -------------------------------------------------------------------- | ---------- |
| 1 | Viết module đánh giá Detection         | `src/detection/evaluate.py`                                          | Hoàn thành |
| 2 | Tính Precision, Recall, mAP            | `src/evaluation/detection_metrics.py`                                | Hoàn thành |
| 3 | Script đánh giá detection              | `experiments/evaluate_detection.py`                                  | Hoàn thành |
| 4 | Hiển thị bounding box                  | `src/visualization/display_detection.py`                             | Hoàn thành |
| 5 | Test module Detection                  | `tests/test_detection.py`                                            | Hoàn thành |
| 6 | Tổng hợp hình + bảng kết quả detection | `results/figures/detection/`, `results/tables/detection_metrics.csv` | Hoàn thành |

## TV5 — U-Net

| # | Công việc                      | File liên quan                                      | Trạng thái |
| - | ------------------------------ | --------------------------------------------------- | ---------- |
| 1 | Nghiên cứu lý thuyết U-Net     | `docs/03_research/unet.md`                          | Hoàn thành |
| 2 | Sơ đồ pipeline U-Net           | `docs/04_design/diagrams/segmentation_pipeline.png` | Hoàn thành |
| 3 | Xây dựng mô hình U-Net         | `src/segmentation/unet.py`                          | Hoàn thành |
| 4 | Huấn luyện mô hình             | `src/segmentation/train.py`                         | Hoàn thành |
| 5 | Dự đoán mask                   | `src/segmentation/predict.py`                       | Hoàn thành |
| 6 | Script chạy train segmentation | `experiments/train_segmentation.py`                 | Hoàn thành |
| 7 | Notebook thử nghiệm U-Net      | `notebooks/04_unet.ipynb`                           | Hoàn thành |

## TV6 — Đánh giá Segmentation + visualization

| #  | Công việc                         | File liên quan                              | Trạng thái |
| -- | --------------------------------- | ------------------------------------------- | ---------- |
| 1  | Kiến trúc tổng thể hệ thống       | `docs/04_design/system_architecture.md`     | Hoàn thành |
| 2  | Quy trình xử lý (pipeline)        | `docs/04_design/processing_pipeline.md`     | Hoàn thành |
| 3  | Sơ đồ tổng thể hệ thống           | `docs/04_design/diagrams/system_flow.png`   | Hoàn thành |
| 4  | Viết module đánh giá Segmentation | `src/segmentation/evaluate.py`              | Hoàn thành |
| 5  | Tính IoU, Dice                    | `src/evaluation/segmentation_metrics.py`    | Hoàn thành |
| 6  | Hiển thị mask                     | `src/visualization/display_segmentation.py` | Hoàn thành |
| 7  | Lưu kết quả                       | `src/visualization/save_results.py`         | Hoàn thành |
| 8  | Script đánh giá segmentation      | `experiments/evaluate_segmentation.py`      | Hoàn thành |
| 9  | Test module Segmentation          | `tests/test_segmentation.py`                | Hoàn thành |
| 10 | Phân tích kết quả segmentation    | `docs/06_results/segmentation_results.md`   | Hoàn thành |

## TV7 — Metrics + testing + đánh giá tổng thể

| # | Công việc                                     | File liên quan                                    | Trạng thái |
| - | --------------------------------------------- | ------------------------------------------------- | ---------- |
| 1 | Yêu cầu thí nghiệm + đánh giá                 | `docs/02_requirements/experiment_requirements.md` | Hoàn thành |
| 2 | Nghiên cứu các metric đánh giá                | `docs/03_research/evaluation_metrics.md`          | Hoàn thành |
| 3 | Kế hoạch kiểm thử                             | `docs/05_testing/test_plan.md`                    | Hoàn thành |
| 4 | Các trường hợp kiểm thử                       | `docs/05_testing/test_cases.md`                   | Hoàn thành |
| 5 | Báo cáo kiểm thử                              | `docs/05_testing/test_report.md`                  | Hoàn thành |
| 6 | Test các metric                               | `tests/test_metrics.py`                           | Hoàn thành |
| 7 | Test toàn bộ pipeline                         | `tests/test_pipeline.py`                          | Hoàn thành |
| 8 | Notebook thử nghiệm đánh giá                  | `notebooks/05_evaluation.ipynb`                   | Hoàn thành |
| 9 | Tổng hợp báo cáo kết quả cuối cùng (cùng TV1) | `results/reports/final_results.md`                | Hoàn thành |
