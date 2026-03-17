# ConvertPDF — PDF to Markdown Pipeline

Chuyển đổi văn bản pháp quy tiếng Việt (PDF) sang Markdown chất lượng cao.  
Hỗ trợ hai chế độ: **Online** (Gemini AI) và **Offline** (marker-pdf + Ollama).

---

## Cấu trúc dự án

```
ConvertPDF/
├── run.py                    # CLI entry — Online pipeline
├── requirements.txt          # Dependencies (online)
├── .env                      # GEMINI_API_KEY
├── inputs/                   # PDF gốc
├── outputs/                  # Kết quả online
├── temp/                     # Cache Gemini
├── src/                      # Source online pipeline
│   ├── config.py             # Config & API key
│   ├── analyzer.py           # Phân tích PDF (trang, bảng, ảnh, scanned)
│   ├── planner.py            # Chia chunk theo chiến lược
│   ├── renderer.py           # Render PDF → ảnh PNG
│   ├── extractor.py          # Trích xuất table hints (PyMuPDF + Tesseract)
│   ├── converter.py          # Gọi Gemini chuyển đổi chunk → Markdown
│   ├── prompts.py            # Prompt templates cho Gemini
│   ├── assembler.py          # Ghép nối chunks thành file hoàn chỉnh
│   ├── postprocess.py        # Hậu xử lý Markdown (loại rác, sửa bảng)
│   ├── quality.py            # Chấm điểm chất lượng tổng thể
│   ├── spot_check.py         # Kiểm tra ngẫu nhiên bằng AI
│   ├── auto_fix.py           # Tự động sửa lỗi phát hiện được
│   └── pipeline.py           # Điều phối toàn bộ pipeline
├── offline/                  # Source offline pipeline
│   ├── config_offline.py     # Config offline
│   ├── converter_marker.py   # Convert bằng marker-pdf / pdftext
│   ├── polisher_ollama.py    # Polish & sửa dấu tiếng Việt bằng Ollama
│   ├── quality_offline.py    # Chấm điểm chất lượng (offline)
│   ├── spot_check_offline.py # Kiểm tra ngẫu nhiên bằng Ollama
│   ├── auto_fix_offline.py   # Tự động sửa lỗi bằng Ollama
│   ├── run_offline.py        # CLI entry — Offline pipeline
│   └── requirements.txt      # Dependencies (offline)
└── outputs_offline/          # Kết quả offline
```

---

## Online Pipeline (Gemini AI)

Sử dụng Google Gemini để chuyển đổi PDF sang Markdown với chất lượng cao nhất.  
Yêu cầu: kết nối internet + `GEMINI_API_KEY`.

### Lưu đồ

```mermaid
flowchart TD
    PDF[PDF Input] --> P1

    subgraph conversion [Chuyển đổi]
        P1["Phase 1: Phân tích PDF<br/>analyzer.py"] --> P2
        P2["Phase 2: Lên kế hoạch chunk<br/>planner.py"] --> P3
        P3["Phase 3: Render trang → ảnh<br/>renderer.py"] --> P4
        P4["Phase 4: Trích xuất table hints<br/>extractor.py + Tesseract OCR"] --> P5
        P5["Phase 5: Gemini convert<br/>converter.py + prompts.py"] --> P6
        P6["Phase 6: Ghép nối Markdown<br/>assembler.py + postprocess.py"]
    end

    P6 --> P7

    subgraph verification [Kiểm tra & Sửa lỗi]
        P7["Phase 7: Spot-check<br/>spot_check.py<br/>(N vị trí ngẫu nhiên)"]
        P7 -->|"critical + warning > 0"| P8
        P7 -->|"all OK"| P10
        P8["Phase 8: Auto-fix<br/>auto_fix.py<br/>(sửa lỗi lần 1)"]
        P8 -->|"còn CRITICAL"| P9
        P8 -->|"hết CRITICAL"| P10
        P9["Phase 9: Final Check Loop<br/>(spot-check lại → fix lại)<br/>Tối đa N vòng"]
        P9 -->|"hết CRITICAL hoặc hết vòng"| P10
    end

    P10["Phase 10: Quality Check<br/>quality.py<br/>(chấm điểm /10 trên bản cuối)"]

    P10 --> Output["Markdown + Reports<br/>pipeline.log<br/>quality_report.json<br/>spot_check_report.json<br/>auto_fix_report.json"]
```

### Final Check Loop (Phase 9)

```mermaid
flowchart TD
    Start["Bắt đầu Final Check"] --> SpotCheck
    SpotCheck["Spot-check lại"] --> Check{Còn CRITICAL?}
    Check -->|Không| Done["Kết thúc → Phase 10"]
    Check -->|Có| CountCheck{"Đã đủ max_fix_rounds<br/>vòng chưa?"}
    CountCheck -->|Chưa| Fix["Auto-fix vòng N"]
    CountCheck -->|Rồi| ForceEnd["Kết thúc<br/>(vẫn còn CRITICAL)"]
    Fix --> SpotCheck
```

### Sử dụng

```bash
# Chạy tất cả PDF trong inputs/
python run.py

# Chạy 1 file cụ thể
python run.py -i "inputs/document.pdf"

# Tùy chỉnh
python run.py --model gemini-2.5-pro --max-fix-rounds 5 --dpi 300

# Dry-run (không gọi Gemini)
python run.py --dry-run

# Chỉ phân tích
python run.py --analyze-only
```

### Tham số CLI

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--input, -i` | - | File PDF cụ thể |
| `--input-dir` | `inputs/` | Thư mục chứa PDF |
| `--output-dir` | `outputs/` | Thư mục lưu kết quả |
| `--model` | `gemini-3-flash-preview` | Model chuyển đổi |
| `--verify-model` | `gemini-2.5-flash-lite` | Model kiểm tra |
| `--chunk-size` | `10` | Số trang mỗi chunk |
| `--dpi` | `300` | DPI render ảnh |
| `--max-fix-rounds` | `3` | Số vòng final check loop tối đa |
| `--no-quality-check` | - | Bỏ qua chấm điểm |
| `--no-spot-check` | - | Bỏ qua kiểm tra ngẫu nhiên |
| `--no-auto-fix` | - | Bỏ qua tự động sửa |
| `--clear-cache` | - | Xóa cache, convert lại từ đầu |
| `--verbose, -v` | - | Log chi tiết |

### Output cho mỗi file

```
outputs/<tên_file>/
├── output_<tên_file>.md        # Markdown cuối cùng
├── pipeline.log                # Log toàn bộ pipeline
├── quality_report.json         # Điểm chất lượng
├── spot_check_report.json      # Kết quả spot-check
├── spot_check.log              # Log chi tiết spot-check
├── auto_fix_report.json        # Kết quả auto-fix
├── auto_fix.log                # Log chi tiết auto-fix
├── final_check_round_N.json    # Report vòng N (nếu có)
├── auto_fix_round_N.json       # Fix report vòng N (nếu có)
└── images/                     # Ảnh trích xuất
```

---

## Offline Pipeline (marker-pdf + Ollama)

Chạy hoàn toàn offline, không cần internet. Sử dụng `marker-pdf` cho chuyển đổi  
và `Ollama` (local LLM) cho kiểm tra + sửa lỗi.

### Lưu đồ

```mermaid
flowchart TD
    PDF[PDF Input] --> P1

    subgraph conversion [Chuyển đổi]
        P1["Phase 1: Smart Convert<br/>converter_marker.py"]
        P1 --> Strategy{Chiến lược?}

        Strategy -->|pdftext OK| DirectExtract["pdftext extract<br/>(text trực tiếp)"]
        Strategy -->|scanned / low quality| MarkerOCR["marker-pdf OCR<br/>(Surya OCR)"]
        Strategy -->|auto| AutoDetect["Tự phát hiện<br/>kiểm tra VN%"]

        DirectExtract --> P2
        MarkerOCR --> P2
        AutoDetect --> P2
    end

    P2["Phase 2: Post-processing<br/>postprocess.py"] --> VNCheck

    VNCheck{"VN diacritics<br/>ratio < 5%?"}
    VNCheck -->|Có| P3["Phase 3: Ollama<br/>diacritics repair"]
    VNCheck -->|Không| P3b["Phase 3: Ollama polish<br/>(optional)"]
    P3 --> P4
    P3b --> P4

    subgraph checks [Kiểm tra & Sửa lỗi]
        P4["Phase 4: Quality Check<br/>quality_offline.py"]
        P4 --> P5["Phase 5: Spot-check<br/>spot_check_offline.py<br/>(Ollama)"]
        P5 -->|"có lỗi"| P6["Phase 6: Auto-fix<br/>auto_fix_offline.py<br/>(Ollama)"]
        P5 -->|"all OK"| Save
    end

    P6 --> Save["Lưu output + reports"]

    Save --> Output["Markdown + Reports<br/>pipeline.log<br/>quality_report.json<br/>spot_check_report.json<br/>auto_fix_report.json"]
```

### Smart Text Extraction (Phase 1)

```mermaid
flowchart TD
    Input["PDF Page"] --> Check{"Trang có text<br/>embedded?"}
    Check -->|Có| PDFText["pdftext extract<br/>(giữ nguyên Unicode)"]
    Check -->|Không| Marker["marker-pdf OCR"]

    PDFText --> VNCheck{"Kiểm tra VN%"}
    VNCheck -->|"> 5%"| Good["Text tốt → dùng"]
    VNCheck -->|"< 5%"| Marker

    Marker --> Result["Kết quả"]
    Good --> Result
```

### Sử dụng

```bash
# Chạy tất cả PDF
python -m offline.run_offline

# Chạy 1 file cụ thể
python -m offline.run_offline -i "inputs/document.pdf"

# Với Ollama polish + spot-check + auto-fix
python -m offline.run_offline --use-ollama

# Chọn chiến lược trích xuất
python -m offline.run_offline --strategy pdftext

# Trên máy có GPU
python -m offline.run_offline --device cuda
```

### Tham số CLI

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--input, -i` | - | File PDF cụ thể |
| `--input-dir` | `inputs/` | Thư mục chứa PDF |
| `--output-dir` | `outputs_offline/` | Thư mục lưu kết quả |
| `--device` | `mps` | Thiết bị: `mps` / `cpu` / `cuda` |
| `--strategy` | `auto` | Chiến lược: `auto` / `marker` / `pdftext` / `hybrid` |
| `--use-ollama` | - | Bật Ollama local LLM |
| `--ollama-model` | `qwen3-vl:8b` | Model Ollama |
| `--force-ocr` | - | Force OCR tất cả trang |
| `--no-quality-check` | - | Bỏ qua chấm điểm |
| `--no-spot-check` | - | Bỏ qua kiểm tra ngẫu nhiên |
| `--no-auto-fix` | - | Bỏ qua tự động sửa |
| `--verbose, -v` | - | Log chi tiết |

---

## So sánh Online vs Offline

| Tiêu chí | Online (Gemini) | Offline (marker-pdf) |
|----------|----------------|---------------------|
| Chất lượng chuyển đổi | Rất cao | Tốt (phụ thuộc PDF) |
| Yêu cầu internet | Có | Không |
| Chi phí | API cost | Miễn phí |
| Tốc độ | Chậm hơn (API calls) | Nhanh hơn (local) |
| Xử lý bảng phức tạp | Tốt (vision AI) | Trung bình |
| Tiếng Việt dấu thanh | Tốt | Cần smart extract + repair |
| Kiểm tra chất lượng | Gemini spot-check | Ollama spot-check |
| Final check loop | Có (tối đa N vòng) | Chưa có |

---

## Cài đặt

```bash
# Clone & setup
git clone <repo-url>
cd ConvertPDF
python -m venv .venv
source .venv/bin/activate

# Online pipeline
pip install -r requirements.txt
echo "GEMINI_API_KEY=your_key_here" > .env

# Offline pipeline
pip install -r offline/requirements.txt

# Offline + Ollama (optional)
# Cài Ollama: https://ollama.com
ollama pull qwen3-vl:8b
```

---

## Logging

Cả hai pipeline đều ghi log đầy đủ:

- **Console**: Rich progress bars + bảng tóm tắt
- **pipeline.log**: Log chi tiết toàn bộ phase (thời gian, số liệu, lỗi)
- **spot_check.log**: Chi tiết từng vị trí kiểm tra
- **auto_fix.log**: Chi tiết từng lỗi đã sửa / không sửa được
- **JSON reports**: Dữ liệu có cấu trúc để phân tích sau
