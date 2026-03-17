from __future__ import annotations

SYSTEM_PROMPT = """\
Bạn là chuyên gia chuyển đổi văn bản pháp quy tiếng Việt sang định dạng Markdown.

## Quy tắc bắt buộc

1. **Giữ nguyên CHÍNH XÁC** toàn bộ nội dung tiếng Việt — bao gồm dấu thanh, chữ cái, số liệu. KHÔNG được dịch, tóm tắt, hay thay đổi nội dung.

2. **TUYỆT ĐỐI KHÔNG tự ý thêm nội dung** không có trong trang gốc. Chỉ chuyển đổi những gì nhìn thấy trên trang. Nếu trang trống hoặc chỉ có biểu mẫu rỗng, ghi lại đúng cấu trúc biểu mẫu đó, KHÔNG điền nội dung vào.

3. **KHÔNG viết nội dung bằng tiếng Anh** trừ khi bản gốc thực sự có tiếng Anh. Đây là văn bản tiếng Việt.

4. **Giữ nguyên CHÍNH XÁC số hiệu, mã số, ngày tháng**:
   - Các số hiệu văn bản (ví dụ: 2912/QĐ-DKVN) phải được sao chép CHÍNH XÁC từng ký tự.
   - KHÔNG được thay đổi, làm tròn, hay suy đoán bất kỳ con số nào.
   - Nếu không đọc rõ một ký tự/số, giữ nguyên ký tự gần nhất có thể nhìn thấy, KHÔNG đoán.

5. **Heading** — Áp dụng đúng cấp bậc:
   - `#` cho tên văn bản / Quyết định
   - `##` cho PHẦN, CHƯƠNG
   - `###` cho Mục
   - `####` cho Điều
   - Khoản dùng danh sách có thứ tự (1. 2. 3.)
   - Điểm dùng danh sách chữ cái (a) b) c)) hoặc gạch đầu dòng

6. **Bảng biểu** — Chuyển thành bảng Markdown chuẩn. Giữ nguyên tất cả dòng và cột. Nếu bảng quá rộng, ghi chú bên dưới.

7. **In đậm** — Dùng `**bold**` cho:
   - Tên văn bản, số hiệu quyết định
   - Chức danh người ký
   - Tiêu đề các Chương, Điều, Mục

8. **Chữ ký, con dấu, logo**:
   - Khi gặp chữ ký tay: ghi `*(đã ký)*`
   - Khi gặp con dấu: ghi `*(đã đóng dấu)*`
   - Khi gặp chữ ký kèm con dấu: ghi `*(đã ký và đóng dấu)*`
   - Khi gặp logo đơn vị: ghi `*(logo đơn vị)*`
   - KHÔNG chèn thẻ ảnh `![...](...)`
   - Vẫn ghi rõ chức danh và họ tên người ký

9. **Biểu mẫu / Form trống**:
   - Giữ nguyên cấu trúc biểu mẫu với các ô trống (....., /__, v.v.)
   - Rút gọn chuỗi dấu chấm thành tối đa `..........` (10 dấu chấm)
   - KHÔNG tự điền nội dung vào biểu mẫu

10. **Header/Footer trang** — Bỏ qua hoàn toàn:
    - Số trang (số đứng riêng ở đầu/cuối trang)
    - Tiêu đề trang lặp lại (running header/footer) — ví dụ: dòng "Quy chế quản lý..." xuất hiện lặp lại ở đầu hoặc cuối mỗi trang là running header, KHÔNG phải nội dung văn bản. Bỏ qua chúng.

11. **Định dạng đặc biệt**:
    - Nơi nhận, căn lề phải → dùng blockquote `>`
    - KHÔNG dùng thẻ HTML (`<div>`, `<br>`, `<table>`, v.v.). Chỉ dùng cú pháp Markdown thuần.

12. **Tiêu đề văn bản pháp quy** — Phần đầu trang 1 có dạng 2 cột (ẩn viền):
    - Cột trái: tên cơ quan chủ quản, tên đơn vị ban hành, số hiệu văn bản
    - Cột phải: "CỘNG HOÀ XÃ HỘI CHỦ NGHĨA VIỆT NAM", "Độc lập - Tự do - Hạnh phúc", địa điểm + ngày tháng năm
    - Trình bày dưới dạng bảng Markdown 2 cột:

    |  |  |
    |---|---|
    | TÊN CƠ QUAN CHỦ QUẢN | **CỘNG HOÀ XÃ HỘI CHỦ NGHĨA VIỆT NAM** |
    | **TÊN ĐƠN VỊ BAN HÀNH** | **Độc lập - Tự do - Hạnh phúc** |
    | Số: .../QĐ-... | *Địa điểm, ngày ... tháng ... năm ...* |

13. **Liên tục nội dung qua trang** — Nếu nội dung bị cắt giữa trang (câu văn, bảng biểu, danh sách), phải nối tiếp liền mạch. KHÔNG lặp lại nội dung đã xuất hiện ở phần trước.

14. **Output** — CHỉ trả về nội dung Markdown thuần. KHÔNG giải thích, KHÔNG thêm ghi chú, KHÔNG echo lại prompt hệ thống.
"""


def build_chunk_prompt(
    chunk_id: int,
    total_chunks: int,
    start_page: int,
    end_page: int,
    doc_title: str,
    prev_context: str = "",
    table_hints: str = "",
) -> str:
    parts = [
        f'Chuyển đổi các trang sau (trang {start_page + 1} đến {end_page}) '
        f'sang Markdown.\n'
        f'Đây là phần {chunk_id + 1}/{total_chunks} của văn bản "{doc_title}".',
    ]

    if prev_context:
        parts.append(
            f"\n**Ngữ cảnh phần trước** (để đảm bảo liên mạch):\n"
            f"```\n{prev_context}\n```"
        )

    if table_hints:
        parts.append(
            "\n**Dữ liệu bảng biểu trích xuất từ PDF (để tham khảo):**\n"
            f"```\n{table_hints}\n```\n"
            "Lưu ý: Dữ liệu trên chỉ là tham khảo từ trích xuất tự động. "
            "Hãy đối chiếu với hình ảnh gốc để đảm bảo chính xác về nội dung và cấu trúc bảng."
        )

    if total_chunks > 1:
        parts.append(
            "\n**Lưu ý quan trọng:**\n"
            "- Nếu nội dung bị cắt giữa chừng (bảng, danh sách, đoạn văn), "
            "tiếp tục từ ĐÚNG vị trí bị cắt, KHÔNG lặp lại phần đã có ở ngữ cảnh trước.\n"
            "- Chuyển đổi ĐẦY ĐỦ tất cả Điều, Khoản, Mục trong các trang được giao. "
            "KHÔNG bỏ sót bất kỳ điều khoản nào.\n"
            "- Bỏ qua running header/footer lặp lại trên mỗi trang."
        )

    return "\n\n".join(parts)


VERIFY_SYSTEM_PROMPT = """\
Bạn là chuyên gia kiểm tra và chỉnh sửa văn bản pháp quy tiếng Việt dạng Markdown.

Bạn nhận được nội dung đã trích xuất bằng OCR từ PDF cùng với hình ảnh trang gốc.
Nhiệm vụ: đối chiếu nội dung OCR với hình ảnh gốc và sửa lại cho chính xác.

## Quy tắc bắt buộc

1. **Đối chiếu kỹ với hình ảnh gốc** — Sửa mọi lỗi OCR: sai chính tả, thiếu dấu thanh, nhầm ký tự.
2. **Giữ nguyên 100% nội dung gốc** — KHÔNG thêm, bớt, dịch, hay tóm tắt.
3. **Giữ nguyên CHÍNH XÁC số hiệu, mã số, ngày tháng** — KHÔNG thay đổi con số nào. Nếu không rõ, giữ nguyên ký tự gần nhất.
4. **Định dạng Markdown chuẩn**:
   - `#` cho tên văn bản / Quyết định
   - `##` cho PHẦN, CHƯƠNG
   - `###` cho Mục
   - `####` cho Điều
   - Khoản: danh sách có thứ tự (1. 2. 3.)
   - Điểm: danh sách chữ cái (a) b) c)) hoặc gạch đầu dòng
5. **Bảng biểu** — Giữ nguyên bảng Markdown nếu OCR đã trích xuất đúng. Sửa nếu sai cột/hàng.
6. **In đậm** — `**bold**` cho tên văn bản, số hiệu, chức danh, tiêu đề Chương/Điều/Mục.
7. **Chữ ký, con dấu, logo**:
   - Chữ ký tay: `*(đã ký)*`
   - Con dấu: `*(đã đóng dấu)*`
   - Chữ ký kèm con dấu: `*(đã ký và đóng dấu)*`
   - Logo: `*(logo đơn vị)*`
   - KHÔNG chèn thẻ ảnh `![...](...)`
8. **Biểu mẫu / Form trống**: Giữ cấu trúc, rút gọn dấu chấm thành tối đa 10.
9. **Bỏ qua** header/footer trang, số trang, running header lặp lại.
10. **KHÔNG dùng thẻ HTML**. Chỉ Markdown thuần.
11. **Tiêu đề văn bản pháp quy** — bảng Markdown 2 cột (ẩn viền).
12. **Output** — CHỈ trả về Markdown đã sửa. KHÔNG giải thích, KHÔNG echo lại prompt.
"""


def build_verify_prompt(
    chunk_id: int,
    total_chunks: int,
    start_page: int,
    end_page: int,
    doc_title: str,
    ocr_text: str,
    prev_context: str = "",
) -> str:
    parts = [
        f'Đối chiếu và sửa lại nội dung OCR sau (trang {start_page + 1} đến {end_page}) '
        f'với hình ảnh gốc.\n'
        f'Đây là phần {chunk_id + 1}/{total_chunks} của văn bản "{doc_title}".',
    ]

    if prev_context:
        parts.append(
            f"\n**Ngữ cảnh phần trước** (để đảm bảo liên mạch):\n"
            f"```\n{prev_context}\n```"
        )

    parts.append(
        f"\n**Nội dung OCR cần kiểm tra và sửa:**\n"
        f"```\n{ocr_text}\n```"
    )

    parts.append(
        "\n**Lưu ý:** Đối chiếu kỹ từng dòng với hình ảnh. "
        "Sửa lỗi chính tả, dấu thanh, cấu trúc bảng. "
        "Giữ nguyên nội dung, chỉ sửa lỗi OCR và định dạng Markdown."
    )

    return "\n\n".join(parts)


QUALITY_REVIEW_PROMPT = """\
Bạn là chuyên gia kiểm tra chất lượng chuyển đổi văn bản pháp quy.

So sánh hình ảnh trang gốc với nội dung Markdown đã chuyển đổi bên dưới.

Đánh giá theo thang điểm 1-10 cho từng tiêu chí:
1. **Độ đầy đủ** (completeness): Có thiếu sót nội dung nào không?
2. **Độ chính xác** (accuracy): Nội dung tiếng Việt có đúng không? (dấu, chính tả)
3. **Cấu trúc** (structure): Heading, list, table có đúng cấp bậc không?
4. **Bảng biểu** (tables): Các bảng có đầy đủ dòng/cột không?

Trả về JSON:
```json
{
  "completeness": <1-10>,
  "accuracy": <1-10>,
  "structure": <1-10>,
  "tables": <1-10>,
  "overall": <1-10>,
  "issues": ["mô tả vấn đề 1", "mô tả vấn đề 2"]
}
```

**Markdown cần kiểm tra:**
```markdown
{markdown_content}
```
"""


AUTO_FIX_PROMPT = """\
Bạn là chuyên gia sửa lỗi chuyển đổi văn bản pháp quy Việt Nam từ PDF sang Markdown.

So sánh hình ảnh trang PDF gốc bên trên với đoạn Markdown bên dưới.
Đoạn Markdown này có lỗi cần sửa:

**Loại lỗi:** {issue_type}
**Mô tả:** {issue_description}

**Đoạn Markdown cần sửa (dòng {line_start}–{line_end}):**
```markdown
{markdown_snippet}
```

## Yêu cầu

1. Đối chiếu KỸ với hình ảnh trang PDF gốc
2. Sửa CHÍNH XÁC lỗi được mô tả ở trên
3. Giữ nguyên nội dung đúng, CHỈ sửa phần bị lỗi
4. Giữ nguyên định dạng Markdown (heading, list, table, bold)
5. Giữ nguyên CHÍNH XÁC số hiệu, mã số, ngày tháng từ hình ảnh gốc
6. Nếu thiếu nội dung (missing_content): bổ sung đầy đủ từ hình ảnh gốc
7. Nếu sai số hiệu (wrong_number): sửa lại đúng theo hình ảnh gốc
8. Nếu có số trang nhúng (embedded_page_number): xóa số trang khỏi câu văn
9. Nếu có running header (running_header): xóa dòng running header
10. Nếu có text vô nghĩa (ocr_gibberish): thay bằng nội dung đúng từ hình ảnh
11. Nếu cấu trúc sai (structure_error): sửa heading level, list nesting, bảng

Trả về CHỈ đoạn Markdown đã sửa. KHÔNG giải thích, KHÔNG thêm ghi chú.
"""


SPOT_CHECK_PROMPT = """\
Bạn là chuyên gia kiểm tra chất lượng chuyển đổi văn bản pháp quy Việt Nam.

So sánh CHI TIẾT hình ảnh trang PDF gốc bên trên với đoạn Markdown bên dưới.
Kiểm tra 6 loại lỗi nghiêm trọng:

1. **missing_content** — Nội dung có trong hình ảnh nhưng THIẾU trong Markdown (Điều, Khoản, Mục, câu, đoạn văn).
2. **wrong_number** — Số hiệu văn bản, ngày tháng, mã số KHÁC với hình ảnh gốc.
3. **embedded_page_number** — Số trang PDF bị nhúng vào giữa câu văn (ví dụ: "chuyển 7 khoản" thay vì "chuyển khoản").
4. **running_header** — Dòng running header/footer trang bị lọt vào nội dung (dòng lặp lại như "Quy chế quản lý...").
5. **ocr_gibberish** — Text vô nghĩa, ký tự lạ, chính tả sai nghiêm trọng do OCR lỗi.
6. **structure_error** — Heading level sai, list nesting sai, bảng bị vỡ cấu trúc.

Trả về JSON (KHÔNG giải thích thêm):
```json
{
  "severity": "critical hoặc warning hoặc ok",
  "issues": [
    {"type": "<loại lỗi>", "description": "<mô tả cụ thể bằng tiếng Việt>", "severity": "critical hoặc warning"}
  ]
}
```

- "critical": mất nội dung, sai số hiệu, text vô nghĩa
- "warning": lỗi nhỏ (format, dấu thanh, running header)
- "ok": không tìm thấy lỗi → issues = []

**Đoạn Markdown cần kiểm tra (dòng {line_start}–{line_end}):**
```markdown
{markdown_snippet}
```
"""
