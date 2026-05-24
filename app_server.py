# app_server.py
# 零依赖 HTTP 服务器 – 文本检测与识别（修复 501 错误）
import os
import sys
import urllib.parse
import base64
import tempfile
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
import cv2
import numpy as np

from config import *
from detector import EASTTextDetector
from ocr_recognizer import OCRRecognizer

HTML_UPLOAD = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Text Detection & OCR</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#1a1a2e; color:#eee; font-family:'Segoe UI',sans-serif; min-height:100vh; display:flex; justify-content:center; align-items:center; }}
  .container {{ background:#16213e; padding:2rem; border-radius:1rem; box-shadow:0 10px 30px rgba(0,0,0,0.5); width:90%; max-width:700px; text-align:center; }}
  h1 {{ font-size:2rem; margin-bottom:0.5rem; color:#e94560; }}
  p {{ color:#a0a0b0; margin-bottom:2rem; }}
  .drop-zone {{ border:2px dashed #e94560; border-radius:0.8rem; padding:2rem; cursor:pointer; transition:0.2s; }}
  .drop-zone:hover {{ background:#0f3460; }}
  input[type="file"] {{ display:none; }}
  button {{ background:#e94560; color:white; border:none; padding:0.8rem 2rem; border-radius:0.5rem; font-size:1rem; cursor:pointer; margin-top:1rem; }}
  button:disabled {{ background:#555; cursor:not-allowed; }}
  .note {{ margin-top:1.5rem; font-size:0.8rem; color:#555; }}
</style>
</head>
<body>
<div class="container">
  <h1>📷 Text Detection &amp; OCR</h1>
  <p>Upload an image to detect and recognize text / 上传图片，检测并识别文字</p>
  <form method="post" action="/" enctype="application/x-www-form-urlencoded" id="uploadForm">
    <div class="drop-zone" id="dropZone">
      📁 Drag &amp; drop an image here or click to browse / 拖拽图片或点击浏览
      <input type="file" id="fileInput" accept="image/*">
    </div>
    <input type="hidden" name="image_data" id="imageData" value="">
    <button type="submit" id="submitBtn" disabled>🔍 Analyze</button>
  </form>
  <div class="note">EAST + EasyOCR + preprocessing</div>
</div>
<script>
  const fileInput = document.getElementById('fileInput');
  const dropZone = document.getElementById('dropZone');
  const submitBtn = document.getElementById('submitBtn');
  const imageData = document.getElementById('imageData');
  const reader = new FileReader();

  function handleFile(file) {{
    dropZone.innerText = file.name;
    reader.onload = function(e) {{
      imageData.value = e.target.result;
      submitBtn.disabled = false;
    }};
    reader.readAsDataURL(file);
  }}

  fileInput.addEventListener('change', (e) => {{
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
  }});
  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('dragover', (e) => {{ e.preventDefault(); dropZone.style.background = '#0f3460'; }});
  dropZone.addEventListener('dragleave', () => dropZone.style.background = '');
  dropZone.addEventListener('drop', (e) => {{
    e.preventDefault();
    dropZone.style.background = '';
    if (e.dataTransfer.files.length > 0) {{
      fileInput.files = e.dataTransfer.files;
      handleFile(e.dataTransfer.files[0]);
    }}
  }});
</script>
</body>
</html>"""

HTML_RESULT = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Result</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#1a1a2e; color:#eee; font-family:'Segoe UI',sans-serif; padding:2rem; }}
  h2 {{ color:#e94560; margin-bottom:1rem; }}
  .row {{ display:flex; gap:2rem; flex-wrap:wrap; margin-bottom:2rem; }}
  .card {{ background:#16213e; border-radius:1rem; padding:1rem; flex:1; min-width:300px; box-shadow:0 4px 15px rgba(0,0,0,0.4); }}
  img {{ width:100%; border-radius:0.5rem; }}
  pre {{ background:#0f3460; padding:1rem; border-radius:0.5rem; overflow-x:auto; max-height:400px; white-space:pre-wrap; }}
  a {{ display:inline-block; margin-top:1rem; background:#e94560; color:white; text-decoration:none; padding:0.8rem 2rem; border-radius:0.5rem; }}
  a:hover {{ background:#ff6b81; }}
</style>
</head>
<body>
<h2>Original Image / 原图</h2>
<div class="row">
  <div class="card"><img src="data:image/jpeg;base64,{original_b64}" alt="original"></div>
</div>
<h2>Detection &amp; Recognition Result / 检测识别结果</h2>
<div class="row">
  <div class="card"><img src="data:image/jpeg;base64,{result_b64}" alt="result"></div>
  <div class="card">
    <h3>Recognized Text / 识别文字</h3>
    <pre>{text_lines}</pre>
    <a href="data:text/plain;base64,{text_b64}" download="ocr_result.txt">⬇ Download TXT / 下载文本</a>
  </div>
</div>
<a href="/">← Upload another / 重新上传</a>
</body>
</html>"""


class OCRRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """处理所有 GET 请求，返回上传页面"""
        try:
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_UPLOAD.encode("utf-8"))
        except Exception:
            traceback.print_exc()

    def do_POST(self):
        """处理图片上传并返回识别结果"""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            params = urllib.parse.parse_qs(body.decode("utf-8"))
            img_b64 = params.get("image_data", [None])[0]
            if not img_b64:
                self.send_error(400, "No image data")
                return

            if "," in img_b64:
                img_b64 = img_b64.split(",", 1)[1]

            img_bytes = base64.b64decode(img_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if original_img is None:
                self.send_error(400, "Cannot decode image")
                return

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(img_bytes)
                tmp_path = tmp.name

            detector = EASTTextDetector(
                model_path=MODEL_PATH,
                conf_thresh=MIN_CONFIDENCE,
                resize_width=RESIZE_WIDTH,
                resize_height=RESIZE_HEIGHT
            )
            result_img, boxes = detector.detect(tmp_path)
            os.unlink(tmp_path)

            recognizer = OCRRecognizer(languages=OCR_LANGUAGES, gpu=False)
            recognized = recognizer.recognize_boxes(original_img, boxes)

            annotated = result_img.copy()
            for (box, text, _) in recognized:
                if text:
                    x1, y1, x2, y2 = box
                    y_t = y1 - 10 if y1 > 20 else y2 + 20
                    cv2.putText(annotated, text, (x1, y_t),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)

            lines = []
            for i, (box, text, _) in enumerate(recognized, 1):
                if text:
                    lines.append(f"Box {i}: {box} -> {text}")
                else:
                    lines.append(f"Box {i}: {box} -> [No text]")
            full_text = "\n".join(lines)

            def img_to_b64(img):
                _, buf = cv2.imencode(".jpg", img)
                return base64.b64encode(buf).decode()

            orig_b64 = img_to_b64(original_img)
            res_b64  = img_to_b64(annotated)
            text_b64 = base64.b64encode(full_text.encode()).decode()

            response = HTML_RESULT.format(
                original_b64=orig_b64,
                result_b64=res_b64,
                text_lines=full_text.replace("\n", "<br>"),
                text_b64=text_b64
            )
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(response.encode())
        except Exception:
            traceback.print_exc()
            error_html = f"<html><body><h1>500 Internal Server Error</h1><pre>{traceback.format_exc()}</pre></body></html>"
            self.send_response(500)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(error_html.encode())


def run_server(host="localhost", port=8000):
    server = HTTPServer((host, port), OCRRequestHandler)
    print(f"✅ Server running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped.")
        server.server_close()


if __name__ == "__main__":
    run_server()