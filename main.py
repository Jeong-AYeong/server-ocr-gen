from fastapi import FastAPI, File, UploadFile
from paddleocr import PaddleOCR
from PIL import Image
import io
import uvicorn
import os
import numpy as np

app = FastAPI()

ocr = PaddleOCR(
        lang="korean",
        det_db_thresh=0.2,
        det_db_box_thresh=0.4,
        use_angle_cls=True
    )

def perform_ocr(image: Image.Image) -> str:
    # [TODO] 여기에 원래 구현했던 OCR 로직 넣기
    # 예: return pytesseract.image_to_string(image)
    ################################################
    print(f"이미지 크기: {image.size} 처리 중...")
    image_np = np.array(image)
    result = ocr.predict(image)
    data = result[0]
    texts = data['rec_texts']
    scores = data['rec_scores']

    result_texts = []
    for text, score in zip(texts, scores):
        result_texts.append(f"{text} (confidence: {score:.2f})")
        print(f"{text} (confidence: {score:.2f})")

    return "\n".join(result_texts) if result_texts else "텍스트를 찾을 수 없습니다."
    # return "Python에서 인식된 텍스트입니다."  # TODO:임시 더미 반환값

@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    image = Image.open(io.BytesIO(image_bytes))
    
    result_text = perform_ocr(image)
    
    return {"text": result_text}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
