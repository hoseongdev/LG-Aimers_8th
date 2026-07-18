import os
import torch
import shutil
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# --- [1. 기본 경로 및 하이퍼파라미터 설정] ---
MODEL_ID = "/workspace/base_model"
OUT_DIR = "/workspace/kim/model_final"
ZIP_NAME = "submit_final_admin"

# 1.2B 모델이 문맥을 충분히 파악하도록 2048 세팅 고정
NUM_CALIBRATION_SAMPLES = 2048
MAX_SEQUENCE_LENGTH = 2048

# --- [2. 모델 및 토크나이저 로드] ---
print("[INFO] 모델 및 토크나이저 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# --- [3. 캘리브레이션 데이터셋 전처리] ---
print("[INFO] 캘리브레이션 데이터셋(MANTA-1M) 전처리 중...")
ds = load_dataset("LGAI-EXAONE/MANTA-1M", split=f"train[:{NUM_CALIBRATION_SAMPLES}]")

def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["conversations"],
            add_generation_prompt=True,
            tokenize=False
        )
    }

ds = ds.map(preprocess)

# --- [4. 정밀 타격형 IGNORE 리스트 구축] ---
# 글로벌 노름을 빼서 용량이 늘어나는 걸 막고,
# 실제 model.named_modules() 구조와 100% 매칭되는 이름만 엄선함.
IGNORE = [
    "lm_head",              # 최종 토큰 생성 헤드
    "model.embed_tokens",   # 8비트로 깎이면 성능 급락하는 임베딩 층
    "model.norm",           # 출력 직전 최종 레이어 노름

    # 기초 문맥을 받아들이는 '입구' 레이어 전체 보호 (BF16 유지)
    "model.layers.0",
    "model.layers.1",

    # 최종 답변의 논리를 정제하는 '출구' 레이어 전체 보호 (BF16 유지)
    "model.layers.30",
    "model.layers.31"       # EXAONE-1.2B는 총 32개(0~31) 레이어
]

# --- [5. W8A16 레시피 빌드 및 양자화 실행] ---
recipe = [
    QuantizationModifier(
        scheme="W8A16",     # 가중치만 8비트(용량 세이브), 연산은 BF16(성능 유지)
        targets="Linear",
        ignore=IGNORE
    )
]

print(f"[INFO] Oneshot 알고리즘 실행 중... (샘플: {NUM_CALIBRATION_SAMPLES}개)")
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# --- [6. 모델 저장 및 제출용 압축] ---
print(f"[INFO] 양자화 완료된 모델 저장 중 -> {OUT_DIR}")
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

print(f"[INFO] 제출용 {ZIP_NAME}.zip 압축 파일 생성 중...")
shutil.make_archive(
    base_name=f"/workspace/kim/{ZIP_NAME}",
    format="zip",
    root_dir="/workspace/kim",
    base_dir="model_final"
)

print("✅ 로컬 파이프라인 완료. (제출은 별도 스크립트/UI에서 진행)")

# 참고: 원본 코드에는 이 지점에 데이콘 자동 제출 API 호출부가 있었으며
# 여기에 개인 API 토큰이 하드코딩되어 있었음.
# 포트폴리오 공개 목적상 해당 블록과 토큰은 제거함.
# (제출은 데이콘 웹 UI 또는 환경변수로 토큰을 관리하는 별도 스크립트로 수행 권장)