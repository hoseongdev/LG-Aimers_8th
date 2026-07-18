# LG Aimers 8th — 온라인 해커톤 (LLM 경량화, 13위 → 오프라인 진출)

> Efficiency First: 지능 방어와 실전 가속을 위한 W8A8 양자화 전략
> 팀 "언발란스" — 하태영, 김호성, 김관호 (충남대학교 컴퓨터융합학부)

## 문제 정의

1.2B 파라미터급 소형 언어모델은 파라미터 제약으로 정확도 향상에 명확한 임계치가 존재함을 사전 분석으로 확인했다. 이에 따라 미미한 정확도 상승에 리소스를 낭비하기보다, 실제 서비스 환경에서 사용자가 체감할 수 있는 **추론 속도**를 최우선 목표로 설정했다 (Accuracy < Speed).

## 접근 방식

### 1. 수치적 안정성 확보
FP16 대신 BF16으로 모델을 로드해 양자화 눈금(Scale) 계산 시 발생하는 초기 오차를 원천 차단.

### 2. 캘리브레이션 데이터 전략
- EXAONE 전용 데이터셋 [MANTA-1M](https://huggingface.co/datasets/LGAI-EXAONE/MANTA-1M) 사용, Chat Template 적용
- 256 샘플 / 시퀀스 길이 256으로 캘리브레이션 — 무조건적인 데이터 증량이 오히려 양자화 통계에 노이즈를 유발함을 실험으로 확인하고 이 지점을 최적점으로 선정

### 3. W8A8 양자화 설계
- `llmcompressor`의 `QuantizationModifier` 사용 (GPTQ 대신 채택)
- 가중치(Weight) + 활성화(Activation) 모두 8비트로 양자화해 GPU Tensor Core 활용도 극대화
- `embed_tokens`, `lm_head`는 양자화 대상에서 제외(Ignore List) — 정보의 핵심 관문을 BF16 정밀도로 보호해 전체 용량 증가는 1% 미만으로 억제하면서 생성 품질 방어
- Oneshot PTQ 방식으로 추가 재학습 없이 단일 패스 교정, batch_size=1로 개별 샘플의 미세한 수치 특성을 보존

```python
recipe = [
    QuantizationModifier(
        scheme="W8A8",
        targets=["Linear"],
        ignore=["embed_tokens", "lm_head"],
    )
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=256,
    num_calibration_samples=256,
)
```

전체 코드는 [`quantize.py`](./quantize.py) 참고. Baseline 원본은 [`baseline_reference.ipynb`](./baseline_reference.ipynb)에서 비교 가능.

## 결과

| 항목 | 값 |
|---|---|
| 리더보드 점수 | 0.6346584803 |
| 추론 처리 시간 | 9분 48초 |
| 제출일 | 2026-02-19 |
| 용량 증가율 | 1% 미만 |

동일 자원 대비 2배 이상의 처리 효율을 가진 실전형 경량화 모델을 구현, 온라인 예선 통과 후 오프라인 해커톤에 진출해 훈련지원금을 받았다.

## 한계 및 향후 방향

발표 자료에는 KV Cache 8비트 양자화를 통한 메모리 50% 절감 방안도 설계 요소로 포함했으나, 실제 최종 제출 모델(`config.json`의 `kv_cache_scheme: null`로 확인)에는 반영되지 못했다. 시간 제약으로 W8A8 weight/activation 양자화까지만 검증을 마쳤고, KV Cache 압축은 후속 개선 과제로 남겨두었다.

## 팀에서의 역할

대회 측 제공 baseline 코드 대비 아래 변경을 직접 설계·구현했다.

| 항목 | Baseline | 최종 제출본 |
|---|---|---|
| 양자화 알고리즘 | GPTQModifier | QuantizationModifier |
| Scheme | W4A16 | W8A8 |
| Calibration Sequence Length | 512 | 256 |
| 평가 서버 대응 | 없음 | base_model의 `.py` 파일을 결과 폴더로 복사하는 로직 추가 |

- **양자화 방식 전환**: baseline의 GPTQ(W4A16) 대신 `QuantizationModifier`(W8A8)로 교체해 가중치뿐 아니라 활성화까지 8비트로 양자화, GPU Tensor Core 활용도를 높이는 방향으로 재설계
- **캘리브레이션 설정 튜닝**: 시퀀스 길이를 512→256으로 조정하며 여러 조합을 실험해 256/256이 최적점임을 도출 (과도한 데이터가 오히려 양자화 통계에 노이즈를 유발함을 확인)
- **평가 서버 이슈 해결**: 인터넷이 차단된 채점 서버 환경에서 `trust_remote_code` 모델의 커스텀 `.py` 파일이 로드되지 않는 문제를 파악하고, 해당 파일들을 산출물 폴더로 함께 복사하도록 baseline에 없던 로직 추가

## 개발 환경

```
torch==2.9.0+cu128
transformers==4.57.3
llmcompressor
compressed-tensors==0.13.0
datasets==4.4.1
vllm==0.14.1
```
OS: Ubuntu 22.04.5 LTS