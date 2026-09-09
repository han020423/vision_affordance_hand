# AGENTS.md

이 파일은 이 저장소에서 작업하는 Codex와 다른 코딩 에이전트가 따라야 할 프로젝트별 지침이다.

## 1. 작업 시작 규칙

1. 모든 작업 전에 `PROJECT_CONTEXT.md`를 처음부터 끝까지 읽는다.
2. 현재 저장소의 파일, 변경 상태, 실행 환경을 먼저 확인한다.
3. 사용자의 기존 코드와 설정을 보존한다. 요청과 관계없는 파일은 수정하지 않는다.
4. 실제 데이터, 장치 연결 상태, 측정값을 확인하지 않고 성공 결과를 가정하거나 만들어내지 않는다.
5. 요구사항이 불명확할 때는 구현 결과를 크게 바꾸는 질문만 사용자에게 묻는다.

## 2. 변하지 않는 프로젝트 결정

- 최종 목표는 Affordance Segmentation 결과를 실제 Brunel Hand의 적응형 파지까지 연결하는 것이다.
- 카메라는 외부 고정 RGB 카메라다.
- 로봇팔과 완전한 6-DoF 파지 계획은 범위 밖이다.
- 공개 데이터는 Aff-Grasp, UMD, IIT-AFF를 사용한다. IIT-AFF는 2026-08-19 사용자
  승인으로 추가했으며, 물체 ID가 없는 장면 데이터이므로 train(pretrain) 전용으로만
  쓰고 validation/test에는 사용하지 않는다.
- 최종 의미 라벨은 `grasp_region`, `functional_region` 두 개다.
- 의미가 충돌하거나 불명확한 픽셀은 억지로 변환하지 않고 `ignore`로 처리한다.
- UMD의 `contain`은 기본적으로 `ignore`이며, 특히 컵 몸통을 `functional_region`으로 만들지 않는다.
- 머그컵 손잡이와 몸통처럼 복수의 파지 후보는 같은 클래스라도 분리된 마스크/컴포넌트로 유지한다.
- 파지 자세는 `PRECISION`, `WRAP`, `POWER` 세 종류다.
- 최종 주장은 모든 물체에 대한 open-world 일반화가 아니라 미학습 물체 인스턴스와 제한된 미학습 범주에 대한 평가다.

위 결정을 바꾸려면 코드 수정 전에 사용자에게 이유와 영향을 설명하고 승인을 받는다.

## 3. 권장 저장소 구조

기존 구조가 없다면 다음 구성을 우선 사용한다. 기존 구조가 있으면 무리하게 재배치하지 않는다.

```text
.
├── AGENTS.md
├── PROJECT_CONTEXT.md
├── README.md
├── configs/
│   ├── datasets/
│   ├── training/
│   └── hardware/
├── data/
│   ├── raw/              # Git에 올리지 않음
│   ├── interim/          # 통합 mask
│   ├── processed/        # 학습 형식
│   └── splits/
├── scripts/
│   ├── prepare_affgrasp.py
│   ├── prepare_umd.py
│   ├── prepare_custom.py
│   ├── validate_dataset.py
│   └── train_seg.py
├── src/
│   ├── labeling/
│   ├── datasets/
│   ├── perception/
│   ├── grasp_selection/
│   ├── hand_control/
│   └── app/
├── tests/
├── outputs/              # Git에 올리지 않음
└── docs/
```

## 4. 데이터 처리 원칙

- 원본 데이터는 수정하지 않는다.
- 원본, 중간 통합 라벨, 모델별 변환본을 별도 폴더에 둔다.
- 절대경로를 코드에 직접 쓰지 말고 YAML 또는 CLI 인자로 받는다.
- 변환 과정은 재실행 가능하고 결정적이어야 한다. 필요하면 seed를 기록한다.
- 생성된 각 샘플에 `source_dataset`, `source_id`, `object_id`, `original_labels`, `mapped_labels`, `split`, `conversion_status`를 기록한다.
- 제외하거나 `ignore` 처리한 샘플/픽셀의 이유를 로그 또는 manifest에 남긴다.
- 동일한 실제 물체 ID가 Train과 Validation/Test에 동시에 들어가지 않도록 자동 검사한다.
- 연속 프레임과 다중 시점의 중복을 고려한다. 임의 프레임 단위 random split을 사용하지 않는다.
- 데이터셋 수치는 다운로드 후 스크립트로 집계한다. 문서의 예상 수치를 코드 상수로 사용하지 않는다.

## 5. 라벨 변환 규칙

```yaml
classes:
  0: grasp_region
  1: functional_region
ignore_index: 255
```

Aff-Grasp:

```yaml
graspable: grasp_region
functional: functional_region
```

UMD:

```yaml
grasp: grasp_region
wrap-grasp: grasp_region
cut: functional_region
scoop: functional_region
pound: functional_region
support: functional_region
contain: ignore
```

다음 상황은 자동으로 하나의 라벨을 선택하지 않는다.

- 한 픽셀에 상충하는 원본 의미가 중첩됨
- 원본 mask와 이미지 크기 또는 ID가 맞지 않음
- 컵 몸통처럼 자체 파지 정책과 공개 라벨 의미가 충돌함
- 지나치게 작거나 잘린 mask로 의미 판단이 불가능함

이 경우 `ignore`, 샘플 제외 또는 사람 검수 상태 중 하나로 기록한다.

## 6. 모델 및 실험 규칙

- 기본 실시간 모델은 `YOLO11n-seg`다.
- Grounding DINO와 SAM2는 자체 데이터의 오프라인 반자동 라벨링에 사용한다.
- 공개 데이터 사전학습 후 자체 데이터 미세조정을 최종 방식으로 사용한다.
- 최소 비교는 다음 두 실험이다.
  - A: COCO pretrained → 자체 데이터
  - D: Aff-Grasp + UMD → 자체 데이터
- 여유가 있으면 Aff-Grasp 단독과 UMD 단독 사전학습을 추가한다.
- 모든 비교는 가능한 한 같은 split, 입력 크기, seed, 평가 코드로 실행한다.
- 학습 결과와 설정은 실행별 폴더에 저장한다.
- best checkpoint만 보고하지 말고 마지막 설정, seed, 데이터 버전과 실패 실행도 기록한다.
- AED는 학습에 사용하지 않고 외부 평가 세트로 유지한다.

## 7. 비전과 후보 선택 규칙

- 마스크 centroid만 파지점으로 사용하지 않는다.
- distance transform 등으로 경계에서 안전한 내부점을 계산한다.
- 후보 점수는 confidence, 손과의 거리, 접근 방향, 손 크기 적합도, 기능 영역 안전거리를 포함한다.
- `functional_region`은 현재 작업에서 감점 또는 rejection에 사용한다.
- 후보 점수 가중치는 설정 파일로 분리하고 Validation 근거 없이 하드코딩하지 않는다.
- confidence가 낮거나 후보 간 점수 차가 작으면 자동 파지보다 `ALIGN`/재정렬 상태를 선택한다.

## 8. 하드웨어 제어 규칙

- 실제 모터 구동은 사용자가 장치 연결과 전원 상태를 확인한 뒤 실행한다.
- 최초 테스트는 짧은 펄스, 낮은 듀티 또는 안전한 제한값으로 진행한다.
- 모든 구동 루프에 타임아웃과 `STOP` 경로를 둔다.
- 통신 끊김, 비정상 피드백, 대상 소실 시 모터를 정지한다.
- `PRECISION`, `WRAP`, `POWER` 프리셋은 코드와 별도 설정 파일에 둔다.
- 사용자가 실측하지 않은 개구 폭, 피드백 범위, 모터 목표값을 임의로 확정하지 않는다.
- 카메라/모델 코드와 Arduino 코드를 독립적으로 시험한 후 통합한다.

## 9. 코드 품질

- Python 코드는 가능하면 type hint와 짧은 docstring을 사용한다.
- 코드 주석, docstring, 오류 메시지, CLI 도움말, 로그 설명, 문서와 보고서는 원칙적으로 한국어로 작성한다.
- 코드의 의도와 주요 처리 순서를 처음 보는 사람도 바로 이해할 수 있도록, 복잡한 변환·검증·예외 처리에는 짧고 구체적인 한국어 주석을 붙인다.
- 함수명, 변수명, 표준 명령어, 외부 라이브러리 API, 데이터셋 고유명, 고정 라벨(`grasp_region`, `functional_region`, `ignore`)과 호환성을 위해 이미 정해진 manifest/JSON/YAML 키는 영문을 유지할 수 있다.
- 기존 영문 식별자를 단순히 한글로 바꾸어 호환성을 깨뜨리지 않는다. 대신 해당 의미를 한국어 docstring과 주석으로 설명한다.
- 화면에 저장되는 그래프·오버레이의 제목도 글꼴과 외부 도구 호환성이 확보되면 한국어를 사용한다. 호환성 때문에 영문을 유지할 때는 결과 보고서에 한국어 설명을 함께 제공한다.
- 데이터 경로, threshold, 클래스 매핑, 점수 가중치와 serial 설정을 구성 파일로 분리한다.
- 큰 단일 스크립트보다 데이터 변환, 추론, 후보 선택, 하드웨어 통신을 모듈로 나눈다.
- 외부 패키지 버전은 `requirements.txt`, `pyproject.toml` 또는 환경 파일에 고정한다.
- 새 의존성을 추가할 때 목적과 Jetson 호환성을 확인한다.
- GUI가 없어도 데이터 검사와 테스트가 가능하도록 overlay 이미지 저장 옵션을 제공한다.
- 오류를 조용히 무시하지 말고 샘플 ID와 원인을 포함해 보고한다.

## 10. 필수 테스트

구현 시 다음 검사를 우선 작성한다.

1. 원본 라벨에서 통합 라벨로의 매핑 테스트
2. `contain → ignore` 정책 테스트
3. 이미지-mask 크기와 ID 대응 테스트
4. 물체 ID 기반 split 누수 테스트
5. 빈 mask, 다중 component, 경계 접촉 mask 테스트
6. 픽셀-실거리 변환 테스트
7. 후보 점수 정규화 및 rejection 테스트
8. Serial 명령 인코딩 테스트
9. 통신 중단과 timeout에서 `STOP`으로 전환되는 테스트

실제 데이터가 없으면 최소 크기의 합성 fixture는 파서와 수학 함수의 단위 테스트에만 사용한다. 합성 결과를 모델 성능으로 보고하지 않는다.

## 11. 검증과 보고

작업 완료를 말하기 전에 수행 가능한 범위에서 다음을 확인한다.

- 변경 파일 목록
- 정적 검사 또는 문법 검사
- 단위 테스트
- 데이터 변환 dry-run
- mask overlay 샘플
- 학습 설정 출력
- 하드웨어가 없을 때는 mock serial 테스트

최종 보고에는 다음을 간단히 포함한다.

- 무엇을 구현했는지
- 어떤 명령으로 검증했는지
- 성공/실패 결과
- 사용자 실측이나 장치 연결이 필요한 남은 항목

## 12. 첫 구현 순서

사용자가 다른 우선순위를 지정하지 않으면 다음 순서로 진행한다.

1. 저장소와 실행 환경 조사
2. 설정/manifest schema 정의
3. Aff-Grasp와 UMD parser 설계
4. 통합 라벨 변환기 구현
5. split 누수 및 라벨 검증 도구 구현
6. overlay 시각화와 소규모 dry-run
7. YOLO 형식 export
8. baseline 학습 스크립트
9. 자체 데이터 반자동 라벨링
10. 실시간 추론과 후보 선택
11. ArUco 통합
12. Arduino와 Brunel Hand 통합

## 13. 구현 전 확인할 사용자 정보

실제 코드 실행 단계에서 아직 없다면 다음만 요청한다.

- Aff-Grasp와 UMD가 저장된 실제 경로
- Jetson 모델, JetPack, Python/CUDA 환경
- 카메라 장치 번호와 해상도
- Arduino serial port와 baud rate
- Brunel Hand 액추에이터 수와 핀 매핑
- ArUco 마커 ID와 실제 크기
- 손의 실측 최대 개구 폭과 세 파지 프리셋 값

정보가 없더라도 구성 파일 예시와 dry-run 가능한 코드 뼈대까지는 작성할 수 있다. 실제 값을 추측하여 장치를 구동해서는 안 된다.
