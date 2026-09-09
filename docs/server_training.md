# 학습 서버 인계 절차

이 서버 묶음은 실험 D의 공개 데이터 사전학습 단계를 준비한다. 자체 촬영 데이터는 포함하지 않으므로 실험 A와 D의 최종 미세조정은 여기서 시작하지 않는다. 압축파일 옆에 `.sha256` 체크섬 파일을 함께 배포하며, 압축을 풀기 전에 반드시 검증한다.

공개 데이터 사전학습을 마친 뒤 자체 승인 데이터가 준비되면, 실험 A와 D 미세조정은
각각 다음 설정으로 실행한다. 두 설정의 데이터·증강·seed는 같고 초기 가중치만 다르다.

```bash
python scripts/check_training_server.py --config configs/training/custom_finetune_a_coco.yaml
python scripts/train_seg.py --config configs/training/custom_finetune_a_coco.yaml --dry-run

python scripts/check_training_server.py --config configs/training/custom_finetune_d_public.yaml
python scripts/train_seg.py --config configs/training/custom_finetune_d_public.yaml --dry-run
```

긴 학습 전에는 각 설정에 `--smoke-test`를 붙여 1 epoch 연기 시험을 별도로 실행한다.
실제 학습은 반드시 `tmux` 안에서 conda 환경을 활성화한 뒤 실행한다. 서버 자체 업데이트나
`sudo`는 사용하지 않으며, 데이터와 큰 결과는 `/DATA` 또는 `/data2`에 둔다.

## 1. 업로드 및 검증

`artifacts/`의 다음 파일 두 개를 서버에 업로드한다.

- `vision_hand_server_training.tar.gz`
- `vision_hand_server_training.tar.gz.sha256`

Linux에서 실행할 명령:

```bash
sha256sum -c vision_hand_server_training.tar.gz.sha256
tar -xzf vision_hand_server_training.tar.gz
cd vision_hand_server_training
python3 scripts/verify_server_bundle.py
```

## 2. 설치 전 서버 점검

```bash
nvidia-smi
python3 --version
```

PyTorch 공식 설치 선택기에서 서버 드라이버가 지원하는 CUDA wheel 주소를 선택한다. CUDA Toolkit 버전만 보고 결정하면 안 된다. 그다음 다음 예시처럼 격리 환경을 만든다.

```bash
bash scripts/setup_training_server.sh https://download.pytorch.org/whl/cu128
source .venv-server/bin/activate
```

위 URL은 고정한 PyTorch 버전용 예시일 뿐, 확인하지 않은 서버가 CUDA 12.8을 지원한다는 뜻이 아니다. 실제 드라이버를 먼저 확인한다.

## 3. 학습 전 필수 검사

```bash
python scripts/check_training_server.py
python scripts/train_seg.py --dry-run
```

두 명령이 모두 통과하고 CUDA GPU를 보고한 뒤에만 학습한다. 사전 점검은 내보낸 모든 이미지·라벨 쌍과 물체 ID 단위 분할 격리도 다시 검증한다.

## 4. 1 epoch 연기 시험

다음 명령은 학습 데이터 5%를 사용해 실제 학습을 수행한다.

```bash
python scripts/train_seg.py --smoke-test
```

GPU 사용량, 마스크 미리보기, loss, validation 결과와 실행 메타데이터 저장 여부를 확인한다. 이 연기 시험 지표를 모델의 최종 성능으로 보고하면 안 된다.

## 5. 전체 공개 데이터 사전학습

연기 시험을 승인한 뒤에만 실행한다.

```bash
python scripts/train_seg.py
```

초기 설정 파일은 `configs/training/public_pretrain.yaml`이다. YOLO11n-seg, 640픽셀 입력, 80 epoch, seed 42, `lr0=0.001`인 AdamW, AMP, patience 20을 사용한다. Mosaic, MixUp, CutMix, Copy-Paste와 random erasing은 끈다.

`images/test`를 기준으로 설정을 조정하면 안 된다. 설정 판단에는 validation만 사용하고 test는 최종 고정 평가용으로 보존한다.
