# 데이터셋 출처 기록

2026-08-13에 검증했다. 아래 수치는 다운로드한 실제 파일에서 측정했으며 변환 코드의 상수로 사용하지 않는다.

## Aff-Grasp

- 공식 데이터: https://huggingface.co/datasets/Gen1113/Data_for_Aff-Grasp
- 로컬 루트: `data/raw/affgrasp/Data_for_Aff-Grasp`
- Git 리비전: `fcc0faac67bdd0126ebde263e6b07f114c2f599c`
- Hugging Face 표시 저장 용량: 2,752,742,862바이트
- Git LFS 검증: `git lfs fsck` 통과
- `ego_train`: RGB 입력 331개, 라벨 331개, 쌍을 이루는 보조 JPG 331개
- `depth`: 676개
- AED: RGB/라벨 쌍 721개이며 외부 평가용으로 보존
- AED depth에는 대응 RGB가 없는 파일 `cup_003269_graydepth.png`가 하나 더 있다. 평가 파서가 이 파일을 임의로 포함하면 안 된다.

## UMD RGB-D Part Affordance 도구 데이터

- 공식 페이지: https://users.umiacs.umd.edu/~fermulcm/affordance/part-affordance-dataset/index.html
- 공식 압축파일: https://obj.umiacs.umd.edu/part-affordance/part-affordance-dataset-tools.tar.gz
- 압축파일 크기: 2,986,470,339바이트
- SHA-256: `E43C97E37FA1E33DA43ED083059572EFAA3625DFDDE4B038D6010A00987EE0F6`
- 로컬 압축파일: `data/raw/_archives/part-affordance-dataset-tools.tar.gz`
- 압축 해제 루트: `data/raw/umd`
- 물체 수: 105개
- RGB/depth/most-likely 라벨/ranked 라벨: 각각 28,843개이며 파일명 기준으로 모두 대응
- `gt_type=manual`: 마스크 9,632개
- `gt_type=automatic`: 마스크 19,211개이며 기본 변환에서 제외

압축파일의 `category_10_fold.txt`와 `novel_split.txt`는 원본 메타데이터로 보존한다. 이 프로젝트는 프레임을 임의로 분할하지 않는다. 프로젝트용 train/validation/test 분할은 실제 물체 디렉터리 하나를 정확히 하나의 분할에만 배정해야 하며, UMD 변환을 승인하기 전에 누수를 검사한다.

