@echo off
chcp 65001 >nul

REM 어느 위치에서 더블클릭해도 저장소 루트 기준 경로를 사용한다.
cd /d "%~dp0"

python scripts\review_custom_masks.py --workspace outputs\custom_mask_review\new132_v2

REM 오류가 생기면 창이 닫히지 않게 원인을 확인할 시간을 준다.
if errorlevel 1 (
    echo.
    echo 신규 132장 마스크 수정 도구 실행 중 오류가 발생했습니다.
    pause
)
