@echo off
chcp 65001 >nul

REM 배치 파일의 위치를 저장소 루트로 사용해 어느 폴더에서 실행해도 경로가 맞게 한다.
cd /d "%~dp0"

python scripts\review_custom_masks.py

REM Python 또는 필수 패키지 문제가 생기면 창이 바로 닫히지 않게 오류를 보여준다.
if errorlevel 1 (
    echo.
    echo 검수 도구 실행 중 오류가 발생했습니다. 위 오류 내용을 확인하세요.
    pause
)
