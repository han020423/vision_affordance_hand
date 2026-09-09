@echo off
chcp 65001 >nul
cd /d "%~dp0"
python scripts/review_hand_masks.py --base data/interim/v6_object_pseudolabels
pause
