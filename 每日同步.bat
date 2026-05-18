@echo off
chcp 65001 >nul
cd /d "%~dp0"
python sync_data.py --daily
