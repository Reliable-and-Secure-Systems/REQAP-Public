@echo off
title VC707 FPGA Live Edge-AI Demonstration Dashboard
echo ========================================================
echo   Launching VC707 FPGA Streamlit Demonstration UI...
echo ========================================================
cd /d "%~dp0"
streamlit run app.py
pause
