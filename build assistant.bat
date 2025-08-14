@echo off
cd /d "C:\Users\a1368\OneDrive\Bureaublad\Flutter1"
echo 当前目录：%cd%
echo 开始打包 GUI 助手程序...
pyinstaller "C:\Users\a1368\OneDrive\Bureaublad\Flutter1\gui_voice_assistant_FIXED.spec"
pyinstaller gui_voice_assistant_FIXED.spec



echo.
echo 打包完成，按任意键打开输出目录...
pause
start dist
