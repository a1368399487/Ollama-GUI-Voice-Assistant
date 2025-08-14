# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

# ⏬ 自动收集 sounddevice 所需资源
datas, binaries, hiddenimports = collect_all('sounddevice')

block_cipher = None

a = Analysis(
    ['gui_voice_assistant.py'],
    pathex=['C:\\Users\\a1368\\OneDrive\\Bureaublad\\Flutter1'],  # ⚠️ 注意路径要真实存在
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + [
        'scipy',
        'faster_whisper',
        'cffi',
        '_cffi_backend',
        'numpy'
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='心理语音助手',
    debug=True,  # 开启调试
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True  # ✅ 显示控制台窗口
)

