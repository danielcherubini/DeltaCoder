@echo off
setlocal

set QUANTIZE=D:\AI\llama.cpp\build\bin\llama-quantize.exe
set F16=D:\AI\DeltaCoder\qwen3.6\v1.0\gguf\DeltaCoder-9B-Qwen3.6-v1.0-DPO-f16.gguf
set OUT=D:\AI\DeltaCoder\qwen3.6\v1.0\gguf

echo === Quantizing all variants ===

for %%Q in (Q2_K Q3_K_S Q3_K_M Q3_K_L Q4_0 Q4_K_S Q4_K_M Q5_K_S Q5_0 Q5_K_M Q6_K Q8_0 BF16) do (
    echo.
    echo --- %%Q ---
    "%QUANTIZE%" "%F16%" "%OUT%\DeltaCoder-9B-Qwen3.6-v1.0-DPO-%%Q.gguf" %%Q
    if errorlevel 1 (
        echo ERROR: %%Q failed!
    ) else (
        echo OK: %%Q done
    )
)

echo.
echo === All done! ===
dir "%OUT%"
