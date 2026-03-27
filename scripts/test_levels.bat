@echo off
echo ============================================================
echo Testing Different Filtering Levels on Subject5
echo ============================================================
echo.

echo [1/3] Testing Level 0 (No filtering)...
C:/Users/furka/AppData/Local/Programs/Python/Python310/python.exe scripts/validation/batch_validate.py --subjects subject5 --level 0
echo.
echo.

echo [2/3] Testing Level 1 (Motion filtering only)...
C:/Users/furka/AppData/Local/Programs/Python/Python310/python.exe scripts/validation/batch_validate.py --subjects subject5 --level 1
echo.
echo.

echo [3/3] Testing Level 2 (Motion + Stability filtering)...
C:/Users/furka/AppData/Local/Programs/Python/Python310/python.exe scripts/validation/batch_validate.py --subjects subject5 --level 2
echo.
echo.

echo ============================================================
echo All 3 levels tested! Check results above.
echo ============================================================
