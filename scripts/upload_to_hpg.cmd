@echo off
setlocal

set "LOCAL_ROOT=N:\Experimental_Data\yujunchen\projects"
set "REMOTE_ROOT=/blue/mzding/yujunchen/projects"
set "REMOTE_CODE=/blue/mzding/yujunchen/projects/AI_ratings/codes"

echo Uploading code and image datasets to HiPerGator.
echo Local root:  %LOCAL_ROOT%
echo Remote root: %REMOTE_ROOT%
echo.
echo This transfer may ask for your HiPerGator password and Duo approval.
echo It does not run the rating job.
echo.

tar -C "%LOCAL_ROOT%" -cf - ^
  "AI_ratings/codes" ^
  "data/NAPS_H" ^
  "data/IAPS1182" ^
  "data/OASIS" ^
| ssh hpg-codex "mkdir -p '%REMOTE_ROOT%' && tar -C '%REMOTE_ROOT%' -xf -"

if errorlevel 1 (
    echo.
    echo ERROR: upload failed.
    exit /b 1
)

echo.
echo Upload completed. Submitting compute-node verification job only.
echo This verification does not call the model/API and does not run ratings.
echo.

ssh hpg-codex "cd '%REMOTE_CODE%' && sbatch scripts/HPG_verify_image_setup.sh"

if errorlevel 1 (
    echo.
    echo WARNING: upload succeeded, but verification job submission failed.
    echo You can submit it later with:
    echo   cd %REMOTE_CODE%
    echo   sbatch scripts/HPG_verify_image_setup.sh
    exit /b 2
)

echo.
echo Done. The rating job was NOT run.
exit /b 0
