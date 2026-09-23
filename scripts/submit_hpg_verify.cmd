@echo off
setlocal

set "REMOTE_CODE=/blue/mzding/yujunchen/projects/AI_ratings/codes"

echo Submitting HiPerGator compute-node verification job.
echo This does not call the model/API and does not run ratings.
echo.
echo Approve Duo if prompted.
echo.

ssh hpg-codex "cd '%REMOTE_CODE%' && sbatch scripts/HPG_verify_image_setup.sh"

if errorlevel 1 (
    echo.
    echo ERROR: verification submission failed.
    exit /b 1
)

echo.
echo Verification job submitted.
exit /b 0
