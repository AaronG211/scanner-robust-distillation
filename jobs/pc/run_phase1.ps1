# Reviewer-driven additions (Literature_Review_2026-11-25.md): per-cell predictions for recalibration baselines,
# scanner probe of teachers vs students, ConvNeXt-T students (no BatchNorm) for the TGR-without-AdaBN test.
# Waits for run_phase0.ps1 (which waits for run_e17.ps1), so GPU jobs stay serial. DONE jobs are skipped on re-launch.
$ErrorActionPreference = "Stop"
$PY = "C:\midog\venv\Scripts\python.exe"
$A = "C:\midog\repo\axel_model"
$LOG = "C:\midog\logs"; $PROG = "$LOG\progress_phase1.txt"; $WAITFOR = "$LOG\progress_phase0.txt"
$env:PYTHONUNBUFFERED = "1"
function Say($m) { "$(Get-Date -Format 'MM-dd HH:mm:ss')  $m" | Tee-Object -Append $PROG }
function Run($name, [string[]]$argv) {
  if ((Test-Path $PROG) -and (Select-String -Path $PROG -SimpleMatch "DONE  $name" -Quiet)) { return }
  Say "START $name"
  $p = Start-Process -FilePath $PY -ArgumentList $argv -WorkingDirectory $A -NoNewWindow -Wait -PassThru `
       -RedirectStandardOutput "$LOG\$name.out.txt" -RedirectStandardError "$LOG\$name.err.txt"
  if ($p.ExitCode -ne 0) { Say "FAIL  $name (exit $($p.ExitCode)) - see $LOG\$name.err.txt"; exit 1 }
  Say "DONE  $name"
}
Say "WAIT  until run_phase0.ps1 has finished"
$DONE_LINE = '^\d\d-\d\d \d\d:\d\d:\d\d  ALL DONE$'; $FAIL_LINE = '^\d\d-\d\d \d\d:\d\d:\d\d  FAIL '
while (-not ((Test-Path $WAITFOR) -and (Select-String -Path $WAITFOR -Pattern $DONE_LINE -Quiet))) {
  if ((Test-Path $WAITFOR) -and (Select-String -Path $WAITFOR -Pattern $FAIL_LINE -Quiet)) { Say "FAIL  phase0 failed, not starting"; exit 1 }
  Start-Sleep 60
}
$seeds = @("--seeds","0","1","2","3","4","5","6","7","8","9")
Run "cells_r18" @("C:\midog\e0_gpu_diag.py","cells","--arms","r18_control","r18_kd","--out","r18")
Run "e04_probe" @("C:\midog\e0_gpu_diag.py","probe")
foreach ($fd in "XR","S360","CS2") {
  Run "cnx_kd_$fd" (@("distill.py","--holdout",$fd,"--tag","cnx_kd","--backbone","convnext_tiny","--alpha","0","--T","2","--beta","1","--epochs","60","--save","--out","convnext_distill.csv") + $seeds)
}
foreach ($fd in "XR","S360","CS2") {
  Run "cnx_control_$fd" (@("distill.py","--holdout",$fd,"--tag","cnx_control","--backbone","convnext_tiny","--alpha","1","--T","2","--beta","0","--epochs","60","--save","--out","convnext_distill.csv") + $seeds)
}
Run "cells_cnx" @("C:\midog\e0_gpu_diag.py","cells","--arms","cnx_kd","cnx_control","--out","cnx")
Say "ALL DONE"
