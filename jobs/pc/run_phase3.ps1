# ConvNeXt-T KD+TGA students, cross-fitted over halves A/B (MULTI_TEACHER_PROTOCOL.md amendment 1), then the cross-fit
# evaluation of all ConvNeXt arms. Waits for run_phase2.ps1 so GPU jobs stay serial. DONE jobs are skipped on re-launch.
$ErrorActionPreference = "Stop"
$PY = "C:\midog\venv\Scripts\python.exe"
$A = "C:\midog\repo\axel_model"
$LOG = "C:\midog\logs"; $PROG = "$LOG\progress_phase3.txt"; $WAITFOR = "$LOG\progress_phase2.txt"
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
Say "WAIT  until run_phase2.ps1 has finished"
$DONE_LINE = '^\d\d-\d\d \d\d:\d\d:\d\d  ALL DONE$'; $FAIL_LINE = '^\d\d-\d\d \d\d:\d\d:\d\d  FAIL '
while (-not ((Test-Path $WAITFOR) -and (Select-String -Path $WAITFOR -Pattern $DONE_LINE -Quiet))) {
  if ((Test-Path $WAITFOR) -and (Select-String -Path $WAITFOR -Pattern $FAIL_LINE -Quiet)) { Say "FAIL  phase2 failed, not starting"; exit 1 }
  Start-Sleep 60
}
$seeds = @("--seeds","0","1","2","3","4","5","6","7","8","9")
$cnx = @("--backbone","convnext_tiny","--alpha","0","--T","2","--beta","1","--epochs","60","--save","--out","convnext_distill.csv") + $seeds
foreach ($fd in "XR","S360","CS2") {
  foreach ($hv in "A","B") {
    Run "cnx_tga_${fd}_$hv" (@("distill.py","--holdout",$fd,"--tag","cnx_kd_tga$hv","--target_half",$hv) + $cnx)
  }
}
Run "eval_cnx_crossfit" @("cf_eval_any.py","--arms","cnx_control","cnx_kd","cnx_kd_tga","--out","cnx_crossfit")
Say "ALL DONE"
