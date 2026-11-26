# Phase-0 GPU diagnostics (E0.2 AdaBN budget, E0.3 BatchNorm swap). Waits until run_e17.ps1 has finished so GPU jobs
# stay serial. DONE jobs are skipped on re-launch.
$ErrorActionPreference = "Stop"
$PY = "C:\midog\venv\Scripts\python.exe"
$A = "C:\midog\repo\axel_model"
$LOG = "C:\midog\logs"; $PROG = "$LOG\progress_phase0.txt"; $WAITFOR = "$LOG\progress_e17.txt"
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
Say "WAIT  until run_e17.ps1 has finished"
$DONE_LINE = '^\d\d-\d\d \d\d:\d\d:\d\d  ALL DONE$'; $FAIL_LINE = '^\d\d-\d\d \d\d:\d\d:\d\d  FAIL '
while (-not ((Test-Path $WAITFOR) -and (Select-String -Path $WAITFOR -Pattern $DONE_LINE -Quiet))) {
  if ((Test-Path $WAITFOR) -and (Select-String -Path $WAITFOR -Pattern $FAIL_LINE -Quiet)) { Say "FAIL  run_e17 failed, not starting"; exit 1 }
  Start-Sleep 60
}
Run "e02_budget" @("C:\midog\e0_gpu_diag.py","budget")
Run "e03_bnswap" @("C:\midog\e0_gpu_diag.py","bnswap")
Say "ALL DONE"
