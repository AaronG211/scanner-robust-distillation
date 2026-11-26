# TENT baseline (TENT_PROTOCOL.md). Waits for run_phase1.ps1 so GPU jobs stay serial. DONE jobs are skipped on re-launch.
$ErrorActionPreference = "Stop"
$PY = "C:\midog\venv\Scripts\python.exe"
$A = "C:\midog\repo\axel_model"
$LOG = "C:\midog\logs"; $PROG = "$LOG\progress_phase2.txt"; $WAITFOR = "$LOG\progress_phase1.txt"
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
Say "WAIT  until run_phase1.ps1 has finished"
$DONE_LINE = '^\d\d-\d\d \d\d:\d\d:\d\d  ALL DONE$'; $FAIL_LINE = '^\d\d-\d\d \d\d:\d\d:\d\d  FAIL '
while (-not ((Test-Path $WAITFOR) -and (Select-String -Path $WAITFOR -Pattern $DONE_LINE -Quiet))) {
  if ((Test-Path $WAITFOR) -and (Select-String -Path $WAITFOR -Pattern $FAIL_LINE -Quiet)) { Say "FAIL  phase1 failed, not starting"; exit 1 }
  Start-Sleep 60
}
Run "e1_tent" @("C:\midog\e0_gpu_diag.py","tent")
Say "ALL DONE"
