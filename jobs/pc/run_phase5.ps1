# Protocol N (NONINFERIORITY_PROTOCOL.md): ConvNeXt-T students with the Virchow2 teacher (family N2), then teacher TTA
# and the efficiency benchmark. Starts only when the GPU has been free for 10 minutes in a row (memory < 3000 MiB and
# utilisation < 40 %), so it never competes with whatever the user is running. DONE jobs are skipped on re-launch.
$ErrorActionPreference = "Stop"
$PY = "C:\midog\venv\Scripts\python.exe"
$A = "C:\midog\repo\axel_model"
$LOG = "C:\midog\logs"; $PROG = "$LOG\progress_phase5.txt"
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
function RunSoft($name, [string[]]$argv) {
  if ((Test-Path $PROG) -and (Select-String -Path $PROG -SimpleMatch "DONE  $name" -Quiet)) { return }
  Say "START $name"
  $p = Start-Process -FilePath $PY -ArgumentList $argv -WorkingDirectory $A -NoNewWindow -Wait -PassThru `
       -RedirectStandardOutput "$LOG\$name.out.txt" -RedirectStandardError "$LOG\$name.err.txt"
  if ($p.ExitCode -ne 0) { Say "SOFTFAIL $name (exit $($p.ExitCode)) - see $LOG\$name.err.txt" } else { Say "DONE  $name" }
}
Say "WAIT  until the GPU has been free for 10 minutes"
$ok = 0
while ($ok -lt 10) {
  $q = (& nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits) -split ','
  if (([int]$q[1].Trim() -lt 3000) -and ([int]$q[0].Trim() -lt 40)) { $ok++ } else { $ok = 0 }
  Start-Sleep 60
}
Say "GPU   free, starting"
$seeds = @("--seeds","0","1","2","3","4","5","6","7","8","9")
$cnx = @("--backbone","convnext_tiny","--teacher_fm","virchow2","--alpha","0","--T","2","--beta","1","--epochs","60","--save","--out","convnext_distill.csv") + $seeds
foreach ($fd in "XR","S360","CS2") {
  Run "cnx_kd_virchow2_$fd" (@("distill.py","--holdout",$fd,"--tag","cnx_kd_virchow2") + $cnx)
}
foreach ($fd in "XR","S360","CS2") {
  foreach ($hv in "A","B") {
    Run "cnx_tga_virchow2_${fd}_$hv" (@("distill.py","--holdout",$fd,"--tag","cnx_kd_tga_virchow2$hv","--target_half",$hv) + $cnx)
  }
}
Run "eval_cnx_crossfit_virchow2" @("cf_eval_any.py","--arms","cnx_kd_virchow2","cnx_kd_tga_virchow2","--out","cnx_crossfit_virchow2")
# new scripts last, so a bug in them cannot block the students above; a failure here is logged and the queue goes on
RunSoft "teacher_tta" @("C:\midog\teacher_tta.py")
RunSoft "efficiency" @("C:\midog\efficiency.py")
Say "ALL DONE"
