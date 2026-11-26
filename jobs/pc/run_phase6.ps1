# Protocols F and C (TEACHER_SEED_AND_CONTROL_PROTOCOL.md): ConvNeXt-T students without any teacher on the new scanner
# (family C), teacher caches for H-optimus-0 LoRA seeds 1 and 2, then ConvNeXt-T students distilled from those seeds
# (family F). Starts only when the GPU has been free for 10 minutes in a row (memory < 3000 MiB and utilisation < 40 %).
# DONE jobs are skipped on re-launch.
$ErrorActionPreference = "Stop"
$PY = "C:\midog\venv\Scripts\python.exe"
$A = "C:\midog\repo\axel_model"
$LOG = "C:\midog\logs"; $PROG = "$LOG\progress_phase6.txt"
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
Say "WAIT  until the GPU has been free for 10 minutes"
$ok = 0
while ($ok -lt 10) {
  $q = (& nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits) -split ','
  if (([int]$q[1].Trim() -lt 3000) -and ([int]$q[0].Trim() -lt 40)) { $ok++ } else { $ok = 0 }
  Start-Sleep 60
}
Say "GPU   free, starting"
$FOLDS = "XR","S360","CS2"; $HALVES = "A","B"
$seeds = @("--seeds","0","1","2","3","4","5","6","7","8","9")

# 1. early check of the changed self-check in distill_teacher.py (also needed later)
Run "cache_XR_s1" @("distill_teacher.py","--holdout","XR","--seed","1")

# 2. Protocol C: ConvNeXt-T adapted on the unlabelled new-scanner half with the ResNet-18 no-teacher ensemble's answers
$free = @("--source_labels","--alpha","1","--T","2","--beta","0","--target_soft_only","--target_teacher","r18ens",
          "--backbone","convnext_tiny","--epochs","60","--save","--out","convnext_distill.csv") + $seeds
foreach ($fd in $FOLDS) { foreach ($hv in $HALVES) {
  Run "cnx_free_${fd}_$hv" (@("distill.py","--holdout",$fd,"--tag","cnx_free_tga$hv","--target_half",$hv) + $free)
} }
Run "eval_cnx_crossfit_free" @("cf_eval_any.py","--arms","cnx_free_tga","--out","cnx_crossfit_free")

# 3. the remaining teacher caches for LoRA seeds 1 and 2
foreach ($k in "1","2") { foreach ($fd in $FOLDS) {
  Run "cache_${fd}_s$k" @("distill_teacher.py","--holdout",$fd,"--seed",$k)
  Run "cache_${fd}_targetall_s$k" @("distill_teacher.py","--holdout",$fd,"--seed",$k,"--target_all")
} }

# 4. Protocol F: ConvNeXt-T students distilled from H-optimus-0 LoRA seeds 1 and 2 (recipe of cnx_kd / cnx_kd_tga)
foreach ($k in "1","2") {
  $cnx = @("--backbone","convnext_tiny","--teacher_fm","hopt0","--teacher_seed",$k,"--alpha","0","--T","2","--beta","1",
           "--epochs","60","--save","--out","convnext_distill.csv") + $seeds
  foreach ($fd in $FOLDS) { Run "cnx_kd_t${k}_$fd" (@("distill.py","--holdout",$fd,"--tag","cnx_kd_t$k") + $cnx) }
  foreach ($fd in $FOLDS) { foreach ($hv in $HALVES) {
    Run "cnx_tga_t${k}_${fd}_$hv" (@("distill.py","--holdout",$fd,"--tag","cnx_kd_tga_t$k$hv","--target_half",$hv) + $cnx)
  } }
}
Run "eval_cnx_crossfit_tseeds" @("cf_eval_any.py","--arms","cnx_kd_t1","cnx_kd_tga_t1","cnx_kd_t2","cnx_kd_tga_t2","--out","cnx_crossfit_tseeds")
Say "ALL DONE"
