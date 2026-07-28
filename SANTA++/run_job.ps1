# --- Configuration ---
$JOB_NAME = "santa-notebook-test"
$MANIFEST = "jupyter-job.yaml"
$NOTEBOOK = "Transformer_Cluster_QWEN_Local_CUDA_MiniBatchKMeans.ipynb"
$DATAFILE = "monte_cristo.txt"
$OUTPUT   = "executed_notebook.ipynb"

# 1. Submit Job
Write-Host "1. Submitting Job to Nautilus..." -ForegroundColor Cyan
kubectl apply -f $MANIFEST

$POD = $null

try {
    # 2. Wait for Pod container to reach RUNNING state
    Write-Host "2. Waiting for Pod container to reach 'Running' state..." -ForegroundColor Cyan
    while ($true) {
        $POD = (kubectl get pods -l job-name=$JOB_NAME -o jsonpath='{.items[0].metadata.name}' 2>$null)
        if ($POD) {
            $CONTAINER_STATE = (kubectl get pod $POD -o jsonpath='{.status.containerStatuses[0].state.running}' 2>$null)
            if ($CONTAINER_STATE -and $CONTAINER_STATE -ne "") { break }
        }
        Start-Sleep -Seconds 3
    }
    Write-Host "Pod active and container ready: $POD" -ForegroundColor Green

    # 3. Upload Files
    Write-Host "3. Uploading notebook and data files..." -ForegroundColor Cyan
    kubectl cp $NOTEBOOK "${POD}:/tmp/${NOTEBOOK}"
    kubectl cp $DATAFILE "${POD}:/tmp/${DATAFILE}"

    # Signal pod to start execution
    kubectl exec $POD -- touch /tmp/START_SIGNAL

    # 4. Stream Logs in Background + Poll Completion Signal
    Write-Host "4. Executing notebook on GPU..." -ForegroundColor Cyan
    
    $LOG_JOB = Start-Job -ScriptBlock { param($p) kubectl logs -f $p } -ArgumentList $POD

    while ($true) {
        Receive-Job -Job $LOG_JOB | Write-Host
        
        $CHECK = (kubectl exec $POD -- bash -c "if [ -f /tmp/DONE_SIGNAL ]; then echo 'YES'; fi" 2>$null)
        $POD_PHASE = (kubectl get pod $POD -o jsonpath='{.status.phase}' 2>$null)
        
        if ($CHECK -eq "YES" -or $POD_PHASE -eq "Succeeded" -or $POD_PHASE -eq "Failed") {
            Start-Sleep -Seconds 2
            Receive-Job -Job $LOG_JOB | Write-Host
            Stop-Job -Job $LOG_JOB 2>$null
            Remove-Job -Job $LOG_JOB 2>$null
            break
        }
        Start-Sleep -Seconds 3
    }

    # 5. Retrieve Results directly
    Write-Host "5. Downloading output notebook from pod $POD..." -ForegroundColor Cyan
    
    # Attempt 1: Standard kubectl cp using relative container path (fixes Windows tar path bug)
    kubectl cp "${POD}:tmp/executed_notebook.ipynb" "./executed_notebook.ipynb"
    
    # Attempt 2: Direct UTF-8 byte stream fallback if tar fails
    if (-not (Test-Path "./executed_notebook.ipynb") -or (Get-Item "./executed_notebook.ipynb").Length -eq 0) {
        Write-Host "Tar copy missed, streaming raw file content..." -ForegroundColor Yellow
        kubectl exec $POD -- cat /tmp/executed_notebook.ipynb | Set-Content -Encoding UTF8 ./executed_notebook.ipynb
    }

    if ((Test-Path "./executed_notebook.ipynb") -and (Get-Item "./executed_notebook.ipynb").Length -gt 0) {
        Write-Host "All tasks completed successfully! Output saved to ./executed_notebook.ipynb" -ForegroundColor Green
    } else {
        Write-Host "Failed to pull output file locally." -ForegroundColor Red
    }

} finally {
    # # 6. Cleanup
    # Write-Host "6. Cleaning up cloud resources (deleting Job and Pod)..." -ForegroundColor Yellow
    # kubectl delete job $JOB_NAME --ignore-not-found=true
}