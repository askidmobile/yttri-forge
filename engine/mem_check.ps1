$os = Get-CimInstance Win32_OperatingSystem
$free = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
$total = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
Write-Output "RAM: free=$free GB / total=$total GB"
Get-Process | Sort-Object -Property WorkingSet64 -Descending | Select-Object -First 8 Name, Id, @{N = 'MemMB'; E = { [math]::Round($_.WorkingSet64 / 1MB) }} | Format-Table -AutoSize
