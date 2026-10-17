param(
    [string]$GamePath = 'E:\EA SPORTS UFC 5 01.031',
    [switch]$GraphicsDebug,
    [switch]$Performance,
    [switch]$RenderDoc,
    [switch]$GraphicsDump,
    [switch]$Tracy,
    [switch]$ValidationOnly,
    [int]$TestSeconds = 0
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$exe = Join-Path $repo '_Build\windows\install\kyty_emulator.exe'
$runtime = Join-Path $repo '_Build\ufc5\runtime'
$logDir = Join-Path $repo ('_Build\ufc5\runs\' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
if ($Performance -and $GraphicsDebug) { throw 'Choose either -Performance or -GraphicsDebug for this run.' }
if ($ValidationOnly -and ($Performance -or $GraphicsDebug)) { throw 'Use -ValidationOnly by itself, without -Performance or -GraphicsDebug.' }
if (-not (Test-Path -LiteralPath $exe)) { throw 'Build the emulator with scripts\Build-Kyty.ps1 first.' }
if (-not (Test-Path -LiteralPath (Join-Path $GamePath 'eboot.bin'))) { throw "Game executable not found in $GamePath" }
New-Item -ItemType Directory -Force $runtime,$logDir | Out-Null
$printfDirection = if ($ValidationOnly) { 'Silent' } else { 'Console' }
$emulatorArgs = @('--game', $GamePath, '--printf-direction', $printfDirection)
if ($env:PROCESSOR_IDENTIFIER -match 'AuthenticAMD') { $emulatorArgs += '--amd-cpu' }
if ($Performance) {
    $emulatorArgs = @('--game', $GamePath, '--printf-direction', 'Silent',
        '--shader-log-direction', 'Silent', '--vulkan-validation', 'false',
        '--gpu-assisted-validation', 'false', '--shader-validation', 'false',
        '--graphics-debug-dump', $GraphicsDump, '--command-buffer-dump', 'false')
    if ($env:PROCESSOR_IDENTIFIER -match 'AuthenticAMD') { $emulatorArgs += '--amd-cpu' }
}
if ($GraphicsDebug) {
    $emulatorArgs += @('--vulkan-validation', 'true', '--shader-validation', 'true',
        '--graphics-debug-dump', 'true', '--shader-log-direction', 'File',
        '--shader-log-folder', (Join-Path $logDir 'shaders'))
}
if ($RenderDoc) {
        $emulatorArgs += '--rd'
}
if ($Tracy) {
        $emulatorArgs += '--profile'
    }
if ($ValidationOnly) {
    $emulatorArgs += @('--vulkan-validation', 'true', '--shader-validation', 'true',
        '--gpu-assisted-validation', 'false', '--graphics-debug-dump', 'false',
        '--command-buffer-dump', 'false', '--shader-log-direction', 'Silent')
}
@{
    started = (Get-Date -Format o)
    revision = (& git -C $repo rev-parse HEAD)
    executable = $exe
    executableSha256 = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
    game = $GamePath
    workingDirectory = $runtime
    arguments = $emulatorArgs
    testSeconds = $TestSeconds
    performanceMode = [bool]$Performance
    validationOnly = [bool]$ValidationOnly
} | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $logDir 'session.json')
Write-Host "Logs: $logDir"
Write-Host "Runtime data and saves: $runtime"
if ($Performance) { Write-Host 'Performance comparison: detailed logging and graphics diagnostics are disabled. Startup/fatal output and session metadata are still saved.' }
if ($ValidationOnly) { Write-Host 'Quiet validation: routine logs and graphics dumps are disabled. Targeted diagnostics and fatal validation errors are still saved.' }
if ($TestSeconds -gt 0) {
    # Windows Start-Process needs quotes around arguments containing spaces.
    $quotedArgs = $emulatorArgs | ForEach-Object { '"' + $_ + '"' }
    $process = Start-Process -FilePath $exe -ArgumentList $quotedArgs -WorkingDirectory $runtime -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir 'console.log') -RedirectStandardError (Join-Path $logDir 'stderr.log') -PassThru
    $finished = $process.WaitForExit($TestSeconds * 1000)
    if (-not $finished) {
        Stop-Process -Id $process.Id
        $process.WaitForExit()
        "Stopped by test timeout after $TestSeconds seconds; gameplay was not verified." | Set-Content (Join-Path $logDir 'result.txt')
    } else {
        "Emulator exited with code $($process.ExitCode)." | Set-Content (Join-Path $logDir 'result.txt')
    }
    Get-Content (Join-Path $logDir 'result.txt')
} else {
    Push-Location $runtime
    try {
        & $exe @emulatorArgs 2>&1 | Tee-Object (Join-Path $logDir 'console.log')
        "Emulator exited with code $LASTEXITCODE." | Set-Content (Join-Path $logDir 'result.txt')
    } finally { Pop-Location }
}
