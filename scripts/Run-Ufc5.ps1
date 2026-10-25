param(
    [string]$GamePath = 'E:\EA SPORTS UFC 5 01.031',
    [switch]$GraphicsDebug,
    [switch]$Performance,
    [switch]$RenderDoc,
    [switch]$GraphicsDump,
    [switch]$Tracy,
    [switch]$ValidationOnly,
    # The instrumented emulator from scripts\Build-Kyty.ps1 -PgoTraining, writing its counters to
    # _Build\pgo\profiles (scripts\Merge-PgoProfile.ps1 merges them for the next normal build).
    [switch]$PgoTraining,
    # The emulated display's refresh rate (the emulator's default is 60). Not for UFC 5: its speed
    # follows the frame rate it sees, and at 120 Hz the game itself ran too fast.
    [int]$VblankFrequency = 0,
    # KYTY_FLIP_LATCH_STEPS: a frame that misses a vblank is shown at the next of this many steps
    # a vblank period instead of a whole period later (at most one flip a period). The emulator
    # defaults to 2; 1 turns it off.
    [int]$FlipLatchSteps = 0,
    # An in-run A/B (KYTY_AB_B): the B arm's settings as NAME=VALUE;NAME=VALUE, e.g.
    # 'KYTY_FLIP_LATCH_STEPS=4'. Press F9 in the game window to start it (once a fight starts);
    # it alternates A and B every 300 frames and prints "AB block" and "AB summary" lines.
    [string]$AbB = '',
    [int]$TestSeconds = 0
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$exe = if ($PgoTraining) { Join-Path $repo '_Build\pgo\install\kyty_emulator.exe' } else { Join-Path $repo '_Build\windows\install\kyty_emulator.exe' }
$runtime = Join-Path $repo '_Build\ufc5\runtime'
$logDir = Join-Path $repo ('_Build\ufc5\runs\' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
if ($Performance -and $GraphicsDebug) { throw 'Choose either -Performance or -GraphicsDebug for this run.' }
if ($ValidationOnly -and ($Performance -or $GraphicsDebug)) { throw 'Use -ValidationOnly by itself, without -Performance or -GraphicsDebug.' }
if (-not (Test-Path -LiteralPath $exe)) { throw $(if ($PgoTraining) { 'Build the training emulator with scripts\Build-Kyty.ps1 -PgoTraining first.' } else { 'Build the emulator with scripts\Build-Kyty.ps1 first.' }) }
if (-not (Test-Path -LiteralPath (Join-Path $GamePath 'eboot.bin'))) { throw "Game executable not found in $GamePath" }
New-Item -ItemType Directory -Force $runtime,$logDir | Out-Null
if ($PgoTraining) {
    # The run shares the runtime folder, so the shader and pipeline caches are warm: neither is keyed
    # by the emulator build.
    $pgoProfiles = Join-Path $repo '_Build\pgo\profiles'
    New-Item -ItemType Directory -Force $pgoProfiles | Out-Null
    $env:KYTY_PGO_FILE = Join-Path $pgoProfiles ('run-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.profraw')
}
if ($FlipLatchSteps -gt 0) { $env:KYTY_FLIP_LATCH_STEPS = $FlipLatchSteps }
# Only for this run (restored at the end): a B arm left in the shell would make later runs A/Bs.
$previousAbB = $env:KYTY_AB_B
if ($AbB) { $env:KYTY_AB_B = $AbB }
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
if ($VblankFrequency -gt 0) {
    $emulatorArgs += @('--vblank-frequency', $VblankFrequency)
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
    pgoProfile = $(if ($PgoTraining) { $env:KYTY_PGO_FILE } else { $null })
    flipLatchSteps = $env:KYTY_FLIP_LATCH_STEPS
    abB = $env:KYTY_AB_B
} | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $logDir 'session.json')
Write-Host "Logs: $logDir"
Write-Host "Runtime data and saves: $runtime"
if ($Performance) { Write-Host 'Performance comparison: detailed logging and graphics diagnostics are disabled. Startup/fatal output and session metadata are still saved.' }
if ($env:KYTY_FLIP_LATCH_STEPS) { Write-Host "Flip latch steps: $env:KYTY_FLIP_LATCH_STEPS (it stays set in this shell; Remove-Item Env:KYTY_FLIP_LATCH_STEPS goes back to the default, 2)" }
if ($AbB) { Write-Host "A/B: B arm $AbB. Press F9 in the game window to start it; results are the 'AB summary' lines in console.log." }
if ($PgoTraining) { Write-Host "PGO training: the instrumented emulator runs slower than usual. Profile: $env:KYTY_PGO_FILE" }
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
    if ($AbB) { $env:KYTY_AB_B = $previousAbB }
} else {
    Push-Location $runtime
    try {
        & $exe @emulatorArgs 2>&1 | Tee-Object (Join-Path $logDir 'console.log')
        "Emulator exited with code $LASTEXITCODE." | Set-Content (Join-Path $logDir 'result.txt')
    } finally {
        Pop-Location
        if ($AbB) { $env:KYTY_AB_B = $previousAbB }
    }
}
if ($AbB) {
    $summary = Select-String -LiteralPath (Join-Path $logDir 'console.log') -Pattern 'AB summary' | Select-Object -Last 1
    if ($summary) { Write-Host "Last A/B result: $($summary.Line)" } else { Write-Host 'No A/B result: was F9 pressed, and did the run last 600+ frames after it?' }
}
