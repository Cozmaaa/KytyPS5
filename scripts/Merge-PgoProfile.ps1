# Merges the counters of the PGO training runs (scripts\Run-Ufc5.ps1 -PgoTraining), in
# _Build\pgo\profiles, into _Build\pgo\kyty.profdata. scripts\Build-Kyty.ps1 then optimizes with it.
# Runs of an older training build still merge: functions changed since then get no profile.
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$profileDir = Join-Path $repo '_Build\pgo\profiles'
$profiles = @(Get-ChildItem -LiteralPath $profileDir -Filter '*.profraw' -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime)
if ($profiles.Count -eq 0) { throw "No training runs in $profileDir. Run scripts\Run-Ufc5.ps1 -PgoTraining first." }
$vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$vsRoot = & $vswhere -products '*' -version '[17,18)' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1
if (-not $vsRoot) { throw 'Visual Studio 2022 C++ Build Tools were not found.' }
# The llvm-profdata of the compiler that built the training emulator (Build-Kyty.ps1 puts this LLVM
# first on PATH).
$profdata = Join-Path $vsRoot 'VC\Tools\Llvm\x64\bin\llvm-profdata.exe'
if (-not (Test-Path -LiteralPath $profdata)) { throw "llvm-profdata was not found at $profdata" }
$output = Join-Path $repo '_Build\pgo\kyty.profdata'
foreach ($run in $profiles) { Write-Host ('{0}  {1:N1} MB  {2:yyyy-MM-dd HH:mm}' -f $run.Name, ($run.Length / 1MB), $run.LastWriteTime) }
& $profdata merge -o $output @($profiles | ForEach-Object { $_.FullName })
if ($LASTEXITCODE) { throw 'llvm-profdata merge failed.' }
& $profdata show $output
if ($LASTEXITCODE) { throw 'llvm-profdata show failed.' }
Write-Host "Merged $($profiles.Count) run(s) into $output; the next scripts\Build-Kyty.ps1 build optimizes with it."
