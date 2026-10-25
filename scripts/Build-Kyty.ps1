# -NoLto: build without link-time optimization (KYTY_LTO), for an A/B against the default.
# -PgoTraining: an instrumented emulator for a profile-guided optimization training run, installed
#   to _Build\pgo\install (the normal install is left alone); scripts\Run-Ufc5.ps1 -PgoTraining runs it.
# A normal build optimizes with _Build\pgo\kyty.profdata when it exists (scripts\Merge-PgoProfile.ps1
#   makes it from the training runs); -NoPgo builds without it.
param([int]$Jobs = 12, [switch]$SkipInstall, [switch]$NoLto, [switch]$PgoTraining, [switch]$NoPgo)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$logDir = Join-Path $repo ('_Build\ufc5\build-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Force $logDir | Out-Null
$vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$vsRoot = & $vswhere -products '*' -version '[17,18)' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1
if (-not $vsRoot) { throw 'Visual Studio 2022 C++ Build Tools were not found.' }
Import-Module (Join-Path $vsRoot 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
Enter-VsDevShell -VsInstallPath $vsRoot -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
$env:PATH = (Join-Path $vsRoot 'VC\Tools\Llvm\x64\bin') + ';' + $env:PATH
$qt = Join-Path $env:USERPROFILE '.kyty-dev\Qt\6.10.3\msvc2022_64'
if (-not (Test-Path -LiteralPath $qt)) { throw "Qt was not found at $qt" }
$pgoProfile = Join-Path $repo '_Build\pgo\kyty.profdata'
$pgoGenerate = if ($PgoTraining) { 'ON' } else { 'OFF' }
$pgoUse = ''
if (-not $PgoTraining -and -not $NoPgo -and (Test-Path -LiteralPath $pgoProfile)) {
    # Built with a copy named by its contents: a new profile changes the compile commands, so Ninja
    # recompiles. A re-merged kyty.profdata under the same name would leave the old objects.
    $pgoHash = (Get-FileHash -LiteralPath $pgoProfile -Algorithm SHA256).Hash.Substring(0, 12).ToLower()
    $pgoCopy = Join-Path $repo "_Build\pgo\use\kyty-$pgoHash.profdata"
    if (-not (Test-Path -LiteralPath $pgoCopy)) {
        New-Item -ItemType Directory -Force (Split-Path $pgoCopy) | Out-Null
        Copy-Item -LiteralPath $pgoProfile -Destination $pgoCopy
    }
    $pgoUse = $pgoCopy.Replace('\', '/')
}
# The instrumented build is not link-time optimized: its counters are what matters, not its speed.
$lto = if ($NoLto -or $PgoTraining) { 'OFF' } else { 'ON' }
$installDir = if ($PgoTraining) { Join-Path $repo '_Build\pgo\install' } else { Join-Path $repo '_Build\windows\install' }
Push-Location $repo
try {
    & git rev-parse HEAD | Set-Content (Join-Path $logDir 'revision.txt')
    "LTO=$lto PGO_GENERATE=$pgoGenerate PGO_USE=$pgoUse install=$installDir" | Tee-Object (Join-Path $logDir 'options.txt')
    & cmake -S . -B _Build/windows -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl "-DCMAKE_PREFIX_PATH=$qt" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DKYTY_BUILD_LAUNCHER=ON "-DKYTY_LTO=$lto" "-DKYTY_PGO_GENERATE=$pgoGenerate" "-DKYTY_PGO_USE=$pgoUse" 2>&1 | Tee-Object (Join-Path $logDir 'configure.log')
    if ($LASTEXITCODE) { throw "CMake configuration failed. See $logDir" }
    & cmake --build _Build/windows --target launcher --parallel $Jobs 2>&1 | Tee-Object (Join-Path $logDir 'build.log')
    if ($LASTEXITCODE) { throw "Build failed. See $logDir" }
    if ($SkipInstall) {
        'Installation skipped; the existing install directory was not changed.' | Set-Content (Join-Path $logDir 'install.log')
        Write-Host "Built candidate: $repo\_Build\windows\kyty_emulator.exe"
    } else {
        & cmake --install _Build/windows --prefix $installDir 2>&1 | Tee-Object (Join-Path $logDir 'install.log')
        if ($LASTEXITCODE) { throw "Installation failed. See $logDir" }
        Copy-Item -LiteralPath (Join-Path $repo '_Build\windows\kyty_emulator.pdb') -Destination (Join-Path $installDir 'kyty_emulator.pdb')
        Copy-Item -LiteralPath (Join-Path $repo '_Build\windows\kyty_emulator_clang_lld_link.map') -Destination (Join-Path $installDir 'kyty_emulator_clang_lld_link.map')
        Write-Host "Ready: $installDir\launcher.exe"
    }
    Write-Host "Build logs: $logDir"
} finally { Pop-Location }
