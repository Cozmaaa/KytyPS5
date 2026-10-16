param([int]$Jobs = 12, [switch]$SkipInstall)
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
Push-Location $repo
try {
    & git rev-parse HEAD | Set-Content (Join-Path $logDir 'revision.txt')
    & cmake -S . -B _Build/windows -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl "-DCMAKE_PREFIX_PATH=$qt" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DKYTY_BUILD_LAUNCHER=ON 2>&1 | Tee-Object (Join-Path $logDir 'configure.log')
    if ($LASTEXITCODE) { throw "CMake configuration failed. See $logDir" }
    & cmake --build _Build/windows --target launcher --parallel $Jobs 2>&1 | Tee-Object (Join-Path $logDir 'build.log')
    if ($LASTEXITCODE) { throw "Build failed. See $logDir" }
    if ($SkipInstall) {
        'Installation skipped; the existing install directory was not changed.' | Set-Content (Join-Path $logDir 'install.log')
        Write-Host "Built candidate: $repo\_Build\windows\kyty_emulator.exe"
    } else {
        & cmake --install _Build/windows --prefix _Build/windows/install 2>&1 | Tee-Object (Join-Path $logDir 'install.log')
        if ($LASTEXITCODE) { throw "Installation failed. See $logDir" }
        Copy-Item -LiteralPath (Join-Path $repo '_Build\windows\kyty_emulator.pdb') -Destination (Join-Path $repo '_Build\windows\install\kyty_emulator.pdb')
        Copy-Item -LiteralPath (Join-Path $repo '_Build\windows\kyty_emulator_clang_lld_link.map') -Destination (Join-Path $repo '_Build\windows\install\kyty_emulator_clang_lld_link.map')
        Write-Host "Ready: $repo\_Build\windows\install\launcher.exe"
    }
    Write-Host "Build logs: $logDir"
} finally { Pop-Location }
