param(
  [string]$OutDir = (Split-Path -Parent $MyInvocation.MyCommand.Path),
  [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$src = Join-Path $here "geodesic_physics.c"
$out = Join-Path $here "geodesic_physics.dll"

Write-Host "Building $out from $src" -ForegroundColor Cyan

# Preferred: CMake (generator-independent) build
$cmake = Get-Command cmake.exe -ErrorAction SilentlyContinue
if ($cmake) {
  Write-Host "Using CMake" -ForegroundColor Cyan

  $buildDir = Join-Path $here "build"
  New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

  Push-Location $buildDir
  try {
    & cmake.exe -S $here -B $buildDir | Write-Host
    & cmake.exe --build $buildDir --config $Config | Write-Host

    # Locate the built DLL (multi-config generators place it under Config/)
    $dllCandidates = @(
      (Join-Path $buildDir "$Config\geodesic_physics.dll"),
      (Join-Path $buildDir "geodesic_physics.dll")
    )
    $built = $null
    foreach ($c in $dllCandidates) {
      if (Test-Path $c) { $built = $c; break }
    }
    if (-not $built) {
      throw "CMake build succeeded but DLL not found. Tried: $($dllCandidates -join ', ')"
    }

    try {
      Copy-Item -Force $built $out
    } catch {
      Write-Warning "Could not overwrite $out (likely loaded/in use)."
      Write-Warning "Using newly built DLL at: $built"
      Write-Warning "Tip: Update your loader to prefer build output, or close the process holding the DLL and re-run."
    }
  } finally {
    Pop-Location
  }

  Write-Host "Built: $built" -ForegroundColor Green
  exit 0
}

# Prefer MSVC cl if available (Visual Studio Build Tools)
$cl = Get-Command cl.exe -ErrorAction SilentlyContinue
if ($cl) {
  Write-Host "Using MSVC cl.exe" -ForegroundColor Cyan
  Push-Location $here
  try {
    # /O2 optimize, /LD build DLL
    & cl.exe /nologo /O2 /LD /EHsc $src /Fe:$out
  } finally {
    Pop-Location
  }
  Write-Host "Built: $out" -ForegroundColor Green
  exit 0
}

# Fallback: clang if installed
$clang = Get-Command clang.exe -ErrorAction SilentlyContinue
if ($clang) {
  Write-Host "Using clang.exe" -ForegroundColor Cyan
  Push-Location $here
  try {
    & clang.exe -O3 -shared -o $out $src
  } finally {
    Pop-Location
  }
  Write-Host "Built: $out" -ForegroundColor Green
  exit 0
}

throw "No compiler found (cl.exe or clang.exe). Install Visual Studio Build Tools or LLVM/clang."