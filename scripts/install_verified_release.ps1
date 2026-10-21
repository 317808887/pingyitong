param(
    [Parameter(Mandatory = $true)][string]$Version,
    [string]$InstallDirectory = 'F:\讯飞\新建文件夹\_verify_install',
    [string]$ReportDirectory = '',
    [string]$VerifierPython = 'python'
)
$ErrorActionPreference = 'Stop'
$projectRoot = 'F:\讯飞\项目源码'
$installDirectory = [IO.Path]::GetFullPath($InstallDirectory)
$installedExe = Join-Path $installDirectory 'PingYiTong.exe'
$builtExe = Join-Path $projectRoot 'build\dist\PingYiTong\PingYiTong.exe'
$installer = Join-Path $projectRoot "build\Output\屏译通-Setup-$Version.exe"
$dataVolume = Join-Path $projectRoot "build\Output\屏译通-Setup-$Version-1.bin"
$configFile = Join-Path $env:LOCALAPPDATA 'PingYiTong\config.json'
$releaseDate = Get-Date -Format 'yyyyMMdd'
$reportRoot = if ($ReportDirectory) { [IO.Path]::GetFullPath($ReportDirectory) } else { Join-Path $projectRoot 'artifacts' }
[IO.Directory]::CreateDirectory($reportRoot) | Out-Null
$reportFile = Join-Path $reportRoot "install_desktop_release_$Version`_$releaseDate.json"
$installLog = Join-Path $reportRoot "install_desktop_release_$Version`_$releaseDate.log"
$packagedReport = Join-Path $reportRoot "install_packaged_release_$Version`_$releaseDate.json"
$verificationLog = Join-Path $reportRoot "install_packaged_release_$Version`_$releaseDate.log"
$verificationScript = Join-Path $projectRoot 'scripts\verify_packaged_release.py'
if (@($reportFile, $installLog, $packagedReport, $verificationLog) | Where-Object { Test-Path -LiteralPath $_ }) {
    throw 'Evidence already exists. Choose a fresh report directory for this installation.'
}
Get-Command $VerifierPython -ErrorAction Stop | Out-Null
foreach ($requiredFile in @($builtExe, $installer, $dataVolume, $configFile, $verificationScript)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required file missing: $requiredFile"
    }
}
$configBefore = (Get-FileHash -LiteralPath $configFile -Algorithm SHA256).Hash
$builtHash = (Get-FileHash -LiteralPath $builtExe -Algorithm SHA256).Hash
Get-CimInstance Win32_Process -Filter "Name='PingYiTong.exe'" |
    Where-Object { $_.ExecutablePath -eq $installedExe } |
    ForEach-Object {
        $applicationPid = $_.ProcessId
        $modelChildren = Get-CimInstance Win32_Process | Where-Object {
            $_.ParentProcessId -eq $applicationPid -and
            $_.ExecutablePath -like (Join-Path $installDirectory 'runtime\llama\*')
        }
        Stop-Process -Id $applicationPid
        $modelChildren | ForEach-Object { Stop-Process -Id $_.ProcessId -ErrorAction SilentlyContinue }
    }
$arguments = @('/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/SP-',
               "/DIR=$installDirectory", "/LOG=$installLog")
$installerProcess = Start-Process -FilePath $installer -ArgumentList $arguments -WindowStyle Hidden -Wait -PassThru
$installedHash = (Get-FileHash -LiteralPath $installedExe -Algorithm SHA256).Hash
$configAfter = (Get-FileHash -LiteralPath $configFile -Algorithm SHA256).Hash
$registered = Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\{8F3C2A94-7B15-4E6D-9C48-2A1F5E7B3D60}_is1'
# The packaged app is a GUI executable: --version has no reliable stdout.
# Read its embedded version and compare every bundled module/OCR asset instead.
& $VerifierPython -X utf8 $verificationScript $installedExe --version $Version --output $packagedReport > $verificationLog 2>&1
$verificationExit = $LASTEXITCODE
$packaged = Get-Content -LiteralPath $packagedReport -Raw -Encoding UTF8 | ConvertFrom-Json
$installedVersion = $packaged.embedded_version
$registryPathMatches = [IO.Path]::GetFullPath($registered.InstallLocation).TrimEnd('\') -eq $installDirectory.TrimEnd('\')
$report = [ordered]@{
    installer = $installer; data_volume = $dataVolume
    data_volume_bytes = (Get-Item -LiteralPath $dataVolume).Length
    installed_exe = $installedExe; exit_code = $installerProcess.ExitCode
    registry_version = $registered.DisplayVersion; registry_path = $registered.InstallLocation
    registry_path_matches = $registryPathMatches; installed_app_version = $installedVersion
    packaged_checks_passed = $packaged.passed; packaged_verification_exit_code = $verificationExit
    packaged_report = $packagedReport; version_method = 'PyInstaller embedded version and complete module/asset comparison'
    built_exe_sha256 = $builtHash; installed_exe_sha256 = $installedHash
    config_before_sha256 = $configBefore; config_after_sha256 = $configAfter
    binary_matches = ($builtHash -eq $installedHash); config_preserved = ($configBefore -eq $configAfter)
}
$jsonReport = $report | ConvertTo-Json
[IO.File]::WriteAllText($reportFile, $jsonReport, [Text.UTF8Encoding]::new($false))
$jsonReport
if ($installerProcess.ExitCode -ne 0 -or $verificationExit -ne 0 -or -not $packaged.passed -or $builtHash -ne $installedHash -or
    $configBefore -ne $configAfter -or $registered.DisplayVersion -ne $Version -or
    $installedVersion -ne $Version -or -not $registryPathMatches) {
    throw 'Installation verification failed. Inspect the report.'
}
