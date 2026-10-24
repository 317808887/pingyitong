; ---------------------------------------------------------------------------
; 屏译通 安装包脚本（Inno Setup 6）
;
; 编译方式：
;   ISCC.exe /DMyAppVersion=0.6.40 /DMySourceDir=<dist目录> /DMyOutputDir=<输出目录> installer.iss
;
; 或直接通过 Python 打包脚本调用（会自动传入上述参数）：
;   python build/build_exe.py --installer
;
; 设计要点：
;   1. 默认安装到用户目录（PrivilegesRequired=lowest），无需管理员权限，
;      避免公司电脑无管理员账号时无法安装；
;   2. 官方 Spark-X2.5 双模型权重与推理运行时**内置打包**（构建脚本已把
;      models/ 与 runtime/llama/ 拷入 dist，随 [Files] 一起进入安装包），
;      下载安装包一次即获得全部能力，无需单独下载模型；
;   3. 个人数据（配置、历史、日志）在卸载时由用户选择是否删除 ——
;      默认为不删除，避免误删用户积累的术语库与历史。
; ---------------------------------------------------------------------------

#ifndef MyAppVersion
  #define MyAppVersion "0.6.40"
#endif

#ifndef MySourceDir
  #define MySourceDir "..\dist\PingYiTong"
#endif

#ifndef MyOutputDir
  #define MyOutputDir "Output"
#endif

#define MyAppName "屏译通"
#define MyAppNameEn "PingYiTong"
#define MyAppVersionFull MyAppVersion + " (核心功能)"
#define MyAppPublisher "屏译通团队"
#define MyAppExeName "PingYiTong.exe"
#define MyAppId "{{8F3C2A94-7B15-4E6D-9C48-2A1F5E7B3D60}"

[Setup]
AppId={#MyAppId}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersionFull}
AppPublisher={#MyAppPublisher}
VersionInfoVersion={#MyAppVersion}

; 默认安装到用户目录，无需管理员权限
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
PrivilegesRequired=lowest
; 只用 commandline：'dialog' 会在启动时弹权限选择框，
; 静默安装（/VERYSILENT）下该对话框无法显示，安装器会直接退出
PrivilegesRequiredOverridesAllowed=commandline

; 输出
OutputDir={#MyOutputDir}
OutputBaseFilename={#MyAppName}-Setup-{#MyAppVersion}
; lzma2/fast：max 档解压需 10 分钟以上（实测超时），fast 快数倍
Compression=lzma2/fast
SolidCompression=no
; 分段是**必须的**：单文件 3.6GB 的 setup.exe 在 Windows 上无法启动
; （实测进程立即退出、无日志无安装）。Inno 官方对 >2GB 的安装包
; 同样建议启用分段。产物为 setup.exe + setup-1.bin，双击 exe 即可安装，
; 两个文件需一起分发。
DiskSpanning=yes
SlicesPerDisk=1
DiskSliceSize=7300000000
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}

; 安装包自身的图标（与应用图标一致）
SetupIconFile=..\assets\icons\app_icon.ico

; 许可文件（安装时会显示）
LicenseFile=..\LICENSE

[Languages]
Name: "chinese"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
chinese.CreateDesktopIcon=创建桌面快捷方式
chinese.LaunchAfterInstall=安装完成后启动 {#MyAppName}
chinese.ModelHint=模型权重与推理运行时已包含在安装包中，安装即可使用。
chinese.ModelHintTitle=准备模型文件
chinese.KeepDataHint=是否保留个人数据（配置、翻译历史、日志）？%n%n保留后重新安装时会自动沿用；选择"否"将一并删除。
chinese.KeepDataTitle=卸载 {#MyAppName}

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; 主程序与全部依赖（由 PyInstaller 生成）
Source: "{#MySourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\使用手册"; Filename: "{app}\docs\用户手册.md"
Name: "{group}\卸载 {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; 安装完成后可选启动
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchAfterInstall}"; Flags: nowait postinstall skipifsilent

; 首次安装时提示模型目录位置（静默安装时跳过）
Filename: "{app}\{#MyAppExeName}"; Parameters: "--help"; Flags: shellexec runhidden; Check: PromptForModels

[UninstallDelete]
; 仅删除程序自身生成的非用户数据文件；
; 用户数据（配置/历史/日志）在下方 CurUninstallStepChanged 中按用户选择处理。
Type: filesandordirs; Name: "{app}\__pycache__"

[Registry]
; 记录卸载信息，便于"应用和功能"中显示完整名称与版本
Root: HKCU; Subkey: "Software\{#MyAppPublisher}\{#MyAppName}"; ValueType: string; ValueName: "Version"; ValueData: "{#MyAppVersion}"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\{#MyAppPublisher}\{#MyAppName}"; ValueType: string; ValueName: "InstallPath"; ValueData: "{app}"; Flags: uninsdeletekey

[Code]
const
  { 与应用数据目录保持一致：%LOCALAPPDATA%\{#MyAppNameEn} }
  AppDataFolderName = 'PingYiTong';

var
  KeepUserData: Boolean;

{ 判断是否为首次安装（用于决定是否提示模型位置） }
function IsFirstInstall(): Boolean;
begin
  Result := not FileExists(ExpandConstant('{app}\{#MyAppExeName}'));
end;

{ 安装后是否弹出就绪提示（模型已内置，无需手动放置） }
function PromptForModels(): Boolean;
begin
  Result := False;  { 模型已内置，无需提示 }
end;

{ 返回应用数据目录的完整路径 }
function GetAppDataDir(): String;
begin
  Result := ExpandConstant('{localappdata}\' + AppDataFolderName);
end;

{ 安装完成后：若数据目录不存在则创建，并提示模型放置位置 }
procedure CurStepChanged(CurStep: TSetupStep);
var
  DataDir: String;
  ModelsDir: String;
  ErrorCode: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    DataDir := GetAppDataDir();
    ModelsDir := DataDir + '\models';

    { 预先创建目录，让用户能直接找到位置 }
    if not DirExists(DataDir) then
      CreateDir(DataDir);
    if not DirExists(ModelsDir) then
      CreateDir(ModelsDir);

    if PromptForModels() then
    begin
      if MsgBox(CustomMessage('ModelHint'), mbConfirmation, MB_YESNO) = IDYES then
        ShellExec('open', ModelsDir, '', '', SW_SHOWNORMAL, ewNoWait, ErrorCode);
    end;
  end;
end;

{ 卸载前询问是否保留个人数据 }
function InitializeUninstall(): Boolean;
begin
  KeepUserData := True;

  if not UninstallSilent() then
  begin
    if MsgBox(CustomMessage('KeepDataHint'), mbConfirmation, MB_YESNO) = IDNO then
      KeepUserData := False;
  end;

  Result := True;
end;

{ 卸载时按用户选择删除个人数据 }
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    if not KeepUserData then
    begin
      DataDir := GetAppDataDir();
      if DirExists(DataDir) then
        DelTree(DataDir, True, True, True);
    end
    else if not UninstallSilent() then
    begin
      { 提示数据保留位置，方便用户日后手动清理 }
      MsgBox(
        '个人数据已保留在：' + #13#10 + GetAppDataDir() + #13#10 + #13#10 +
        '如需彻底删除，请手动删除该目录。',
        mbInformation, MB_OK
      );
    end;
  end;
end;
