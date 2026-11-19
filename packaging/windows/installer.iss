; Inno Setup script for the NUScope-TOMO Windows installer.
; Built by .github/workflows/app.yml after PyInstaller has produced build\dist\NUScope-TOMO\.
;   iscc /DAppVersion=0.3.3 packaging\windows\installer.iss
; Installs per user by default (no administrator rights needed); an administrator can choose all users.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{6E3F1C2A-6B2E-4C61-9D3B-7A1F0C5E2B41}
AppName=NUScope-TOMO
AppVersion={#AppVersion}
AppPublisher=NUANCE Center, Northwestern University
AppPublisherURL=https://github.com/NUANCE-IT/NUScope-TOMO
AppCopyright=Roberto dos Reis, NUANCE Center, Northwestern University
DefaultDirName={autopf}\NUScope-TOMO
DefaultGroupName=NUScope-TOMO
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\..\build\installer
OutputBaseFilename=NUScope-TOMO-{#AppVersion}-windows-x64-setup
SetupIconFile=..\icons\nuscope-tomo.ico
UninstallDisplayIcon={app}\NUScope-TOMO.exe
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
LicenseFile=..\..\LICENSE

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "..\..\build\dist\NUScope-TOMO\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\NUScope-TOMO"; Filename: "{app}\NUScope-TOMO.exe"
Name: "{group}\Uninstall NUScope-TOMO"; Filename: "{uninstallexe}"
Name: "{autodesktop}\NUScope-TOMO"; Filename: "{app}\NUScope-TOMO.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\NUScope-TOMO.exe"; Description: "Open NUScope-TOMO"; Flags: nowait postinstall skipifsilent
