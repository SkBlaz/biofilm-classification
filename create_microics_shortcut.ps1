$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
$target = Join-Path $root 'run_gui.bat'
$icon = Join-Path $root 'MicroICS.ico'
$desktop = [Environment]::GetFolderPath('Desktop')
$shortcutPath = Join-Path $desktop 'MicroICS GUI.lnk'

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $target
$shortcut.WorkingDirectory = $root
$shortcut.IconLocation = "$icon,0"
$shortcut.Description = 'Launch the MicroICS 3D biofilm classification GUI'
$shortcut.Save()

Write-Host "Created: $shortcutPath"
