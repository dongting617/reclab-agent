param(
    [switch]$Plan,
    [switch]$Web,
    [int]$Port = 8000,
    [string]$Config,
    [string]$EnvFile
)

$ErrorActionPreference = 'Stop'
$utf8 = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$env:PYTHONUTF8 = '1'
$projectRoot = Split-Path -Parent $PSScriptRoot

function Test-AgentPython {
    param([string]$Executable, [string[]]$Prefix = @())
    if (-not $Executable -or $Executable -match '(?i)[\\/]WindowsApps[\\/]') {
        return $false
    }
    try {
        & $Executable @Prefix -B -c 'import sys, venv; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' *> $null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function ConvertTo-AgentWslPath {
    param([string]$Value)
    if ($Value -match '^([A-Za-z]):[\\/](.*)$') {
        return '/mnt/' + $Matches[1].ToLowerInvariant() + '/' + $Matches[2].Replace('\', '/')
    }
    if ($Value.StartsWith('\\')) {
        throw 'Use a project-relative path or a local drive path for WSL.'
    }
    return $Value.Replace('\', '/')
}

$pythonCandidates = @(
    [PSCustomObject]@{
        Executable = Join-Path $projectRoot '.venv\Scripts\python.exe'
        Prefix = @()
    }
)
foreach ($commandName in @('python', 'python3', 'py')) {
    $command = Get-Command $commandName -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($command) {
        $prefix = @()
        if ($commandName -eq 'py') { $prefix = @('-3') }
        $pythonCandidates += [PSCustomObject]@{ Executable = $command.Source; Prefix = $prefix }
    }
}

$selectedPython = $null
foreach ($candidate in $pythonCandidates) {
    if (Test-AgentPython -Executable $candidate.Executable -Prefix $candidate.Prefix) {
        $selectedPython = $candidate
        break
    }
}

Push-Location -LiteralPath $projectRoot
try {
    $agentArguments = @('-B', '-m', 'reclab_agent.start')
    if ($Plan) { $agentArguments += '--plan' }
    if ($Web) { $agentArguments += @('--web', '--port', [string]$Port) }

    if ($selectedPython) {
        if ($Config) { $agentArguments += @('--config', $Config) }
        if ($EnvFile) { $agentArguments += @('--env-file', $EnvFile) }
        $pythonArguments = @($selectedPython.Prefix) + $agentArguments
        & $selectedPython.Executable @pythonArguments
        $agentExitCode = $LASTEXITCODE
    }
    else {
        $wslCommand = Get-Command wsl -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $wslCommand) {
            throw 'Python 3.10+ or WSL is required. Install either, then run start.cmd again.'
        }
        if ($Config) { $agentArguments += @('--config', (ConvertTo-AgentWslPath $Config)) }
        if ($EnvFile) { $agentArguments += @('--env-file', (ConvertTo-AgentWslPath $EnvFile)) }
        $agentArguments += '--windows-ssh'
        $wslArguments = @('--cd', $projectRoot, '--exec', '/usr/bin/python3') + $agentArguments
        & $wslCommand.Source @wslArguments
        $agentExitCode = $LASTEXITCODE
    }
}
catch {
    [Console]::Error.WriteLine('Unable to start Agent: ' + $_.Exception.Message)
    $agentExitCode = 1
}
finally {
    Pop-Location
}
exit $agentExitCode
