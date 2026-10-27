# Short commands for the OpenClinic demo. Run from the app folder:  .\oc help
param([string]$Cmd = "help", [string]$Arg = "")

Set-Location $PSScriptRoot
$env:Path += ";$env:APPDATA\Python\Python312\Scripts"
$sh = "C:\Program Files\Git\bin\sh.exe"
$mysql = "C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe"
$url = "http://127.0.0.1:8100"
$tokenFile = Join-Path $PSScriptRoot ".oc-token"
$py = @("run", "--with", "pymysql", "python")

function Get-Token {
    if ($env:ROBBY_API_TOKEN) { return $env:ROBBY_API_TOKEN }
    if (Test-Path $tokenFile) { return (Get-Content $tokenFile -Raw).Trim() }
    Write-Host "No token yet. Create one on the Integrations page, then run:  .\oc token rbk_..." -ForegroundColor Yellow
    exit 1
}

switch ($Cmd) {
    "token" {
        if (-not $Arg.StartsWith("rbk_")) { Write-Host "Usage: .\oc token rbk_..." -ForegroundColor Yellow; exit 1 }
        Set-Content -Path $tokenFile -Value $Arg -NoNewline
        Write-Host "Token saved (file .oc-token, not committed)."
    }
    "add" {
        $n = if ($Arg) { $Arg } else { "1" }
        uv @py scripts/openclinic_standin/seed.py add $n
    }
    "bad" { uv @py scripts/openclinic_standin/seed.py add 1 --bad }
    "fix" {
        & $mysql -h 127.0.0.1 -P 3307 -u root openclinic -e "update obstetric_record set presentation='Vertex' where presentation='sideways'; select row_count() as fixed"
    }
    "sync" {
        $env:ROBBY_API_TOKEN = Get-Token
        uv @py scripts/openclinic_standin/connector.py --url $url
    }
    "status" {
        $app = & curl.exe -s -o NUL -w "%{http_code}" "$url/login"
        $db = & $sh scripts/openclinic_standin/mysql.sh status 2>&1 | Select-Object -Last 1
        Write-Host "App ($url):      $(if ($app -eq '200') { 'running' } else { 'NOT running' })"
        Write-Host "OpenClinic DB:   $db"
        $count = & $mysql -h 127.0.0.1 -P 3307 -u root openclinic -N -e "select count(*) from encounter e join obstetric_record o using(encounter_id)" 2>$null
        if ($count) { Write-Host "Maternity admissions in OpenClinic: $count" }
    }
    "start" {
        & $sh scripts/openclinic_standin/mysql.sh start
        $app = & curl.exe -s -o NUL -w "%{http_code}" "$url/login"
        if ($app -ne "200") {
            Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$env:Path += ';$env:APPDATA\Python\Python312\Scripts'; Set-Location '$PSScriptRoot'; & '$sh' scripts/demo.sh 8100"
            Write-Host "Demo app starting in a new window (give it 15 seconds)."
        } else { Write-Host "Demo app already running." }
    }
    default {
        Write-Host @"
OpenClinic demo commands (run from the app folder):

  .\oc status      is everything running?
  .\oc start       start the OpenClinic database and the demo app
  .\oc add 3       add 3 new patients to OpenClinic (default 1)
  .\oc sync        send OpenClinic's admissions to the app
  .\oc bad         add one patient with a bad value (to show a rejection)
  .\oc fix         repair the bad value so sync works again
  .\oc token rbk_  save a feed token (once)

Demo flow:  .\oc add 3   then   .\oc sync   then look at Recent admissions.
"@
    }
}
