@echo off
rem Lets you type  .\oc add 3  without changing PowerShell's script policy.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0oc.ps1" %*
