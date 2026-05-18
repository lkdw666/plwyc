$action = New-ScheduledTaskAction -Execute "C:\Users\凌枯大王\Desktop\排列五预测器\每日同步.bat"
$trigger = New-ScheduledTaskTrigger -Daily -At 22:00
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERNAME" -RunLevel Limited
Register-ScheduledTask -TaskName "排列五每日数据同步" -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description "每天22:00自动拉取最新一期排列五开奖数据" -Force
